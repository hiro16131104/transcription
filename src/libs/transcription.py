import json
import math
import mimetypes
import os
from datetime import timedelta
from typing import Iterable

import pandas as pd
import torch
import torchaudio
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from torch import Tensor


class Transcription:
    """
    メディアファイル（音声又は動画）から文字起こしや話者分離を行うクラス

    Attributes:
        media_file_path (str): メディアファイルのパス
        transcriptions (list[dict]): 文字起こし結果を格納するリスト
        speaker_segments (list[dict]): 話者分離の結果を格納するリスト
        merged_results (list[dict]): 文字起こしと話者分離の結果を結合した後に格納するリスト
    """

    def __init__(self, media_file_path: str) -> None:
        """
        クラスの初期化メソッド

        Args:
            media_file_path (str): メディアファイルのパス
        """
        self.media_file_path: str = media_file_path
        self.transcriptions: list[dict] = []
        self.speaker_segments: list[dict] = []
        self.merged_results: list[dict] = []

    def is_video(self) -> bool:
        """
        メディアファイルが動画か否か判定する

        Returns:
            bool: 動画ファイルであればTrue、そうでなければFalse
        """
        mime_type: str | None = mimetypes.guess_type(self.media_file_path)[0]

        if not mime_type:
            return False

        return mime_type.startswith("video")

    def convert_video_to_audio(self) -> None:
        """
        動画ファイルを音声ファイルに変換し、mp3形式で保存する
        """
        video: VideoFileClip = VideoFileClip(self.media_file_path)
        output_path: str = os.path.splitext(self.media_file_path)[0] + ".mp3"

        video.audio.write_audiofile(output_path)
        self.media_file_path = output_path

    def transcribe_audio(self, model_size: str = "medium") -> None:
        """
        音声ファイルを文字起こしする

        Args:
            model_size (str): Whisperモデルのサイズ（"tiny", "base", "small", "medium", "large"）
        """
        device: str = ""
        compute_type: str = ""
        model: WhisperModel = None
        segments: Iterable[Segment] = None

        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"

        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, _ = model.transcribe(self.media_file_path, language="ja")
        self.transcriptions = []

        for segment in segments:
            item: dict = {
                "start_time": segment.start,
                "end_time": segment.end,
                "text": segment.text,
            }
            self.transcriptions.append(item)

    def diarize_audio(self, hugging_face_token: str) -> None:
        """
        音声ファイルを話者分離する
        モデルを認証する際にHugging Faceのトークンが必要になる
        トークンの発行方法
        1. HuggingFace（https://huggingface.co/）のアカウントを作る
        2. pyannoteのモデルの利用申請を行う
          - https://huggingface.co/pyannote/speaker-diarization-3.1
          - https://huggingface.co/pyannote/segmentation-3.0
        3. アクセストークンを発行する（https://huggingface.co/settings/tokens）

        Args:
            hugging_face_token (str): HuggingFaceのアクセストークン
        """
        pipeline: Pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hugging_face_token
        )
        diarization: Annotation = None

        if torch.cuda.is_available():
            audio: tuple[Tensor, int] = torchaudio.load(self.media_file_path)

            pipeline.to(torch.device("cuda"))
            diarization = pipeline({"waveform": audio[0], "sample_rate": audio[1]})
        else:
            diarization = pipeline(self.media_file_path)

        self.speaker_segments = []

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            item: dict = {
                "start_time": segment.start,
                "end_time": segment.end,
                "speaker": speaker,
            }
            self.speaker_segments.append(item)

    def __format_seconds_to_hhmmss(self, seconds: float) -> str:
        """
        秒数の小数点以下を切り捨て、"hh:mm:ss"形式の文字列に変換する

        Args:
            seconds (float): 秒数。

        Returns:
            str: "hh:mm:ss" 形式の文字列
        """
        temp_seconds: int = math.floor(seconds)
        return str(timedelta(seconds=temp_seconds))

    def merge_results(self) -> None:
        """
        文字起こしと話者分離の結果を時間軸に基づいて結合する
        """
        i: int = 0
        j: int = 0

        self.merged_results = []

        while i < len(self.transcriptions) and j < len(self.speaker_segments):
            tr_start: float = float(self.transcriptions[i]["start_time"])
            tr_end: float = float(self.transcriptions[i]["end_time"])
            sp_start: float = float(self.speaker_segments[j]["start_time"])
            sp_end: float = float(self.speaker_segments[j]["end_time"])

            if tr_start < sp_end and tr_end > sp_start:
                item: dict = {
                    "start_time": self.__format_seconds_to_hhmmss(tr_start),
                    "end_time": self.__format_seconds_to_hhmmss(tr_end),
                    "speaker": self.speaker_segments[j]["speaker"],
                    "text": self.transcriptions[i]["text"],
                }

                self.merged_results.append(item)
                i += 1
            elif tr_end <= sp_start:
                i += 1
            else:
                j += 1

    def export_results_to_csv(self, file_path: str, encoding: str = "utf-8") -> None:
        """
        結合した結果をCSV形式で保存する

        Args:
            file_path (str): 保存先のファイルパス
            encoding (str): 保存する際の文字コード
        """
        df: pd.DataFrame = pd.DataFrame(self.merged_results)
        df.to_csv(file_path, index=False, encoding=encoding)

    def export_results_to_json(self, file_path: str, encoding: str = "utf-8") -> None:
        """
        結合した結果をJSON形式で保存する

        Args:
            file_path (str): 保存先のファイルパス
            encoding (str): 保存する際の文字コード
        """
        with open(file_path, "w", encoding=encoding) as file:
            json.dump(self.merged_results, file, indent=4, ensure_ascii=False)

    def __format_values_to_md_table_row(self, values: list[str]) -> str:
        """
        リストの値をMarkdownのテーブル行の形式に変換する

        Args:
            values (list[str]): テーブル行に含める値のリスト

        Returns:
            str: 変換したMarkdownテーブル行
        """
        return f"| {' | '.join(values)} |"

    def export_results_to_md(self, file_path: str, encoding: str = "utf-8") -> None:
        """
        結合した結果をMarkdown形式で保存する

        Args:
            file_path (str): 保存先のファイルパス
            encoding (str): 保存する際の文字コード
        """
        col_names: list[str] = self.merged_results[0].keys()
        separators: list[str] = ["---"] * len(col_names)
        header_row: str = self.__format_values_to_md_table_row(col_names)
        separator_row: str = self.__format_values_to_md_table_row(separators)
        rows: list[str] = [header_row, separator_row]

        for merged_result in self.merged_results:
            row: str = ""
            values: list[str] = []

            for col_name in col_names:
                values.append(str(merged_result[col_name]))

            row = self.__format_values_to_md_table_row(values)
            rows.append(row)

        with open(file_path, "w", encoding=encoding) as file:
            file.write("\n".join(rows))
