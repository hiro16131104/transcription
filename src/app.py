import os
from datetime import datetime

from dotenv import load_dotenv

from libs.transcription import Transcription

load_dotenv()

HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

if __name__ == "__main__":
    prompt: str = "音声ファイル又は動画ファイルの名称を入力してください（拡張子あり）: "
    media_dir_path: str = "./assets/media"
    media_file_name: str = input(prompt)
    media_file_path: str = f"{media_dir_path}/{media_file_name}"
    export_dir_path: str = "./assets/texts"
    export_file_path: str = ""
    prefix: str = ""
    transcription: Transcription = Transcription(media_file_path)
    is_video: bool = transcription.is_video()

    if is_video:
        transcription.convert_video_to_audio()

    transcription.transcribe_audio("large-v3")
    transcription.diarize_audio(HUGGING_FACE_TOKEN)
    transcription.merge_results()

    prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
    export_file_path = f"{export_dir_path}/{prefix}_result"

    transcription.export_results_to_csv(f"{export_file_path}.csv")
    transcription.export_results_to_json(f"{export_file_path}.json")
    transcription.export_results_to_md(f"{export_file_path}.md")

    if is_video:
        os.remove(transcription.media_file_path)

    print("処理が完了しました")
