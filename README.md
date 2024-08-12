## 目次
- [目次](#目次)
- [利用者向け](#利用者向け)
  - [概要](#概要)
  - [前提](#前提)
  - [検証環境](#検証環境)
  - [事前準備](#事前準備)
  - [仮想環境を構築し、起動する](#仮想環境を構築し起動する)
  - [Pythonスクリプトを実行する](#pythonスクリプトを実行する)
  - [仮想環境を終了する](#仮想環境を終了する)
- [開発者向け（自分用メモ）](#開発者向け自分用メモ)
  - [プロジェクトを作成し、git管理にする](#プロジェクトを作成しgit管理にする)
  - [仮想環境を作成する](#仮想環境を作成する)
  - [仮想環境を起動し、パッケージ管理ツールをアップデートする](#仮想環境を起動しパッケージ管理ツールをアップデートする)
  - [フォーマッター、リンター等をインストールする](#フォーマッターリンター等をインストールする)
  - [Pythonファイル、その他の設定を作成する](#pythonファイルその他の設定を作成する)
  - [blackと競合しないようにflake8の設定を変更する](#blackと競合しないようにflake8の設定を変更する)
  - [isortの設定を変更する](#isortの設定を変更する)
  - [その他の依存パッケージを追加、削除する](#その他の依存パッケージを追加削除する)
  - [フォーマッター、リンター等を使用する](#フォーマッターリンター等を使用する)
  - [Pythonファイルを実行する](#pythonファイルを実行する)
  - [仮想環境を終了する](#仮想環境を終了する-1)

<br>

## 利用者向け

### 概要
- 音声ファイルや動画ファイルが対象
- OpenAIが提供しているWhisperモデルを使って文字起こしをする
- pyannoteモデルを使って話者分離（話者認識）をする
- 文字起こしと話者分離の結果を結合する
- 結合した結果をCSV、JSON、Markdown形式で保存する

### 前提
- Pythonがインストール済みである
- Poetry（Pythonのパッケージ管理ツール）がインストール済みである

### 検証環境
- macOS version14.5

### 事前準備
- "./assets/media"にメディアファイル（音声又は動画）を保存する
- ルートディレクトリに.envファイルを作成し、HuggingFaceのトークンを記載する（トークンの発行方法は以下のとおり）
  1. HuggingFace（https://huggingface.co/）のアカウントを作る
  2. pyannoteのモデルの利用申請を行う
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/segmentation-3.0
  3. アクセストークンを発行する（https://huggingface.co/settings/tokens）
  ```
  HUGGING_FACE_TOKEN="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  ```

### 仮想環境を構築し、起動する
```
poetry install
poetry shell
```

### Pythonスクリプトを実行する
```
python ./src/app.py
```

### 仮想環境を終了する
```
exit
```

<br>

## 開発者向け（自分用メモ）
### プロジェクトを作成し、git管理にする
```
mkdir <project-name>
cd ./<project-name>
git init
```

### 仮想環境を作成する
```
poetry init
poetry config virtualenvs.in-project true
poetry env use 3.11
poetry install
```

### 仮想環境を起動し、パッケージ管理ツールをアップデートする
```
poetry shell
poetry self update
```

### フォーマッター、リンター等をインストールする
```
poetry add black flake8 isort
```

### Pythonファイル、その他の設定を作成する
```
mkdir src
touch ./src/app.py
touch .flake8 .env .gitignore
```

### blackと競合しないようにflake8の設定を変更する
``` 
# .flake8に記載する
[flake8]
max-line-length = 88
extend-ignore = E203
```

### isortの設定を変更する
```
# pyproject.tomlに追記する
[tool.isort]
profile = "black"
```

### その他の依存パッケージを追加、削除する
```
poetry add <package-name>
poetry remove <package-name>
```

### フォーマッター、リンター等を使用する
```
black ./src/
isort ./src/
flake8 ./src/
```

### Pythonファイルを実行する
```
python ./src/app.py
```

### 仮想環境を終了する
```
exit
```
