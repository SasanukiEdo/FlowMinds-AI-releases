# Irodori-TTS ブリッジ（FlowMinds://AI 用）

FlowMinds://AI から音声合成エンジン [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（Aratako 氏作）を使うための中継プログラムです。
Irodori-TTS 本体には FlowMinds とつなぐ仕組みが入っていないため、このフォルダの2ファイルを Irodori-TTS のフォルダに追加して使います。

- `bridge.py` … FlowMinds からの読み上げ依頼を受けて Irodori-TTS で音声を作る中継プログラム
- `start-bridge.bat` … 中継プログラムの起動用バッチ

対応モデル: **v4（既定・Irodori-TTS-v4-Small）** と v3（バッチ内の設定で切り替え）

最新版はこちらからダウンロードできます（3ファイルをまとめた ZIP）:
https://github.com/SasanukiEdo/FlowMinds-AI-releases/raw/main/irodori-tts-bridge/irodori-tts-bridge.zip
（ファイル単位で見る場合: https://github.com/SasanukiEdo/FlowMinds-AI-releases/tree/main/irodori-tts-bridge ）

## 必要なもの

- Windows 11
- **NVIDIA の GPU を推奨**（VRAM 6GB 以上が目安）。GPU がなくても動きますが、1回の読み上げに数十秒以上かかり実用的ではありません
- 空きディスク 15GB 程度（Python パッケージ＋モデル）
- [Git](https://git-scm.com/download/win)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)（Python の実行環境管理ツール。PowerShell で `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`）

## セットアップ

1. Irodori-TTS を取得する（例: `C:\Irodori-TTS`）
   - **ドライブ直下など浅い場所に置いてください**。深いフォルダ（デスクトップやダウンロードの奥など）に置くと、Windows のパス長制限で読み上げ時にエラーになることがあります
   ```
   git clone https://github.com/Aratako/Irodori-TTS.git C:\Irodori-TTS
   ```
2. このフォルダの `bridge.py` と `start-bridge.bat` を、取得した Irodori-TTS フォルダ（`irodori_tts` フォルダや `pyproject.toml` がある場所）にコピーする
3. NVIDIA の GPU がない場合だけ、`start-bridge.bat` をメモ帳で開いて `set IRODORI_EXTRA=cu128` を `set IRODORI_EXTRA=cpu` に書き換える
4. `start-bridge.bat` をダブルクリックして一度起動する
   - **初回は Python パッケージとモデル（合わせて数GB）のダウンロードがあるため、10〜30分ほどかかります**。黒い画面に `Uvicorn running on http://127.0.0.1:8080` と出れば起動完了です
   - モデル本体は最初の読み上げ時にダウンロードされます
5. FlowMinds の設定画面 →「音声設定」で
   - TTSプロバイダーに **Irodori-TTS** を選ぶ
   - 「起動バッチのパス」に `start-bridge.bat` の場所（例: `C:\Irodori-TTS\start-bridge.bat`）を指定する。これで次回からは、読み上げ時に未起動なら自動で立ち上がります
   - 「テスト再生」で声が出れば完了です

声の設定（声質キャプション・参照音声）はユーザーマニュアル「04_音声機能ガイド」を参照してください。

## v3 を使いたい場合

`start-bridge.bat` の次の2行の先頭の `REM ` を消してください。

```
REM set IRODORI_CHECKPOINT_IRODORI=Aratako/Irodori-TTS-500M-v3
REM set IRODORI_CHECKPOINT_VOICEDESIGN=Aratako/Irodori-TTS-600M-v3-VoiceDesign
```

v4 は短い参照音声1本での声の再現度が v3 より下がる傾向があります（モデル説明に記載）。参照音声を長めにする・複数用意すると改善しやすいです。

## うまくいかないとき

| 症状 | 対処 |
|---|---|
| `uv is not installed` と出る | uv をインストールし、PC を再起動（またはサインアウト）してから再実行 |
| `"irodori_tts" folder not found` と出る | 2ファイルを置く場所が違います。`irodori_tts` フォルダと同じ階層に置いてください |
| `WinError 1314` で止まる | バッチ経由で起動しているか確認してください（バッチが回避設定を入れています） |
| 読み上げ時に `No such file or directory: ...\.venv\Lib\site-packages\...` と出る | パスが長すぎます。Irodori-TTS フォルダを `C:\Irodori-TTS` など浅い場所に移し、中の `.venv` フォルダを削除してから再起動してください |
| 読み上げが極端に遅い | GPU 版の PyTorch が入っていない可能性があります。`IRODORI_EXTRA=cu128` になっているか確認 |
| FlowMinds のテスト再生がタイムアウトする | 初回はセットアップに時間がかかります。先に手順4でバッチを手動起動し、起動完了を確認してから試してください |

## 利用上の注意

Irodori-TTS の利用規約上、**声優・有名人・公人の無許諾の声のクローンは禁止**されています。参照音声は自分の声、VOICEVOX 等で作った合成音声、許諾を得た音源、CC0 音源などを使ってください。

Irodori-TTS 本体は MIT ライセンスです（各モデルのライセンスは Hugging Face のモデルページを確認してください）。

このフォルダの3ファイル（`bridge.py`・`start-bridge.bat`・`README.md`）は、Irodori-TTS と組み合わせて使う目的で、自由にコピー・改変してかまいません（FlowMinds://AI 本体の利用規約とは別扱いです）。
