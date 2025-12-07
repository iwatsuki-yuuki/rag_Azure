# rag_Azure

Azure OpenAI と Azure AI Search を使った RAG サンプルです。PDF をインデックス化し、Streamlit でチャット UI を動かします。

## 前提
- Python 3.10 以上
- Azure OpenAI（埋め込み・チャットモデル）と Azure AI Search のリソース
- 上記で作成したインデックス（`id`、`content`、`contentVector` などを持つベクトルインデックス）

## セットアップ
1. 依存関係をインストールします。
   - `pip install -r building_practice/requirements.txt`
2. `.env` に環境変数を設定します（`building_practice/.env` を参考）。
   - `SEARCH_SERVICE_ENDPOINT`
   - `SEARCH_SERVICE_API_KEY`
   - `SEARCH_SERVICE_INDEX_NAME`
   - `AOAI_ENDPOINT`
   - `AOAI_API_VERSION`
   - `AOAI_API_KEY`
   - `AOAI_EMBEDDING_MODEL_NAME`
   - `AOAI_CHAT_MODEL_NAME`

## ドキュメントのインデックス化
1. PDF などのソースを用意します。
2. 次のコマンドでテキスト抽出と分割、ベクトル化、AI Search へのアップロードを実行します。
   - `python building_practice/indexer.py <PDFのパス>`

## チャットの起動
- RAG チャットを立ち上げるには、以下を実行してブラウザで開きます。
  - `streamlit run building_practice/orchestrator.py`
  - 画面に表示されるプロンプトから質問を入力すると、インデックス化したドキュメントを根拠に回答します。
