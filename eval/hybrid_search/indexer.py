import os
import wikipedia
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from dotenv import load_dotenv
import uuid

# .envファイルから環境変数を読み込む。
load_dotenv(verbose=True)

# 環境変数から各種Azureリソースへの接続情報を取得する。
SEARCH_SERVICE_ENDPOINT = os.environ.get("SEARCH_SERVICE_ENDPOINT") # Azure AI Searchのエンドポイント
SEARCH_SERVICE_API_KEY = os.environ.get("SEARCH_SERVICE_API_KEY") # Azure AI SearchのAPIキー
SEARCH_SERVICE_INDEX_NAME = os.environ.get("SEARCH_SERVICE_INDEX_NAME") # Azure AI Searchのインデックス名
AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT") # Azure OpenAI Serviceのエンドポイント
AOAI_API_VERSION = os.environ.get("AOAI_API_VERSION") # Azure OpenAI ServiceのAPIバージョン
AOAI_API_KEY = os.environ.get("AOAI_API_KEY") # Azure OpenAI ServiceのAPIキー
AOAI_EMBEDDING_MODEL_NAME = os.environ.get("AOAI_EMBEDDING_MODEL_NAME") # Azure OpenAI Serviceの埋め込みモデル名

# Azure AI SearchのAPIに接続するためのクライアントを生成する。
searchClient = SearchClient(
    endpoint=SEARCH_SERVICE_ENDPOINT,
    index_name=SEARCH_SERVICE_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_SERVICE_API_KEY)
)

# Azure OpenAIのAPIに接続するためのクライアントを生成する。
openAIClient = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY,
    api_version = AOAI_API_VERSION
)

# チャンクを生成する。
def create_chunk(title, chunk_size, chunk_overlap, output_dir='data', lang='ja'):
    # Wikipediaページの取得
    wikipedia.set_lang(lang)

    page = wikipedia.page(title)
    text = page.content

    # テキストをチャンクに分割
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base',
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)

    return chunks

# チャンクをAzure AI Searchに登録する。
def index_docs(title: str, chunk: str):
    # 引数によって渡されたチャンクのリストをベクトル化する。
    response = openAIClient.embeddings.create(
        input = chunk,
        model = AOAI_EMBEDDING_MODEL_NAME
    )

    # チャンクのテキストと、そのチャンクをベクトル化したものをAzure AI Searchに登録する。
    document = {
        "id": str(uuid.uuid4()), 
        "title": title,
        "content": chunk, 
        "contentVector": response.data[0].embedding
    }
    searchClient.upload_documents([document])

characters = [
    "ウィリアム・シェイクスピア",
    "ジョン・ウェブスター",
    "トマス・ダーフィー",
    "ベン・ジョンソン (詩人)",
    "マーガレット・キャヴェンディッシュ",
    "アフラ・ベーン",
    "トマス・ミドルトン",
    "ジョン・ミルトン",
    "ジョン・リリー",
    "ジョン・ドライデン",
    "トマス・ダーフィー",
    "ジョン・ゲイ",
]

# チャンクサイズとオーバーラップを設定
chunk_size = 1000  # チャンクサイズ
chunk_overlap = 50  # チャンクのオーバーラップ

for character in characters:
    chunks = create_chunk(character, chunk_size, chunk_overlap)
    for i, chunk in enumerate(chunks):
        index_docs(f"{character}_{i:02}", chunk)
