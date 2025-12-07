import os
import sys
from enum import Enum
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv

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
AOAI_CHAT_MODEL_NAME = os.environ.get("AOAI_CHAT_MODEL_NAME") # Azure OpenAI Serviceのチャットモデル名

def search(query: str, type: str):
    # Azure AI SearchのAPIに接続するためのクライアントを生成する
    search_client = SearchClient(
        endpoint=SEARCH_SERVICE_ENDPOINT,
        index_name=SEARCH_SERVICE_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_SERVICE_API_KEY)
    )

    # Azure OpenAI ServiceのAPIに接続するためのクライアントを生成する
    openai_client = AzureOpenAI(
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        api_version=AOAI_API_VERSION
    )

    # Azure OpenAI Serviceの埋め込み用APIを用いて、ユーザーからの質問をベクトル化する。
    response = openai_client.embeddings.create(
        input = query,
        model = AOAI_EMBEDDING_MODEL_NAME
    )

    # ベクトル化された質問をAzure AI Searchに対して検索するためのクエリを生成する。
    vector_query = VectorizedQuery(
        vector=response.data[0].embedding,
        k_nearest_neighbors=10,
        fields="contentVector"
    )

    if type == "keyword":
        results = search_client.search(
            search_text = query,
            select=['title', 'content'],
            top=10
        )
    elif type == "vector":
        results = search_client.search(
            vector_queries=[vector_query],
            select=['title', 'content'],
            top=10
        )
    elif type == "hybrid":
        results = search_client.search(
            search_text = query,
            vector_queries=[vector_query],
            select=['title', 'content'],
            top=10
        )
    
    return results

if __name__ == "__main__":
    query = "ロミオとじゅりえっとの作者は？"

    results = search(query, sys.argv[1])

    for i, result in enumerate(results, start=1):
        print(f"Rank: {i}")
        print(f"Score: {result['@search.score']}")
        print(f"Title: {result['title']}")
        print(f"Content: {result['content']}")
        print("\n---------------------------------------------------------\n")

