import os
import sys
import csv
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv

load_dotenv(verbose=True)

SEARCH_SERVICE_ENDPOINT = os.environ.get("SEARCH_SERVICE_ENDPOINT") # Azure AI Searchのエンドポイント
SEARCH_SERVICE_API_KEY = os.environ.get("SEARCH_SERVICE_API_KEY") # Azure AI SearchのAPIキー
SEARCH_SERVICE_INDEX_NAME = os.environ.get("SEARCH_SERVICE_INDEX_NAME") # Azure AI Searchのインデックス名
AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT") # Azure OpenAI Serviceのエンドポイント
AOAI_API_VERSION = os.environ.get("AOAI_API_VERSION") # Azure OpenAI ServiceのAPIバージョン
AOAI_API_KEY = os.environ.get("AOAI_API_KEY") # Azure OpenAI ServiceのAPIキー
AOAI_EMBEDDING_MODEL_NAME = os.environ.get("AOAI_EMBEDDING_MODEL_NAME") # Azure OpenAI Serviceの埋め込みモデル名
AOAI_CHAT_MODEL_NAME = os.environ.get("AOAI_CHAT_MODEL_NAME") # Azure OpenAI Serviceのチャットモデル名

# AIのキャラクターを決めるためのシステムメッセージを定義する。
system_message_chat_conversation = """
あなたはユーザーの質問に回答するチャットボットです。
回答については、「Sources:」以下に記載されている内容に基づいて回答してください。回答は簡潔にしてください。
「Sources:」に記載されている情報以外の回答はしないでください。
情報が複数ある場合は「Sources:」のあとに[Source1]、[Source2]、[Source3]のように記載されますので、それに基づいて回答してください。
また、ユーザーの質問に対して、Sources:以下に記載されている内容に基づいて適切な回答ができない場合は、「すみません。わかりません。」と回答してください。
回答の中に情報源の提示は含めないでください。例えば、回答の中に「[Source1]」や「Sources:」という形で情報源を示すことはしないでください。
"""

# ユーザーの質問に対して回答を生成するための関数を定義する。
# 引数はチャット履歴を表すJSON配列とする。
def search(history):
    question = history[-1].get('content')

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
        input = question,
        model = AOAI_EMBEDDING_MODEL_NAME
    )

    # ベクトル化された質問をAzure AI Searchに対して検索するためのクエリを生成する。
    vector_query = VectorizedQuery(
        vector=response.data[0].embedding,
        k_nearest_neighbors=3,
        fields="contentVector"
    )

    # ベクトル化された質問を用いて、Azure AI Searchに対してベクトル検索を行う。
    results = search_client.search(
        vector_queries=[vector_query],
        select=['id', 'content'])

    # チャット履歴の中からユーザーの質問に対する回答を生成するためのメッセージを生成する。
    messages = []

    # 先頭にAIのキャラ付けを行うシステムメッセージを追加する。
    messages.insert(0, {"role": "system", "content": system_message_chat_conversation})

    # 回答を生成するためにAzure AI Searchから取得した情報を整形する。
    sources = ["[Source" + result["id"] + "]: " + result["content"] for result in results]
    source = "\n".join(sources)

    # ユーザーの質問と情報源を含むメッセージを生成する。
    user_message = """
    {query}

    Sources:
    {source}
    """.format(query=question, source=source)

    # メッセージを追加する。
    messages.append({"role": "user", "content": user_message})

    # Azure OpenAI Serviceに回答生成を依頼する。
    response = openai_client.chat.completions.create(
        model=AOAI_CHAT_MODEL_NAME,
        messages=messages
    )
    answer = response.choices[0].message.content

    # 回答を返す。
    return answer, sources

# ユーザーの質問を読み込むための関数を定義する。
def load_questions(file_path):
    questions = []  # 質問と期待する回答を格納するリスト
    # ファイルを指定されたモードとエンコーディングで開く
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)  # CSVを辞書形式で読み込む
        # 各行の 'question' 列と 'ground_truth' 列の値をリストに追加していく
        for row in reader:
            questions.append((row['question'], row['ground_truth']))
    
    # 質問と期待する回答のリストを返す
    return questions

# ユーザーの質問に対して回答を生成し、
# その回答と情報源を含むコンテキストを生成するための関数を定義する。
def generate_evaluation_dataset(questions):
    # evaluation_dataset.csvというファイルを新規作成または上書きして開く
    with open('evaluation_dataset.csv', 'w', newline='', encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # CSVライターを作成、すべての項目をダブルクオーテーションで囲む
        writer.writerow(['query', 'response', 'context','ground_truth'])  # ヘッダ行をCSVに書き込む（質問、回答、コンテキスト、期待する回答(ground_truth)）
        
        # 質問ごとに処理を行う
        for question, ground_truth in questions:
            history = [{"role": "user", "content": question}]  # 質問を履歴として保持
            response, context = search(history)  # search関数を呼び出して回答とコンテキストを取得
            writer.writerow([question, response.replace('\n', ' '), ' '.join(context).replace('\n', ' '), ground_truth])  # CSVファイルに質問、回答、コンテキストを書き込む

if __name__ == "__main__":
    # プログラムの引数からCSVファイルのパスを取得する
    csv_file_path = sys.argv[1]

    # ユーザーの質問と期待する回答(ground_truth)を読み込む
    questions = load_questions(csv_file_path)

    # ユーザーの質問に対して回答を生成し、その回答と情報源を含むコンテキストを生成する
    generate_evaluation_dataset(questions)

