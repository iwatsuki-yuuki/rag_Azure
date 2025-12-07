import os
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity 
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む。
load_dotenv(verbose=True)

# 環境変数から各種Azureリソースへの接続情報を取得する。
AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT") # Azure OpenAI Serviceのエンドポイント
AOAI_API_VERSION = os.environ.get("AOAI_API_VERSION") # Azure OpenAI ServiceのAPIバージョン
AOAI_API_KEY = os.environ.get("AOAI_API_KEY") # Azure OpenAI ServiceのAPIキー
AOAI_EMBEDDING_MODEL_NAME = os.environ.get("AOAI_EMBEDDING_MODEL_NAME") # Azure OpenAI Serviceの埋め込み用APIのモデル名
AOAI_CHAT_MODEL_NAME = os.environ.get("AOAI_CHAT_MODEL_NAME") # Azure OpenAI Serviceのチャット用APIのモデル名

# HyDEを検証するためのサンプルドキュメント
document = """
古代エジプト文明は、紀元前3000年頃に始まり、ピラミッドの建設やヒエログリフの使用で知られています。
特にギザの大ピラミッドは、世界七不思議の一つとして有名です。
"""

# ドキュメントに対する質問
question = "古代エジプト文明で有名な建築物は何ですか？"

# Azure OpenAI ServiceのAPIに接続するためのクライアントを生成する。
openai_client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY,
    api_version=AOAI_API_VERSION
)

# 質問をベクトル化する。
vectorized_question = openai_client.embeddings.create(
    input = question,
    model = AOAI_EMBEDDING_MODEL_NAME
)

# 質問から仮の回答を生成するためのプロンプトを生成する。
user_message = f"""Please write a passage to answer the question
Question: {question}
Passage:
"""

messages = [
    {
        "role": "system",
        "content": "you are a chatbot that answers user questions."
    },
    {
        "role": "user",
        "content": user_message
    }
]

# LLMを使って仮の回答を生成する。
hypothetical_answer = openai_client.chat.completions.create(
    model=AOAI_CHAT_MODEL_NAME,
    messages=messages
)

# 仮の回答をベクトル化する。
vectorized_hypothetical_answer = openai_client.embeddings.create(
    input = hypothetical_answer.choices[0].message.content,
    model = AOAI_EMBEDDING_MODEL_NAME
)

# ドキュメントをベクトル化する。
vectorized_document = openai_client.embeddings.create(
    input = document,
    model = AOAI_EMBEDDING_MODEL_NAME
)

# ベクトル化された質問とベクトル化されたドキュメントのコサイン類似度を計算する。
similarity1 = cosine_similarity(
    [vectorized_question.data[0].embedding],
    [vectorized_document.data[0].embedding]
)

# ベクトル化された仮の回答とベクトル化されたドキュメントのコサイン類似度を計算する。
similarity2 = cosine_similarity(
    [vectorized_hypothetical_answer.data[0].embedding],
    [vectorized_document.data[0].embedding]
)

# 結果を出力する。
print(f"ベクトル化された質問とベクトル化されたドキュメントのコサイン類似度: {similarity1[0][0]}")
print(f"ベクトル化された仮の回答とベクトル化されたドキュメントのコサイン類似度: {similarity2[0][0]}")