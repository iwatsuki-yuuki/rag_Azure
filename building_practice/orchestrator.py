import os
import pandas as pd
from openai import AzureOpenAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv(verbose=True)

AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT") # Azure OpenAI Serviceのエンドポイント
AOAI_API_VERSION = os.environ.get("AOAI_API_VERSION") # Azure OpenAI ServiceのAPIバージョン
AOAI_API_KEY = os.environ.get("AOAI_API_KEY") # Azure OpenAI ServiceのAPIキー
AOAI_CHAT_MODEL_NAME = os.environ.get("AOAI_CHAT_MODEL_NAME") # Azure OpenAI Serviceのチャットモデル名

# CSVデータの読み込みとマージ
script_dir = os.path.dirname(os.path.abspath(__file__))
df_question = pd.read_csv(os.path.join(script_dir, "hanahira-question.csv"), index_col=0)
df_data = pd.read_csv(os.path.join(script_dir, "hanahira-data.csv"), index_col=0)
df_merged = df_question.merge(df_data[["答え"]], left_index=True, right_index=True)

# 全Q&Aデータを Sources 形式の文字列に整形
sources_text = "\n".join(
    f"[Source{idx}]: カテゴリ: {row['カテゴリ']} / 質問: {row['質問内容']} / 答え: {row['答え']}"
    for idx, row in df_merged.iterrows()
)

system_message_chat_conversation = """
あなたはユーザーの質問に回答するチャットボットです。
回答については、「Sources:」以下に記載されている内容に基づいて回答してください。回答は簡潔にしてください。
「Sources:」に記載されている情報以外の回答はしないでください。
情報が複数ある場合は「Sources:」のあとに[Source1]、[Source2]、[Source3]のように記載されますので、それに基づいて回答してください。
また、ユーザーの質問に対して、Sources:以下に記載されている内容に基づいて適切な回答ができない場合は、「すみません。わかりません。」と回答してください。
回答の中に情報源の提示は含めないでください。例えば、回答の中に「[Source1]」や「Sources:」という形で情報源を示すことはしないでください。
"""

def search(history):
    question = history[-1].get('content')

    # Azure OpenAI ServiceのAPIに接続するためのクライアントを生成する
    openai_client = AzureOpenAI(
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        api_version=AOAI_API_VERSION
    )

    # チャット履歴の中からユーザーの質問に対する回答を生成するためのメッセージを生成する。
    messages = []

    # 先頭にAIのキャラ付けを行うシステムメッセージを追加する。
    messages.insert(0, {"role": "system", "content": system_message_chat_conversation})

    # ユーザーの質問とCSVから読み込んだ全Q&Aデータを含むメッセージを生成する。
    user_message = """
    {query}

    Sources:
    {source}
    """.format(query=question, source=sources_text)

    # メッセージを追加する。
    messages.append({"role": "user", "content": user_message})

    # Azure OpenAI Serviceに回答生成を依頼する。
    response = openai_client.chat.completions.create(
        model=AOAI_CHAT_MODEL_NAME,
        messages=messages
    )
    answer = response.choices[0].message.content

    # 回答を返す。
    return answer

# ここからは画面を構築するためのコード
# チャット履歴を初期化する。
if "history" not in st.session_state:
    st.session_state["history"] = []

# チャット履歴を表示する。
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ユーザーが質問を入力したときの処理を記述する。
if prompt := st.chat_input("質問を入力してください"):

    # ユーザーが入力した質問を表示する。
    with st.chat_message("user"):
        st.write(prompt)

    # ユーザの質問をチャット履歴に追加する
    st.session_state.history.append({"role": "user", "content": prompt})

    # ユーザーの質問に対して回答を生成するためにsearch関数を呼び出す。
    response = search(st.session_state.history)

    # 回答を表示する。
    with st.chat_message("assistant"):
        st.write(response)

    # 回答をチャット履歴に追加する。
    st.session_state.history.append({"role": "assistant", "content": response})
