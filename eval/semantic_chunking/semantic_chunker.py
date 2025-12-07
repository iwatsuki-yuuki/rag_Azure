from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

# サンプルドキュメント（このテキストを各手法でチャンク化する）
document = """
# はじめに
このドキュメントは、テキストを分割するためのサンプルとして使用されています。この文章は、複数の段落やセクションで構成されています。目的は、異なる分割方法によって得られる結果を示すことです。

# 世界の歴史
世界の歴史は非常に広範で、古代から現代に至るまで様々な時代があります。たとえば、古代エジプト文明は紀元前3000年頃に始まり、長い歴史を持っています。また、古代ギリシャやローマの文明も重要な役割を果たしました。その後、中世ヨーロッパにおける封建制度、そして近代における産業革命など、様々な時代にわたる発展がありました。

## 古代文明
古代文明の中でも、特にメソポタミア文明とエジプト文明が重要です。これらの文明は農業、建築、そして宗教において大きな影響を与えました。

## 中世
中世は西暦500年頃から1500年頃までの期間を指します。この時期には、封建制度や騎士文化が栄え、特にヨーロッパではキリスト教の影響が強まりました。

# りんごの栽培
りんごは、世界中で栽培されている果物の一つです。りんごの栽培には、土壌、気候、水などの条件が重要です。また、りんごの品種によっても栽培方法が異なります。

## りんごの種類
りんごにはさまざまな種類があります。代表的な品種には、ふじ、さんふじ、王林などがあります。
"""

# 文字数ベースでのチャンク化を行う関数
def chunk_by_character(document):
    # 複数の区切り文字（改行、句読点、スペースなど）を設定
    separator = ["\n\n", "\n", "。", "、", " ", ""]
    # RecursiveCharacterTextSplitterを使って文字数100ごとに分割
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        separators=separator
    )
    
    # チャンク化を実行し、結果を保存
    char_chunks = char_splitter.split_text(document)
    
    print("Character-based chunks:")
    # 各チャンクを順番に表示
    for i, chunk in enumerate(char_chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")

# Markdownヘッダーに基づいてチャンク化を行う関数
def chunk_by_markdown(document):
    # チャンクを分割する際に使うMarkdownヘッダーを指定
    headers_to_split_on = [
        ("#", "Header 1"),  # ヘッダー1で分割
        ("##", "Header 2")  # ヘッダー2で分割
    ]
    
    # MarkdownHeaderTextSplitterを使用してチャンク化
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # チャンク化を実行し、結果を保存
    markdown_chunks = markdown_splitter.split_text(document)

    print("Markdown-based chunks:")
    # 各チャンクを順番に表示
    for i, chunk in enumerate(markdown_chunks):
        print(f"Chunk {i+1}:\n{chunk.page_content}\n")

# セマンティックチャンク化を行う関数（LLMを使用）
def chunk_by_semantics(document):
    # .envファイルから環境変数を読み込む
    load_dotenv(verbose=True)
    # Azure OpenAI Embeddingsを使用してセマンティックに基づいたチャンク化を設定
    text_splitter = SemanticChunker(
        AzureOpenAIEmbeddings(model="text-embedding-ada-002"),
        sentence_split_regex=r"。|\n"
    )
    
    # ドキュメントをセマンティックベースで分割し、結果を保存
    docs = text_splitter.create_documents([document])

    print("Semantic chunks:")
    # 各チャンクを順番に表示
    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}:\n{doc.page_content}\n")

# メイン関数
if __name__ == "__main__":
    chunk_by_character(document)  # 文字数ベースでチャンク化
    
    chunk_by_markdown(document)  # Markdownヘッダーでチャンク化
    
    chunk_by_semantics(document)  # セマンティックチャンク化

