import os
import streamlit as st
import openai
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlitページ設定（スクリプトの最初のStreamlitコマンドとして配置）
st.set_page_config(
    page_title="ドキュメント検索",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# カスタムCSS
st.markdown("""
<style>
    body {
        color: #a388ee;
        background-color: #ffffff;
    }
    .main {
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .stApp {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #ffffff;
    }
    .search-container {
        display: flex;
        margin-bottom: 20px;
    }
    .search-input {
        flex-grow: 1;
        padding: 10px;
        font-size: 16px;
        border: 2px solid #ffd1dc;
        border-radius: 24px;
        outline: none;
        background-color: #fffafa;
    }
    .search-button {
        margin-left: 10px;
        padding: 10px 20px;
        font-size: 16px;
        background-color: #ffd1dc;
        border: none;
        border-radius: 24px;
        color: #ffffff;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .search-button:hover {
        background-color: #ffb6c1;
        transform: scale(1.05);
    }
    .result-item {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 15px;
        background-color: #f0f8ff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .result-title {
        color: #ff69b4;
        font-size: 20px;
        margin-bottom: 5px;
    }
    .result-url {
        color: #20b2aa;
        font-size: 14px;
        margin-bottom: 5px;
    }
    .result-snippet {
        color: #778899;
        font-size: 16px;
    }
    .highlight {
        background-color: #fffacd;
        padding: 2px 4px;
        border-radius: 4px;
    }
    .stButton>button {
        background-color: #ffd1dc;
        color: white;
        border: none;
        border-radius: 24px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ffb6c1;
        transform: scale(1.05);
    }
    .css-1dp5vir {
        background-color: #ffffff;
    }
    .css-18e3th9 {
        padding-top: 1rem;
        padding-bottom: 10rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
        padding-right: 1rem;
        padding-bottom: 3.5rem;
        padding-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# OpenAI APIキーの設定
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# サイドバーにAPIキー入力を配置
with st.sidebar:
    st.title("🌈 OpenAI API キー入力")
    api_key = st.text_input("APIキーを入力してください:", value=st.session_state.api_key, type="password")
    if st.button("APIキーを設定"):
        st.session_state.api_key = api_key
        openai.api_key = api_key
        st.success("APIキーが設定されました ✨")

# ベクトルデータベースの構築
@st.cache_resource(show_spinner=False)
def build_vector_database():
    with st.spinner(text="✨ ベクトルデータベースを構築中... ✨"):
        reader = SimpleDirectoryReader(
            input_dir="./data",
            recursive=True,
            encoding="utf-8",  # UTF-8エンコーディングを明示的に指定
            errors="ignore"    # デコードエラーを無視
        )
        docs = reader.load_data()
        embed_model = OpenAIEmbedding(openai_api_key=openai.api_key)
        index = VectorStoreIndex.from_documents(docs)
        return index

# テキストをチャンクに分割する関数
def split_into_chunks(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# チャンクのスコアリング関数
def score_chunks(chunks, query):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks + [query])
    query_vector = tfidf_matrix[-1]
    chunk_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(chunk_vectors, query_vector)
    return similarities.flatten()

# 最も関連性の高いチャンクを選択する関数
def select_best_chunks(chunks, scores, num_chunks=2):
    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
    return sorted_chunks[:num_chunks]

# ハイライト関数
def highlight_text(text, query):
    words = query.lower().split()
    for word in words:
        text = re.sub(f'(?i){re.escape(word)}', lambda m: f'<span class="highlight">{m.group()}</span>', text)
    return text

# メイン処理
if st.session_state.api_key:
    try:
        index = build_vector_database()
        
        st.title("🌸 かわいい類似ドキュメント検索 🌸")
        
        # 検索フォーム
        with st.form(key='search_form'):
            query = st.text_input("🔍 検索クエリを入力してください:", key="search_input")
            submit_button = st.form_submit_button(label='検索 🚀')
        
        if submit_button and query:
            with st.spinner("🔍 かわいく検索中..."):
                # クエリエンジンの設定
                retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=10,
                )
                response_synthesizer = get_response_synthesizer()
                query_engine = RetrieverQueryEngine(
                    retriever=retriever,
                    response_synthesizer=response_synthesizer,
                    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
                )
                
                # クエリ実行
                response = query_engine.query(query)
                
                # 検索結果の表示
                for i, node in enumerate(response.source_nodes, 1):
                    content = node.node.get_content()
                    score = node.score if hasattr(node, 'score') else 'N/A'
                    file_name = node.node.metadata.get('file_name', '不明')
                    
                    # コンテンツをチャンクに分割
                    chunks = split_into_chunks(content)
                    
                    # チャンクのスコアリング
                    chunk_scores = score_chunks(chunks, query)
                    
                    # 最も関連性の高いチャンクを選択
                    best_chunks = select_best_chunks(chunks, chunk_scores)
                    
                    # 選択されたチャンクをハイライト
                    highlighted_content = "...".join([highlight_text(chunk, query) for chunk in best_chunks])
                    
                    # 結果表示
                    st.markdown(f"""
                    <div class="result-item">
                        <div class="result-title">📄 {file_name}</div>
                        <div class="result-url">💖 類似度スコア: {score:.2f}</div>
                        <div class="result-snippet">{highlighted_content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 元の資料へのリンク
                    file_path = os.path.join("data", file_name)
                    if os.path.exists(file_path):
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            file_content = f.read()
                        st.download_button(
                            label=f"📥 元の資料をダウンロード",
                            data=file_content,
                            file_name=file_name,
                            mime="text/plain",
                            key=f"download_button_{i}"
                        )
                
                # 総合回答の表示
                st.subheader("✨ 総合回答:")
                st.write(response)
    except Exception as e:
        st.error(f"😢 エラーが発生しました: {str(e)}")
        st.error("ファイルのエンコーディングに問題がある可能性があります。すべてのファイルがUTF-8でエンコードされていることを確認してください。")
else:
    st.warning("🔑 OpenAI APIキーを設定してください。サイドバーを開いて設定してください。")
