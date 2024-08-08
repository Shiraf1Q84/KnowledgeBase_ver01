import os
import streamlit as st
import openai
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlitページ設定
st.set_page_config(
    page_title="高度な類似ドキュメント検索",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 重要なディレクトリパスの設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

# ディレクトリの存在確認と作成
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)


# カスタムCSS
st.markdown("""
<style>
    .main {
        font-family: Arial, sans-serif;
    }
    .stApp {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    .search-container {
        display: flex;
        margin-bottom: 20px;
    }
    .search-input {
        flex-grow: 1;
        padding: 10px;
        font-size: 16px;
        border: 1px solid #dfe1e5;
        border-radius: 24px;
        outline: none;
    }
    .search-button {
        margin-left: 10px;
        padding: 10px 20px;
        font-size: 16px;
        background-color: #f8f9fa;
        border: 1px solid #f8f9fa;
        border-radius: 4px;
        color: #3c4043;
        cursor: pointer;
    }
    .search-button:hover {
        box-shadow: 0 1px 1px rgba(0,0,0,.1);
        background-color: #f8f9fa;
        border: 1px solid #dadce0;
        color: #202124;
    }
    .result-item {
        margin-bottom: 20px;
    }
    .result-title {
        color: #1a0dab;
        font-size: 18px;
        margin-bottom: 5px;
    }
    .result-url {
        color: #006621;
        font-size: 14px;
        margin-bottom: 5px;
    }
    .result-snippet {
        color: #545454;
        font-size: 14px;
    }
    .highlight {
        background-color: #ffffcc;
    }
</style>
""", unsafe_allow_html=True)



# OpenAI APIキーの設定
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# サイドバーにAPIキー入力を配置
with st.sidebar:
    st.title("OpenAI API キー入力")
    api_key = st.text_input("APIキーを入力してください:", value=st.session_state.api_key, type="password")
    if st.button("APIキーを設定"):
        st.session_state.api_key = api_key
        openai.api_key = api_key
        st.success("APIキーが設定されました")

# ベクトルデータベースの構築または読み込み
@st.cache_resource(show_spinner=False)
def get_vector_index():
    try:
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        if os.path.exists(os.path.join(STORAGE_DIR, "docstore.json")):
            # 既存のインデックスを読み込む
            index = load_index_from_storage(storage_context)
            st.sidebar.success("既存のベクトルデータベースを読み込みました。")
        else:
            # 新しいインデックスを作成する
            with st.spinner(text="ベクトルデータベースを構築中..."):
                if not os.listdir(DATA_DIR):
                    st.error(f"データディレクトリ {DATA_DIR} が空です。データファイルを追加してください。")
                    return None
                reader = SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True)
                docs = reader.load_data()
                embed_model = OpenAIEmbedding(openai_api_key=openai.api_key)
                index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
                index.storage_context.persist()
            st.sidebar.success("新しいベクトルデータベースを作成しました。")
        return index
    except Exception as e:
        st.error(f"ベクトルデータベースの処理中にエラーが発生しました: {str(e)}")
        return None

# その他の関数（split_into_chunks, score_chunks, select_best_chunks, highlight_text）は変更なし...

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
    index = get_vector_index()
    
    if index is not None:
        st.title("高度な類似ドキュメント検索")
        
        # 検索フォーム
        with st.form(key='search_form'):
            query = st.text_input("検索クエリを入力してください:", key="search_input")
            submit_button = st.form_submit_button(label='検索')
        
        if submit_button and query:
            with st.spinner("検索中..."):
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
                
                # 検索結果の表示（変更なし）...
                

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
                    <div class="result-title">{file_name}</div>
                    <div class="result-url">類似度スコア: {score:.2f}</div>
                    <div class="result-snippet">{highlighted_content}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 元の資料へのリンク
                file_path = os.path.join("data", file_name)
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        file_content = f.read()
                    st.download_button(
                        label=f"元の資料をダウンロード",
                        data=file_content,
                        file_name=file_name,
                        mime="text/plain",
                        key=f"download_button_{i}"
                    )
                    
                # 総合回答の表示
                st.subheader("総合回答:")
                st.write(response)
    else:
        st.warning("ベクトルデータベースの初期化に失敗しました。上記のエラーメッセージを確認してください。")
else:
    st.warning("OpenAI APIキーを設定してください。サイドバーを開いて設定してください。")

# データベース更新ボタン
if st.sidebar.button("ベクトルデータベースを更新"):
    # セッションステートをクリアしてキャッシュを無効化
    st.cache_resource.clear()
    # ストレージディレクトリを削除
    import shutil
    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
    st.sidebar.success("ベクトルデータベースが更新されました。ページを再読み込みしてください。")
    st.experimental_rerun()