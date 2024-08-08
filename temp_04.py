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

# カスタムCSS（省略）...

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
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    if os.path.exists("./storage"):
        # 既存のインデックスを読み込む
        index = load_index_from_storage(storage_context)
        st.sidebar.success("既存のベクトルデータベースを読み込みました。")
    else:
        # 新しいインデックスを作成する
        with st.spinner(text="ベクトルデータベースを構築中..."):
            reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
            docs = reader.load_data()
            embed_model = OpenAIEmbedding(openai_api_key=openai.api_key)
            index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
            index.storage_context.persist()
        st.sidebar.success("新しいベクトルデータベースを作成しました。")
    return index

# その他の関数（split_into_chunks, score_chunks, select_best_chunks, highlight_text）は変更なし...

# メイン処理
if st.session_state.api_key:
    index = get_vector_index()
    
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
            
            # 総合回答の表示
            st.subheader("総合回答:")
            st.write(response)
else:
    st.warning("OpenAI APIキーを設定してください。サイドバーを開いて設定してください。")

# データベース更新ボタン
if st.sidebar.button("ベクトルデータベースを更新"):
    # セッションステートをクリアしてキャッシュを無効化
    st.cache_resource.clear()
    # ストレージディレクトリを削除
    import shutil
    if os.path.exists("./storage"):
        shutil.rmtree("./storage")
    st.sidebar.success("ベクトルデータベースが更新されました。ページを再読み込みしてください。")
    st.experimental_rerun()
