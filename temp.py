import os
import streamlit as st
import openai
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Streamlitページ設定
st.set_page_config(
    page_title="類似ドキュメント検索",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="auto",
)

# OpenAI APIキーの設定
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

st.title("OpenAI API キー入力")
api_key = st.text_input("APIキーを入力してください:", value=st.session_state.api_key, type="password")

if st.button("APIキーを設定"):
    st.session_state.api_key = api_key
    openai.api_key = api_key
    st.success("APIキーが設定されました")

# ベクトルデータベースの構築
@st.cache_resource(show_spinner=False)
def build_vector_database():
    with st.spinner(text="ベクトルデータベースを構築中..."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        embed_model = OpenAIEmbedding(openai_api_key=openai.api_key)
        index = VectorStoreIndex.from_documents(docs)
        return index

# ポップアップウィンドウを生成するJavaScript
def get_popup_script(content):
    return f"""
    <script>
    function openPopup() {{
        var w = window.open('', '_blank', 'width=600,height=400');
        w.document.write(`<pre>{content}</pre>`);
        w.document.close();
    }}
    </script>
    """

# メイン処理
if st.session_state.api_key:
    index = build_vector_database()
    
    st.title("類似ドキュメント検索")
    query = st.text_input("検索クエリを入力してください:")
    
    if query:
        with st.spinner("検索中..."):
            # 類似度の高い上位6件を取得
            results = index.as_retriever().retrieve(query, similarity_top_k=6)
            
            st.subheader("検索結果:")
            for i, node in enumerate(results, 1):
                content = node.node.get_content()
                # アイコンボタンとポップアップスクリプトを生成
                icon = "📄"  # ドキュメントアイコン
                popup_script = get_popup_script(content)
                st.markdown(f"{popup_script}<button onclick='openPopup()'>{icon} 結果 {i}</button>", unsafe_allow_html=True)
                st.markdown("---")
else:
    st.warning("OpenAI APIキーを設定してください。")
