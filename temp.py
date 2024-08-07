import os
import streamlit as st
import openai
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import re

# Streamlitページ設定
st.set_page_config(
    page_title="高度な類似ドキュメント検索",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="auto",
)

# OpenAI APIキーの設定
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

st.sidebar.title("OpenAI API キー入力")
api_key = st.sidebar.text_input("APIキーを入力してください:", value=st.session_state.api_key, type="password")

if st.sidebar.button("APIキーを設定"):
    st.session_state.api_key = api_key
    openai.api_key = api_key
    st.sidebar.success("APIキーが設定されました")

# ベクトルデータベースの構築
@st.cache_resource(show_spinner=False)
def build_vector_database():
    with st.spinner(text="ベクトルデータベースを構築中..."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        embed_model = OpenAIEmbedding(openai_api_key=openai.api_key)
        index = VectorStoreIndex.from_documents(docs)
        return index

# ハイライト関数
def highlight_text(text, query):
    words = query.lower().split()
    for word in words:
        text = re.sub(f'(?i){re.escape(word)}', lambda m: f'<span style="background-color: yellow;">{m.group()}</span>', text)
    return text

# ポップアップウィンドウを生成するJavaScript
def get_popup_script(content, index):
    return f"""
    <script>
    function openPopup{index}() {{
        var w = window.open('', '_blank', 'width=600,height=400');
        w.document.write(`<pre>{content}</pre>`);
        w.document.close();
    }}
    </script>
    """

# メイン処理
if st.session_state.api_key:
    index = build_vector_database()
    
    st.title("高度な類似ドキュメント検索")
    query = st.text_input("検索クエリを入力してください:")
    
    if query:
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
            
            st.subheader("検索結果:")
            
            # ソースノードの処理
            for i, node in enumerate(response.source_nodes, 1):
                content = node.node.get_content()
                score = node.score if hasattr(node, 'score') else 'N/A'
                
                # 重要箇所の抽出（この例では、クエリに最も関連する1-2文を抽出）
                sentences = content.split('。')
                relevant_sentences = [sent for sent in sentences if any(word in sent.lower() for word in query.lower().split())][:2]
                highlighted_content = highlight_text('。'.join(relevant_sentences), query)
                
                # 結果表示
                st.markdown(f"**結果 {i}:** (類似度スコア: {score})")
                st.markdown(highlighted_content, unsafe_allow_html=True)
                
                # ポップアップボタン
                popup_script = get_popup_script(content, i)
                st.markdown(f"{popup_script}<button onclick='openPopup{i}()'>📄 全文を表示</button>", unsafe_allow_html=True)
                
                st.markdown("---")
            
            # 総合回答の表示
            st.subheader("総合回答:")
            st.write(response)
else:
    st.warning("OpenAI APIキーを設定してください。")
