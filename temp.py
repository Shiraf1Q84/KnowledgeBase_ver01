from llama_index.core import ListIndex, Document

# --- ベクトルデータベースの構築 (事前構築) ---
@st.cache_resource(show_spinner=False)
def build_list_index():
    with st.spinner(text="Wait minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        
        # 各ドキュメントにファイル名をメタデータとして追加
        for doc in docs:
            doc.metadata["filename"] = os.path.basename(doc.metadata["file_path"])
        
        # ListIndexの作成
        index = ListIndex.from_documents(docs)
        return index

# ListIndexを構築
index = build_list_index()

# --- チャットエンジンの初期化 ---
if "chat_engine" not in st.session_state.keys() and openai.api_key is not None:
    system_prompt = """
    あなたはナレッジベースに提供されている書類に関する情報を提供するチャットボットです。
    利用者の質問に対して、関連するファイルの内容を表示します。
    LLMを使用せず、単純にキーワードマッチングで関連ファイルを探します。
    """
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        verbose=True,
        system_prompt=system_prompt
    )

# チャット処理
if openai.api_key is not None:
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                # キーワード検索を実行
                retriever = index.as_retriever(similarity_top_k=5)
                nodes = retriever.retrieve(prompt)
                
                # 検索結果を表示
                st.write("関連するファイルが見つかりました：")
                for node in nodes:
                    st.write(f"**ファイル名: {node.metadata['filename']}**")
                    st.markdown(node.get_content())
                
                # メッセージを追加
                message = {"role": "assistant", "content": "上記のファイルが関連していると思われます。詳細な情報が必要な場合は、さらに質問してください。"}
                st.session_state.messages.append(message)