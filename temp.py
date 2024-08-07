import os
import streamlit as st
import openai
from llama_index.llms import OpenAI
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    Document,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding

# OpenAI API key setup
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Llama-index settings
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding()

# Service context setup
service_context = ServiceContext.from_defaults(
    llm=Settings.llm,
    embed_model=Settings.embed_model,
)

# Function to build index without chunking
@st.cache_resource(show_spinner=False)
def build_file_based_index(directory="./data"):
    with st.spinner(text="Building index. This may take a few minutes..."):
        reader = SimpleDirectoryReader(input_dir=directory, recursive=True)
        documents = []
        for file in reader.input_files:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                metadata = {"filename": os.path.basename(file)}
                doc = Document(text=content, metadata=metadata)
                documents.append(doc)
        
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        return index

# Main Streamlit app logic
def main():
    st.title("Document Search Application")

    # Build the index
    index = build_file_based_index()

    # Search interface
    query = st.text_input("Enter your search query:")
    if query:
        # Perform the search
        retriever = index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query)

        # Display results
        st.subheader("Search Results:")
        for i, node in enumerate(nodes, 1):
            st.write(f"**Result {i}**")
            st.write(f"File: {node.metadata['filename']}")
            st.write(f"Content: {node.text[:500]}...")  # Display first 500 characters
            st.write("---")

if __name__ == "__main__":
    main()
