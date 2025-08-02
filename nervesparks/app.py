import streamlit as st
from document_parser import process_document
from rag_pipeline import RAGPipeline

st.title("📄 Visual Document Analysis RAG")

uploaded_file = st.file_uploader("Upload a PDF/Image", type=["pdf", "png", "jpg", "jpeg"])
query = st.text_input("Ask a question about the document")

if uploaded_file and query:
    with st.spinner("Processing document..."):
        chunks = process_document(uploaded_file)
        rag = RAGPipeline()
        rag.index_chunks(chunks)
        response, context = rag.query(query)

    st.subheader("Answer:")
    st.write(response)

    with st.expander("Retrieved Chunks"):
        for i, chunk in enumerate(context):
            st.markdown(f"**Chunk {i+1}:** {chunk}")