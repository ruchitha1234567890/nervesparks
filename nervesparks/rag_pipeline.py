from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import openai

class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.db = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
        self.collection = self.db.get_or_create_collection(name="rag_docs")

    def index_chunks(self, chunks):
        embeddings = self.embedder.encode(chunks).tolist()
        self.collection.delete()
        self.collection.add(documents=chunks, embeddings=embeddings, ids=[str(i) for i in range(len(chunks))])

    def query(self, question):
        q_emb = self.embedder.encode([question])[0].tolist()
        results = self.collection.query(query_embeddings=[q_emb], n_results=3)
        context_chunks = results["documents"][0]
        context = "\n".join(context_chunks)
        response = self.generate_answer(question, context)
        return response, context_chunks

    def generate_answer(self, question, context):
        openai.api_key = "YOUR_API_KEY"
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res["choices"][0]["message"]["content"].strip()