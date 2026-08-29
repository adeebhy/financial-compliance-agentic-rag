import glob
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

class RegulatoryVectorDB:
    def __init__(self, docs_dir="data/regulatory_docs/"):
        self.docs_dir = docs_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None

    def build(self):
        docs = []
        for file_path in glob.glob(os.path.join(self.docs_dir, "*.txt")):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append(Document(page_content=text, metadata={"source": os.path.basename(file_path)}))

        splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        return self.vector_store

    def query(self, query_str: str, k: int = 2) -> str:
        if not self.vector_store:
            self.build()
        results = self.vector_store.similarity_search(query_str, k=k)
        return "\n\n".join([f"[{r.metadata['source']}] {r.page_content}" for r in results])
