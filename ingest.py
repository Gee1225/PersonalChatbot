import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import config

def ingest_and_embed():
    if not os.path.exists("data/bio.txt"):
        print("🚫 bio.txt not found!")
        return

    print("📂 Loading bio.txt...")
    loader = TextLoader("data/bio.txt")
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} raw documents")

    if not docs:
        print("🚫 No content found in bio.txt")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    print(f"✂️ Split into {len(split_docs)} chunks")

    if not split_docs:
        print("🚫 No chunks generated from documents")
        return

    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)

    # test embedding
    try:
        print("🔑 Testing embedding call...")
        _ = embeddings.embed_query("Test embedding")
        print("✅ Embedding service is working")
    except Exception as e:
        print("❌ Embedding failed:", e)
        return

    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local("vectorstore")
    print("✅ Vectorstore saved at:", os.path.abspath("vectorstore"))

if __name__ == "__main__":
    ingest_and_embed()
