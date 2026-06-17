import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURATION ---
DB_PATH = "./chroma_db"

def inspect_database():
    # 1. Check if DB exists
    if not os.path.exists(DB_PATH):
        print("❌ Database folder not found! Did you run ingest.py?")
        return

    # 2. Connect to the DB
    print("🔌 Connecting to database...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

    # 3. Get basic stats
    # Chroma doesn't have a direct "count" method in all versions, 
    # so we do a dummy fetch to get the collection.
    count = vector_db._collection.count()
    print(f"\n📊 Total Text Chunks Stored: {count}")

    if count == 0:
        print("⚠️ The database is empty!")
        return

    # 4. Peek at a few random records
    print("\n🔍 Peeking at the first 3 chunks to verify content...")
    
    # We fetch the first few IDs to inspect them
    data = vector_db._collection.get(limit=3)
    
    ids = data['ids']
    metadatas = data['metadatas']
    documents = data['documents']

    for i in range(len(ids)):
        print(f"\n--- Chunk {i+1} ---")
        print(f"📄 Content (First 100 chars): \"{documents[i][:100]}...\"")
        print(f"🏷️  Metadata: {metadatas[i]}")
        print("-------------------")

    # 5. Test the 'Timeline Filter' logic
    print("\n🧪 Testing Timeline Filter (Book 1 only)...")
    results = vector_db.similarity_search("Harry", k=1, filter={"book_index": 1})
    
    if results:
        print(f"✅ Filter Success! Found data tagged as Book 1: {results[0].metadata}")
    else:
        print("❌ Filter Warning: No data found for Book 1. Check your ingestion.")

if __name__ == "__main__":
    inspect_database()