"""Quick diagnostic script to test imports."""
print("Testing imports...")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence_transformers imported successfully")
except Exception as e:
    print(f"❌ sentence_transformers import failed: {e}")

try:
    import faiss
    print("✅ faiss imported successfully")
except Exception as e:
    print(f"❌ faiss import failed: {e}")

try:
    from src.retrieval.embeddings import EmbeddingRetriever
    print("✅ EmbeddingRetriever imported successfully")
except Exception as e:
    print(f"❌ EmbeddingRetriever import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
