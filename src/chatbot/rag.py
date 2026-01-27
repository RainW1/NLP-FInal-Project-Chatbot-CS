"""
Person B: RAG (Retrieval-Augmented Generation) Module
=====================================================

YOUR TASK:
1. Create knowledge base from documents (FAQ, policies, etc.)
2. Convert documents to embeddings and store in vector DB
3. Retrieve relevant documents for user queries

LIBRARIES TO USE:
- sentence-transformers (for embeddings)
- chromadb or faiss (vector database)

SETUP:
pip install sentence-transformers chromadb

KNOWLEDGE BASE CONTENT IDEAS:
- FAQ (frequently asked questions)
- Shipping policy
- Return/refund policy
- Product catalog
- Contact information
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from interfaces import RetrievedDoc
from typing import List

# ============================================================
# STUB IMPLEMENTATION - Replace with real code!
# ============================================================

# Dummy knowledge base for testing
DUMMY_KNOWLEDGE_BASE = [
    {
        "content": "Pengiriman standar membutuhkan 3-5 hari kerja untuk Pulau Jawa dan 5-7 hari kerja untuk luar Jawa. Pengiriman express tersedia dengan tambahan biaya Rp 15.000.",
        "source": "shipping_policy.txt"
    },
    {
        "content": "Untuk melakukan return/refund, pelanggan harus mengajukan dalam waktu 7 hari setelah barang diterima. Barang harus dalam kondisi original dan belum digunakan. Proses refund membutuhkan 3-5 hari kerja setelah barang diterima kembali.",
        "source": "return_policy.txt"
    },
    {
        "content": "Jam operasional customer service: Senin-Jumat 08:00-20:00 WIB, Sabtu-Minggu 09:00-17:00 WIB. Untuk pertanyaan urgent, silakan hubungi hotline 021-12345678.",
        "source": "contact_info.txt"
    },
    {
        "content": "Status pesanan dapat dicek melalui menu 'Pesanan Saya' di aplikasi atau website. Anda juga bisa tracking dengan memasukkan nomor resi di halaman tracking.",
        "source": "faq.txt"
    },
    {
        "content": "Jika barang yang diterima rusak atau tidak sesuai, silakan foto bukti kerusakan dan ajukan komplain melalui aplikasi dalam waktu 2x24 jam. Tim kami akan memproses penggantian atau refund.",
        "source": "faq.txt"
    },
    {
        "content": "Pembayaran dapat dilakukan melalui transfer bank, virtual account, e-wallet (GoPay, OVO, Dana), atau kartu kredit. Semua metode pembayaran aman dan terenkripsi.",
        "source": "payment_info.txt"
    },
]


def retrieve_knowledge(query: str, top_k: int = 3) -> List[RetrievedDoc]:
    """
    STUB: Returns dummy documents based on keyword matching.
    
    TODO (Person B):
    1. Load sentence-transformer model for embeddings
    2. Setup ChromaDB or FAISS vector store
    3. Index all knowledge base documents
    4. Retrieve top-k similar documents for query
    
    Real implementation example:
    ```python
    from sentence_transformers import SentenceTransformer
    import chromadb
    
    # Initialize (do once at startup)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("knowledge_base")
    
    # Index documents (do once)
    def index_documents(documents):
        for i, doc in enumerate(documents):
            embedding = embed_model.encode(doc["content"]).tolist()
            collection.add(
                embeddings=[embedding],
                documents=[doc["content"]],
                metadatas=[{"source": doc["source"]}],
                ids=[f"doc_{i}"]
            )
    
    # Retrieve
    def retrieve_knowledge(query: str, top_k: int = 3) -> List[RetrievedDoc]:
        query_embedding = embed_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        docs = []
        for i in range(len(results['documents'][0])):
            docs.append(RetrievedDoc(
                content=results['documents'][0][i],
                source=results['metadatas'][0][i]['source'],
                relevance=1.0 - results['distances'][0][i]  # Convert distance to similarity
            ))
        return docs
    ```
    """
    # STUB - Simple keyword matching for testing
    query_lower = query.lower()
    
    scored_docs = []
    for doc in DUMMY_KNOWLEDGE_BASE:
        # Simple relevance scoring based on keyword overlap
        doc_lower = doc["content"].lower()
        query_words = set(query_lower.split())
        doc_words = set(doc_lower.split())
        overlap = len(query_words.intersection(doc_words))
        
        # Boost score for specific keywords
        if "pengiriman" in query_lower and "pengiriman" in doc_lower:
            overlap += 3
        if "refund" in query_lower and "refund" in doc_lower:
            overlap += 3
        if "return" in query_lower and "return" in doc_lower:
            overlap += 3
        if "status" in query_lower and "status" in doc_lower:
            overlap += 3
        if "rusak" in query_lower and "rusak" in doc_lower:
            overlap += 3
            
        if overlap > 0:
            scored_docs.append((doc, overlap))
    
    # Sort by relevance and take top_k
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for doc, score in scored_docs[:top_k]:
        # Normalize score to 0-1 range
        relevance = min(score / 10.0, 1.0)
        results.append(RetrievedDoc(
            content=doc["content"],
            source=doc["source"],
            relevance=relevance
        ))
    
    # If no matches, return top general docs
    if not results:
        for doc in DUMMY_KNOWLEDGE_BASE[:top_k]:
            results.append(RetrievedDoc(
                content=doc["content"],
                source=doc["source"],
                relevance=0.3
            ))
    
    return results


def add_document_to_knowledge_base(content: str, source: str) -> bool:
    """
    Adds a new document to the knowledge base.
    
    TODO (Person B): Implement real document addition with embedding
    """
    # STUB
    DUMMY_KNOWLEDGE_BASE.append({
        "content": content,
        "source": source
    })
    return True


def get_knowledge_base_stats() -> dict:
    """
    Returns statistics about the knowledge base.
    Useful for debugging and monitoring.
    """
    sources = {}
    for doc in DUMMY_KNOWLEDGE_BASE:
        source = doc["source"]
        sources[source] = sources.get(source, 0) + 1
    
    return {
        "total_documents": len(DUMMY_KNOWLEDGE_BASE),
        "sources": sources
    }


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    # Test stub function
    test_queries = [
        "pesanan saya belum sampai",
        "gimana cara refund",
        "status ORDER123",
        "barang rusak mau komplain",
    ]
    
    print("Testing retrieve_knowledge stub:\n")
    for query in test_queries:
        print(f"Query: {query}")
        docs = retrieve_knowledge(query, top_k=2)
        for i, doc in enumerate(docs):
            print(f"  Doc {i+1}: [{doc.source}] (relevance: {doc.relevance:.2f})")
            print(f"          {doc.content[:80]}...")
        print("-" * 60)
    
    print("\nKnowledge Base Stats:")
    print(get_knowledge_base_stats())
    
    print("\n✅ Stubs working! Ready for real implementation.")
