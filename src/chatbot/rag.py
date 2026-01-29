"""
Person B: RAG (Retrieval-Augmented Generation) Module
=====================================================

REAL IMPLEMENTATION using:
- ChromaDB for vector storage
- Sentence-Transformers for embeddings

SETUP:
pip install chromadb sentence-transformers
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List

# ============================================================
# IMPORTS - Install if not available
# ============================================================

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("⚠️ Please install: pip install chromadb sentence-transformers")

# ============================================================
# DATA CLASS (compatible with interfaces.py)
# ============================================================

class RetrievedDoc:
    """Document retrieved from RAG"""
    def __init__(self, content: str, source: str, relevance: float):
        self.content = content
        self.source = source
        self.relevance = relevance

# ============================================================
# KNOWLEDGE BASE CONTENT (ENGLISH)
# ============================================================

KNOWLEDGE_BASE = [
    # ----- SHIPPING POLICY -----
    {
        "content": "Standard shipping takes 3-5 business days for domestic orders and 7-14 business days for international orders. We partner with FedEx, UPS, DHL, and local carriers.",
        "source": "shipping_policy",
        "category": "shipping"
    },
    {
        "content": "Express shipping (same day or next day) is available for select areas with an additional fee of $15. Orders must be placed before 12:00 PM local time.",
        "source": "shipping_policy",
        "category": "shipping"
    },
    {
        "content": "Free shipping is available for orders over $50 for domestic delivery. This promotion applies to standard shipping only.",
        "source": "shipping_policy",
        "category": "shipping"
    },
    {
        "content": "If your order hasn't arrived within the estimated delivery time, please contact customer service with your order number. Our team will help track your package and coordinate with the carrier.",
        "source": "shipping_policy",
        "category": "shipping"
    },
    {
        "content": "Tracking numbers are sent via email within 24 hours after your order is shipped. You can also check your order status in the 'My Orders' section of your account.",
        "source": "shipping_policy",
        "category": "shipping"
    },
    
    # ----- RETURN & REFUND POLICY -----
    {
        "content": "Returns and refunds can be requested within 30 days of receiving your order. Items must be in original condition, unused, and with all tags attached.",
        "source": "return_policy",
        "category": "return"
    },
    {
        "content": "Refunds are processed within 5-7 business days after we receive and verify the returned item. The refund will be credited to your original payment method.",
        "source": "return_policy",
        "category": "return"
    },
    {
        "content": "For damaged or defective products, we offer a full refund or free replacement. Please take photos of the damage and submit a complaint through our app or website.",
        "source": "return_policy",
        "category": "return"
    },
    {
        "content": "Return shipping costs are the responsibility of the customer, except for cases of damaged, defective, or incorrectly shipped items where we cover the return shipping.",
        "source": "return_policy",
        "category": "return"
    },
    {
        "content": "To initiate a return, go to 'My Orders', select the item you want to return, and click 'Request Return'. You'll receive a return shipping label via email.",
        "source": "return_policy",
        "category": "return"
    },
    
    # ----- ORDER & TRACKING -----
    {
        "content": "You can check your order status in the 'My Orders' section on our app or website. You can also track your package using the tracking number provided in your shipping confirmation email.",
        "source": "faq",
        "category": "order"
    },
    {
        "content": "Orders can be cancelled as long as the status is still 'Processing' or 'Pending Payment'. Once the order status changes to 'Shipped', cancellation is no longer possible.",
        "source": "faq",
        "category": "order"
    },
    {
        "content": "To modify your order (change address, add items, etc.), please contact customer service immediately. Changes can only be made before the order is shipped.",
        "source": "faq",
        "category": "order"
    },
    
    # ----- PAYMENT -----
    {
        "content": "We accept various payment methods including Credit/Debit Cards (Visa, MasterCard, AmEx), PayPal, Apple Pay, Google Pay, and Bank Transfer.",
        "source": "payment_info",
        "category": "payment"
    },
    {
        "content": "Payment must be completed within 24 hours of placing your order. Orders will be automatically cancelled if payment is not received within this timeframe.",
        "source": "payment_info",
        "category": "payment"
    },
    {
        "content": "Installment payment options (Buy Now, Pay Later) are available through Klarna and Afterpay for orders over $100. No interest if paid within the promotional period.",
        "source": "payment_info",
        "category": "payment"
    },
    {
        "content": "All transactions are secured with SSL encryption. We do not store your complete credit card information on our servers.",
        "source": "payment_info",
        "category": "payment"
    },
    
    # ----- PRODUCT & WARRANTY -----
    {
        "content": "All electronic products come with a minimum 1-year manufacturer warranty. Warranty claims can be made at authorized service centers with proof of purchase.",
        "source": "product_info",
        "category": "product"
    },
    {
        "content": "Fashion items can be exchanged for a different size within 14 days if they don't fit. Items must be unworn, unwashed, and have original tags attached.",
        "source": "product_info",
        "category": "product"
    },
    {
        "content": "Product availability and stock information is updated in real-time on our website. If an item is out of stock, you can sign up for notifications when it becomes available again.",
        "source": "product_info",
        "category": "product"
    },
    
    # ----- CONTACT & SUPPORT -----
    {
        "content": "Customer service hours: Monday-Friday 8:00 AM - 9:00 PM, Saturday-Sunday 9:00 AM - 6:00 PM (all times in local timezone). For inquiries outside business hours, please email support@store.com.",
        "source": "contact_info",
        "category": "contact"
    },
    {
        "content": "For urgent assistance, call our hotline at 1-800-123-4567 or chat with us through WhatsApp at +1-234-567-8900. Response time is typically within 1 hour during business hours.",
        "source": "contact_info",
        "category": "contact"
    },
    {
        "content": "You can also reach us through our social media channels: Facebook, Twitter, and Instagram @OfficialStore. We typically respond within 2-4 hours.",
        "source": "contact_info",
        "category": "contact"
    },
    
    # ----- COMPLAINT HANDLING -----
    {
        "content": "If you received a damaged or incorrect item, please take photos and submit a complaint within 48 hours of delivery. Our team will process your case within 1 business day.",
        "source": "complaint_policy",
        "category": "complaint"
    },
    {
        "content": "All complaints are handled with high priority. We will contact you within 24 hours to confirm the issue and provide a solution (replacement or refund).",
        "source": "complaint_policy",
        "category": "complaint"
    },
    {
        "content": "For quality issues discovered after use, please contact us within 7 days with photos and description of the problem. We'll evaluate each case individually.",
        "source": "complaint_policy",
        "category": "complaint"
    },
    
    # ----- ACCOUNT & MEMBERSHIP -----
    {
        "content": "Creating an account is free and gives you access to order tracking, faster checkout, exclusive deals, and loyalty rewards.",
        "source": "account_info",
        "category": "account"
    },
    {
        "content": "Our loyalty program rewards you with 1 point for every $1 spent. Accumulate 100 points to get a $5 discount on your next purchase.",
        "source": "account_info",
        "category": "account"
    },
    {
        "content": "To reset your password, click 'Forgot Password' on the login page. You'll receive an email with instructions to create a new password.",
        "source": "account_info",
        "category": "account"
    },
]


# ============================================================
# RAG SYSTEM CLASS
# ============================================================

class RAGSystem:
    """
    RAG System using ChromaDB and Sentence Transformers
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG system
        
        Args:
            embedding_model: Sentence transformer model name
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Please install: pip install chromadb sentence-transformers")
        
        print("🔄 Loading embedding model...")
        self.embed_model = SentenceTransformer(embedding_model)
        
        print("🔄 Setting up ChromaDB...")
        self.client = chromadb.Client()  # In-memory client
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"description": "E-commerce customer service knowledge base"}
        )
        
        # Index knowledge base if empty
        if self.collection.count() == 0:
            self._index_knowledge_base()
        
        print(f"✅ RAG System ready! ({self.collection.count()} documents indexed)")
    
    def _index_knowledge_base(self):
        """Index all documents in knowledge base"""
        print("🔄 Indexing knowledge base...")
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(KNOWLEDGE_BASE):
            documents.append(doc["content"])
            metadatas.append({
                "source": doc["source"],
                "category": doc["category"]
            })
            ids.append(f"doc_{i}")
        
        # Generate embeddings
        embeddings = self.embed_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Indexed {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of RetrievedDoc objects
        """
        # Generate query embedding
        query_embedding = self.embed_model.encode(query).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to RetrievedDoc objects
        docs = []
        for i in range(len(results['documents'][0])):
            # Convert distance to similarity (ChromaDB returns L2 distance)
            distance = results['distances'][0][i]
            relevance = max(0, 1 - (distance / 2))  # Normalize to 0-1
            
            docs.append(RetrievedDoc(
                content=results['documents'][0][i],
                source=results['metadatas'][0][i]['source'],
                relevance=round(relevance, 3)
            ))
        
        return docs
    
    def add_document(self, content: str, source: str, category: str = "general") -> bool:
        """Add a new document to the knowledge base"""
        try:
            doc_id = f"doc_{self.collection.count()}"
            embedding = self.embed_model.encode(content).tolist()
            
            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[{"source": source, "category": category}],
                ids=[doc_id]
            )
            return True
        except Exception as e:
            print(f"Error adding document: {e}")
            return False


# ============================================================
# GLOBAL RAG INSTANCE (Singleton)
# ============================================================

_rag_system = None

def get_rag_system() -> RAGSystem:
    """Get or create RAG system singleton"""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system


# ============================================================
# MAIN FUNCTION (matches interface contract)
# ============================================================

def retrieve_knowledge(query: str, top_k: int = 3) -> List[RetrievedDoc]:
    """
    Retrieves relevant documents from knowledge base using RAG.
    
    Args:
        query: Search query (usually user's message)
        top_k: Number of documents to retrieve
        
    Returns:
        List of RetrievedDoc objects
        
    Example:
        >>> docs = retrieve_knowledge("how long does shipping take")
        >>> print(docs[0].content)
        "Standard shipping takes 3-5 business days..."
    """
    try:
        rag = get_rag_system()
        return rag.retrieve(query, top_k)
    except Exception as e:
        print(f"⚠️ RAG error: {e}")
        return []


def add_document_to_knowledge_base(content: str, source: str, category: str = "general") -> bool:
    """Add a new document to the knowledge base"""
    try:
        rag = get_rag_system()
        return rag.add_document(content, source, category)
    except Exception as e:
        print(f"⚠️ Error adding document: {e}")
        return False


def get_knowledge_base_stats() -> dict:
    """Returns statistics about the knowledge base"""
    try:
        rag = get_rag_system()
        return {
            "total_documents": rag.collection.count(),
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_db": "ChromaDB"
        }
    except:
        return {"status": "not_initialized"}


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing RAG System (English Knowledge Base)")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "my order hasn't arrived yet",
        "how do I get a refund",
        "how long does shipping take",
        "I received a damaged product",
        "what payment methods do you accept",
        "how can I track my order",
    ]
    
    print("\n🧪 Running retrieval tests...\n")
    
    for query in test_queries:
        print(f"Query: \"{query}\"")
        docs = retrieve_knowledge(query, top_k=2)
        
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] ({doc.source}) [relevance: {doc.relevance:.2f}]")
            print(f"      {doc.content[:80]}...")
        print("-" * 60)
    
    print("\n📊 Knowledge Base Stats:")
    print(get_knowledge_base_stats())
    
    print("\n✅ RAG System working!")