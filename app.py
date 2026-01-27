"""
E-Commerce Customer Service Chatbot
====================================

Main application file that integrates all modules.

RUN WITH:
    streamlit run app.py

REQUIREMENTS:
    pip install streamlit
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from modules
from interfaces import Message, Intent, Entities
from src.speech.transcriber import transcribe_audio, process_uploaded_audio
from src.speech.classifier import classify_intent, get_intent_description
from src.chatbot.rag import retrieve_knowledge
from src.chatbot.llm import generate_response
from src.nlp.ner import extract_entities
from src.nlp.summarizer import summarize_conversation

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="E-Commerce Customer Service",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🛒 E-Commerce CS Bot")
    st.markdown("---")
    
    # Conversation Summary
    st.subheader("📝 Conversation Summary")
    if len(st.session_state.conversation_history) >= 2:
        summary = summarize_conversation(st.session_state.conversation_history)
        st.info(summary)
    else:
        st.write("_Mulai percakapan untuk melihat ringkasan_")
    
    st.markdown("---")
    
    # Debug Info (can hide in production)
    with st.expander("🔧 Debug Info"):
        st.write(f"Total messages: {len(st.session_state.messages)}")
        if st.session_state.messages:
            last_msg = st.session_state.messages[-1]
            if "intent" in last_msg:
                st.write(f"Last intent: {last_msg['intent']}")
            if "entities" in last_msg:
                st.write(f"Last entities: {last_msg['entities']}")
    
    # Clear conversation button
    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

# ============================================================
# MESSAGE PROCESSING FUNCTION (defined before use)
# ============================================================

def process_user_message(user_text: str):
    """
    Main processing pipeline for user messages.
    """
    # 1. Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_text
    })
    
    # 2. Classify intent (Person A)
    intent = classify_intent(user_text)
    intent_desc = get_intent_description(intent.label)
    
    # 3. Extract entities (Person C)
    entities = extract_entities(user_text)
    
    # Update last message with detected info
    st.session_state.messages[-1]["intent"] = f"{intent.label} ({intent.confidence:.0%})"
    st.session_state.messages[-1]["entities"] = {
        k: v for k, v in {
            "order_id": entities.order_id,
            "product": entities.product_name,
            "date": entities.date,
            "amount": entities.amount
        }.items() if v is not None
    }
    
    # 4. Retrieve relevant knowledge (Person B)
    retrieved_docs = retrieve_knowledge(user_text)
    
    # 5. Generate response (Person B)
    response = generate_response(
        user_message=user_text,
        intent=intent,
        entities=entities,
        retrieved_docs=retrieved_docs,
        conversation_history=st.session_state.conversation_history
    )
    
    # 6. Add bot response to chat
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    
    # 7. Update conversation history for summary
    st.session_state.conversation_history.append(
        Message(role="user", content=user_text)
    )
    st.session_state.conversation_history.append(
        Message(role="assistant", content=response)
    )
    
    # Rerun to update UI
    st.rerun()


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================

st.title("🛒 Customer Service Chatbot")
st.caption("Powered by AI | E-Commerce Support")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show detected info for user messages
        if message["role"] == "user" and "intent" in message:
            with st.expander("🔍 Detected Info", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Intent:** {message['intent']}")
                with col2:
                    if message.get("entities"):
                        st.write(f"**Entities:** {message['entities']}")

# ============================================================
# INPUT METHODS
# ============================================================

st.markdown("---")

# Create tabs for different input methods
tab_text, tab_audio = st.tabs(["💬 Text Input", "🎤 Voice Input"])

with tab_text:
    # Text input
    user_input = st.chat_input("Ketik pesan Anda di sini...")
    
    if user_input:
        # Process text input
        process_user_message(user_input)

with tab_audio:
    st.write("**Rekam suara Anda:**")
    
    # Audio input (Streamlit's built-in audio recorder)
    audio_value = st.audio_input("Tekan untuk merekam")
    
    if audio_value:
        # Process audio
        with st.spinner("🎤 Transcribing audio..."):
            # Convert audio to text
            transcribed_text = process_uploaded_audio(audio_value)
        
        st.success(f"**Transcribed:** {transcribed_text}")
        
        if st.button("📤 Send Transcribed Text"):
            process_user_message(transcribed_text)

# ============================================================
# QUICK ACTION BUTTONS
# ============================================================

st.markdown("---")
st.subheader("⚡ Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📦 Cek Status Pesanan", use_container_width=True):
        process_user_message("Saya mau cek status pesanan saya")

with col2:
    if st.button("🚚 Keluhan Pengiriman", use_container_width=True):
        process_user_message("Pesanan saya belum sampai, sudah lama")

with col3:
    if st.button("↩️ Return/Refund", use_container_width=True):
        process_user_message("Saya mau refund pesanan saya")

with col4:
    if st.button("❓ Bantuan Lainnya", use_container_width=True):
        process_user_message("Saya butuh bantuan")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("🎓 NLP Final Project | E-Commerce Customer Service Chatbot")
st.caption("Built with Streamlit, Whisper, RAG, and LLM")
