"""
Person A: Speech Transcription Module
=====================================

YOUR TASK:
1. Implement real-time microphone recording
2. Transcribe audio using OpenAI Whisper
3. Return clean text

LIBRARIES TO USE:
- whisper (OpenAI's Whisper model)
- sounddevice or pyaudio (for microphone)
- numpy (audio processing)

SETUP:
pip install openai-whisper sounddevice numpy

NOTE: Whisper model will download on first run (~1.5GB for 'base')
"""

import numpy as np
from typing import Optional

# ============================================================
# STUB IMPLEMENTATION - Replace with real code!
# ============================================================

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    STUB: Returns dummy text for testing.
    
    TODO (Person A):
    1. Load Whisper model (recommend 'base' or 'small' for speed)
    2. Convert audio_bytes to numpy array
    3. Run whisper transcription
    4. Return transcribed text
    
    Real implementation example:
    ```python
    import whisper
    
    model = whisper.load_model("base")
    
    def transcribe_audio(audio_bytes: bytes) -> str:
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # Transcribe
        result = model.transcribe(audio_np, language="id")
        
        return result["text"].strip()
    ```
    """
<<<<<<< HEAD
    try:
        print("[Whisper] Loading model for English transcription...")
        
        # Load Whisper model - using 'base' for speed/balance
        # Change to 'small' for better accuracy if needed
        model = whisper.load_model("base")
        
        print(f"[Whisper] Processing {len(audio_bytes)} bytes of audio...")
        
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        
        if len(audio_np) == 0:
            print("[Whisper] Empty audio data")
            return ""
        
        # Normalize audio volume (prevent distortion)
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val * 0.9  # Scale to 90%
            print(f"[Whisper] Normalized audio (peak: {max_val:.3f})")
        
        # Transcribe with English language focus
        result = model.transcribe(
            audio_np,
            language="en",           # Force English language
            task="transcribe",
            fp16=False,              # Disable for CPU compatibility
            temperature=0.0,         # Less randomness, more consistent
            best_of=3,               # Better accuracy
            beam_size=3,             # Beam search for accuracy
            initial_prompt="This is an English conversation about e-commerce customer service. The user is asking about orders, delivery, returns, or products."  # Context helps accuracy
        )
        
        raw_text = result["text"].strip()
        print(f"[Whisper] Raw transcription: '{raw_text}'")
        
        # ENGLISH TEXT CLEANING AND PROCESSING
        import re
        
        # 1. Remove non-English characters (keep only Latin alphabet, numbers, punctuation)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'":;-]', '', raw_text)
        
        # 2. Remove common filler words and sounds
        filler_words = ['um', 'uh', 'ah', 'er', 'mm', 'hm', 'like', 'you know', 'actually', 'basically']
        for word in filler_words:
            text = re.sub(r'\b' + word + r'\b', '', text, flags=re.IGNORECASE)
        
        # 3. Fix common e-commerce abbreviations
        abbreviations = {
            r'\bpls\b': 'please',
            r'\bthx\b': 'thanks',
            r'\bty\b': 'thank you',
            r'\basap\b': 'as soon as possible',
            r'\bapprox\b': 'approximately',
            r'\binfo\b': 'information',
            r'\bref\b': 'reference',
            r'\bqty\b': 'quantity',
            r'\bdeliv\b': 'delivery',
            r'\bprod\b': 'product',
            r'\border\b': 'order',  # Ensure correct spelling
            r'\brefund\b': 'refund'
        }
        
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 4. Standardize informal contractions
        informal_to_formal = {
            r'\bwanna\b': 'want to',
            r'\bgonna\b': 'going to',
            r'\bdunno\b': "don't know",
            r'\blemme\b': 'let me',
            r'\bgimme\b': 'give me',
            r'\bgotta\b': 'got to',
            r'\bhafta\b': 'have to',
            r'\bkinda\b': 'kind of',
            r'\bsorta\b': 'sort of',
            r'\boutta\b': 'out of'
        }
        
        for informal, formal in informal_to_formal.items():
            text = re.sub(informal, formal, text, flags=re.IGNORECASE)
        
        # 5. Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # 6. Capitalize first letter if needed
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        
        # 7. Ensure proper spacing after punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
        
        print(f"[Whisper] Cleaned English: '{text}'")
        return text
        
    except Exception as e:
        print(f"[ERROR] English transcription failed: {str(e)}")
        # Return empty string as fallback
        return ""
=======
    # STUB - Remove this and implement real code
    return "Halo, pesanan saya ORDER123 belum sampai sudah 5 hari, tolong dicek dong"

>>>>>>> parent of cd924e0 (Person A: Speech Transcription Module)

def record_from_microphone(duration: int = 10, sample_rate: int = 16000) -> bytes:
    """
    Records audio from microphone.
    
    STUB: Returns dummy bytes for testing.
    
    TODO (Person A):
    1. Use sounddevice to record from mic
    2. Return audio as bytes
    
    Real implementation example:
    ```python
    import sounddevice as sd
    
    def record_from_microphone(duration: int = 5, sample_rate: int = 16000) -> bytes:
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()  # Wait until recording is finished
        print("Recording complete!")
        return audio.tobytes()
    ```
    """
<<<<<<< HEAD
    try:
        print(f"[Recording] Starting {duration}-second recording...")
        print("🎤 Please speak in English (e-commerce related)")
        print("💡 Example: 'Hello, I want to check my order status'")
        
        # Check available audio devices
        try:
            devices = sd.query_devices()
            print(f"[Recording] Using audio device: {sd.default.device}")
        except:
            print("[Recording] Using default audio device")
        
        # Record audio
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,           # Mono recording
            dtype=np.float32,     # Whisper expects float32
            device=sd.default.device[0] if hasattr(sd.default, 'device') else None
        )
        
        # Show recording progress
        print("[Recording] 🎙️ Recording...", end='', flush=True)
        sd.wait()  # Wait for recording to complete
        print(" ✅ Complete!")
        
        # Convert to bytes
        audio_bytes = audio.tobytes()
        print(f"[Recording] Captured {len(audio_bytes)} bytes")
        
        # Simple audio quality check
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        if len(audio_np) > 0:
            volume = np.max(np.abs(audio_np))
            if volume < 0.05:
                print("⚠️  Warning: Low volume detected")
                print("💡 Tip: Speak closer to the microphone")
            elif volume > 0.95:
                print("⚠️  Warning: High volume (possible clipping)")
                print("💡 Tip: Move slightly away from microphone")
            else:
                print("✅ Audio volume: Good")
        
        return audio_bytes
        
    except Exception as e:
        print(f"[ERROR] Microphone recording failed: {str(e)}")
        print("💡 Troubleshooting:")
        print("   1. Check if microphone is connected")
        print("   2. Grant microphone permissions")
        print("   3. Try different audio device")
        return b""  # Return empty bytes on error
=======
    # STUB - Remove this and implement real code
    return b"dummy_audio_bytes"
>>>>>>> parent of cd924e0 (Person A: Speech Transcription Module)


# ============================================================
# FOR STREAMLIT INTEGRATION
# ============================================================

def process_uploaded_audio(uploaded_file) -> str:
    """
    Process audio file uploaded via Streamlit.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Transcribed text
        
    TODO (Person A): Implement this for Streamlit file upload
    """
<<<<<<< HEAD
    try:
        # Check if pydub is available for format conversion
        try:
            from pydub import AudioSegment
            import io
            
            # Read audio file
            audio_bytes = uploaded_file.getvalue()
            
            # Convert to WAV if needed
            if uploaded_file.name.lower().endswith(('.mp3', '.m4a', '.ogg', '.flac')):
                print(f"[Upload] Converting {uploaded_file.name} to WAV format...")
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                
                # Convert to WAV: 16kHz mono for Whisper
                audio = audio.set_frame_rate(16000).set_channels(1)
                
                # Export to WAV bytes
                buffer = io.BytesIO()
                audio.export(buffer, format="wav")
                audio_bytes = buffer.getvalue()
            
            print(f"[Upload] Processing {uploaded_file.name} ({len(audio_bytes)} bytes)")
            
        except ImportError:
            # Fallback: assume it's already WAV format
            print(f"[Upload] Processing {uploaded_file.name} (assuming WAV format)")
            audio_bytes = uploaded_file.getvalue()
        
        # Use the same transcription function
        text = transcribe_audio(audio_bytes)
        
        return text
        
    except Exception as e:
        print(f"[ERROR] Audio upload processing failed: {str(e)}")
        return ""
=======
    # STUB
    return "Pesanan ORDER456 saya rusak, minta refund"
>>>>>>> parent of cd924e0 (Person A: Speech Transcription Module)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    # Test stub functions
    print("Testing transcribe_audio stub:")
    result = transcribe_audio(b"test")
    print(f"Result: {result}")
    print(f"Type: {type(result)}")
    
    print("\nTesting record_from_microphone stub:")
    audio = record_from_microphone(duration=3)
    print(f"Audio bytes length: {len(audio)}")
    
    print("\n✅ Stubs working! Ready for real implementation.")
<<<<<<< HEAD

# ============================================================
# TESTING FUNCTIONS
# ============================================================

def test_transcription():
    """Test transcription with synthetic audio"""
    print("\n" + "="*50)
    print("TEST 1: Transcription Test")
    print("="*50)
    
    # Create 1 second of synthetic speech (sine wave)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate a simple tone (440 Hz)
    synthetic_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    audio_bytes = synthetic_audio.tobytes()
    
    print(f"Generated {len(audio_bytes)} bytes of synthetic audio")
    result = transcribe_audio(audio_bytes)
    print(f"Synthetic audio transcription: '{result}'")
    
    return result


def test_recording():
    """Test microphone recording functionality"""
    print("\n" + "="*50)
    print("TEST 2: Microphone Recording Test")
    print("="*50)
    
    # Record for 3 seconds
    audio_bytes = record_from_microphone(duration=3)
    
    if audio_bytes and len(audio_bytes) > 0:
        print(f"✅ Recording successful: {len(audio_bytes)} bytes")
        return audio_bytes
    else:
        print("❌ Recording failed or no audio captured")
        return None


def test_english_pipeline():
    """Full English e-commerce speech pipeline test"""
    print("\n" + "="*50)
    print("TEST 3: English E-commerce Pipeline Test")
    print("="*50)
    
    # Common e-commerce phrases in English
    test_phrases = [
        "Hello, I want to check my order status",
        "My order hasn't arrived yet",
        "The product I received is damaged",
        "I want a refund for my purchase",
        "How much does this product cost?",
        "When will my order be delivered?",
        "Can I return this item?",
        "Do you have this in stock?",
        "Where is my package?",
        "I need to cancel my order"
    ]
    
    import random
    selected_phrase = random.choice(test_phrases)
    
    print(f"\n💬 TRY SAYING THIS ENGLISH PHRASE:")
    print(f"   '{selected_phrase}'")
    print("\nPress Enter when ready to record...")
    input()
    
    # Step 1: Record
    print("\n1. 🎤 Recording audio...")
    audio_bytes = record_from_microphone(duration=4)
    
    if not audio_bytes or len(audio_bytes) == 0:
        print("⚠️  No audio recorded, using synthetic audio for demo")
        # Generate synthetic audio as fallback
        sample_rate = 16000
        duration = 3
        t = np.linspace(0, duration, int(sample_rate * duration))
        synthetic_audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        audio_bytes = synthetic_audio.tobytes()
    
    # Step 2: Transcribe
    print("\n2. 🔄 Transcribing to English text...")
    text = transcribe_audio(audio_bytes)
    
    print(f"\n🎯 FINAL TRANSCRIPTION RESULT:")
    print(f"   '{text}'")
    
    # Save for reference
    if text:
        with open("english_transcription.txt", "w", encoding="utf-8") as f:
            f.write(f"Test phrase: {selected_phrase}\n")
            f.write(f"Transcription: {text}\n")
            f.write(f"Audio bytes: {len(audio_bytes)}\n")
            f.write(f"Timestamp: {__import__('datetime').datetime.now()}\n")
        print("💾 Saved to 'english_transcription.txt'")
    
    return text


# ============================================================
# MAIN TEST RUNNER
# ============================================================

if __name__ == "__main__":
    print("\n" + "🔊" * 20)
    print("ENGLISH SPEECH-TO-TEXT MODULE TEST")
    print("🔊" * 20)
    print("E-commerce Customer Service Chatbot")
    print("=" * 50)
    
    # Run basic transcription test
    test_transcription()
    
    # Ask user for recording test
    response = input("\nTest microphone recording? (y/n): ").lower().strip()
    if response == 'y':
        audio_data = test_recording()
        
        # If recording successful, ask to transcribe
        if audio_data and len(audio_data) > 0:
            transcribe = input("\nTranscribe the recording? (y/n): ").lower().strip()
            if transcribe == 'y':
                text = transcribe_audio(audio_data)
                print(f"\n📝 Your English recording transcribed as: '{text}'")
    
    # Test full English e-commerce pipeline
    run_full = input("\nRun full English e-commerce pipeline test? (y/n): ").lower().strip()
    if run_full == 'y':
        test_english_pipeline()
    
    print("\n" + "✅" * 20)
    print("ALL TESTS COMPLETED!")
    print("✅" * 20)
    print("\n🎉 English Speech-to-Text module is ready!")
=======
>>>>>>> parent of cd924e0 (Person A: Speech Transcription Module)
