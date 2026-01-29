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

import whisper
import numpy as np
import sounddevice as sd
import tempfile
import os
import wave
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
    try:
        model = whisper.load_model("base")
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        
        if len(audio_np) == 0:
            return ""
        
        # SIMPLE NORMALIZATION
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val * 0.9
        
        result = model.transcribe(
            audio_np,
            language="id",
            task="transcribe",
            fp16=False,
            temperature=0.0  # Important: less randomness
        )
        
        text = result["text"].strip()
        
        # SIMPLE CLEANING
        import re
        text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove weird chars
        text = ' '.join(text.split())  # Remove extra spaces
        
        return text
        
    except Exception as e:
        print(f"Error: {e}")
        return ""

def record_from_microphone(duration: int = 5, sample_rate: int = 16000) -> bytes:
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
    try:
        print(f"[Recording] Starting {duration}s recording...")
        print("🎤 Speak now (Indonesian): 'Halo, saya mau cek pesanan'")
        
        # Record audio
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,  # Mono
            dtype=np.float32
        )
        
        # Wait for recording to complete
        sd.wait()
        
        print("[Recording] Complete!")
        
        # Convert to bytes
        audio_bytes = audio.tobytes()
        print(f"[Recording] Captured {len(audio_bytes)} bytes")
        
        return audio_bytes
        
    except Exception as e:
        print(f"[ERROR] Recording failed: {str(e)}")
        print("💡 Tips: Check microphone connection and permissions")
        return b""  # Return empty bytes on error


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
    try:
        # Read audio bytes from uploaded file
        audio_bytes = uploaded_file.getvalue()
        
        print(f"[Upload] Processing {uploaded_file.name} ({len(audio_bytes)} bytes)")
        
        # Use same transcription function
        text = transcribe_audio(audio_bytes)
        
        return text
        
    except Exception as e:
        print(f"[ERROR] Upload processing failed: {str(e)}")
        return ""


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

# ============================================================
# TEST FUNCTIONS (Run directly in this file)
# ============================================================

def test_transcription():
    """Test transcription with dummy audio"""
    print("\n" + "="*50)
    print("TEST 1: Transcription with silence")
    print("="*50)
    
    # Create 1 second of silence
    sample_rate = 16000
    samples = sample_rate * 1  # 1 second
    silence = np.zeros(samples, dtype=np.float32)
    audio_bytes = silence.tobytes()
    
    result = transcribe_audio(audio_bytes)
    print(f"Silence transcription: '{result}'")
    return result


def test_recording():
    """Test microphone recording"""
    print("\n" + "="*50)
    print("TEST 2: Microphone Recording")
    print("="*50)
    
    # Record for 3 seconds
    audio_bytes = record_from_microphone(duration=3)
    
    if audio_bytes:
        print(f"✅ Recording successful: {len(audio_bytes)} bytes")
        return audio_bytes
    else:
        print("❌ Recording failed")
        return None


def test_full_pipeline():
    """Full test: Record → Transcribe"""
    print("\n" + "="*50)
    print("TEST 3: Full Pipeline (Record + Transcribe)")
    print("="*50)
    
    # Step 1: Record
    print("\n1. Recording...")
    audio_bytes = record_from_microphone(duration=4)
    
    if not audio_bytes:
        print("⚠️ Using dummy audio for testing")
        # Create dummy audio with some noise
        sample_rate = 16000
        duration = 4
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Simple sine wave (440 Hz)
        dummy_audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio_bytes = dummy_audio.tobytes()
    
    # Step 2: Transcribe
    print("\n2. Transcribing...")
    text = transcribe_audio(audio_bytes)
    
    print(f"\n🎯 FINAL TRANSCRIPTION: '{text}'")
    
    # Save result
    if text:
        with open("last_transcription.txt", "w", encoding="utf-8") as f:
            f.write(f"Audio bytes: {len(audio_bytes)}\n")
            f.write(f"Transcription: {text}\n")
        print("💾 Saved to 'last_transcription.txt'")
    
    return text


# ============================================================
# MAIN TEST RUNNER
# ============================================================

if __name__ == "__main__":
    print("\n" + "🎤"*20)
    print("SPEECH-TO-TEXT MODULE TEST")
    print("🎤"*20)
    
    # Run tests
    test_transcription()
    
    # Ask user if they want to test recording
    response = input("\nTest microphone recording? (y/n): ").lower()
    if response == 'y':
        audio_data = test_recording()
        
        # If recording successful, ask to transcribe
        if audio_data and len(audio_data) > 0:
            transcribe = input("\nTranscribe the recording? (y/n): ").lower()
            if transcribe == 'y':
                text = transcribe_audio(audio_data)
                print(f"\n📝 Your recording transcribed as: '{text}'")
    
    # Test full pipeline
    run_full = input("\nRun full pipeline test? (y/n): ").lower()
    if run_full == 'y':
        test_full_pipeline()
    
    print("\n" + "✅"*20)
    print("ALL TESTS COMPLETED!")
    print("✅"*20)
    print("\nYour module is ready to be integrated with app.py")
