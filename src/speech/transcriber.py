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
    # STUB - Remove this and implement real code
    return "Halo, pesanan saya ORDER123 belum sampai sudah 5 hari, tolong dicek dong"


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
    # STUB - Remove this and implement real code
    return b"dummy_audio_bytes"


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
    # STUB
    return "Pesanan ORDER456 saya rusak, minta refund"


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
