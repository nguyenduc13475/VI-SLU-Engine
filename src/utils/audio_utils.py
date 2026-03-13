import speech_recognition as sr
from typing import Optional

def transcribe_audio_from_mic(timeout: int = 5, language: str = "vi-VN") -> Optional[str]:
    """
    Listens to the microphone and converts speech to text using Google Speech Recognition.
    
    Args:
        timeout (int): Maximum seconds to wait for speech.
        language (str): Language code.
        
    Returns:
        Optional[str]: Transcribed text, or None if failed.
    """
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.8
    recognizer.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        print("\n🎙️ Listening for commands...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            text = recognizer.recognize_google(audio, language=language)
            return text.lower()
        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
        except sr.RequestError as e:
            print(f"❌ Could not request results; {e}")
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            
    return None