"""
Speech Subsystem for Saarthi AI
Provides offline Text-to-Speech (TTS via pyttsx3 / Windows SAPI5) and Speech Recognition (STT via SpeechRecognition).
Engineered to be fully non-blocking and thread-safe for GUI integration.
"""

import threading
import queue
import time
from saarthi.config import DEFAULT_VOICE_RATE, DEFAULT_VOICE_VOLUME

# Dynamic import for speech_recognition to gracefully support text-only environments
try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

class SpeechEngine:
    """
    Manages audio input from microphone and spoken audio output via TTS.
    """
    def __init__(self, voice_gender="female", rate=DEFAULT_VOICE_RATE, volume=DEFAULT_VOICE_VOLUME):
        self.voice_gender = voice_gender
        self.rate = rate
        self.volume = volume
        self.is_speaking = False
        self.is_listening = False

        # TTS Queue for non-blocking speech
        self.speech_queue = queue.Queue()
        self._tts_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._tts_thread.start()

        # Speech Recognizer
        self.recognizer = None
        self.microphone = None
        self._init_stt()

    def _init_stt(self):
        """Initializes microphone and speech recognition instance if supported."""
        if HAS_SPEECH_RECOGNITION:
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.energy_threshold = 300
                self.recognizer.pause_threshold = 0.8
                self.microphone = sr.Microphone()
                print("[Voice Engine] Speech recognition subsystem ready.")
            except Exception as e:
                print(f"[Voice Engine Warning] Microphone initialization failed: {str(e)}")
                self.microphone = None
        else:
            print("[Voice Engine] SpeechRecognition package not loaded. Text input active.")

    def _speech_worker(self):
        """Worker thread dedicated to handling pyttsx3 speech tasks without blocking GUI."""
        if not HAS_PYTTSX3:
            return

        while True:
            text = self.speech_queue.get()
            if text is None:
                break

            try:
                self.is_speaking = True
                engine = pyttsx3.init()
                engine.setProperty('rate', self.rate)
                engine.setProperty('volume', self.volume)

                voices = engine.getProperty('voices')
                if voices:
                    # Select David (Male) or Zira (Female)
                    selected_voice = voices[0].id
                    for v in voices:
                        if self.voice_gender == "female" and ("zira" in v.name.lower() or "female" in v.name.lower()):
                            selected_voice = v.id
                            break
                        elif self.voice_gender == "male" and ("david" in v.name.lower() or "male" in v.name.lower()):
                            selected_voice = v.id
                            break
                    engine.setProperty('voice', selected_voice)

                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[TTS Error] {str(e)}")
            finally:
                self.is_speaking = False
                self.speech_queue.task_done()

    def speak(self, text: str, block: bool = False):
        """Queues text to be spoken via Text-To-Speech."""
        if not text or not HAS_PYTTSX3:
            return

        # Clean text from Markdown or special symbols before speaking
        clean_text = text.replace("*", "").replace("#", "").replace("`", "").strip()
        self.speech_queue.put(clean_text)

        if block:
            self.speech_queue.join()

    def set_voice_gender(self, gender: str):
        """Switches between 'male' and 'female' TTS voices."""
        if gender.lower() in ["male", "female"]:
            self.voice_gender = gender.lower()

    def set_rate(self, rate: int):
        """Adjusts speech tempo (words per minute)."""
        self.rate = max(100, min(300, rate))

    def listen_from_mic(self, timeout: int = 5, phrase_time_limit: int = 7) -> str:
        """
        Captures speech from microphone and returns transcribed text.
        Returns empty string or raises exception if speech is undetected.
        """
        if not HAS_SPEECH_RECOGNITION or not self.microphone or not self.recognizer:
            raise RuntimeError("Microphone or SpeechRecognition library is not available.")

        with self.microphone as source:
            self.is_listening = True
            try:
                # Adjust for ambient noise briefly
                self.recognizer.adjust_for_ambient_noise(source, duration=0.4)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                self.is_listening = False

                # Transcribe with Google Speech Recognition
                transcription = self.recognizer.recognize_google(audio)
                return transcription.strip()
            except sr.WaitTimeoutError:
                self.is_listening = False
                return ""
            except sr.UnknownValueError:
                self.is_listening = False
                return ""
            except Exception as e:
                self.is_listening = False
                raise e
