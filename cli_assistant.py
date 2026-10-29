"""
Saarthi AI - Command Line Interactive Interface
Ideal for headless testing, low-latency terminal interactions, and server deployments.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from saarthi.core.intent_classifier import IntentClassifier
from saarthi.core.executor import ActionExecutor
from saarthi.core.speech_engine import SpeechEngine

def run_cli():
    print("=" * 65)
    print("   SAARTHI AI - INTERACTIVE TERMINAL CONSOLE")
    print("   Type 'exit', 'quit', or 'q' to end the session.")
    print("   Type 'voice' to trigger microphone speech input.")
    print("=" * 65)

    classifier = IntentClassifier()
    executor = ActionExecutor(classifier)
    speech = SpeechEngine()

    speech.speak("Saarthi terminal interface ready. How may I help you?")

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q", "shutdown"]:
                print("\nSaarthi > Goodbye! Have a great day.")
                speech.speak("Goodbye! Have a great day.", block=True)
                break

            if user_input.lower() == "voice":
                print("\n[Voice Mode] Listening to microphone for 5 seconds...")
                try:
                    user_input = speech.listen_from_mic(timeout=5, phrase_time_limit=6)
                    print(f"You (Spoken) > {user_input}")
                except Exception as e:
                    print(f"[Mic Error] {str(e)}")
                    continue

            result = executor.process_query(user_input)

            # Display rich telemetry
            print("-" * 65)
            print(f"🧠 [Neural Telemetry] Intent: [{result['intent']}] | Confidence: {result['confidence']}% | Latency: {result['latency_ms']} ms")
            print(f"⚡ [Action Executed] {result['action_taken']}")
            print(f"💬 Saarthi > {result['response_text']}")
            print("-" * 65)

            # Speak response
            speech.speak(result['response_text'])

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated by user.")
            break
        except Exception as e:
            print(f"\n[Error] {str(e)}")

if __name__ == "__main__":
    run_cli()
