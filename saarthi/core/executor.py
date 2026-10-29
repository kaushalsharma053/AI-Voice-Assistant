"""
Action Execution and Intent Dispatch Router for Saarthi AI
Maps deep learning intent predictions to system, web, calculation, and conversational skills.
Answers questions in-place without forcefully hijacking the user's browser.
"""

from saarthi.core.intent_classifier import IntentClassifier
from saarthi.skills import system_skills, web_skills, info_skills, productivity_skills, chat_skills

class ActionExecutor:
    """
    Directs classified intents to specific skill handlers with parameter extraction and intelligent in-place answers.
    """
    def __init__(self, classifier: IntentClassifier = None):
        self.classifier = classifier or IntentClassifier()

    def process_query(self, user_input: str, open_browser_on_search: bool = False) -> dict:
        """
        End-to-end intent processing:
        NLP Preprocessing -> Neural Prediction -> Skill Dispatch -> Response Generation.
        Answers queries in-place without disruptive desktop popups.
        """
        raw_text = user_input.strip()
        if not raw_text:
            return {
                "response_text": "I didn't catch that. Please speak or type a command.",
                "intent": "none",
                "confidence": 0.0,
                "latency_ms": 0.0,
                "action_taken": "none"
            }

        # Step 1: Deep Learning Intent Classification
        prediction = self.classifier.predict(raw_text)
        intent = prediction.get("intent", "unknown")
        confidence = prediction.get("confidence", 0.0)
        latency_ms = prediction.get("latency_ms", 0.0)
        top_k = prediction.get("top_k", [])

        response_text = ""
        action_taken = intent

        # Step 2: Intent Dispatch Logic
        if intent == "greeting":
            response_text = self.classifier.get_default_response("greeting") or "Hello! I am Saarthi. How can I assist you today?"
        
        elif intent == "goodbye":
            response_text = self.classifier.get_default_response("goodbye") or "Goodbye! Have a great day."

        elif intent == "identity":
            response_text = self.classifier.get_default_response("identity") or "I am Saarthi AI, your multimodal deep learning desktop assistant."

        elif intent == "creator":
            response_text = self.classifier.get_default_response("creator") or "I was engineered as a 7th-semester Computer Science capstone project."

        elif intent == "capabilities":
            response_text = (
                "Here is what I can do: 1) System Control (Apps, Volume, Screenshot, CPU/RAM stats), "
                "2) Factual Knowledge (Wikipedia summaries, Live Weather & News), "
                "3) Safe Math Calculations, 4) Notes & Viva Assistance."
            )

        elif intent == "system_status":
            response_text = system_skills.get_system_status()

        elif intent == "open_app":
            response_text = system_skills.open_application(raw_text)

        elif intent == "volume_up":
            response_text = system_skills.control_volume("up")

        elif intent == "volume_down":
            response_text = system_skills.control_volume("down")

        elif intent == "volume_mute":
            response_text = system_skills.control_volume("mute")

        elif intent == "take_screenshot":
            response_text = system_skills.take_screenshot()

        elif intent == "lock_pc":
            response_text = system_skills.lock_screen()

        elif intent == "get_time":
            response_text = info_skills.get_current_time()

        elif intent == "get_date":
            response_text = info_skills.get_current_date()

        elif intent == "get_weather":
            response_text = info_skills.get_weather(raw_text)

        elif intent == "get_news":
            response_text = info_skills.get_news_headlines()

        elif intent == "calculate":
            response_text = info_skills.calculate_expression(raw_text)

        elif intent == "wikipedia_search":
            response_text = web_skills.search_wikipedia(raw_text)

        elif intent == "play_youtube":
            response_text = web_skills.play_youtube(raw_text, open_browser=open_browser_on_search)

        elif intent == "web_search":
            # For general searches, first attempt direct knowledge lookup
            wiki_ans = web_skills.search_wikipedia(raw_text)
            if "could not find a direct encyclopedia summary" not in wiki_ans and "What topic would you like" not in wiki_ans:
                response_text = wiki_ans
                action_taken = "knowledge_search"
            else:
                response_text = web_skills.search_google(raw_text, open_browser=open_browser_on_search)

        elif intent == "take_note":
            response_text = productivity_skills.take_note(raw_text)

        elif intent == "read_notes":
            response_text = productivity_skills.read_notes()

        elif intent == "clear_notes":
            response_text = productivity_skills.clear_notes()

        elif intent == "joke":
            response_text = self.classifier.get_default_response("joke") or "Why did the neural network cross the road? To optimize its loss function!"

        elif intent == "quote":
            response_text = chat_skills.get_motivational_quote()

        elif intent == "college_viva_helper":
            response_text = chat_skills.get_viva_explanation(raw_text)

        elif intent in ["how_are_you", "thanks", "compliment"]:
            response_text = self.classifier.get_default_response(intent) or "Thank you! I am always here to assist."

        else:
            # Step 3: Intelligent In-Place Fallback (No intrusive browser popups)
            query_lower = raw_text.lower()
            
            # Sub-check A: Mathematical operations
            if any(symbol in query_lower for symbol in ["+", "-", "*", "/", "sqrt", "calculate"]):
                response_text = info_skills.calculate_expression(raw_text)
                action_taken = "math_fallback"
            
            # Sub-check B: Direct Knowledge / Wikipedia lookup
            else:
                wiki_ans = web_skills.search_wikipedia(raw_text)
                if "could not find a direct encyclopedia summary" not in wiki_ans and "What topic would you like" not in wiki_ans:
                    response_text = wiki_ans
                    action_taken = "wikipedia_knowledge"
                else:
                    response_text = (
                        f"I don't have that specific topic in my local training model yet (Confidence: {confidence:.1f}%). "
                        f"You can explore more online: https://www.google.com/search?q={raw_text.replace(' ', '+')}"
                    )
                    action_taken = "conversational_fallback"

        return {
            "response_text": response_text,
            "intent": intent,
            "confidence": round(confidence, 1),
            "latency_ms": latency_ms,
            "top_k": top_k,
            "action_taken": action_taken,
            "raw_text": raw_text
        }
