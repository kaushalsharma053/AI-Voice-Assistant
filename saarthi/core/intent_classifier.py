"""
Deep Learning Intent Classification and Neural Inference Engine
Loads trained Keras neural network, performs real-time tokenization, and computes intent confidence.
"""

import os
import time
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

from saarthi.config import (
    MODEL_FILE,
    TOKENIZER_FILE,
    LABEL_ENCODER_FILE,
    INTENTS_FILE,
    MAX_SEQUENCE_LENGTH,
    CONFIDENCE_THRESHOLD
)

class IntentClassifier:
    """
    Manages loading of Deep Learning assets and executes real-time inference.
    """
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.intents_data = None
        self.is_loaded = False
        self.load_assets()

    def load_assets(self):
        """Loads model, tokenizer, label encoder, and intents metadata."""
        if not (os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE) and os.path.exists(LABEL_ENCODER_FILE)):
            print(f"[Warning] Trained DL model files not found. Please execute train_model.py first.")
            return False

        try:
            print("[DL Engine] Loading Deep Learning model from disk...")
            self.model = keras.models.load_model(MODEL_FILE)

            with open(TOKENIZER_FILE, 'rb') as f:
                self.tokenizer = pickle.load(f)

            with open(LABEL_ENCODER_FILE, 'rb') as f:
                self.label_encoder = pickle.load(f)

            if os.path.exists(INTENTS_FILE):
                with open(INTENTS_FILE, 'r', encoding='utf-8') as f:
                    self.intents_data = json.load(f)

            self.is_loaded = True
            print(f"[DL Engine] Neural Network loaded successfully. Classes: {len(self.label_encoder.classes_)}")
            return True
        except Exception as e:
            print(f"[DL Engine Error] Could not load model: {str(e)}")
            self.is_loaded = False
            return False

    def predict(self, text: str) -> dict:
        """
        Classifies input text sentence and outputs predicted intent, confidence, and latency.
        """
        if not self.is_loaded or not text.strip():
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "latency_ms": 0.0,
                "top_k": [],
                "raw_text": text
            }

        start_time = time.perf_counter()

        cleaned_text = text.strip().lower()
        seq = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

        # Neural forward pass
        predictions = self.model.predict(padded, verbose=0)[0]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Top predicted class
        top_idx = int(np.argmax(predictions))
        top_intent = self.label_encoder.inverse_transform([top_idx])[0]
        confidence = float(predictions[top_idx]) * 100.0

        # Top 3 predictions for academic transparency
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_k = [
            (self.label_encoder.inverse_transform([idx])[0], float(predictions[idx]) * 100.0)
            for idx in top_3_indices
        ]

        # Apply confidence threshold
        is_confident = (confidence / 100.0) >= CONFIDENCE_THRESHOLD

        return {
            "intent": top_intent if is_confident else "low_confidence",
            "detected_intent": top_intent,
            "confidence": confidence,
            "latency_ms": round(latency_ms, 2),
            "top_k": top_k,
            "raw_text": text,
            "is_confident": is_confident
        }

    def get_default_response(self, intent_tag: str):
        """Fetches canned conversational responses from intents.json if available."""
        if not self.intents_data:
            return None

        import random
        for intent in self.intents_data.get('intents', []):
            if intent['tag'] == intent_tag:
                responses = intent.get('responses', [])
                if responses:
                    return random.choice(responses)
        return None
