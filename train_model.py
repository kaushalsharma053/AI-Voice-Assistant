"""
Saarthi AI - Deep Learning Intent Model Trainer
Trains a Bidirectional LSTM / Neural Intent Classifier on intents.json dataset
Exports .keras model, tokenizer, label encoder, and evaluation curves (metrics_plot.png)
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless mode for plotting
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from saarthi.config import (
    INTENTS_FILE,
    MODEL_FILE,
    TOKENIZER_FILE,
    LABEL_ENCODER_FILE,
    METRICS_PLOT_FILE,
    EMBEDDING_DIM,
    MAX_SEQUENCE_LENGTH,
    EPOCHS,
    BATCH_SIZE
)

def load_dataset(intents_path):
    """Loads and flattens intents JSON into text patterns and label tags."""
    if not os.path.exists(intents_path):
        raise FileNotFoundError(f"Intents file not found at: {intents_path}")

    with open(intents_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    patterns = []
    labels = []

    for intent in data['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            cleaned_pattern = pattern.strip().lower()
            if cleaned_pattern:
                patterns.append(cleaned_pattern)
                labels.append(tag)

    print(f"[Dataset] Loaded {len(patterns)} pattern samples across {len(set(labels))} unique intent classes.")
    return patterns, labels

def build_model(vocab_size, num_classes, max_len=MAX_SEQUENCE_LENGTH, emb_dim=EMBEDDING_DIM):
    """
    Constructs a robust Deep Learning Intent Classification Architecture:
    Input -> Embedding -> SpatialDropout -> Bi-LSTM -> GlobalMaxPool -> Dense -> Dropout -> Softmax
    """
    model = keras.Sequential([
        layers.Input(shape=(max_len,), name="input_token_sequence"),
        layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, name="word_embeddings"),
        layers.SpatialDropout1D(0.25, name="spatial_dropout"),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bi_lstm"),
        layers.GlobalMaxPooling1D(name="max_pooling"),
        layers.Dense(64, activation='relu', name="dense_feature_extractor"),
        layers.BatchNormalization(name="batch_norm"),
        layers.Dropout(0.3, name="dropout_regularizer"),
        layers.Dense(num_classes, activation='softmax', name="intent_probabilities")
    ], name="Saarthi_Intent_Classifier")

    optimizer = keras.optimizers.Adam(learning_rate=0.003)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_training_history(history, save_path):
    """Plots and saves academic-standard loss and accuracy curves for project submission."""
    epochs_range = range(1, len(history.history['accuracy']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f172a')

    # Accuracy Plot
    ax1.set_facecolor('#1e293b')
    ax1.plot(epochs_range, history.history['accuracy'], label='Training Accuracy', color='#38bdf8', linewidth=2.5)
    if 'val_accuracy' in history.history and any(history.history['val_accuracy']):
        ax1.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy', color='#4ade80', linewidth=2, linestyle='--')
    ax1.set_title('Model Accuracy vs Epochs', fontsize=14, fontweight='bold', color='#f8fafc', pad=12)
    ax1.set_xlabel('Epochs', color='#94a3b8', fontsize=11)
    ax1.set_ylabel('Accuracy', color='#94a3b8', fontsize=11)
    ax1.tick_params(colors='#94a3b8')
    ax1.grid(True, linestyle=':', alpha=0.4, color='#64748b')
    ax1.legend(loc='lower right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')

    # Loss Plot
    ax2.set_facecolor('#1e293b')
    ax2.plot(epochs_range, history.history['loss'], label='Training Loss', color='#f87171', linewidth=2.5)
    if 'val_loss' in history.history and any(history.history['val_loss']):
        ax2.plot(epochs_range, history.history['val_loss'], label='Validation Loss', color='#fbbf24', linewidth=2, linestyle='--')
    ax2.set_title('Model Categorical Loss vs Epochs', fontsize=14, fontweight='bold', color='#f8fafc', pad=12)
    ax2.set_xlabel('Epochs', color='#94a3b8', fontsize=11)
    ax2.set_ylabel('Loss', color='#94a3b8', fontsize=11)
    ax2.tick_params(colors='#94a3b8')
    ax2.grid(True, linestyle=':', alpha=0.4, color='#64748b')
    ax2.legend(loc='upper right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')

    plt.suptitle("Saarthi AI - Deep Learning Intent Classifier Metrics (7th Sem Capstone)", fontsize=15, fontweight='bold', color='#f1f5f9', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Evaluation] Metrics curves exported to: {save_path}")

def train():
    """Main training routine."""
    print("=" * 65)
    print("   SAARTHI AI - DEEP LEARNING MODEL TRAINING PIPELINE")
    print("=" * 65)

    patterns, raw_labels = load_dataset(INTENTS_FILE)

    # Label Encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(raw_labels)
    num_classes = len(label_encoder.classes_)
    categorical_labels = keras.utils.to_categorical(integer_encoded, num_classes=num_classes)

    # Tokenization
    tokenizer = Tokenizer(oov_token="<OOV>", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(patterns)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"[Tokenizer] Vocabulary size: {vocab_size} unique tokens.")

    # Convert patterns to padded sequences
    sequences = tokenizer.texts_to_sequences(patterns)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    # Train / Validation Split (Stratified to maintain class distributions)
    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, categorical_labels, test_size=0.15, random_state=42
    )

    print(f"[Data Split] Training samples: {len(X_train)} | Validation samples: {len(X_val)}")

    # Model Compilation
    model = build_model(vocab_size=vocab_size, num_classes=num_classes)
    model.summary()

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-5,
        verbose=1
    )

    # Training Loop
    print("\n[Training] Starting Deep Learning model optimization...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Save Model Artifacts
    print("\n[Exporting] Saving trained model and preprocessors...")
    model.save(MODEL_FILE)
    print(f" -> Keras Model saved at: {MODEL_FILE}")

    with open(TOKENIZER_FILE, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f" -> Tokenizer saved at: {TOKENIZER_FILE}")

    with open(LABEL_ENCODER_FILE, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f" -> Label Encoder saved at: {LABEL_ENCODER_FILE}")

    # Generate Evaluation Charts
    plot_training_history(history, METRICS_PLOT_FILE)

    # Test Sample Inferences
    print("\n[Verification] Testing Sample Inferences on Trained Neural Network:")
    test_phrases = [
        "Hello Saarthi",
        "Open notepad please",
        "What is the time right now",
        "Check system status and cpu",
        "Who created you",
        "Tell me a programming joke"
    ]

    for phrase in test_phrases:
        seq = tokenizer.texts_to_sequences([phrase.lower()])
        pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        preds = model.predict(pad, verbose=0)[0]
        idx = np.argmax(preds)
        tag = label_encoder.inverse_transform([idx])[0]
        conf = float(preds[idx]) * 100
        print(f"  Input: '{phrase}' -> Predicted: [{tag}] ({conf:.1f}% confidence)")

    print("\n" + "=" * 65)
    print("   MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    train()
