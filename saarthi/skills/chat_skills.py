"""
Conversational and Academic Viva Preparation Skills for Saarthi AI
Provides answers for academic concepts (Deep Learning, Neural Networks, NLP) and small talk.
"""

import random

VIVA_KNOWLEDGE_BASE = {
    "deep learning": (
        "Deep learning is a subset of machine learning based on artificial neural networks "
        "with representation learning. It uses multiple layers of non-linear processing units "
        "to extract feature hierarchies directly from raw input data."
    ),
    "backpropagation": (
        "Backpropagation is short for 'backward propagation of errors'. It is the algorithm "
        "used in supervised learning to compute the gradients of the loss function with respect "
        "to all weights in the network using the mathematical chain rule, updating weights via gradient descent."
    ),
    "activation function": (
        "Activation functions introduce non-linearity into a neural network, enabling it to learn "
        "complex mapping functions. Popular functions include ReLU to prevent vanishing gradients, "
        "Sigmoid for binary probabilities, and Softmax for multi-class probability distributions."
    ),
    "overfitting": (
        "Overfitting happens when a machine learning model learns the training data and noise too well, "
        "failing to generalize to unseen test data. It is mitigated using Dropout, L1/L2 regularization, "
        "Early Stopping, and data augmentation."
    ),
    "recurrent neural network": (
        "A Recurrent Neural Network (RNN) processes sequential data like time-series or sentences "
        "by maintaining an internal hidden state memory across steps. Bi-directional LSTMs overcome "
        "the vanishing gradient problem of standard RNNs by using gating mechanisms."
    ),
    "saarthi architecture": (
        "Saarthi AI utilizes a Bi-directional Long Short-Term Memory (Bi-LSTM) network with "
        "an embedding layer, spatial dropout, and global max pooling. It predicts intent class "
        "probabilities via a Softmax layer and extracts query parameters to trigger modular skills."
    )
}

def get_viva_explanation(query):
    """Matches academic viva concepts and returns crisp, textbook-accurate definitions."""
    query_lower = query.lower()
    for concept, explanation in VIVA_KNOWLEDGE_BASE.items():
        if concept in query_lower:
            return f"Viva Explanation: {explanation}"

    # Default fallback academic response
    return (
        "In Deep Learning, neural networks mimic biological neurons to map inputs to outputs. "
        "Our model processes word token sequences through an embedding layer and Bi-LSTM "
        "to compute intent probabilities."
    )

def get_motivational_quote():
    """Returns an inspiring quote."""
    quotes = [
        "\"The only way to do great work is to love what you do.\" - Steve Jobs",
        "\"Success is walking from failure to failure with no loss of enthusiasm.\" - Winston Churchill",
        "\"Technology is best when it brings people together.\" - Matt Mullenweg",
        "\"The best way to predict the future is to invent it.\" - Alan Kay",
        "\"Opportunities don't happen. You create them.\" - Chris Grosser"
    ]
    return random.choice(quotes)
