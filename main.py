"""
Saarthi AI (सारथी) - Main Application Launcher
7th Semester Academic Capstone Project: Multimodal Deep Learning Desktop Assistant
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from saarthi.config import MODEL_FILE

def check_environment():
    """Validates that trained model exists before launching GUI."""
    if not os.path.exists(MODEL_FILE):
        print("=" * 60)
        print(" [NOTICE] Pre-trained model not detected!")
        print(" Initiating automatic deep learning model training...")
        print("=" * 60)
        from train_model import train
        train()

def main():
    print("=" * 60)
    print("   SAARTHI AI (सारथी) - INTELLIGENT DEEP LEARNING ASSISTANT")
    print("   7th-Semester Academic Capstone Project")
    print("=" * 60)

    check_environment()

    # Launch GUI
    try:
        from saarthi.ui.app_ui import run_app
        print("[Launcher] Starting CustomTkinter Graphical User Interface...")
        run_app()
    except Exception as e:
        print(f"[Launcher Warning] GUI failed to start ({str(e)}). Falling back to CLI console mode...")
        from cli_assistant import run_cli
        run_cli()

if __name__ == "__main__":
    main()
