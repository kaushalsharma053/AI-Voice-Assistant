# Saarthi AI (सारथी) 🎙️🌐🤖
### Multimodal Deep Learning Voice & Web Assistant (FastAPI + Cyber HUD)
*7th Semester Academic Capstone Project (B.Tech / B.E. in CSE / IT / AI & Data Science)*

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Backend](https://img.shields.io/badge/Web%20Backend-FastAPI%20%2B%20Uvicorn-009688)
![Deep Learning](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras%203-orange)
![Architecture](https://img.shields.io/badge/Model-Bi--LSTM%20Intent%20Classifier-green)
![Frontend](https://img.shields.io/badge/Web%20HUD-HTML5%20%2B%20Glassmorphism-cyan)
![Platform](https://img.shields.io/badge/OS-Windows%2010%20%2F%2011%20%2F%20Web-blueviolet)

---

## 🌟 Overview
**Saarthi AI (सारथी)** is an intelligent, multimodal voice assistant engineered specifically for **7th-semester engineering capstone presentation** and practical deployment. In Sanskrit, *Saarthi* means a trusted charioteer or guide. 

The project replaces rule-based if-else bots with an end-to-end **Deep Learning Intent Classifier** built on a **Bidirectional Long Short-Term Memory (Bi-LSTM)** neural network. It offers two client presentation interfaces:
1. **Interactive Web Dashboard**: High-performance **FastAPI** backend with a futuristic **Cyber HUD** frontend (Web Speech API microphone listening, animated concentric glowing AI core, live CPU/RAM telemetry gauges, and an academic viva modal).
2. **Desktop GUI Application**: Sleek **CustomTkinter** dark mode window with native microphone capture and SAPI5 offline speech synthesis.
3. **Headless Terminal Console (CLI)**: Lightweight console mode for server or low-spec lab environments.

---

## 🏗️ System Architecture

```
                                  ┌───────────────────────────┐
                                  │  Web Browser / Client UI  │
                                  │ (Microphone / Web Speech) │
                                  └─────────────┬─────────────┘
                                                │ REST / WebSockets
┌───────────────────────────┐                   ▼
│   Desktop HUD / Console   ├──────► [ FastAPI Asynchronous Server ]
└───────────────────────────┘        [     (server.py)             ]
                                                │
                                                ▼
                                  [ Text Preprocessing & Cleaning ]
                                                │
                                                ▼
                                    [ Keras Tokenizer (OOV) ]
                                                │
                                                ▼
                                   [ pad_sequences(maxlen=18) ]
                                                │
                                                ▼
                                ┌───────────────────────────────┐
                                │   Deep Learning Neural Net    │
                                │   ------------------------    │
                                │   • Embedding Layer (64D)     │
                                │   • SpatialDropout1D (0.25)   │
                                │   • Bidirectional LSTM (64)   │
                                │   • GlobalMaxPooling1D        │
                                │   • Dense ReLU + BatchNorm    │
                                │   • Softmax (29 Classes)      │
                                └───────────────┬───────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
    High Confidence (>= 55%)                                      Low Confidence (< 55%)
                 │                                                             │
                 ▼                                                             ▼
     [ Modular Action Dispatcher ]                                [ Conversational & Web Fallback ]
                 │                                                             │
 ┌───────────────┼───────────────┬───────────────┐                             │
 ▼               ▼               ▼               ▼                             │
System Skills   Web Skills     Info Skills   Productivity                      │
(Apps, Stats,  (Wikipedia,    (Time, Date,   (Notes, Timers)                   │
Volume, Lock)  YouTube, Web)  Math, Weather)                                   │
 └───────────────┼───────────────┴───────────────┴─────────────────────────────┘
                 │
                 ▼
     [ Response Synthesizer ]
                 │
                 ├──────────────────────────────┐
                 ▼                              ▼
    [ Web Audio / Browser TTS ]        [ Live Cyber HUD & Telemetry ]
    [ + Offline SAPI5 for Desktop ]    [ Intent, Confidence %, Latency ]
```

---

## 📂 Project Structure

```
AI Voice Assistant/
├── data/
│   └── intents.json               # 29 intent categories with 300+ pattern variations
├── models/
│   ├── saarthi_intent_model.keras # Trained deep learning model
│   ├── tokenizer.pickle           # Fitted Keras text tokenizer
│   └── label_encoder.pickle       # Fitted label encoder
├── saarthi/
│   ├── __init__.py
│   ├── config.py                  # Global configurations, paths, thresholds
│   ├── core/
│   │   ├── __init__.py
│   │   ├── speech_engine.py       # Speech recognition & offline TTS engine
│   │   ├── intent_classifier.py   # Deep Learning inference & tokenization
│   │   └── executor.py            # Intent router & skill dispatcher
│   ├── skills/
│   │   ├── __init__.py
│   │   ├── system_skills.py       # Volume, battery, memory, app launch, screenshot
│   │   ├── web_skills.py          # Google search, Wikipedia summary, YouTube
│   │   ├── info_skills.py         # Time, date, weather, news, safe math solver
│   │   ├── productivity_skills.py # Notes manager, reminder alerts
│   │   └── chat_skills.py         # Conversational responses & viva explanations
│   └── ui/
│       ├── __init__.py
│       └── app_ui.py              # CustomTkinter interface with telemetry meters
├── web/
│   ├── index.html                 # Modern Cyber HUD web application
│   └── static/
│       ├── css/
│       │   └── style.css          # Glassmorphism, animations, glowing AI orb
│       └── js/
│           └── app.js             # Web Speech API, telemetry rendering, REST client
├── tests/
│   ├── __init__.py
│   ├── test_intent_model.py       # Unit tests for DL model inference & classes
│   ├── test_skills.py             # Unit tests for system, web, and math skills
│   └── test_web_server.py         # Unit tests for FastAPI REST endpoints
├── server.py                      # FastAPI web server entrypoint
├── train_model.py                 # Deep learning model trainer with metrics plotting
├── main.py                        # Primary desktop application launcher
├── cli_assistant.py               # Terminal console version for headless runs
├── requirements.txt               # Complete dependencies specification
├── metrics_plot.png               # Training/validation accuracy & loss curves
├── PROJECT_REPORT.md              # 7th-Semester Academic Project Report & Viva Q&A
└── README.md                      # Comprehensive project documentation
```

---

## ⚡ Quick Start

### 1. Requirements & Installation
Install all dependencies in one command:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard (FastAPI) — Recommended for Teacher Presentation!
```bash
python server.py
```
Open your browser at **`http://localhost:8000`** or **`http://127.0.0.1:8000`**.
- Click the glowing **`🎤 Mic`** button to speak through the browser.
- Click any of the **Quick Test Chips** (*"Check System Status"*, *"Who was Einstein"*, *"Calculate 25 * 4"*, *"Open Notepad"*, *"Viva: Backpropagation"*).
- Click **`🎓 Academic Viva Specs`** in the top navbar to show your teacher the project architecture, training loss/accuracy graphs, and mathematical equations.

### 3. Launch Desktop GUI (CustomTkinter HUD)
```bash
python main.py
```

### 4. Launch Terminal Console (Headless Mode)
```bash
python cli_assistant.py
```

### 5. Run Complete Test Suite
```bash
python -m unittest discover -s tests
```
*(All 19 unit tests pass in ~3 seconds).*

---

## 💬 Voice & Web Command Demonstrations

| Category | Example Commands | Action Performed |
| :--- | :--- | :--- |
| **System Apps** | *"Open notepad"*, *"Launch calculator"*, *"Open chrome"* | Launches Windows executables |
| **Diagnostics** | *"Check system status"*, *"RAM usage"*, *"Battery level"* | Live CPU & RAM gauges update |
| **Audio & Screen** | *"Volume up"*, *"Mute volume"*, *"Take a screenshot"* | Native Windows multimedia control |
| **Knowledge** | *"Who was Albert Einstein"*, *"Define quantum computing"* | Extracts concise Wikipedia summary |
| **Media** | *"Play lo-fi hip hop on YouTube"*, *"Open youtube"* | Opens YouTube search stream |
| **Math** | *"Calculate 25 * 4 + 10"*, *"Square root of 144"* | AST-safe numeric evaluation |
| **Productivity** | *"Take a note buy resistors"*, *"Read my notes"* | Timestamped text note storage |
| **Academic Viva** | *"Explain backpropagation for viva"*, *"What is overfitting"* | Textbook viva definitions |

---

## 🎓 Academic Submission Details
- **Project Title**: Saarthi AI: A Multimodal Deep Learning Desktop & Web Voice Assistant
- **Course**: 7th Semester B.Tech / B.E. Capstone Project
- **Key Modules**: FastAPI Asynchronous Backend, Bi-LSTM NLP Intent Classifier, Web Speech API, SAPI5 TTS, AST Math Engine, CustomTkinter GUI.
- **Project Report**: See [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for full literature review, mathematical derivations (LSTM gate formulas, cross-entropy), experimental results, and 25 viva questions with answers.
