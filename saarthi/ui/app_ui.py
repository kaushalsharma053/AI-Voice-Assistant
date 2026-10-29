"""
Modern Futuristic Desktop HUD for Saarthi AI
Built with CustomTkinter for 7th-Semester Academic Capstone Presentation.
Features live Deep Learning telemetry, conversation stream, voice/text dual input, and voice settings.
"""

import sys
import threading
import time
import tkinter as tk
from datetime import datetime
import customtkinter as ctk

from saarthi.core.intent_classifier import IntentClassifier
from saarthi.core.executor import ActionExecutor
from saarthi.core.speech_engine import SpeechEngine, HAS_SPEECH_RECOGNITION

# Set theme styling
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SaarthiAssistantUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Saarthi AI - Deep Learning Multimodal Assistant (7th Sem Capstone)")
        self.geometry("1080x720")
        self.minsize(920, 600)

        # Core Engines
        self.classifier = IntentClassifier()
        self.executor = ActionExecutor(self.classifier)
        self.speech_engine = SpeechEngine(voice_gender="female")

        self.is_listening_active = False
        self._audio_thread = None

        # Grid configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_panel()

        # Initial Welcome Message
        self.after(500, self._show_welcome_message)

    def _build_sidebar(self):
        """Builds left telemetry and controls sidebar."""
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color="#0f172a")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        # Header Branding
        self.brand_title = ctk.CTkLabel(
            self.sidebar_frame,
            text="SAARTHI AI",
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
            text_color="#38bdf8"
        )
        self.brand_title.grid(row=0, column=0, padx=20, pady=(20, 2))

        self.brand_sub = ctk.CTkLabel(
            self.sidebar_frame,
            text="सारथी • Deep Learning Assistant",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#94a3b8"
        )
        self.brand_sub.grid(row=1, column=0, padx=20, pady=(0, 15))

        # Status Indicator Pill
        self.status_pill = ctk.CTkLabel(
            self.sidebar_frame,
            text="● SYSTEM READY",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#4ade80",
            fg_color="#1e293b",
            corner_radius=12,
            height=28
        )
        self.status_pill.grid(row=2, column=0, padx=20, pady=(0, 15), sticky="ew")

        # Telemetry Card
        self.telemetry_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="#1e293b", corner_radius=10)
        self.telemetry_frame.grid(row=3, column=0, padx=15, pady=5, sticky="ew")

        ctk.CTkLabel(
            self.telemetry_frame,
            text="NEURAL TELEMETRY",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#64748b"
        ).pack(anchor="w", padx=12, pady=(10, 4))

        # Intent Tag
        self.intent_label = ctk.CTkLabel(
            self.telemetry_frame,
            text="Intent: [idle]",
            font=ctk.CTkFont(family="Consolas", size=13, weight="bold"),
            text_color="#f8fafc"
        )
        self.intent_label.pack(anchor="w", padx=12, pady=2)

        # Confidence Progress Bar
        self.confidence_label = ctk.CTkLabel(
            self.telemetry_frame,
            text="Confidence: 0.0%",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8"
        )
        self.confidence_label.pack(anchor="w", padx=12, pady=(4, 2))

        self.confidence_bar = ctk.CTkProgressBar(self.telemetry_frame, progress_color="#38bdf8", fg_color="#334155")
        self.confidence_bar.pack(fill="x", padx=12, pady=(0, 6))
        self.confidence_bar.set(0.0)

        # Latency
        self.latency_label = ctk.CTkLabel(
            self.telemetry_frame,
            text="Inference Latency: 0 ms",
            font=ctk.CTkFont(size=11),
            text_color="#64748b"
        )
        self.latency_label.pack(anchor="w", padx=12, pady=(0, 10))

        # Controls & Settings Divider
        ctk.CTkLabel(
            self.sidebar_frame,
            text="ASSISTANT SETTINGS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#64748b"
        ).grid(row=4, column=0, padx=20, pady=(15, 6), sticky="w")

        # Voice Selector
        self.voice_gender_seg = ctk.CTkSegmentedButton(
            self.sidebar_frame,
            values=["Female Voice", "Male Voice"],
            command=self._on_voice_change,
            selected_color="#0284c7"
        )
        self.voice_gender_seg.set("Female Voice")
        self.voice_gender_seg.grid(row=5, column=0, padx=15, pady=6, sticky="ew")

        # Quick Actions
        self.btn_clear = ctk.CTkButton(
            self.sidebar_frame,
            text="Clear Conversation",
            fg_color="#334155",
            hover_color="#475569",
            command=self._clear_chat
        )
        self.btn_clear.grid(row=6, column=0, padx=15, pady=(10, 6), sticky="ew")

        self.btn_help = ctk.CTkButton(
            self.sidebar_frame,
            text="Suggested Commands",
            fg_color="#334155",
            hover_color="#475569",
            command=self._show_help_modal
        )
        self.btn_help.grid(row=7, column=0, padx=15, pady=6, sticky="ew")

        # Footer Version
        self.footer_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="v1.0.0 • Bi-LSTM Neural Engine",
            font=ctk.CTkFont(size=10),
            text_color="#475569"
        )
        self.footer_label.grid(row=9, column=0, padx=15, pady=15)

    def _build_main_panel(self):
        """Builds central interactive chat and input HUD."""
        self.main_frame = ctk.CTkFrame(self, fg_color="#090d16", corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Header Status Banner
        self.header_banner = ctk.CTkFrame(self.main_frame, height=55, fg_color="#0f172a", corner_radius=0)
        self.header_banner.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.header_banner.grid_columnconfigure(0, weight=1)

        self.activity_wave_label = ctk.CTkLabel(
            self.header_banner,
            text="~ ~ ~ SAARTHI MULTIMODAL CONSOLE ~ ~ ~",
            font=ctk.CTkFont(family="Consolas", size=13, weight="bold"),
            text_color="#38bdf8"
        )
        self.activity_wave_label.grid(row=0, column=0, pady=12)

        # Scrollable Chat Stream
        self.chat_scroll = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        self.chat_scroll.grid(row=1, column=0, sticky="nsew", padx=20, pady=15)

        # Bottom Input Dock
        self.input_dock = ctk.CTkFrame(self.main_frame, fg_color="#0f172a", corner_radius=12)
        self.input_dock.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.input_dock.grid_columnconfigure(1, weight=1)

        # Voice Mic Toggle Button
        self.mic_btn = ctk.CTkButton(
            self.input_dock,
            text="🎤 Mic",
            width=75,
            height=40,
            fg_color="#0284c7",
            hover_color="#0369a1",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._toggle_voice_input
        )
        self.mic_btn.grid(row=0, column=0, padx=(10, 8), pady=10)

        # Text Query Entry
        self.query_entry = ctk.CTkEntry(
            self.input_dock,
            placeholder_text="Type your command or question (e.g., 'open notepad', 'who is einstein', '25 * 4')...",
            height=40,
            font=ctk.CTkFont(size=13),
            border_color="#334155"
        )
        self.query_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        self.query_entry.bind("<Return>", lambda event: self._send_text_query())

        # Send Button
        self.send_btn = ctk.CTkButton(
            self.input_dock,
            text="Send ➔",
            width=80,
            height=40,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._send_text_query
        )
        self.send_btn.grid(row=0, column=2, padx=(8, 10), pady=10)

    def _show_welcome_message(self):
        """Renders initial greeting message in chat log."""
        self._add_message(
            "Saarthi AI",
            "Namaste! I am Saarthi, your Deep Learning AI voice assistant. "
            "You can speak via the microphone or type any command in the input box below. "
            "Try saying: 'What can you do?', 'Check system status', or 'Open notepad'.",
            is_user=False,
            intent="greeting",
            confidence=99.0
        )
        self.speech_engine.speak("Namaste! Saarthi AI is initialized and ready.")

    def _on_voice_change(self, value):
        """Switches TTS voice gender."""
        if "Female" in value:
            self.speech_engine.set_voice_gender("female")
            self.speech_engine.speak("Switched to female voice.")
        else:
            self.speech_engine.set_voice_gender("male")
            self.speech_engine.speak("Switched to male voice.")

    def _set_status(self, text, color="#38bdf8"):
        """Updates the status pill safely on main thread."""
        self.status_pill.configure(text=text, text_color=color)

    def _update_telemetry(self, intent, confidence, latency_ms):
        """Updates DL telemetry metrics in left sidebar."""
        self.intent_label.configure(text=f"Intent: [{intent}]")
        self.confidence_label.configure(text=f"Confidence: {confidence:.1f}%")
        self.confidence_bar.set(min(1.0, confidence / 100.0))
        self.latency_label.configure(text=f"Inference Latency: {latency_ms:.1f} ms")

    def _add_message(self, sender, text, is_user=False, intent=None, confidence=None):
        """Appends a styled bubble card to the scrollable conversation log."""
        bubble = ctk.CTkFrame(
            self.chat_scroll,
            fg_color="#1e293b" if not is_user else "#0284c7",
            corner_radius=10
        )
        bubble.pack(fill="x", padx=5, pady=6, anchor="e" if is_user else "w")

        # Top Header of Card
        header_frame = ctk.CTkFrame(bubble, fg_color="transparent")
        header_frame.pack(fill="x", padx=12, pady=(8, 2))

        timestamp = datetime.now().strftime("%I:%M %p")
        sender_label = ctk.CTkLabel(
            header_frame,
            text=f"{sender} • {timestamp}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#94a3b8" if not is_user else "#e0f2fe"
        )
        sender_label.pack(side="left")

        if intent and not is_user:
            badge_text = f"[{intent}] {confidence:.0f}%" if confidence else f"[{intent}]"
            badge = ctk.CTkLabel(
                header_frame,
                text=badge_text,
                font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
                text_color="#38bdf8",
                fg_color="#0f172a",
                corner_radius=6,
                padx=6,
                pady=1
            )
            badge.pack(side="right")

        # Content Text
        content_label = ctk.CTkLabel(
            bubble,
            text=text,
            font=ctk.CTkFont(size=13),
            text_color="#f8fafc",
            wraplength=640,
            justify="left"
        )
        content_label.pack(anchor="w", padx=12, pady=(2, 10))

        # Auto-scroll to bottom
        self.after(50, lambda: self.chat_scroll._parent_canvas.yview_moveto(1.0))

    def _send_text_query(self):
        """Processes manual text prompt from entry box."""
        query = self.query_entry.get().strip()
        if not query:
            return

        self.query_entry.delete(0, tk.END)
        self._add_message("You", query, is_user=True)
        self._set_status("● INFERENCE RUNNING", "#fbbf24")

        # Process in separate thread to prevent GUI lockup
        threading.Thread(target=self._execute_query_worker, args=(query,), daemon=True).start()

    def _toggle_voice_input(self):
        """Toggles microphone listening mode in background thread."""
        if not HAS_SPEECH_RECOGNITION:
            self._add_message(
                "Saarthi AI",
                "Microphone voice recognition package is still finalizing installation. You can type commands in the box below!",
                is_user=False,
                intent="system_notice"
            )
            return

        if self.is_listening_active:
            return

        self.is_listening_active = True
        self.mic_btn.configure(text="🔴 Listening...", fg_color="#ef4444")
        self._set_status("● LISTENING TO MIC...", "#ef4444")
        self.activity_wave_label.configure(text="~ ~ ~ 🎙️ LISTENING FOR VOICE INPUT ~ ~ ~", text_color="#ef4444")

        self._audio_thread = threading.Thread(target=self._voice_listen_worker, daemon=True)
        self._audio_thread.start()

    def _voice_listen_worker(self):
        """Background thread that captures audio from microphone."""
        try:
            transcribed_text = self.speech_engine.listen_from_mic(timeout=5, phrase_time_limit=7)
            if transcribed_text:
                self.after(0, lambda: self._add_message("You (Voice)", transcribed_text, is_user=True))
                self.after(0, lambda: self._execute_query_worker(transcribed_text))
            else:
                self.after(0, lambda: self._set_status("● IDLE", "#4ade80"))
                self.after(0, lambda: self._add_message("Saarthi AI", "No speech detected. Please speak clearly or try again.", is_user=False))
        except Exception as e:
            self.after(0, lambda: self._add_message("Saarthi AI", f"Microphone error: {str(e)}", is_user=False))
        finally:
            self.is_listening_active = False
            self.after(0, self._reset_mic_ui)

    def _reset_mic_ui(self):
        """Restores mic button and banner styling."""
        self.mic_btn.configure(text="🎤 Mic", fg_color="#0284c7")
        self.activity_wave_label.configure(text="~ ~ ~ SAARTHI MULTIMODAL CONSOLE ~ ~ ~", text_color="#38bdf8")

    def _execute_query_worker(self, query):
        """Executes intent classification, skills dispatch, and audio speech synthesis."""
        result = self.executor.process_query(query)

        response_text = result.get("response_text", "")
        intent = result.get("intent", "unknown")
        confidence = result.get("confidence", 0.0)
        latency_ms = result.get("latency_ms", 0.0)

        # Update GUI on main thread
        self.after(0, lambda: self._update_telemetry(intent, confidence, latency_ms))
        self.after(0, lambda: self._add_message("Saarthi AI", response_text, is_user=False, intent=intent, confidence=confidence))
        self.after(0, lambda: self._set_status("● SPEAKING...", "#38bdf8"))

        # Speak response via TTS
        self.speech_engine.speak(response_text)

        # Reset status back to ready
        self.after(2000, lambda: self._set_status("● SYSTEM READY", "#4ade80"))

    def _clear_chat(self):
        """Clears all conversation bubbles."""
        for widget in self.chat_scroll.winfo_children():
            widget.destroy()
        self._add_message("Saarthi AI", "Conversation cleared. Ready for new commands.", is_user=False)

    def _show_help_modal(self):
        """Displays quick reference suggestions."""
        help_window = ctk.CTkToplevel(self)
        help_window.title("Saarthi AI - Quick Commands Reference")
        help_window.geometry("540x480")
        help_window.transient(self)

        content = (
            "🎯 CORE VOICE & TEXT COMMANDS FOR DEMO:\n\n"
            "• System Control:\n"
            "   - 'Open notepad' or 'Launch calculator'\n"
            "   - 'Check system status' or 'RAM usage'\n"
            "   - 'Increase volume' / 'Mute volume'\n"
            "   - 'Take a screenshot' / 'Lock workstation'\n\n"
            "• Web & Media:\n"
            "   - 'Who was Albert Einstein' (Wikipedia summary)\n"
            "   - 'Play lo-fi hip hop on YouTube'\n"
            "   - 'Search Google for machine learning'\n\n"
            "• Calculations & Productivity:\n"
            "   - 'Calculate 25 * 4 + 10' or 'Square root of 144'\n"
            "   - 'What is the time' / 'What is today\\'s date'\n"
            "   - 'Weather in Mumbai' or 'Top news headlines'\n"
            "   - 'Take a note buy electronics components'\n"
            "   - 'Read my notes' or 'Clear notes'\n\n"
            "• Deep Learning Viva Assistance:\n"
            "   - 'Explain backpropagation for viva'\n"
            "   - 'What is an activation function'\n"
            "   - 'Tell me a programming joke'"
        )

        label = ctk.CTkLabel(
            help_window,
            text=content,
            font=ctk.CTkFont(family="Consolas", size=12),
            justify="left",
            text_color="#f1f5f9"
        )
        label.pack(padx=20, pady=20, fill="both", expand=True)

def run_app():
    """Entry function for the GUI application."""
    app = SaarthiAssistantUI()
    app.mainloop()

if __name__ == "__main__":
    run_app()
