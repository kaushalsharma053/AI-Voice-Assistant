/**
 * Saarthi AI - Client-Side Interactive Engine
 * Handles Web Speech API, real-time FastAPI telemetry, and audio feedback.
 */

class SaarthiWebClient {
    constructor() {
        this.isListening = false;
        this.audioEnabled = true;
        this.recognition = null;
        this.synth = window.speechSynthesis;

        this.initDOMElements();
        this.initSpeechRecognition();
        this.bindEvents();
        this.startSystemTelemetryLoop();
        this.loadModelInfo();
    }

    initDOMElements() {
        this.orb = document.getElementById("ai-orb");
        this.orbStatus = document.getElementById("orb-status");
        this.chatStream = document.getElementById("chat-stream");
        this.queryInput = document.getElementById("query-input");
        this.btnSend = document.getElementById("btn-send");
        this.btnMic = document.getElementById("btn-mic");
        this.btnAudioToggle = document.getElementById("btn-audio-toggle");
        this.btnClear = document.getElementById("btn-clear");
        this.btnVivaModal = document.getElementById("btn-viva-modal");
        this.btnCloseModal = document.getElementById("btn-close-modal");
        this.vivaModal = document.getElementById("viva-modal");

        // Telemetry Elements
        this.telemetryIntent = document.getElementById("telemetry-intent");
        this.telemetryConfidence = document.getElementById("telemetry-confidence");
        this.telemetryBar = document.getElementById("telemetry-bar");
        this.telemetryLatency = document.getElementById("telemetry-latency");
        this.telemetryAction = document.getElementById("telemetry-action");

        // System Gauges
        this.gaugeCpu = document.getElementById("gauge-cpu");
        this.gaugeRam = document.getElementById("gauge-ram");
        this.gaugeBattery = document.getElementById("gauge-battery");
    }

    initSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = "en-US";

            this.recognition.onstart = () => {
                this.isListening = true;
                this.setOrbState("listening", "LISTENING TO MIC...");
                this.btnMic.classList.add("recording");
            };

            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                this.queryInput.value = transcript;
                this.submitQuery(transcript);
            };

            this.recognition.onerror = (event) => {
                console.warn("[WebSpeech Error]", event.error);
                this.setOrbState("idle", "SYSTEM ONLINE");
                this.btnMic.classList.remove("recording");
                this.isListening = false;
            };

            this.recognition.onend = () => {
                this.isListening = false;
                this.btnMic.classList.remove("recording");
                if (this.orb.className.includes("state-listening")) {
                    this.setOrbState("idle", "SYSTEM ONLINE");
                }
            };
        } else {
            console.warn("[WebSpeech] Browser does not natively support SpeechRecognition.");
            this.btnMic.title = "Browser does not support direct mic capture; use text input.";
        }
    }

    bindEvents() {
        // Send button & enter key
        this.btnSend.addEventListener("click", () => this.handleInputSubmit());
        this.queryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") this.handleInputSubmit();
        });

        // Microphone toggle
        this.btnMic.addEventListener("click", () => this.toggleMicrophone());

        // Audio Speech Toggle
        this.btnAudioToggle.addEventListener("click", () => {
            this.audioEnabled = !this.audioEnabled;
            this.btnAudioToggle.innerHTML = this.audioEnabled 
                ? "<span>🔊 Voice ON</span>" 
                : "<span>🔇 Voice MUTE</span>";
            if (!this.audioEnabled && this.synth) {
                this.synth.cancel();
            }
        });

        // Clear Chat
        this.btnClear.addEventListener("click", () => {
            this.chatStream.innerHTML = "";
            this.appendMessage("Saarthi AI", "Chat stream reset. Ready for your commands.", false);
        });

        // Quick Suggestion Chips
        document.querySelectorAll(".chip").forEach(chip => {
            chip.addEventListener("click", () => {
                const prompt = chip.getAttribute("data-prompt") || chip.innerText;
                this.queryInput.value = prompt;
                this.submitQuery(prompt);
            });
        });

        // Viva Modal
        this.btnVivaModal.addEventListener("click", () => {
            this.vivaModal.style.display = "flex";
        });
        this.btnCloseModal.addEventListener("click", () => {
            this.vivaModal.style.display = "none";
        });
        window.addEventListener("click", (e) => {
            if (e.target === this.vivaModal) {
                this.vivaModal.style.display = "none";
            }
        });
    }

    setOrbState(state, text) {
        this.orb.className = "ai-orb";
        if (state !== "idle") {
            this.orb.classList.add(`state-${state}`);
        }
        if (text) {
            this.orbStatus.innerText = text;
        }
    }

    toggleMicrophone() {
        if (!this.recognition) {
            alert("Your browser does not support native Web Speech Recognition. Please type your query in the prompt box!");
            return;
        }

        if (this.isListening) {
            this.recognition.stop();
        } else {
            try {
                this.recognition.start();
            } catch (err) {
                console.warn(err);
            }
        }
    }

    handleInputSubmit() {
        const text = this.queryInput.value.trim();
        if (text) {
            this.queryInput.value = "";
            this.submitQuery(text);
        }
    }

    async submitQuery(query) {
        this.appendMessage("You", query, true);
        this.setOrbState("processing", "NEURAL INFERENCE...");

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query })
            });

            if (!res.ok) {
                throw new Error(`Server returned HTTP ${res.status}`);
            }

            const data = await res.json();
            
            // Update Telemetry
            this.updateTelemetry(data.intent, data.confidence, data.latency_ms, data.action_taken);

            // Append response message
            this.appendMessage("Saarthi AI", data.response_text, false, data.intent, data.confidence);

            // Speak response via browser audio if enabled
            if (this.audioEnabled) {
                this.speak(data.response_text);
            } else {
                this.setOrbState("idle", "SYSTEM ONLINE");
            }
        } catch (err) {
            console.error(err);
            this.appendMessage("Saarthi AI", `Error communicating with backend: ${err.message}`, false);
            this.setOrbState("idle", "SYSTEM READY");
        }
    }

    speak(text) {
        if (!this.synth) {
            this.setOrbState("idle", "SYSTEM ONLINE");
            return;
        }

        this.synth.cancel(); // Stop any ongoing speech
        const clean = text.replace(/[*#`_]/g, "").trim();
        const utterance = new SpeechSynthesisUtterance(clean);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;

        // Try selecting female or natural voice if available
        const voices = this.synth.getVoices();
        const preferred = voices.find(v => v.name.includes("Zira") || v.name.includes("Google") || v.name.includes("Natural"));
        if (preferred) utterance.voice = preferred;

        utterance.onstart = () => {
            this.setOrbState("speaking", "SPEAKING RESPONSE...");
        };

        utterance.onend = () => {
            this.setOrbState("idle", "SYSTEM ONLINE");
        };

        utterance.onerror = () => {
            this.setOrbState("idle", "SYSTEM ONLINE");
        };

        this.synth.speak(utterance);
    }

    appendMessage(sender, text, isUser, intent = null, confidence = null) {
        const bubble = document.createElement("div");
        bubble.className = `message-bubble ${isUser ? "bubble-user" : "bubble-saarthi"}`;

        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        let badgeHTML = "";
        if (intent && !isUser) {
            badgeHTML = `<span class="bubble-badge">[${intent}] ${confidence ? confidence.toFixed(0) + '%' : ''}</span>`;
        }

        let formattedText = text;
        if (text && text.includes("http")) {
            formattedText = text.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer" style="color: #38bdf8; text-decoration: underline; font-weight: 600;">[🔗 View on Web]</a>');
        }

        bubble.innerHTML = `
            <div class="bubble-meta">
                <span>${sender} • ${time}</span>
                ${badgeHTML}
            </div>
            <div class="bubble-content">${formattedText}</div>
        `;

        this.chatStream.appendChild(bubble);
        this.chatStream.scrollTop = this.chatStream.scrollHeight;
    }

    updateTelemetry(intent, confidence, latency, action) {
        this.telemetryIntent.innerText = `[${intent}]`;
        this.telemetryConfidence.innerText = `${confidence.toFixed(1)}%`;
        this.telemetryBar.style.width = `${Math.min(100, confidence)}%`;
        this.telemetryLatency.innerText = `${latency.toFixed(1)} ms`;
        this.telemetryAction.innerText = action || intent;
    }

    async startSystemTelemetryLoop() {
        const updateGauges = async () => {
            try {
                const res = await fetch("/api/system-status");
                if (res.ok) {
                    const data = await res.json();
                    this.gaugeCpu.innerText = `${data.cpu_percent}%`;
                    this.gaugeRam.innerText = `${data.ram_percent}%`;
                    if (data.battery && data.battery.has_battery) {
                        this.gaugeBattery.innerText = `${data.battery.percent}%`;
                    } else {
                        this.gaugeBattery.innerText = "AC Power";
                    }
                }
            } catch (e) {
                // Ignore background polling errors
            }
        };

        await updateGauges();
        setInterval(updateGauges, 4000);
    }

    async loadModelInfo() {
        try {
            const res = await fetch("/api/model-info");
            if (res.ok) {
                const data = await res.json();
                const container = document.getElementById("modal-model-specs");
                if (container) {
                    container.innerHTML = `
                        <p><strong>Neural Architecture:</strong> ${data.architecture}</p>
                        <p><strong>Framework:</strong> ${data.framework}</p>
                        <p><strong>Embedding Dimensionality:</strong> ${data.embedding_dim}D</p>
                        <p><strong>Intent Classes (${data.classes_count}):</strong> ${data.classes ? data.classes.slice(0, 10).join(", ") + ", ..." : "29 classes"}</p>
                        <p><strong>Vocabulary Size:</strong> ${data.vocabulary_size} unique tokens</p>
                    `;
                }
            }
        } catch (e) {
            console.warn("Could not load model info", e);
        }
    }
}

// Instantiate client on DOM readiness
document.addEventListener("DOMContentLoaded", () => {
    window.saarthi = new SaarthiWebClient();
});
