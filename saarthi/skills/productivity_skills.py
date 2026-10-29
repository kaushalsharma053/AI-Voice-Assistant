"""
Productivity and Task Management Skills for Saarthi AI
Features timestamped notes creation, retrieval, note wiping, and countdown timers.
"""

import os
import threading
import time
from datetime import datetime
from saarthi.config import NOTES_FILE

def take_note(query):
    """Saves a timestamped note to saarthi_notes.txt."""
    cleaned = query.lower()
    for phrase in ["take a note", "write this down", "remember this", "make a note", "note down", "add note"]:
        cleaned = cleaned.replace(phrase, "")
    
    note_content = cleaned.strip()
    if not note_content:
        return "What would you like me to note down?"

    timestamp = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    entry = f"[{timestamp}] {note_content}\n"

    try:
        with open(NOTES_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
        return f"Note recorded: '{note_content}'."
    except Exception as e:
        return f"Could not write to notes file: {str(e)}"

def read_notes():
    """Reads all recorded user notes from disk."""
    if not os.path.exists(NOTES_FILE):
        return "You do not have any notes recorded yet."

    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            return "Your notes file is currently empty."

        # Return the last 5 notes if there are many
        recent = lines[-5:]
        formatted = " | ".join(recent)
        return f"Here are your latest notes: {formatted}"
    except Exception as e:
        return f"Could not read notes: {str(e)}"

def clear_notes():
    """Clears all saved notes."""
    try:
        if os.path.exists(NOTES_FILE):
            with open(NOTES_FILE, "w", encoding="utf-8") as f:
                f.write("")
        return "All notes have been cleared."
    except Exception as e:
        return f"Could not clear notes: {str(e)}"

def set_timer(seconds, callback=None):
    """Launches an asynchronous timer thread and invokes callback on completion."""
    def _timer_worker():
        time.sleep(seconds)
        if callback:
            callback(f"Timer Alert: Your {seconds}-second timer has ended!")

    thread = threading.Thread(target=_timer_worker, daemon=True)
    thread.start()
    return f"Timer set for {seconds} seconds."
