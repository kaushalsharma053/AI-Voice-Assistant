"""
System Automation Skills for Saarthi AI
Handles OS telemetry, app launching, volume adjustments, screenshots, and workstation locking.
"""

import os
import subprocess
import ctypes
from datetime import datetime
import psutil
import pyautogui

from saarthi.config import APP_MAPPING, SCREENSHOTS_DIR

def get_system_status():
    """Fetches CPU, RAM, and Battery telemetry."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.3)
        memory = psutil.virtual_memory()
        ram_percent = memory.percent
        ram_available_gb = memory.available / (1024 ** 3)

        status_msg = f"CPU utilization is at {cpu_percent}%. RAM usage is at {ram_percent}% with {ram_available_gb:.1f} GB free."

        battery = psutil.sensors_battery()
        if battery:
            plugged = "charging" if battery.power_plugged else "on battery power"
            status_msg += f" Battery level is at {battery.percent}%, currently {plugged}."
        else:
            status_msg += " Running on AC desktop power."

        return status_msg
    except Exception as e:
        return f"Unable to fetch complete system status: {str(e)}"

def open_application(query):
    """Identifies and launches application based on spoken or typed command."""
    query_lower = query.lower()
    
    # Strip trigger words
    for trigger in ["open", "launch", "start", "run", "please"]:
        query_lower = query_lower.replace(trigger, "").strip()

    target_app = None
    target_cmd = None

    # Check mapping
    for key, cmd in APP_MAPPING.items():
        if key in query_lower:
            target_app = key
            target_cmd = cmd
            break

    if not target_cmd and query_lower:
        target_app = query_lower
        target_cmd = f"start {query_lower}"

    if target_cmd:
        try:
            if target_cmd.startswith("start "):
                os.system(target_cmd)
            else:
                subprocess.Popen(target_cmd, shell=True)
            return f"Launching {target_app}."
        except Exception as e:
            return f"Could not launch {target_app}: {str(e)}"
    
    return "Please specify which application you would like me to open."

def control_volume(action="up"):
    """Adjusts or mutes system volume using native Windows media key commands."""
    try:
        if action == "up":
            for _ in range(5):
                pyautogui.press("volumeup")
            return "Volume increased."
        elif action == "down":
            for _ in range(5):
                pyautogui.press("volumedown")
            return "Volume decreased."
        elif action == "mute":
            pyautogui.press("volumemute")
            return "Volume muted or unmuted."
        else:
            return "Unknown volume command."
    except Exception as e:
        return f"Could not adjust volume: {str(e)}"

def take_screenshot():
    """Captures the current screen and saves to screenshots directory."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = SCREENSHOTS_DIR / filename
        
        screenshot = pyautogui.screenshot()
        screenshot.save(str(filepath))
        return f"Screenshot captured successfully and saved to {filename}."
    except Exception as e:
        return f"Failed to capture screenshot: {str(e)}"

def lock_screen():
    """Locks the Windows workstation securely."""
    try:
        ctypes.windll.user32.LockWorkStation()
        return "Workstation locked successfully."
    except Exception as e:
        return f"Could not lock screen: {str(e)}"
