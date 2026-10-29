"""
Web Intelligence and Search Skills for Saarthi AI
Features Wikipedia knowledge extraction, YouTube media streaming, and Google search automation.
"""

import urllib.parse
import webbrowser
import requests

def extract_search_terms(raw_query, remove_words):
    """Utility function to cleanly isolate the user's intended target query."""
    cleaned = raw_query.lower()
    for word in remove_words:
        cleaned = cleaned.replace(word, "")
    return cleaned.strip()

def search_wikipedia(query):
    """
    Fetches real-time concise encyclopedic summaries from Wikipedia REST API.
    Directly answers user questions in-place without disruptive browser popups.
    """
    target = extract_search_terms(
        query,
        ["search wikipedia for", "search wikipedia", "wikipedia", "who was", "who is",
         "what is", "tell me about", "define", "according to wikipedia", "summarize",
         "information about", "where is", "explain", "history of", "search for", "meaning of"]
    )
    
    if not target or len(target) < 2:
        target = query.strip()

    headers = {
        "User-Agent": "SaarthiAI/1.0 (College Engineering Capstone; academic research)"
    }

    try:
        # Step 1: Direct summary query via Wikipedia REST API
        encoded = urllib.parse.quote(target)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        response = requests.get(url, headers=headers, timeout=6)

        if response.status_code == 200:
            data = response.json()
            extract = data.get("extract")
            if extract:
                # Keep first 2-3 sentences for clean speech and chat bubble
                sentences = extract.split(". ")
                summary = ". ".join(sentences[:2]) + "."
                return f"According to knowledge sources: {summary}"

        # Step 2: Fallback to Opensearch API if direct title was not exact
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={encoded}&limit=1&namespace=0&format=json"
        search_res = requests.get(search_url, headers=headers, timeout=6)
        if search_res.status_code == 200:
            search_data = search_res.json()
            if len(search_data) >= 3 and search_data[2]:
                first_summary = search_data[2][0]
                if first_summary:
                    title_found = search_data[1][0] if search_data[1] else target
                    return f"For {title_found}: {first_summary}"

        return f"I could not find a direct encyclopedia summary for '{target}'."
    except Exception as e:
        return f"Knowledge service temporarily unavailable: {str(e)}"

def play_youtube(query, open_browser: bool = True):
    """Searches and opens YouTube to play videos or songs requested by the user."""
    target = extract_search_terms(
        query,
        ["play on youtube", "play song on youtube", "play video on youtube", "play on youtube and stream",
         "play song", "play video", "play music", "play", "stream", "on youtube"]
    )

    if not target:
        url = "https://www.youtube.com"
        if open_browser:
            webbrowser.open(url)
        return "Opening YouTube."

    encoded = urllib.parse.quote_plus(target)
    url = f"https://www.youtube.com/results?search_query={encoded}"
    if open_browser:
        webbrowser.open(url)
    return f"Searching and playing '{target}' on YouTube."

def search_google(query, open_browser: bool = False):
    """
    Returns search query URL or optionally opens browser tab only if explicitly requested.
    Does NOT disrupt the conversation by forcefully hijacking the screen.
    """
    target = extract_search_terms(
        query,
        ["google search for", "search google for", "search on google for",
         "google search", "search for", "search google", "look up", "find online"]
    )

    if not target:
        target = query.strip()

    encoded = urllib.parse.quote_plus(target)
    url = f"https://www.google.com/search?q={encoded}"

    if open_browser:
        webbrowser.open(url)
        return f"Opening Google search results for '{target}'."

    return f"I've searched for '{target}'. You can view results here: {url}"
