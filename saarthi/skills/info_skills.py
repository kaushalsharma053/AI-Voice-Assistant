"""
Information Retrieval and Safe Math Calculation Skills for Saarthi AI
Provides live clock, calendar, weather, headlines, and AST-parsed mathematical evaluation.
"""

import ast
import operator
import math
import re
from datetime import datetime
import requests

def get_current_time():
    """Returns formatted 12-hour time with AM/PM."""
    now = datetime.now()
    return f"The current time is {now.strftime('%I:%M %p')}."

def get_current_date():
    """Returns full day and date representation."""
    now = datetime.now()
    return f"Today is {now.strftime('%A, %B %d, %Y')}."

def get_weather(query=""):
    """
    Fetches real-time weather information using Open-Meteo / wttr.in API.
    Zero API key required for seamless demonstration during college evaluations.
    """
    city = "Delhi"
    # Extract city name if specified
    city_match = re.search(r"(?:in|for|at)\s+([a-zA-Z\s]+)", query, re.IGNORECASE)
    if city_match:
        extracted = city_match.group(1).strip()
        if len(extracted) > 2 and extracted.lower() not in ["today", "tomorrow", "now", "outside"]:
            city = extracted.title()

    try:
        url = f"https://wttr.in/{city}?format=j1"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            curr = data['current_condition'][0]
            temp_c = curr['temp_C']
            weather_desc = curr['weatherDesc'][0]['value']
            humidity = curr['humidity']
            feels_like = curr['FeelsLikeC']
            return (f"Weather in {city}: {weather_desc}, temperature is {temp_c}°C "
                    f"(feels like {feels_like}°C) with {humidity}% humidity.")
    except Exception:
        pass

    # Fallback to simple text format
    try:
        res = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        if res.status_code == 200 and res.text.strip():
            return f"Weather report: {res.text.strip()}"
    except Exception as e:
        return f"Unable to retrieve weather data right now: {str(e)}"

    return f"Could not retrieve weather for {city}. Please check internet connectivity."

def get_news_headlines():
    """Fetches top technology and world news headlines."""
    try:
        # Fetch HackerNews top stories or general headlines via free public feed
        url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            top_ids = res.json()[:3]
            headlines = []
            for item_id in top_ids:
                item_res = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json", timeout=4)
                if item_res.status_code == 200:
                    title = item_res.json().get("title")
                    if title:
                        headlines.append(title)

            if headlines:
                bullet_points = " 1) " + " 2) ".join(headlines)
                return f"Top technology headlines right now:{bullet_points}"
    except Exception:
        pass

    return "Currently unable to reach the live news feed. Please verify internet access."

# Safe Math AST Evaluator
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod
}

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log10,
    "ln": math.log,
    "abs": abs,
    "round": round
}

def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported math operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else ""
        if func_name in SAFE_FUNCTIONS:
            args = [_eval_node(a) for a in node.args]
            return SAFE_FUNCTIONS[func_name](*args)
        raise ValueError(f"Function '{func_name}' is not permitted.")
    else:
        raise ValueError(f"Expression type '{type(node).__name__}' cannot be evaluated.")

def calculate_expression(query):
    """
    Safely parses natural language arithmetic and algebraic expressions.
    Converts phrases like 'what is 25 times 4' or 'square root of 144' to numeric output.
    """
    cleaned = query.lower()
    for word in ["calculate", "solve", "what is", "evaluate", "math"]:
        cleaned = cleaned.replace(word, "")

    # Lexical normalization of spoken terms
    cleaned = re.sub(r"\bsquare root of\s+(\d+(?:\.\d+)?)", r"sqrt(\1)", cleaned)
    cleaned = cleaned.replace("plus", "+")
    cleaned = cleaned.replace("minus", "-")
    cleaned = cleaned.replace("times", "*")
    cleaned = cleaned.replace("multiplied by", "*")
    cleaned = cleaned.replace("divided by", "/")
    cleaned = cleaned.replace("into", "*")
    cleaned = cleaned.replace("x", "*")
    cleaned = cleaned.replace("^", "**")
    cleaned = cleaned.replace("power of", "**")

    # Keep only safe mathematical characters
    expression = re.sub(r"[^0-9\+\-\*\/\(\)\.\s\,a-z]", "", cleaned).strip()

    if not expression:
        return "Please specify the mathematical expression you want me to solve."

    try:
        parsed_ast = ast.parse(expression, mode='eval')
        result = _eval_node(parsed_ast.body)
        
        # Pretty formatting for float / integer results
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        elif isinstance(result, float):
            result = round(result, 4)

        return f"The result of {expression} is {result}."
    except Exception as e:
        return f"I couldn't solve that mathematical expression: {str(e)}"
