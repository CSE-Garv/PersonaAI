import os
import time
from flask import Flask, request, jsonify, render_template, session
import google.generativeai as genai
from flask_session import Session
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"  # Store session on disk
app.config["SESSION_PERMANENT"] = False  # Reset session on refresh
Session(app)

# Configure Gemini API key from environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_system_prompt():
    """Ensures the selected character and source are properly used."""
    character = session.get("character", "Yoda")  # Default to Yoda
    source = session.get("source", "Star Wars")
    return f"You are {character} from {source}. Stay in character and never break your speaking style."


def generate_response(user_input):
    """Generates a response from the AI model using past conversation context."""
    
    if "conversation_history" not in session:
        session["conversation_history"] = []

    model = genai.GenerativeModel("gemini-2.0-flash")
    system_message = get_system_prompt()
    
    # Keep only the last 10 exchanges for context
    history = session["conversation_history"][-10:]

    # Convert history into a list of strings (Gemini expects plain text)
    formatted_history = [f"User: {h['text']}" if h["role"] == "user" else f"AI: {h['text']}" for h in history]

    session["conversation_history"].append({"role": "user", "text": user_input})

    try:
        response = model.generate_content(
            [system_message] + formatted_history + [f"User: {user_input}"],
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 124,
            }
        )

        # Extract response text
        response_text = response.text if hasattr(response, "text") else "Sorry, I couldn't generate a response."

        # Remove character name if present at the beginning
        character = session.get("character", "Mickey Mouse")
        response_text = re.sub(rf"^{re.escape(character)}:\s*", "", response_text, flags=re.IGNORECASE)

    except Exception as e:
        response_text = f"Error generating response: {str(e)}"

    # Store response in session
    session["conversation_history"].append({"role": "model", "text": response_text})
    session.modified = True

    return response_text



@app.route("/")
def home():
    """Render the main page."""
    session.clear()  # Reset session when the page is refreshed
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_conversation():
    """Sets the character for the conversation."""
    data = request.json
    session["character"] = data.get("character", "Yoda")
    session["source"] = data.get("source", "Star Wars")
    session["conversation_history"] = []  # Reset history
    session.modified = True
    return jsonify({"message": f"Character set to {session['character']}."})


@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat interactions with the bot."""
    if "character" not in session:
        return jsonify({"error": "Character not set. Please start a new session."})

    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please enter a message."})

    if user_input.lower() == "bye":
        return jsonify({"response": f"Goodbye! Stay in character as {session['character']}!"})

    time.sleep(1)  # Simulate processing delay

    bot_response = generate_response(user_input)

    return jsonify({
        "response": bot_response,
        "history": [c["text"] for c in session["conversation_history"]]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
