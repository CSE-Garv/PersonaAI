# Persona AI Chatbot

A Flask-based chatbot that mimics fictional characters' speaking styles using Google's Gemini AI. Users can select a character, and the AI will generate responses while staying in character.

## Features
- Allows users to choose a fictional character and its source.
- Maintains conversation history for context-aware responses.
- Uses Google Gemini AI for text generation.
- Stores session data to maintain chat continuity.

## Tech Stack
- **Backend:** Flask, Flask-Session
- **AI Model:** Google Gemini API
- **Environment Management:** dotenv
- **Frontend:** HTML/CSS (with Flask rendering)

## Installation
### 1. Clone the Repository
```sh
git clone https://github.com/your-repo/persona-ai-chatbot.git
cd persona-ai-chatbot
```

### 2. Create a Virtual Environment
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```sh
pip install flask flask-session google-generativeai python-dotenv
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory and add your Gemini API key:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 5. Run the Application
```sh
python app.py
```
Access the chatbot at `http://127.0.0.1:5000/`.

## File Structure
```
persona-ai-chatbot/
│── env/                # Virtual environment folder
│── templates/          # HTML templates for frontend
│── Persona_AI.py       # Main Flask application file
│── .env                # Environment variables
│── README.md           # Project documentation
```

## API Endpoints
### `POST /start`
- Initializes a new session and sets the character.
- **Request:** `{ "character": "Yoda", "source": "Star Wars" }`
- **Response:** `{ "message": "Character set to Yoda." }`

### `POST /chat`
- Sends a message and receives an in-character response.
- **Request:** `{ "message": "Hello there!" }`
- **Response:** `{ "response": "Greetings, young padawan!" }`

## Contributing
Feel free to fork this repository and submit pull requests with improvements.
