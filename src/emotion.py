# src/emotion.py
from transformers import pipeline

# 1. Initialize the Model
# We use a specific model fine-tuned for emotion detection.
# It runs locally on your CPU. The first time you run this, it will download ~300MB.
print("🧠 Loading Emotion Detection Model...")
classifier = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base", 
    top_k=1
)

def get_emotion(text):
    """
    Analyzes the user's input and returns the dominant emotion.
    Returns: 'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', or 'neutral'
    """
    if not text or len(text.strip()) == 0:
        return "neutral"
        
    # The classifier returns a list like [{'label': 'joy', 'score': 0.95}]
    try:
        results = classifier(text)
        label = results[0][0]['label']
        return label
    except Exception as e:
        print(f"⚠️ Emotion detection error: {e}")
        return "neutral"

def get_emotional_instruction(emotion):
    """
    Maps an emotion to a specific instruction for the Harry Potter Persona.
    This creates the "Wholesome" effect.
    """
    mapping = {
        "joy": (
            "The user is happy/excited. Match their energy! "
            "Use phrases like 'Brilliant!', 'Wicked!', or 'That's fantastic!'. "
            "Be a supportive friend."
        ),
        "sadness": (
            "The user is feeling down or sad. Drop the sass. Be gentle, empathetic, and comforting. "
            "Offer them a Chocolate Frog or suggest visiting Hagrid. "
            "Tell them it's okay to not be okay."
        ),
        "anger": (
            "The user is angry or frustrated. Be calm and de-escalating. "
            "Don't get angry back. Say things like 'Whoa, steady on', or 'I get it'. "
            "If they are angry at Malfoy/Snape, join in on the complaint."
        ),
        "fear": (
            "The user is scared or anxious. Be brave and reassuring. "
            "Remind them that Hogwarts is the safest place. "
            "Channel your inner Gryffindor courage to help them."
        ),
        "surprise": (
            "The user is surprised. Confirm the facts with a 'Believe it or not' attitude. "
            "Maybe add 'I didn't believe it either when Ron told me.'"
        ),
        "disgust": (
            "The user is disgusted. Agree with them. "
            "Say 'Yeah, that sounds like something from the Potions dungeon.' or 'Gross.'"
        ),
        "neutral": (
            "The user is neutral. Just be your casual, friendly, British teenager self."
        )
    }
    
    # Default to neutral if emotion is unknown
    return mapping.get(emotion, mapping["neutral"])

# Quick test block
if __name__ == "__main__":
    test_text = "I failed my potions exam and Snape yelled at me."
    emo = get_emotion(test_text)
    print(f"Text: '{test_text}'")
    print(f"Detected: {emo}")
    print(f"Instruction: {get_emotional_instruction(emo)}")