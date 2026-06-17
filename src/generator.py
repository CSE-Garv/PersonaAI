from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.config import GROQ_API_KEY

def auto_generate_system_prompt(character_name, book_excerpt):
    """
    Reads the book and writes a professional System Prompt for that character.
    """
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert AI Persona Designer. \n"
        "Your task is to write a 'System Prompt' for a chatbot based on the character '{char}'.\n"
        "Analyze this text sample from their book:\n"
        "'''{excerpt}'''\n\n"
        "Write a system prompt that includes:\n"
        "1. Role definition (You are {char}).\n"
        "2. Personality traits (Tone, slang, catchphrases).\n"
        "3. Strict Logic Rules (Family definitions, Enemies).\n"
        "4. Output strictly the prompt text, nothing else."
    )
    
    chain = prompt | llm
    response = chain.invoke({"char": character_name, "excerpt": book_excerpt[:2000]})
    return response.content