import json
import os
from src.config import JSON_PATH

def load_personalities():
    if not os.path.exists(JSON_PATH):
        return {}
    with open(JSON_PATH, "r") as f:
        return json.load(f)

def save_personality(name, book_file, system_prompt, description):
    data = load_personalities()
    data[name] = {
        "book_source": book_file,
        "system_prompt": system_prompt,
        "description": description
    }
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)
    return True

def delete_personality(name):
    data = load_personalities()
    if name in data:
        del data[name]
        with open(JSON_PATH, "w") as f:
            json.dump(data, f, indent=4)