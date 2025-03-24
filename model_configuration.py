import json
import os

def load_json(file_path):
    """Loads a JSON file if it exists, otherwise returns an empty dictionary."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def save_json(file_path, data):
    """Saves data to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def get_settings_from_config():
    """Retrieve model names and other settings from config.json."""
    # Load available models and config
    available_models = load_json("available_models.json")
    config = load_json("config.json")

    settings = {}

    # Handle LLM model selection
    llm_model_key = config.get("llm_model")
    if llm_model_key and isinstance(llm_model_key, dict):
        model_name = next(iter(llm_model_key))

        if "gpt" in model_name:
            llm_model_name = model_name  # Directly use GPT model name
        else:
            # Get the model name from available_models.json  "huggingface"
            llm_model_name = available_models["llm_models"].get(model_name, {}).get("ollama") or \
                             available_models["llm_models"].get(model_name, {}).get("huggingface") or \
                             model_name  # Fallback to original model name
        
        settings["llm_model_name"] = llm_model_name

    # Handle embeddings model selection
    embeddings_model_key = config.get("embeddings_model")
    if embeddings_model_key and isinstance(embeddings_model_key, dict):
        embeddings_model_name = next(iter(embeddings_model_key))  # Extract model key
        settings["embeddings_model_name"] = available_models["embeddings_models"].get(embeddings_model_name, embeddings_model_name)
    
    elif isinstance(embeddings_model_key, str):  
        settings["embeddings_model_name"] = available_models["embeddings_models"].get(embeddings_model_key, embeddings_model_key)

    # Add all other settings from config.json
    for key, value in config.items():
        if key not in ["llm_model", "embeddings_model"]:
            settings[key] = value

    return settings