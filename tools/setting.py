HUGGING_FACE_TOKEN = "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"
SAVE_FOLDER = "./model"
MODEL_LIST = {
    "rag": {"owner": "sentence-transformers", "model_name": "all-MiniLM-L6-v2"},
    "gen_text": {"owner": "HuggingFaceH4", "model_name": "zephyr-7b-beta"},
    "img_text2text": {"owner": "openai", "model_name": "clip-vit-base-patch16"},
}
