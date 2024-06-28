from huggingface_hub import hf_hub_download, snapshot_download
from setting import MODEL_LIST, SAVE_FOLDER

# REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
# FILENAME = "all-MiniLM-L6-v2.joblib"


# Mode1: Download one file.
# model = hf_hub_download(repo_id=REPO_ID, filename=FILENAME,cache_dir=save_folder)

# Mode2: Download all repository.
for model in MODEL_LIST:
    snapshot_download(model, cache_dir=SAVE_FOLDER)
