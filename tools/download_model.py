from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
FILENAME = "all-MiniLM-L6-v2.joblib"
save_folder = "./test/"

# Mode1: Download one file.
# model = hf_hub_download(repo_id=REPO_ID, filename=FILENAME,cache_dir=save_folder)

# Mode2: Download all repository.
snapshot_download(REPO_ID, cache_dir=save_folder)
