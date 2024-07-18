import os

from huggingface_hub import hf_hub_download, snapshot_download

from tools.setting import MODEL_LIST, SAVE_FOLDER

# REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
# FILENAME = "all-MiniLM-L6-v2.joblib"


# Mode1: Download one file.
# model = hf_hub_download(repo_id=REPO_ID, filename=FILENAME,cache_dir=save_folder)


def model_exist(model_owner: str, model_name: str) -> bool:
    """Check model wether  download or not.

    Args:
        model_name (str): Full model name.

    Returns:
        bool: exist or not.
    """

    if os.path.exists(
        os.path.join(SAVE_FOLDER, "models--" + model_owner + "--" + model_name)
    ):
        return True
    return False


def build_path(root: str, model_info: dict) -> str:
    """Generate model path

    Args:
        root (str) : root path
        model_info (dict): Contain model owner and model name.

    Returns:
        str: model path
    """
    model_dir = "models--" + model_info["owner"] + "--" + model_info["model_name"]
    model_root_dir = os.path.join(root, model_dir, "snapshots")

    snapshot_dirs = [
        d
        for d in os.listdir(model_root_dir)
        if os.path.isdir(os.path.join(model_root_dir, d))
    ]
    latest_snapshot_dir = max(
        snapshot_dirs, key=lambda d: os.path.getmtime(os.path.join(model_root_dir, d))
    )
    path = os.path.join(model_root_dir, latest_snapshot_dir)

    return path


# Mode2: Download all repository.
def download_model() -> dict:
    """Download model from hugging face.

    Returns:
        dict: Add model path.
    """
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER, exist_ok=True)

    for model_usage, model_info in MODEL_LIST.items():
        if model_exist(
            model_owner=model_info["owner"], model_name=model_info["model_name"]
        ):
            print(f'{model_info["model_name"]} is exist!')

        else:
            model_repository = model_info["owner"] + "/" + model_info["model_name"]
            snapshot_download(
                model_repository,
                cache_dir=SAVE_FOLDER,
            )
            print(f'Success download {model_info["model_name"]}!')

        model_path = build_path(root=SAVE_FOLDER, model_info=model_info)
        model_info["path"] = model_path
    return MODEL_LIST


if __name__ == "__main__":
    a = download_model()
    print(a)
