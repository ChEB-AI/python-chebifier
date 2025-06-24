import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

# Updated registry: use a list of filenames if you're downloading a folder
MODEL_REGISTRY = {
    "electra": {
        "repo_id": "aditya0by0/python-chebifier",
        "subfolder": "electra",
        "filenames": ["electra.ckpt", "classes.txt"],
    }
}

DOWNLOAD_PATH = Path(__file__).resolve().parent / "api_models"


def download_model(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_info = MODEL_REGISTRY[model_name]
    repo_id = model_info["repo_id"]
    subfolder = model_info["subfolder"]
    filenames = model_info["filenames"]

    local_paths = []
    for filename in filenames:
        local_model_path = DOWNLOAD_PATH / model_name / filename
        if local_model_path.exists():
            print(f"File already exists: {local_model_path}")
            local_paths.append(local_model_path)
            continue

        print(f"Downloading: {repo_id}/{filename} (subfolder: {subfolder})")
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
        )

        local_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(downloaded_file, local_model_path)
        print(f"Saved to: {local_model_path}")
        local_paths.append(local_model_path)

    return local_paths


if __name__ == "__main__":
    paths = download_model("electra")
    print("Downloaded files:")
    for p in paths:
        print(p)
