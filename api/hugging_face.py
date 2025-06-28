import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def download_model_files(model_config: dict, download_path: Path):
    repo_id = model_config["repo_id"]
    subfolder = model_config["subfolder"]
    filenames = model_config["files"]

    local_paths = {}
    for file_type, filename in filenames.items():
        local_file_path = download_path / filename
        if local_file_path.exists():
            print(f"File already exists: {local_file_path}")
            local_paths[file_type] = local_file_path
            continue

        print(
            f"Downloading file from: https://huggingface.co/{repo_id}/{subfolder}/{filename}"
        )
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
        )

        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(downloaded_file, local_file_path)
        print(f"Saved to: {local_file_path}")
        local_paths[file_type] = local_file_path

    return local_paths
