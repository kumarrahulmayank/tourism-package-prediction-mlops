
from huggingface_hub import HfApi

HF_USER = "rahulmayank"
SPACE_REPO = f"{HF_USER}/tourism-wellness-space"

api = HfApi()

api.create_repo(
    repo_id=SPACE_REPO,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True
)

api.upload_folder(
    repo_id=SPACE_REPO,
    repo_type="space",
    folder_path="master_folder/deployment",
    commit_message="Upload deployment files"
)

print("Deployment files uploaded successfully")
