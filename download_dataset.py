from huggingface_hub import snapshot_download

# Replace 'YOUR_USERNAME' with your actual Hugging Face username
# or the specific organization/repo where the dataset is hosted.
local_dir = snapshot_download(
    repo_id="Amu21/MadhvaMinds-Dataset",
    repo_type="dataset",
    local_dir="./Dataset"
)

print(f"Dataset successfully downloaded to: {local_dir}")
