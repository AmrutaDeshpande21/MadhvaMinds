from huggingface_hub import hf_hub_download, HfApi
import os
import glob

# Ensure the destination directory exists
dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Dataset", "ucf-crime-videos"))
os.makedirs(dest_dir, exist_ok=True)

repo_id = "Amu21/MadhvaMinds-Dataset"
api = HfApi()

try:
    print(f"Fetching file list from {repo_id}...")
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    
    mp4_files = [f for f in files if f.endswith('.mp4')]
    
    if not mp4_files:
        print("No .mp4 files found in the dataset.")
    else:
        print(f"Found {len(mp4_files)} mp4 files. Downloading...")
        
        for file in mp4_files:
            print(f"Downloading {file}...")
            local_path = hf_hub_download(repo_id=repo_id, filename=file, repo_type="dataset", local_dir=dest_dir)
            print(f"Saved to {local_path}")
            
    print("Download process completed.")
except Exception as e:
    print(f"Error during download: {e}")
