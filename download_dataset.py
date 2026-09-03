import os
import argparse
from huggingface_hub import snapshot_download

def download_dataset(repo_id="jinmang2/ucf-crime-tencrop-i3d", local_dir="./Dataset/ucf-crime-i3d", max_workers=16):
    print(f"Downloading dataset '{repo_id}' to '{local_dir}' with {max_workers} parallel workers...")
    os.makedirs(local_dir, exist_ok=True)
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        max_workers=max_workers
    )
    print(f"Dataset successfully downloaded to: {downloaded_path}")
    return downloaded_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from Hugging Face")
    parser.add_argument("--repo_id", type=str, default="jinmang2/ucf-crime-tencrop-i3d", help="Hugging Face repo ID")
    parser.add_argument("--local_dir", type=str, default="./Dataset/ucf-crime-i3d", help="Local directory path")
    parser.add_argument("--max_workers", type=int, default=16, help="Number of parallel download threads")
    args = parser.parse_args()
    
    download_dataset(repo_id=args.repo_id, local_dir=args.local_dir, max_workers=args.max_workers)


