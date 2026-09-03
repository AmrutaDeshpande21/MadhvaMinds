import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

HF_REPO_ID = "jinmang2/ucf-crime-tencrop-i3d"
HF_REPO_TYPE = "dataset"
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset", "ucf-crime-i3d"))

def download_one(filename):
    local_path = os.path.join(DATASET_DIR, filename)
    if os.path.exists(local_path):
        return True
    try:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=filename,
            local_dir=DATASET_DIR
        )
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def preload_balanced_subset(max_normal=150, max_abnormal=150, max_test=50, max_workers=20):
    print(f"Preloading dataset subset to {DATASET_DIR} using {max_workers} threads...")
    api = HfApi()
    repo_files = [f for f in api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE) if f.endswith('.npy')]

    train_normal = [f for f in repo_files if f.startswith('UCF_Train') and 'Normal' in f][:max_normal]
    train_abnormal = [f for f in repo_files if f.startswith('UCF_Train') and 'Normal' not in f][:max_abnormal]
    test_files = [f for f in repo_files if f.startswith('UCF_Test')][:max_test]

    targets = train_normal + train_abnormal + test_files
    print(f"Target files to ensure locally: {len(targets)} (Normal: {len(train_normal)}, Abnormal: {len(train_abnormal)}, Test: {len(test_files)})")

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, f): f for f in targets}
        for future in as_completed(futures):
            if future.result():
                success_count += 1

    print(f"Preload complete. {success_count}/{len(targets)} files verified/downloaded.")

if __name__ == "__main__":
    preload_balanced_subset()
