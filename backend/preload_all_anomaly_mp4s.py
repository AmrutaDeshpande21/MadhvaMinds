import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

HF_REPO_ID = "shahadalll/UCF-crime-binary"
HF_REPO_TYPE = "dataset"
DATASET_VIDEO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset", "ucf-crime-videos"))

def discover_and_download():
    os.makedirs(DATASET_VIDEO_DIR, exist_ok=True)
    api = HfApi()
    
    print(f"Querying {HF_REPO_ID} for all anomaly MP4 videos...")
    try:
        repo_files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
        mp4_files = [f for f in repo_files if f.endswith('.mp4')]
        print(f"Found {len(mp4_files)} total MP4 files in repo.")
        
        abnormal = [f for f in mp4_files if "abnormal" in f or "Normal" not in f][:15]
        normal = [f for f in mp4_files if "normal" in f or "Normal" in f][:5]
        
        targets = abnormal + normal
        print(f"Downloading {len(targets)} selected MP4 video files...")
        
        def dl(hf_path):
            fn = os.path.basename(hf_path)
            dest = os.path.join(DATASET_VIDEO_DIR, fn)
            if os.path.exists(dest) and os.path.getsize(dest) > 100000:
                return fn, True
            try:
                downloaded = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    repo_type=HF_REPO_TYPE,
                    filename=hf_path
                )
                shutil.copy(downloaded, dest)
                print(f"Downloaded: {fn}")
                return fn, True
            except Exception as e:
                print(f"Failed {fn}: {e}")
                return fn, False

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(dl, path) for path in targets]
            for f in as_completed(futures):
                f.result()
                
        print("All target MP4 videos updated in Dataset/ucf-crime-videos!")
    except Exception as e:
        print(f"Error querying HF repo: {e}")

if __name__ == "__main__":
    discover_and_download()
