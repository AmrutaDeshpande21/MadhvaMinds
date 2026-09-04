import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download

HF_REPO_ID = "shahadalll/UCF-crime-binary"
HF_REPO_TYPE = "dataset"
DATASET_VIDEO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset", "ucf-crime-videos"))

TARGET_VIDEOS = [
    "data/test/normal/Normal_Videos_907_x264.mp4",
    "data/test/normal/Normal_Videos_908_x264.mp4",
    "data/test/normal/Normal_Videos_909_x264.mp4",
    "data/test/normal/Normal_Videos_910_x264.mp4",
    "data/test/abnormal/Fighting042_x264.mp4",
    "data/test/abnormal/Fighting043_x264.mp4",
    "data/test/abnormal/Abuse041_x264.mp4",
    "data/test/abnormal/Arrest001_x264.mp4",
    "data/test/abnormal/Burglary001_x264.mp4",
    "data/test/abnormal/Robbery001_x264.mp4"
]

def download_video(hf_path):
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
        print(f"Error downloading {fn}: {e}")
        return fn, False

def preload_real_ucf_mp4s():
    os.makedirs(DATASET_VIDEO_DIR, exist_ok=True)
    print(f"Ensuring real UCF-Crime test set videos in {DATASET_VIDEO_DIR}...")
    
    # Remove any old dummy sample file if present
    dummy_path = os.path.join(DATASET_VIDEO_DIR, "Normal_sample.mp4")
    if os.path.exists(dummy_path):
        try:
            os.remove(dummy_path)
            print("Removed old Normal_sample.mp4")
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(download_video, path) for path in TARGET_VIDEOS]
        for f in as_completed(futures):
            f.result()
    print("UCF-Crime MP4 video preloading complete!")

if __name__ == "__main__":
    preload_real_ucf_mp4s()
