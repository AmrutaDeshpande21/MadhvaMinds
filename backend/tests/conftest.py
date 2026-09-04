import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Disable Telegram dispatch and mock camera hardware for automated tests
os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ["TELEGRAM_CHAT_ID"] = ""

@pytest.fixture(scope="session", autouse=True)
def mock_hardware_and_camera():
    """Prevent automated tests from attempting to open local webcam hardware or spawn background threads."""
    with patch("camera_worker.camera_worker_process", MagicMock()), \
         patch("main.run_vision_pipeline", MagicMock()), \
         patch("multiprocessing.Process.start", MagicMock()):
        yield
