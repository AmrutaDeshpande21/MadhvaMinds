import pytest
from starlette.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "running" in data["message"].lower()

def test_get_incidents_endpoint():
    response = client.get("/api/incidents")
    assert response.status_code == 200
    data = response.json()
    assert "incidents" in data
    assert isinstance(data["incidents"], list)

def test_dataset_videos_endpoint():
    response = client.get("/api/dataset/videos")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "videos" in data
    assert isinstance(data["videos"], list)
    assert data["count"] >= 0

def test_simulation_status_endpoint():
    response = client.get("/api/simulation/status")
    assert response.status_code == 200
    data = response.json()
    assert "running" in data

def test_analyze_video_missing_filename():
    response = client.post("/api/dataset/analyze-video", json={})
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert "filename parameter required" in data["error"]

def test_analyze_video_nonexistent_file():
    response = client.post("/api/dataset/analyze-video", json={"filename": "does_not_exist_999.mp4"})
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert "not found" in data["error"].lower()
