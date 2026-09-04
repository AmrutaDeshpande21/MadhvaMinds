import pytest
import time
import database

@pytest.mark.asyncio
async def test_init_db_graceful_fallback():
    # Should complete without throwing exceptions even if PostgreSQL service is offline
    await database.init_db()

@pytest.mark.asyncio
async def test_save_and_retrieve_incident():
    database.IN_MEMORY_INCIDENTS.clear()
    
    test_incident = {
        "id": 999901,
        "camera_id": "test-cam-01",
        "type": "Violence/Fight",
        "severity": 5,
        "confidence": 0.94,
        "timestamp": time.time()
    }
    
    await database.save_incident(test_incident)
    incidents = await database.get_incidents(limit=10)
    
    assert len(incidents) >= 1
    recent = incidents[0]
    assert recent["camera_id"] == "test-cam-01"
    assert recent["event_type"] == "Violence/Fight"
    assert recent["severity"] == 5
    assert recent["confidence"] == 0.94
    assert "started_at" in recent

@pytest.mark.asyncio
async def test_get_incidents_limit_and_ordering():
    database.IN_MEMORY_INCIDENTS.clear()
    
    now = time.time()
    for i in range(10):
        await database.save_incident({
            "id": i + 1,
            "camera_id": f"cam-{i}",
            "type": "Intrusion",
            "severity": 3,
            "confidence": 0.80,
            "timestamp": now + i
        })
        
    all_five = await database.get_incidents(limit=5)
    assert len(all_five) == 5
    # Should be ordered descending (latest timestamp first)
    assert all_five[0]["camera_id"] == "cam-9"
