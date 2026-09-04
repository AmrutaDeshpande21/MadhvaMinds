import pytest
import math
import numpy as np
from shapely.geometry import Polygon, box

from violence.test_real_video import format_anomaly_category, detect_temporal_anomaly_windows


def check_intrusion_math(bbox, restricted_polygon, overlap_thresh=0.10):
    x1, y1, x2, y2 = bbox
    person_box = box(x1, y1, x2, y2)
    if not person_box.intersects(restricted_polygon):
        return False, 0.0
    inter_area = person_box.intersection(restricted_polygon).area
    overlap_ratio = inter_area / person_box.area
    return overlap_ratio >= overlap_thresh, overlap_ratio


def calculate_torso_angle_math(shoulder_mid, hip_mid):
    dx = abs(hip_mid[0] - shoulder_mid[0])
    dy = abs(hip_mid[1] - shoulder_mid[1]) + 1e-6
    return math.degrees(math.atan2(dx, dy))


def is_fall_math(angle_history, centroid_history, angle_thresh=60, still_frames=8, vel_thresh=5.0):
    if len(angle_history) < still_frames or angle_history[-1] < angle_thresh:
        return False
    recent = list(centroid_history)[-still_frames:]
    if len(recent) < still_frames:
        return False
    velocities = [math.dist(recent[i], recent[i-1]) for i in range(1, len(recent))]
    return np.mean(velocities) < vel_thresh


class TestIntrusionLogic:
    def test_person_completely_outside_restricted_zone(self):
        polygon = Polygon([(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)])
        bbox_outside = [0.05, 0.05, 0.20, 0.20]
        is_intruder, ratio = check_intrusion_math(bbox_outside, polygon)
        assert not is_intruder
        assert ratio == 0.0

    def test_person_inside_restricted_zone(self):
        polygon = Polygon([(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)])
        bbox_inside = [0.4, 0.4, 0.6, 0.6]
        is_intruder, ratio = check_intrusion_math(bbox_inside, polygon)
        assert is_intruder
        assert ratio > 0.95

    def test_person_partially_overlapping_zone(self):
        polygon = Polygon([(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)])
        # Half inside: x from 0.2 to 0.4, y from 0.3 to 0.5
        bbox_overlap = [0.2, 0.3, 0.4, 0.5]
        is_intruder, ratio = check_intrusion_math(bbox_overlap, polygon, overlap_thresh=0.10)
        assert is_intruder
        assert 0.4 <= ratio <= 0.6


class TestFallDetectionLogic:
    def test_upright_posture_angle(self):
        shoulder_mid = (100.0, 50.0)
        hip_mid = (100.0, 150.0)
        angle = calculate_torso_angle_math(shoulder_mid, hip_mid)
        # Perfectly vertical torso angle should be 0 degrees
        assert abs(angle) < 0.1

    def test_fallen_horizontal_posture_angle(self):
        shoulder_mid = (50.0, 100.0)
        hip_mid = (150.0, 100.0)
        angle = calculate_torso_angle_math(shoulder_mid, hip_mid)
        # Horizontal torso angle should be close to 90 degrees
        assert abs(angle - 90.0) < 0.5

    def test_fall_heuristic_triggers_on_collapse_and_stillness(self):
        angle_history = [15.0] * 20 + [75.0] * 10
        # Posture collapsed and subject is stationary on the floor
        centroid_history = [(200.0, 300.0)] * 30
        assert is_fall_math(angle_history, centroid_history)

    def test_fall_heuristic_rejects_walking_or_running_fast(self):
        angle_history = [15.0] * 20 + [70.0] * 10
        # High velocity movement
        centroid_history = [(float(i * 20), float(i * 20)) for i in range(30)]
        assert not is_fall_math(angle_history, centroid_history)


class TestAnomalyCategoryAndTemporalWindows:
    def test_format_anomaly_normal_video(self):
        cat_clean, event_type, label = format_anomaly_category("Normal_Videos_907_x264.mp4")
        assert cat_clean == "Normal"
        assert event_type == "normal"
        assert label == "NORMAL ACTIVITY"

    def test_format_anomaly_fighting_video(self):
        cat_clean, event_type, label = format_anomaly_category("Fighting042_x264.mp4")
        assert cat_clean == "Fighting"
        assert event_type == "fighting"
        assert "FIGHTING" in label

    def test_format_anomaly_abuse_video(self):
        cat_clean, event_type, label = format_anomaly_category("Abuse041_x264.mp4")
        assert cat_clean == "Abuse"
        assert event_type == "abuse"
        assert "ABUSE" in label

    def test_detect_temporal_windows_for_normal_stream_is_empty(self):
        dummy_feats = np.random.randn(32, 2048).astype(np.float32)
        timestamps = [float(i) for i in range(32)]
        windows = detect_temporal_anomaly_windows(dummy_feats, timestamps, duration=32.0, is_anom=False)
        assert windows == []

    def test_detect_temporal_windows_for_anomaly_stream_returns_intervals(self):
        # Create features with high energy spike in the middle
        dummy_feats = np.ones((32, 2048), dtype=np.float32) * 0.1
        dummy_feats[12:18] = 5.0 # Energy spike
        timestamps = [float(i * 2) for i in range(32)]
        windows = detect_temporal_anomaly_windows(dummy_feats, timestamps, duration=64.0, is_anom=True)
        assert len(windows) > 0
        start, end = windows[0]
        assert start < end
        assert end <= 64.0
