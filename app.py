import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from utils import (
    calculate_fall_score,
    calculate_panic_score,
    calculate_conflict_score,
    draw_visuals,
    N_FRAMES_FOR_ANALYSIS,
)

# --- PAGE & AI CONFIGURATION ---
VIDEO_SOURCE = "/app/input/test_video_panic.mp4"
MODEL_PATH = "/app/models/yolov8n-pose.onnx"
CONFIDENCE_THRESHOLD = 0.3
MAX_HISTORY = 50

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Aegis AI - Final Demo", page_icon="🛡️", layout="wide")

# --- STYLES ---
st.markdown("""<style>.stMetric {border: 1px solid #2e3440; border-radius: 10px; padding: 10px; background-color: #2e3440;}</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Loads the YOLO model once and caches it."""
    try:
        model = YOLO(model_path)
        print("✅ AI Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ CRITICAL: Could not load the YOLO model: {e}")
        return None

def main():
    st.title("🛡️ Aegis AI - Crowd Intelligence Platform")
    st.info("🔵 Monolithic Demo | Analyzing Pre-recorded Video File")

    # --- Load Model ---
    model = load_model(MODEL_PATH)
    if model is None:
        return

    # --- LAYOUT ---
    video_col, analysis_col = st.columns([3, 1])
    with video_col:
        st.header("Live Feed Analysis")
        video_placeholder = st.empty()
    with analysis_col:
        st.header("Real-Time Analytics")
        # These are now just placeholders that will be overwritten in the loop
        count_metric = st.empty()
        panic_metric = st.empty()
        conflict_metric = st.empty()
        st.divider()
        st.subheader("⚠️ High-Risk Alerts")
        alerts_placeholder = st.empty()
        alerts_placeholder.info("Awaiting analysis...")

    # --- VIDEO & PROCESSING LOOP ---
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        st.error("❌ CRITICAL: Error opening video source. Check file path and codecs.")
        return

    track_history = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        results = model.track(frame, persist=True, verbose=False, conf=CONFIDENCE_THRESHOLD)
        analytics_payload = {"person_count": 0, "persons": [], "overall_panic_score": 0, "conflict_score": 0}

        if results[0].boxes.id is not None:
            tracked_boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            keypoints = results[0].keypoints.xy.cpu().numpy()
            current_person_data = {}
            for box, track_id, kpts in zip(tracked_boxes, track_ids, keypoints):
                if track_id not in track_history: track_history[track_id] = []
                history = track_history[track_id]
                history.append({"box": box.tolist(), "keypoints": kpts.tolist()})
                if len(history) > MAX_HISTORY: history.pop(0)
                fall_score = calculate_fall_score(history)
                panic_score, speed = calculate_panic_score(history)
                current_person_data[track_id] = {"id": int(track_id), "box": box.tolist(), "keypoints": kpts.tolist(), "fall_score": fall_score, "panic_score": panic_score, "speed_kph": speed}
            analytics_payload["persons"] = list(current_person_data.values())
            analytics_payload["person_count"] = len(current_person_data)
            if current_person_data:
                analytics_payload["overall_panic_score"] = np.mean([p['panic_score'] for p in current_person_data.values()])
                analytics_payload["conflict_score"] = calculate_conflict_score(list(current_person_data.values()))
        
        annotated_frame = draw_visuals(frame, analytics_payload.get('persons', []), analytics_payload['overall_panic_score'], analytics_payload['conflict_score'])
        
        # --- UPDATE STREAMLIT DASHBOARD (CORRECTED) ---
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, use_column_width=True)
        
        # To update metrics, we re-declare them inside their placeholder containers
        count_metric.metric("👥 Person Count", f"{analytics_payload['person_count']}")
        panic_metric.metric("🔥 Overall Panic Score", f"{analytics_payload['overall_panic_score']:.2f}")
        conflict_metric.metric("⚔️ Conflict Score", f"{analytics_payload['conflict_score']:.2f}")

        high_risk_persons = []
        for p_data in analytics_payload.get('persons', []):
            if p_data['fall_score'] > 0.9:
                high_risk_persons.append({"ID": f"Track-{p_data['id']}", "Status": "🔴 FALL DETECTED", "Speed (kph)": f"{p_data['speed_kph']:.1f}"})
            elif p_data['panic_score'] > 0.7:
                high_risk_persons.append({"ID": f"Track-{p_data['id']}", "Status": "🟠 High Speed / Panic", "Speed (kph)": f"{p_data['speed_kph']:.1f}"})
        
        if high_risk_persons:
            alerts_placeholder.dataframe(pd.DataFrame(high_risk_persons), use_container_width=True)
        else:
            alerts_placeholder.success("✅ All individuals appear normal.")

    cap.release()

if __name__ == "__main__":
    main()