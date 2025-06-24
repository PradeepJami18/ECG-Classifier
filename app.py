import streamlit as st
import joblib
import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from PIL import Image
import tempfile

# Load model and feature list
model = joblib.load(os.path.join("models", "model.pkl"))
features = joblib.load(os.path.join("models", "features.pkl"))

# Function to extract ECG features
def extract_ecg_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image file or unsupported format.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    h = cleaned.shape[0]
    lead_ii = cleaned[int(h * 0.75):int(h * 0.85), :]
    center_line = lead_ii.shape[0] // 2
    trace = 255 - lead_ii[center_line, :]
    signal = gaussian_filter1d(trace.astype(float), sigma=2)

    r_peaks, _ = find_peaks(signal, height=50, distance=30)
    inv_signal = -signal
    q_peaks, _ = find_peaks(inv_signal, height=-30, distance=20)

    if len(r_peaks) < 1:
        raise ValueError("No R-peaks found. Please upload a clearer ECG image.")

    rr_intervals = np.diff(r_peaks) / 25.0
    avg_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else 1
    heart_rate = 60 / avg_rr if avg_rr > 0 else 0

    def interval_in_ms(p1, p2):
        return round((p2 - p1) * 1000 / 25, 1)

    r = r_peaks[1] if len(r_peaks) > 1 else r_peaks[0]
    q = max([p for p in q_peaks if p < r], default=r - 10)
    s = min([p for p in q_peaks if p > r], default=r + 10)
    p = q - 15
    t = s + 30

    return {
        'P_wave_duration_ms': interval_in_ms(p, q),
        'PR_interval_ms': interval_in_ms(p, r),
        'QRS_duration_ms': interval_in_ms(q, s),
        'ST_segment_duration_ms': interval_in_ms(s, t - 20),
        'T_wave_duration_ms': interval_in_ms(t - 20, t),
        'QT_interval_ms': interval_in_ms(q, t),
        'RR_interval_sec': round(avg_rr, 3),
        'heart_rate_bpm': round(heart_rate, 1)
    }

def predict_ecg(image_path):
    features_dict = extract_ecg_features(image_path)
    input_vector = [features_dict[feat] for feat in features]
    input_df = pd.DataFrame([input_vector], columns=features)
    prediction = model.predict(input_df)[0]
    return "Abnormal" if prediction == 1 else "Normal", features_dict

# --- Streamlit UI ---
st.set_page_config(page_title="🪀 ECG Classifier", layout="centered")
st.markdown("<h1 style='text-align: center;'>🪀 ECG Image Classification</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Upload an ECG image to classify it as Normal or Abnormal</h4>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("\ud83d\udcc4 Upload ECG Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    # Save uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    # Display image
    with col1:
        st.image(temp_path, caption="ECG Image", use_container_width=True)

    # Prediction
    with col2:
        with st.spinner("\ud83d\udd0d Analyzing ECG..."):
            try:
                prediction, feature_vals = predict_ecg(temp_path)
                st.success(f"\u2705 Prediction: **{prediction}**", icon="💡")

                with st.expander("\ud83d\udcca Show Extracted Features"):
                    st.dataframe(pd.DataFrame([feature_vals]), use_container_width=True)

            except Exception as e:
                st.error(f"\u274c Error during prediction: {e}")

        # Optional: cleanup
        os.remove(temp_path)
else:
    st.info("\ud83d\udcc1 Please upload a valid ECG image to get started.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Made with 💖 by Jami Pradeep | Powered by Streamlit</div>", unsafe_allow_html=True)
