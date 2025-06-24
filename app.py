import tempfile

# --- Streamlit UI ---

st.set_page_config(page_title="🫀 ECG Classifier", layout="centered")
st.markdown("<h1 style='text-align: center;'>🫀 ECG Image Classification</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Upload an ECG image to classify it as Normal or Abnormal</h4>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload ECG Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    col1, col2 = st.columns([1, 2])

    # Show image
    with col1:
        image = Image.open(temp_path)
        st.image(image, caption="ECG Image", use_container_width=True)

    # Make prediction
    with col2:
        with st.spinner("🔍 Analyzing ECG..."):
            try:
                prediction, feature_vals = predict_ecg(temp_path)
                st.success(f"✅ Prediction: **{prediction}**", icon="💡")

                with st.expander("📊 Show Extracted Features"):
                    st.dataframe(pd.DataFrame([feature_vals]), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
else:
    st.info("📁 Please upload a valid ECG image to get started.")

st.markdown("---")
st.markdown("<div style='text-align: center;'>Made with 💖 by Jami Pradeep | Powered by Streamlit</div>", unsafe_allow_html=True)
