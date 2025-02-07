import streamlit as st
import requests
import io
from PIL import Image

st.sidebar.header("API Settings")
REGISTER_API_URL = st.sidebar.text_input("Register API URL", "http://127.0.0.1:8000/register/")
RECOGNIZE_API_URL = st.sidebar.text_input("Recognize API URL", "http://127.0.0.1:8000/recognize/")

st.title("Face Recognition System")

option = st.radio("Choose Image Input Method:", ("Use Camera", "Upload Image"))

image = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif option == "Use Camera":
    image = st.camera_input("Take a picture")
    if image is not None:
        image = Image.open(image)

if image:
    st.image(image, caption="Selected Image", use_container_width=True)

    img_byte_arr = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(img_byte_arr, format="JPEG", quality=85)
    img_byte_arr = img_byte_arr.getvalue()

    col1, col2 = st.columns(2)
    verify_clicked = col1.button("Verify")
    register_clicked = col2.button("Register")

    if verify_clicked:
        with st.spinner("Verifying..."):
            response = requests.post(RECOGNIZE_API_URL, files={"file": img_byte_arr})
            result = response.json()
            if result.get("message") == "✅ Recognized":
                st.success(f"🎉 Recognized: {result.get('recognized_user', 'Unknown')}")
            else:
                st.warning("⚠️ Not recognized. Please register the person.")

    if register_clicked:
        st.session_state.register_mode = True

if "register_mode" in st.session_state and st.session_state.register_mode:
    person_name = st.text_input("Enter the person's name for registration:")
    confirm_register = st.button("Confirm Registration")

    if confirm_register:
        if not person_name.strip():
            st.error("⚠️ Name cannot be empty. Please enter a valid name.")
        else:
            with st.spinner("Registering..."):
                response = requests.post(REGISTER_API_URL, files={"file": img_byte_arr}, data={"name": person_name})
                result = response.json()

                if "message" in result:
                    st.success(result["message"])
                    del st.session_state.register_mode
                else:
                    st.error(result["error"])
