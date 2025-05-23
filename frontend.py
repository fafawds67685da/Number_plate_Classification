import streamlit as st
import requests
from PIL import Image
import base64

# Read local background image and convert to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image_base64 = get_base64_of_bin_file("NQ.jpg")  # Adjust path if needed

# Inject CSS for background image and white text
st.markdown(
    f"""
    <style>
    body, .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        position: relative;
        color: white;
        height: 100vh;
    }}

    /* Overlay to darken the background for better contrast */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: rgba(0, 0, 0, 0.5); /* darker overlay */
        z-index: -1;
    }}

    /* Force all headers and text to white */
    h1, h2, h3, p, label, div, span {{
        color: white !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Indian Number Plate Classifier")
st.markdown("Upload a number plate image (any format), and it will predict the state of registration.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

rto_to_state = {
    "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh", "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli and Daman and Diu", "GA": "Goa", "GJ": "Gujarat",
    "HP": "Himachal Pradesh", "HR": "Haryana", "JH": "Jharkhand", "JK": "Jammu and Kashmir",
    "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh", "MH": "Maharashtra", "ML": "Meghalaya",
    "MN": "Manipur", "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha",
    "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu",
    "TR": "Tripura", "TS": "Telangana", "UK": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal"
}

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Sending image to model..."):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post("http://localhost:8000/predict", files=files)
                if response.status_code == 200:
                    result = response.json()
                    acronym = result.get("label")
                    confidence = result.get("confidence", 0)
                    full_state = rto_to_state.get(acronym, "Unknown State")

                    st.success(
                        f"The model predicts that this number plate is registered in **{full_state}** "
                        f"({acronym}) with a confidence level of {confidence:.2%}."
                    )
                else:
                    st.error("Prediction failed. Please try again.")
            except requests.exceptions.RequestException as e:
                st.error(f"Request error: {e}")
