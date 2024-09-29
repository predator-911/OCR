
import streamlit as st
from PIL import Image
import easyocr

# Load OCR readers
@st.cache_resource
def load_easyocr_model():
    reader = easyocr.Reader(['en', 'hi'])  # Specify languages you need
    return reader

reader = load_easyocr_model()

# Function to perform OCR using EasyOCR
@st.cache_data
def perform_ocr_easyocr(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    result = reader.readtext(image, detail=0)  # Get the text without extra details
    extracted_text = " ".join(result)
    return extracted_text

# Function to search keywords in the extracted text
def search_keywords(extracted_text, keyword):
    if keyword.lower() in extracted_text.lower():
        return f"Keyword '{keyword}' found!"
    else:
        return f"Keyword '{keyword}' not found."

# Streamlit interface
def main():
    st.title("OCR with Keyword Search (EasyOCR)")

    uploaded_file = st.file_uploader("Upload an image (JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform OCR using EasyOCR
        with st.spinner('Extracting text...'):
            extracted_text = perform_ocr_easyocr(uploaded_file)

        # Display the extracted text
        st.subheader("Extracted Text:")
        st.write(extracted_text)

        # Keyword search functionality
        keyword = st.text_input("Enter keyword to search in the extracted text")
        if st.button("Search"):
            result = search_keywords(extracted_text, keyword)
            st.write(result)

if __name__ == "__main__":
    main()
