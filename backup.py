import streamlit as st
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Load the OCR model and processor
@st.cache_resource
def load_model():
    # Use the correct pre-trained model from Hugging Face
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model

processor, model = load_model()

# Function to perform OCR on uploaded image
def perform_ocr(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to("cuda" if torch.cuda.is_available() else "cpu")
    
    outputs = model.generate(pixel_values)
    extracted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return extracted_text

# Function to search keywords in the extracted text
def search_keywords(extracted_text, keyword):
    if keyword.lower() in extracted_text.lower():
        return f"Keyword '{keyword}' found!"
    else:
        return f"Keyword '{keyword}' not found."

# Streamlit interface
def main():
    st.title("Multilingual OCR with Keyword Search (Hindi and English)")
    
    uploaded_file = st.file_uploader("Upload an image (JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Perform OCR
        with st.spinner('Extracting text...'):
            extracted_text = perform_ocr(uploaded_file)
        
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
