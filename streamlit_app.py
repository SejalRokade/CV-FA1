import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os

import sys
import os
# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import to_grayscale_enhanced, denoise
from src.segment import otsu_segment, largest_component
from src.features import compute_shape_features, compute_texture_features, compute_local_descriptors, descriptors_to_bovw


# BloodMNIST class labels
BLOOD_CLASSES = {
    0: "Basophil",
    1: "Eosinophil", 
    2: "Erythroblast",
    3: "Immature Granulocytes",
    4: "Lymphocyte",
    5: "Monocyte",
    6: "Neutrophil",
    7: "Platelet"
}

# Group into main categories
CELL_CATEGORIES = {
    "RBC": [2],  # Erythroblast
    "WBC": [0, 1, 3, 4, 5, 6],  # All white blood cell types
    "Platelet": [7]  # Platelet
}


def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV format (BGR)"""
    # Convert PIL to RGB numpy array
    rgb_image = np.array(pil_image)
    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image


def cv2_to_pil(cv2_image):
    """Convert OpenCV image (BGR) to PIL format (RGB)"""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert to PIL
    return Image.fromarray(rgb_image)


def classify_single_image(image, model_data):
    """Classify a single image using the trained model"""
    clf = model_data["clf"]
    codebook = model_data["codebook"]
    
    # Preprocess
    gray = to_grayscale_enhanced(image)
    gray = denoise(gray, ksize=5)
    
    # Segment
    mask = otsu_segment(gray)
    mask = largest_component(mask)
    
    # Extract features
    shape_features = compute_shape_features(mask)
    texture_features = compute_texture_features(gray, mask)
    
    if codebook is not None:
        # Extract local descriptors
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        descriptors = compute_local_descriptors(masked, method="sift")
        
        # Convert to BoVW
        bovw_features = descriptors_to_bovw(descriptors, codebook)
        
        # Combine all features
        features = np.concatenate([shape_features, texture_features, bovw_features])
    else:
        features = np.concatenate([shape_features, texture_features])
    
    # Predict
    prediction = clf.predict([features])[0]
    probability = np.max(clf.predict_proba([features]))
    
    return prediction, probability


def get_cell_category(prediction):
    """Get the main cell category from the prediction"""
    for category, class_indices in CELL_CATEGORIES.items():
        if prediction in class_indices:
            return category
    return "Unknown"


def main():
    st.set_page_config(
        page_title="Blood Cell Classifier",
        page_icon="🩸",
        layout="wide"
    )
    
    st.title("🩸 Blood Cell Morphology Classifier")
    st.markdown("Upload blood smear images to classify them as RBC, WBC, or Platelet")
    
    # Sidebar for model upload
    st.sidebar.header("Model Configuration")
    
    # Check if model exists
    model_path = "blood_cell_model.pkl"
    model_data = None
    
    if os.path.exists(model_path):
        try:
            import joblib
            model_data = joblib.load(model_path)
            st.sidebar.success("✅ Model loaded successfully!")
            st.sidebar.info(f"Model: {type(model_data['clf']).__name__}")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
    else:
        st.sidebar.warning("⚠️ No trained model found!")
        st.sidebar.info("Please train a model first using:")
        st.sidebar.code("python -m src.train_eval --feature sift --clf svm")
        st.sidebar.info("Or upload a trained model file.")
        
        # Allow manual model upload
        uploaded_model = st.sidebar.file_uploader(
            "Upload trained model (.pkl)", 
            type=['pkl'],
            help="Upload a trained model file from the training script"
        )
        
        if uploaded_model is not None:
            try:
                import joblib
                model_data = joblib.load(uploaded_model)
                st.sidebar.success("✅ Model uploaded successfully!")
                st.sidebar.info(f"Model: {type(model_data['clf']).__name__}")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading uploaded model: {str(e)}")
    
    # Main content
    if model_data is None:
        st.error("Please load or upload a trained model to continue.")
        return
    
    # File uploader
    st.header("Image Upload")
    uploaded_files = st.file_uploader(
        "Choose blood smear images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload one or more blood smear images for classification"
    )
    
    if uploaded_files:
        st.header("Classification Results")
        
        # Process each image
        for i, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"Image {i+1}: {uploaded_file.name}")
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Image:**")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with col2:
                st.write("**Analysis:**")
                
                try:
                    # Convert to OpenCV format
                    cv_image = pil_to_cv2(image)
                    
                    # Classify
                    prediction, probability = classify_single_image(cv_image, model_data)
                    
                    # Get class name and category
                    class_name = BLOOD_CLASSES.get(prediction, "Unknown")
                    cell_category = get_cell_category(prediction)
                    
                    # Display results
                    st.success(f"**Classification:** {cell_category}")
                    st.info(f"**Specific Type:** {class_name}")
                    st.metric("**Confidence:**", f"{probability:.2%}")
                    
                    # Color-coded category display
                    if cell_category == "RBC":
                        st.markdown("🔴 **Red Blood Cell**")
                    elif cell_category == "WBC":
                        st.markdown("⚪ **White Blood Cell**")
                    elif cell_category == "Platelet":
                        st.markdown("🟡 **Platelet**")
                    
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    st.exception(e)
            
            st.divider()
    
    # Instructions
    if not uploaded_files:
        st.info("👆 Upload blood smear images above to get started!")
        
        st.header("How to Use")
        st.markdown("""
        1. **Upload Images**: Use the file uploader above to select blood smear images
        2. **Automatic Classification**: The system will process each image and classify it
        3. **Results**: View the classification results including:
           - Main category (RBC, WBC, or Platelet)
           - Specific cell type
           - Confidence score
        """)
        
        st.header("Supported Image Formats")
        st.markdown("- PNG, JPG, JPEG")
        st.markdown("- Images should be clear blood smear microscopy images")
        st.markdown("- The system works best with well-focused, stained blood cells")


if __name__ == "__main__":
    main()
