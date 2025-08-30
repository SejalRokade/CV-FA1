# Single-Image Blood Smear Cell Morphology Features

End-to-end classic CV pipeline for BloodMNIST: preprocessing, segmentation, feature extraction (Hu moments, eccentricity, SIFT/ORB), and SVM/RandomForest classification.

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

Download will be handled automatically by `medmnist` on first run.

**Training:**
```bash
python -m src.train_eval --feature sift --clf auto --augment
```

**Options:**
- `--feature {sift,orb,none}`: local descriptors aggregated with Bag-of-Visual-Words; `none` uses only shape features.
- `--clf {auto,svm,rf,gb,mlp}`: classifier type. `auto` tries all and picks the best.
- `--val-split 0.1`: fraction from train set for validation.
- `--augment`: use data augmentation for better generalization.
- `--model-out blood_cell_model.pkl`: save trained model to file.

### 2. Web Interface (Streamlit)

Launch the interactive web app for image classification directly:

```bash
streamlit run src/streamlit_app.py
```

**Features:**
- Upload multiple blood smear images
- Real-time classification as RBC, WBC, or Platelet
- Confidence scores and detailed cell type information
- Visual results with color-coded categories

## Project Structure

```
src/
  data.py           # BloodMNIST loading, splits
  preprocess.py     # color conversion, denoise
  segment.py        # Otsu + morphology
  features.py       # Enhanced shape, texture, SIFT/ORB + BoVW
  train_eval.py     # CLI: extract features, train, evaluate
  streamlit_app.py  # Web interface for image classification
run_streamlit.py    # Launcher script for Streamlit app
test_improved.py    # Test script for improved classifier
```

## Workflow

1. **Train Model**: Use `test_improved.py` for quick start or `train_eval.py` for manual control
2. **Save Model**: Model is automatically saved as `blood_cell_model.pkl`
3. **Launch Web App**: Use `run_streamlit.py` to start the interface
4. **Upload Images**: Drag & drop blood smear images for classification
5. **View Results**: Get instant classification with confidence scores

## Improvements Made

- **Enhanced Features**: Added texture features (GLCM, LBP, Gabor filters)
- **Better Classifiers**: SVM, Random Forest, Gradient Boosting, Neural Network
- **Hyperparameter Tuning**: Automatic optimization using GridSearchCV
- **Data Augmentation**: Rotation, flip, brightness/contrast adjustments
- **Feature Normalization**: Robust scaling for better generalization
- **Auto-selection**: Automatically picks the best performing classifier

## Notes
- SIFT requires `opencv-contrib-python`.
- If SIFT is unavailable, the code falls back to ORB.
- This project focuses on classical CV features aligned with syllabus topics.
- The web interface automatically loads the trained model if available.
