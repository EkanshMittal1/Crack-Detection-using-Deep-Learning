# Crack-Detection-using-Deep-Learning

# Surface Crack Detection using Deep Learning (MobileNetV2)

A computer vision project that detects surface cracks in images using Transfer Learning with MobileNetV2. The model classifies images as **Crack** or **Non-Crack** and is optimized for deployment on mobile and embedded devices via TFLite.

## Files
- `Project_EST_final.ipynb` — Main Jupyter notebook (full pipeline: preprocessing → training → evaluation → deployment)
- `Dataset.zip` — Dataset containing labeled surface images (`Cr_` prefix = crack, others = non-crack)

## Pipeline
1. **Data Loading & Extraction** — Loads images from Dataset.zip
2. **Image Preprocessing** — Resizes all images to 224×224, converts to RGB
3. **Labeling** — Auto-labels based on filename prefix (`Cr_` → Crack, others → Non-Crack)
4. **Model Training** — Transfer Learning using pretrained MobileNetV2 (two models compared)
5. **Evaluation** — Confusion matrix and classification report
6. **Grad-CAM Visualization** — Highlights regions the model focuses on for predictions
7. **TFLite Export** — Converts model to 3 formats for edge deployment:
   - `crack_detector_float32.tflite`
   - `crack_detector_quant.tflite` (Quantized)
   - `crack_detector_fp16.tflite` (FP16 GPU-friendly)

## Tech Stack
- Python, TensorFlow/Keras
- MobileNetV2 (Transfer Learning)
- OpenCV, PIL, NumPy, Matplotlib, Seaborn
- Scikit-learn (evaluation metrics)
- TensorFlow Lite (edge deployment)

## How to Run
1. Open `Project_EST_final.ipynb` in Google Colab
2. Upload `Dataset.zip` to `/content/`
3. Run all cells sequentially

## Output
- Trained model: `crack_detector_mobilenet.h5`
- Lightweight TFLite models ready for mobile/embedded deployment
