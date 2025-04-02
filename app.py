# TDA-based Shape Classification: Circle vs Square (with giotto-tda + Gradio demo)

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
import gradio as gr
import os
import pandas as pd

def preprocess_image(img):
    """Preprocess image with consistent steps"""
    # Convert to grayscale if needed
    if img.ndim == 3:
        img = rgb2gray(img)
    
    # Normalize to [0,1]
    if img.max() > 1.0:
        img = img / 255.0
    
    # Apply Otsu's thresholding
    thresh = threshold_otsu(img)
    binary = img > thresh
    
    return binary.astype(float)

def extract_shape_features(img):
    """Extract shape features including contour points"""
    # Find contours
    contours = find_contours(img, 0.5)
    
    if not contours:
        return np.array([[0, 0]])
    
    # Use the longest contour
    contour = max(contours, key=len)
    
    # Sample points from contour if too many
    if len(contour) > 100:
        indices = np.linspace(0, len(contour)-1, 100).astype(int)
        contour = contour[indices]
    
    return contour

def load_image(image_path):
    """Load and preprocess image"""
    img = imread(image_path)
    return preprocess_image(img)

def load_images_from_directory(directory):
    """Load all images from a directory"""
    images = []
    valid_extensions = ['.png', '.jpg', '.jpeg']
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            img_path = os.path.join(directory, filename)
            try:
                img = load_image(img_path)
                images.append(img)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return images

# Load images from directories
circle_images = load_images_from_directory('dataset/circle')
square_images = load_images_from_directory('dataset/square')

# Create training data
X_images = circle_images + square_images
y_labels = [0] * len(circle_images) + [1] * len(square_images)

print(f"Loaded {len(circle_images)} circle images and {len(square_images)} square images")

# Convert images to point clouds using contours
point_clouds = []
for img in X_images:
    points = extract_shape_features(img)
    point_clouds.append(points)

# Pad point clouds to the same shape
max_points = max(len(pc) for pc in point_clouds)
X_pc_padded = np.zeros((len(point_clouds), max_points, 2))
for i, pc in enumerate(point_clouds):
    pad_len = max_points - len(pc)
    if pad_len > 0:
        pc = np.pad(pc, ((0, pad_len), (0, 0)), mode='constant')
    X_pc_padded[i] = pc

# Apply Vietoris-Rips and extract persistence entropy
VR = VietorisRipsPersistence(
    homology_dimensions=[0, 1], 
    n_jobs=-1,
    metric='euclidean'
)
diagrams = VR.fit_transform(X_pc_padded)
PE = PersistenceEntropy()
X_features = PE.fit_transform(diagrams)

# Add shape-specific features
shape_features = []
for img in X_images:
    # Calculate aspect ratio and area
    points = np.column_stack(np.where(img > 0.5))
    if len(points) > 0:
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        width = max_coords[1] - min_coords[1]
        height = max_coords[0] - min_coords[0]
        aspect_ratio = width / (height + 1e-6)
        area = len(points)
    else:
        aspect_ratio = 1.0
        area = 0
    
    shape_features.append([aspect_ratio, area])

# Combine TDA features with shape features
X_combined = np.hstack([X_features, shape_features])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_labels, test_size=0.3, random_state=42)

# Train classifier with more trees and balanced class weights
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Save models
joblib.dump(model, 'tda_shape_classifier.pkl')
joblib.dump(VR, 'tda_vr.pkl')
joblib.dump(PE, 'tda_entropy.pkl')

# Print classification report
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Circle', 'Square']))

# Plot confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig = plt.figure(figsize=(8, 6))
plt.imshow(conf_mat, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ['Circle', 'Square'])
plt.yticks([0, 1], ['Circle', 'Square'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_mat[i, j], ha='center', va='center')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.close(fig)  # Close the figure after creating it

def predict_image(image):
    """Predict shape from input image"""
    # Preprocess image
    processed_img = preprocess_image(image)
    
    # Create debug visualization
    fig = plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Input Image')
    
    plt.subplot(132)
    plt.imshow(processed_img, cmap='gray')
    plt.title('Processed Image')
    
    # Extract features
    contour = extract_shape_features(processed_img)
    
    plt.subplot(133)
    plt.imshow(processed_img, cmap='gray')
    plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
    plt.title('Detected Contour')
    
    plt.tight_layout()
    plt.close(fig)  # Close the figure after creating it
    
    # Pad contour points
    max_len = VR.n_vertices_ if hasattr(VR, 'n_vertices_') else 100
    pad_len = max_len - len(contour)
    if pad_len > 0:
        points = np.pad(contour, ((0, pad_len), (0, 0)), mode='constant')
    else:
        points = contour[:max_len]
    
    # Extract TDA features
    points = points.reshape(1, -1, 2)
    diag = VR.transform(points)
    tda_feat = PE.transform(diag)
    
    # Extract shape features
    points = np.column_stack(np.where(processed_img > 0.5))
    if len(points) > 0:
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        width = max_coords[1] - min_coords[1]
        height = max_coords[0] - min_coords[0]
        aspect_ratio = width / (height + 1e-6)
        area = len(points)
    else:
        aspect_ratio = 1.0
        area = 0
    
    # Combine features
    feat = np.hstack([tda_feat, [[aspect_ratio, area]]])
    
    # Get prediction and confidence
    pred_proba = model.predict_proba(feat)[0]
    pred_class = np.argmax(pred_proba)
    confidence = pred_proba[pred_class]
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': ['TDA1', 'TDA2', 'Aspect Ratio', 'Area'],
        'Importance': model.feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('Importance', ascending=False))
    
    result = "Circle" if pred_class == 0 else "Square"
    return f"{result} (Confidence: {confidence:.2f})"

# Create Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload shape image"),
    outputs=gr.Textbox(label="Predicted Shape"),
    title="Shape Classifier with TDA",
    description="Upload a black and white image of a shape (circle or square) to classify.",
    examples=[
        ["dataset/circle/circle_000.png"],
        ["dataset/square/square_000.png"]
    ]
)

demo.launch()
