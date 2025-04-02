# TDA-based Shape Classification: Circle vs Square (with giotto-tda + Gradio demo)

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage.io import imread
from skimage.color import rgb2gray
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
import gradio as gr
import os

def load_image(image_path):
    """Load and preprocess image"""
    img = imread(image_path)
    if img.ndim == 3:
        img = rgb2gray(img)
    if img.max() > 1.0:
        img = img / 255.0
    return img

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

# Convert images to point clouds
point_clouds = []
for img in X_images:
    points = np.column_stack(np.where(img > 0.5))
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
VR = VietorisRipsPersistence(homology_dimensions=[0, 1])
diagrams = VR.fit_transform(X_pc_padded)
PE = PersistenceEntropy()
X_features = PE.fit_transform(diagrams)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.3, random_state=42)

# Train classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save models
joblib.dump(model, 'tda_shape_classifier.pkl')
joblib.dump(VR, 'tda_vr.pkl')
joblib.dump(PE, 'tda_entropy.pkl')

# Print classification report
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Circle', 'Square']))

# Visualize sample images
n_samples = min(5, len(X_images))
fig, axes = plt.subplots(1, n_samples, figsize=(2*n_samples, 2))
if n_samples == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    if i < len(X_images):
        img_normalized = (X_images[i] - X_images[i].min()) / (X_images[i].max() - X_images[i].min())
        ax.imshow(img_normalized, cmap='gray')
        label = 'Circle' if y_labels[i] == 0 else 'Square'
        ax.set_title(label)
        ax.axis('off')
plt.tight_layout()
plt.show()

def predict_image(image):
    if image.ndim == 3:
        image = rgb2gray(image)
    if image.max() > 1.0:
        image = image / 255.0
    points = np.column_stack(np.where(image > 0.5))
    max_len = VR.n_vertices_ if hasattr(VR, 'n_vertices_') else 500
    pad_len = max_len - len(points)
    if pad_len > 0:
        points = np.pad(points, ((0, pad_len), (0, 0)), mode='constant')
    else:
        points = points[:max_len]
    points = points.reshape(1, -1, 2)
    diag = VR.transform(points)
    feat = PE.transform(diag)
    pred = model.predict(feat)[0]
    return "Circle" if pred == 0 else "Square"

# Create Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload shape image"),
    outputs=gr.Label(label="Predicted Shape"),
    title="Shape Classifier with TDA",
    description="Upload a black and white image of a shape (circle or square) to classify."
)

demo.launch()
