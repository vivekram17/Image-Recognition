# Image-Recognition
This project demonstrates building and evaluating Convolutional Neural Networks (CNNs) for various image classification tasks. It covers dataset preprocessing, model training, performance evaluation, and prediction on new images.

📌 Features
Load and preprocess datasets — MNIST, CIFAR-10, and Cats vs. Dogs.

Build CNN models from scratch for image classification.

Apply data augmentation to improve performance and reduce overfitting.

Use transfer learning with a pre-trained MobileNetV2 model for Cats vs. Dogs classification.

Evaluate models with:

Accuracy

Confusion Matrix

Classification Report

ROC Curve

Save and load trained models for reuse.

Make predictions on new images.

📂 Datasets
MNIST — Handwritten digits dataset.

CIFAR-10 — 10-class natural image dataset.

Cats vs. Dogs — Binary classification dataset.

⚙️ Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/cnn-image-classification.git
cd cnn-image-classification
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
▶️ Usage
Run the notebook or Python scripts for training and evaluation:

bash
Copy
Edit
python train_cnn.py
To make predictions on new images:

bash
Copy
Edit
python predict.py --image path/to/image.jpg
📊 Evaluation
Models are evaluated using:

Accuracy — Overall prediction correctness.

Confusion Matrix — True/false positives/negatives.

Classification Report — Precision, recall, F1-score.

ROC Curve — Model discrimination ability.

💡 Transfer Learning
For the Cats vs. Dogs dataset, a MobileNetV2 model pre-trained on ImageNet is fine-tuned for better performance with fewer training epochs.

💾 Model Saving and Loading
Save trained models:

python
Copy
Edit
model.save('model_name.h5')
Load trained models:

python
Copy
Edit
from tensorflow.keras.models import load_model
model = load_model('model_name.h5')
