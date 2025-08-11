# Image-Recognition
This project demonstrates the end-to-end process of building, training, and evaluating Convolutional Neural Networks (CNNs) for various image classification tasks. It covers both training models from scratch and using transfer learning to achieve better performance.

#📌 **Key Features**
* **Dataset Preparation**
    *Loading and preprocessing MNIST, CIFAR-10, and Cats vs. Dogs datasets.

* **Model Development**
   *Building CNN architectures from scratch.
   *Implementing data augmentation to reduce overfitting and improve generalization.
   *Applying transfer learning with a pre-trained MobileNetV2 model for Cats vs. Dogs classification.

* **Model Evaluation**
   *Accuracy score.
   *Confusion matrix.
   *Classification report.
   *ROC curve.
* **Model Management**
   *Saving and loading trained models for future use.
* **Prediction on New Data**
   *Making predictions on unseen images.

**🎯 Purpose**
This repository serves as a practical example of applying deep learning techniques to solve real-world image recognition problems, showcasing best practices in dataset handling, model training, and performance evaluation.
Copy
Edit
from tensorflow.keras.models import load_model
model = load_model('model_name.h5')
