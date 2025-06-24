📊 Deep Learning-Based Multi-Class Classification of Diabetic Retinopathy Using Fundus Images
👩‍💻 Author
Jeeva Jose C

📝 Project Overview
This project applies a deep learning approach using Convolutional Neural Networks (CNNs) to classify diabetic retinopathy stages based on fundus images. Automated detection can assist healthcare professionals in early diagnosis and grading, potentially reducing the risk of vision loss.

The model classifies images into four categories:

No DR signs

Severe NPDR

Very Severe NPDR

Advanced PDR

📂 Project Workflow
1️⃣ Data Preparation
Images resized to 256x256 pixels and normalized.

Labels assigned based on directory structure.

One-hot encoding applied for multi-class classification.

Dataset shuffled and split into training, validation, and test sets with stratification.

2️⃣ Model Architecture
Custom CNN model inspired by AlexNet.

Two convolutional layers with max pooling.

Two dense (fully connected) layers.

Dropout layers to prevent overfitting.

Softmax output layer for 4-class prediction.

3️⃣ Training
Adam optimizer with categorical cross-entropy loss.

Data augmentation techniques used:

Random rotations (±20°)

Horizontal/vertical shifts (±10%)

Zoom (±10%)

Horizontal flip

Custom evaluation metrics implemented:

Accuracy

Precision

Recall

Dice Coefficient

Intersection over Union (IoU)

Trained for 100 epochs, batch size of 32.

Best model saved based on highest validation accuracy.

4️⃣ Evaluation
Evaluated on training, validation, and test sets.

Performance tracked using multiple metrics.

Loss curves plotted to monitor model behavior.

📊 Results
Dataset	Accuracy	Loss	Dice	IoU	Precision	Recall
Training	82.26%	0.4938	0.7419	0.5920	87.56%	78.21%
Validation	77.59%	0.9650	0.7035	0.5630	81.13%	74.14%
Test	74.58%	0.6332	0.7032	0.5435	77.78%	71.19%


🛠 Tools and Libraries
Python 3.x

TensorFlow / Keras

NumPy

OpenCV

PIL

scikit-learn

Matplotlib

🚀 Future Work
Use larger and more diverse datasets.

Experiment with advanced architectures (e.g., transfer learning models like ResNet, EfficientNet).

Perform hyperparameter tuning.

Apply explainable AI (XAI) methods for clinical interpretability.

📧 Contact
For any queries or collaborations:

Jeeva Jose C
Email: jeevajosec@gmail.com

