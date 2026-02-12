# Adversarial Attack on MNIST (FGSM)

## 📌 Project Overview

This project demonstrates a practical implementation of the **Fast Gradient Sign Method (FGSM)** adversarial attack on a Convolutional Neural Network (CNN) trained on the MNIST dataset.

The goal is to slightly perturb an input image in a direction that maximizes model loss, causing the model to misclassify the image — even though it still looks unchanged to a human.

---

## 🎯 Objective

- Train a CNN model on MNIST
- Generate adversarial noise using FGSM
- Create adversarial examples
- Demonstrate successful misclassification

---

## 🧠 Key Concept

FGSM formula:

x_adv = x + ε * sign(∇x J(x, y))

Where:
- x = original image
- ε = perturbation strength
- ∇x J(x, y) = gradient of loss w.r.t input
- sign() = direction of maximum increase

---

## 🏗 Project Structure

adversarial-mnist/
│
├── train_model.py
├── attack.py
├── requirements.txt
├── README.md
│
├── models/
│ └── mnist_victim_model.keras
│
└── screenshots/
├── original.png
├── adversarial.png
└── perturbation.png


---

## 🚀 How to Run

### 1️⃣ Install Dependencies

pip install tensorflow numpy matplotlib


Or use:

pip install -r requirements.txt


---

### 2️⃣ Train the Model

python train_model.py


This creates:

models/mnist_victim_model.keras


---

### 3️⃣ Run the Attack

python attack.py


This generates:

- original.png
- adversarial.png
- perturbation.png

---

## 📊 Results

### Original Image
Model correctly predicts digit 9.

![Original](screenshots/original.png)

---

### Adversarial Image
The same image with minimal perturbation.
Model misclassifies digit 9 as digit 4.

![Adversarial](screenshots/adversarial.png)

---

### Perturbation Pattern
Noise added using gradient direction.

![Perturbation](screenshots/perturbation.png)

---

## 🔍 Observations

- The adversarial image still visually appears as digit 9.
- The CNN confidently predicts it as digit 4.
- This demonstrates model vulnerability to gradient-based attacks.

---

## 🛡 Real-World Implication

Such attacks highlight vulnerabilities in:
- Autonomous driving systems
- Facial recognition
- Medical AI diagnosis
- Security systems

Defenses include:
- Adversarial training
- Defensive distillation
- Input preprocessing techniques

---

## 🧩 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- MNIST Dataset

---

## 📌 Author

Aman Lodha  
Integrated M.Tech CSE  
Focus: Systems, Security, and Applied AI
