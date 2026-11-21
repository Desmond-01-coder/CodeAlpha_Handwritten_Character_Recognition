# Handwritten Character Recognition Using CNN

A deep learning project that uses a **Convolutional Neural Network (CNN)** to identify handwritten digits from image data. Developed using Python and TensorFlow/Keras as part of the CodeAlpha Machine Learning Internship.

## 1. Project Overview

Handwritten digit recognition is an image classification problem in which a computer learns to identify numerical digits from handwritten images.

This project implements a CNN that processes grayscale images and predicts which digit (0–9) each image represents.

The model is trained and evaluated using the MNIST dataset.

## 2. Project Objectives

- Understand how deep learning can be applied to image recognition.
- Implement a Convolutional Neural Network using TensorFlow/Keras.
- Preprocess and normalize image data.
- Train a model to classify handwritten digits.
- Evaluate the model using test data and accuracy metrics.

## 3. Dataset

The project uses the **MNIST handwritten digit dataset**.

| Dataset | Number of Images |
|---|---:|
| Training set | 60,000 |
| Testing set | 10,000 |
| Image dimensions | 28 × 28 pixels |
| Image format | Grayscale |
| Classes | 10 (digits 0–9) |

Each image represents a handwritten numerical digit.

## 4. Technologies and Tools

- **Python** – Programming language
- **TensorFlow** – Deep learning framework
- **Keras** – Neural network implementation
- **NumPy** – Numerical operations
- **Google Colab** – Model training and execution environment

## 5. Methodology

The project follows these stages:

1. **Data Loading:** Load the MNIST dataset.
2. **Data Preprocessing:** Convert image values to floating-point numbers and normalize pixel values to the range 0–1.
3. **Reshaping:** Reshape images to include the grayscale channel dimension.
4. **Model Building:** Create a CNN using convolutional, pooling, and dense layers.
5. **Model Training:** Train the network using the Adam optimizer.
6. **Model Evaluation:** Evaluate performance using the test dataset.
7. **Model Saving:** Save the trained model in Keras format.

## 6. Model Architecture

The CNN consists of the following layers:

| Layer | Configuration |
|---|---|
| Input | 28 × 28 × 1 |
| Conv2D | 32 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 64 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Flatten | Flatten feature maps |
| Dense | 64 neurons, ReLU |
| Output | 10 neurons, Softmax |

**Total trainable parameters:** 121,930

## 7. Training Configuration

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Epochs: 5
- Batch Size: 64
- Validation Split: 10%

## 8. Model Performance

The model achieved the following results:

| Evaluation Metric | Result |
|---|---:|
| Training Accuracy | 99.28% |
| Validation Accuracy | 99.05% |
| Test Accuracy | 98.92% |
| Test Loss | 0.0341 |

The model correctly classified approximately 98.92% of the unseen test images.

## 9. Project Structure

```text
CodeAlpha_Handwritten_Character_Recognition/
│
├── main.py
├── handwritten_character_recognition.keras
└── README.md
```

## 10. Installation and Execution

Install the required dependencies in a compatible Python environment:

```bash
pip install tensorflow numpy
```

Run the training script:

```bash
python main.py
```

The script loads the MNIST dataset, builds and trains the CNN, evaluates its performance, and saves the trained model as:

`handwritten_character_recognition.keras`

**Execution Environment:** Model training and evaluation were performed using Google Colab.

## 11. Future Improvements

Possible extensions to this project include:

- Extending recognition beyond numerical digits to handwritten alphabets using the EMNIST dataset.
- Building a graphical user interface for drawing and recognizing digits.
- Improving model performance through hyperparameter tuning.
- Deploying the model as a web-based application.

## 12. Acknowledgements

- CodeAlpha Machine Learning Internship
- TensorFlow/Keras documentation
- MNIST handwritten digit dataset

---

**Project Category:** Machine Learning / Deep Learning / Computer Vision

**Task:** Handwritten Character Recognition

**Developed with:** Python and TensorFlow/Keras
