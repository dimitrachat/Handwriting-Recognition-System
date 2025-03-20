# Handwriting-Recognition-System

## Overview
This project aims to create a handwriting recognition system that converts handwritten characters and digits into machine-readable text. It uses traditional computer vision techniques and deep learning models, including a Convolutional Neural Network (CNN), to achieve accurate recognition.

## Introduction
Handwriting recognition is a challenging task due to the variability in writing styles. This project uses a dataset of handwritten names and applies computer vision and deep learning techniques to recognize and convert handwritten text into digital form. The system includes preprocessing, character segmentation, and classification using a CNN.

## Dataset
The dataset consists of over 200,000 images of handwritten names, divided into training, testing, and validation sets. Each image is mapped to a transcribed name in CSV files. The dataset is sourced from Kaggle.

## Object Recognition with Pytesseract
The project uses the Pytesseract library for Optical Character Recognition (OCR) to extract text from images. OpenCV is used for image preprocessing, including grayscale conversion, Gaussian blur, and morphological transformations.

## Preprocessing
The preprocessing steps include:
- Grayscale conversion.
- Noise reduction using Gaussian blur.
- Morphological transformations to remove small artifacts.
- Otsu’s thresholding for foreground-background separation.
- Character segmentation using bounding box coordinates.

## Character Classification with CNN
A Convolutional Neural Network (CNN) is designed to classify segmented characters. The model includes:
- Convolutional layers for feature extraction.
- Max pooling layers to reduce spatial dimensions.
- Dropout layers to prevent overfitting.
- Fully connected layers for classification.
- Softmax activation for multi-class classification.

## Evaluation
The model achieves an accuracy of 92% on the test set. The evaluation metrics include:
- Accuracy: 0.92
- Loss: 0.35
- Precision and recall for word-level predictions.

## Future Work
- Improve the object detection model for better bounding box predictions.
- Explore advanced deep learning architectures like Transformers.
- Increase the dataset size for better generalization.

## References
- [Build a Handwritten Text Recognition System using TensorFlow](https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5)
- [Machine Learning for Beginners: An Introduction to Neural Networks](https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9)

- ## Requirements
- Python 3.x  
- OpenCV  
- Pytesseract  
- TensorFlow/Keras  
- NumPy, Pandas  
- (Other dependencies are listed in the `requirements.txt` file)
