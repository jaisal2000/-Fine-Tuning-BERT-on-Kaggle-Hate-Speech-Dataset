# Fine-Tuning BERT on Kaggle Hate Speech Dataset

This project demonstrates the process of fine-tuning the BERT model for hate speech detection on the Kaggle Twitter Hate Speech dataset. It utilizes the Hugging Face Transformers library and PyTorch to build, train, and evaluate a BERT-based model for binary text classification (Hate Speech vs. Non-Hate Speech).

## Project Overview

This project aims to train a BERT model to classify tweets into hate speech and non-hate speech categories. The workflow includes data preprocessing, model training, and evaluation using the Hugging Face `transformers` library and PyTorch. Additionally, a **Streamlit web app** is created to deploy the model and classify tweets in real time.

- **Dataset**: Kaggle Twitter Hate Speech Dataset (You can download it from Kaggle)
- **Model**: BERT (`bert-base-uncased`)
- **Framework**: Hugging Face Transformers, PyTorch
- **Metrics**: Accuracy, Precision, Recall, F1-Score

## Features

- Preprocessing of the Kaggle hate speech dataset to fine-tune BERT.
- BERT model fine-tuned on the hate speech dataset.
- Model evaluation on the test set after training.
- Real-time hate speech classification using a Streamlit web application.
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score.
