# Sentiment Classification Models Comparison

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Models](#models)
6. [Data](#data)
7. [License](#license)

## Introduction
This project aims to compare three different models for sentiment classification, identifying emotions such as "sadness", "joy", "fear", and "anger". The project includes three separate Jupyter notebooks for training and testing each model: an LSTM model, a Fully Neural Network (FNN) model, and a Transformer model based on BERT.

## Project Structure
The project is organized as follows:
 sentiment-classification
┣ 📂data
┃ ┗ 📜 sentiment_data.csv
┣ 📂models
┃ ┗ 📜 (trained models to be added here)
┣ 📜 sentiment_classification_LSTM.ipynb
┣ 📜 sentiment_classification_FNN.ipynb
┣ 📜 sentiment_classification_Transformer.ipynb
┣ 📜 main.py
┣ 📜 requirements.txt
┗ 📜 README.md


## Installation
To set up the environment and install the necessary dependencies, follow these steps:

1. **Install Anaconda**: Download and install Anaconda from [the official website](https://www.anaconda.com/products/distribution).

2. **Clone the repository**:
   ```sh
   git clone https://github.com/MarcBarreto/sentiment-classification.git
   cd sentiment-classification

Claro! Aqui está todo o conteúdo no formato Markdown para você copiar e colar diretamente no seu arquivo README.md:

markdown
Copiar código
# Sentiment Classification Models Comparison

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Models](#models)
6. [Data](#data)
7. [License](#license)

## Introduction
This project aims to compare three different models for sentiment classification, identifying emotions such as "sadness", "joy", "fear", and "anger". The project includes three separate Jupyter notebooks for training and testing each model: an LSTM model, a Fully Neural Network (FNN) model, and a Transformer model based on BERT.

## Project Structure
The project is organized as follows:
📦 sentiment-classification
┣ 📂data
┃ ┗ 📜 sentiment_data.csv
┣ 📂models
┃ ┗ 📜 (trained models to be added here)
┣ 📜 sentiment_classification_LSTM.ipynb
┣ 📜 sentiment_classification_FNN.ipynb
┣ 📜 sentiment_classification_Transformer.ipynb
┣ 📜 main.py
┣ 📜 requirements.txt
┗ 📜 README.md

## Installation
To set up the environment and install the necessary dependencies, follow these steps:

1. **Install Anaconda**: Download and install Anaconda from [the official website](https://www.anaconda.com/products/distribution).

2. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/sentiment-classification.git
   cd sentiment-classification
   ```
4. **Create a virtual environment**:
   ```sh
   conda create --name sentiment_env python=3.10
   conda activate sentiment_env ```
6. **Install dependencies**:
   pip install -r requirements.txt
7. **Add trained models: Place the trained models into the models directory.**

## Usage
To deploy the models and analyze a sentence, follow these steps:

1. **Ensure the trained models are in the models directory.**
2. **Run the main.py script with the sentence to be analyzed**
   ```sh
   python main.py --sentence "Your sentence here" ```
## Models

# LSTM Model
- Notebook: sentiment_classification_LSTM.ipynb
- Description: This notebook includes the creation, training, and testing of an LSTM model for sentiment classification.
# FNN Model
- Notebook: sentiment_classification_FNN.ipynb
- Description: This notebook covers the creation, training, and testing of a Fully Neural Network model for sentiment classification.
# Transformer Model
- Notebook: sentiment_classification_Transformer.ipynb
- Description: This notebook demonstrates the creation, training, and testing of a Transformer model based on BERT from Hugging Face's TensorFlow implementation.

## Data
  The dataset used for training and testing the models is located in the data directory under the file sentiment_data.csv.

##License
  MIT License
   
