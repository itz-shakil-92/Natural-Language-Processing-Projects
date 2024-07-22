# SMS Spam Classification Project

Welcome to the SMS Spam Classification Project repository! This project aims to classify SMS messages as spam or ham (not spam) using natural language processing techniques. The dataset used for this project is downloaded from the UCI Machine Learning Repository.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Author](#Author)
- [Contact](#contact)

## Overview

The SMS Spam Classification Project utilizes natural language processing (NLP) and machine learning techniques to classify SMS messages as spam or ham. The model is trained on the SMS Spam Collection dataset from the UCI Machine Learning Repository.

## Features

- Data preprocessing and cleaning
- Text vectorization using Bag of words
- Text vectorization using TF-IDF
- Classification using Naive Bayes machine learning models
- Evaluation of model performance

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/itz-shakil-92/Natural-Language-Processing-Projects.git
    cd Natural-Language-Processing-Projects/SMS Spam Classification Project
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv myenv

    #Unix/Macos
    source myenv/bin/activate 

    # On Windows 
    myenv\Scripts\activate

    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the ```sms_spam_classifier_project.py ``` to preprocess the data, train the model, and evaluate its performance with TF-IDF text processing.

2. Run the ```sms_spam_classifier_project_using_TF-IDF_text_preprocessing.py ```  to preprocess the data, train the model, and evaluate its performance with TF-IDF text processing.
   

## Dataset

The dataset used for this project is the SMS Spam Collection dataset from the UCI Machine Learning Repository. You can download the dataset from [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

The dataset contains one set of SMS messages in English, tagged according to being legitimate (ham) or spam.

## Model

The project employs various machine learning models for classification, including:

- Naive Bayes Classifier

Text data is vectorized using TF-IDF and Bag of words before being fed into the models.

## Results

The model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. 

Condusion Matrix for bag of words text processing:-

![Confusion Matrix](/SMS%20Spam%20Classification%20Project/screenshot/image1.png)

Accuracy Score:-
![Accuracy Score](/SMS%20Spam%20Classification%20Project/screenshot/image2.png)

Condusion Matrix for TF-IDF text processing:-

![Confusion Matrix](/SMS%20Spam%20Classification%20Project/screenshot/image3.png)

Accuracy Score:-
![Accuracy Score](/SMS%20Spam%20Classification%20Project/screenshot/image4.png)

## Author
- [Shakil Kathat](https://www.github.com/itz-shakil-92)

## Contact
If you have any questions or need further assistance, please feel free to contact us at shakilkathat5603@gmail.com
