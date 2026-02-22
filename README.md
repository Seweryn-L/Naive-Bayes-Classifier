# Multilingual Text Classification using Naive Bayes

## Project Overview
This project focuses on building a Natural Language Processing (NLP) pipeline to classify text messages. It features a custom implementation of the **Multinomial Naive Bayes** algorithm and compares it with the standard `scikit-learn` implementation. The project also handles multilingual data, including language detection and specialized preprocessing.

## Key Features

### 1. Advanced Text Preprocessing
To improve model accuracy, a robust cleaning pipeline was implemented:
* **Language Detection**: Uses `langdetect` to identify the language of each record.
* **Multilingual Stopwords**: Dynamically removes stopwords based on the detected language (English, Spanish, Polish, French, etc.) using `nltk`.
* **Lemmatization & Cleaning**: Includes noise removal (punctuation, special characters) and word normalization using `WordNetLemmatizer`.

### 2. Custom Naive Bayes Implementation
Beyond using standard libraries, the project includes a manual implementation of the **Multinomial Naive Bayes** logic:
* **Log-Likelihood Calculation**: Uses log-probabilities to prevent numerical underflow.
* **Laplace Smoothing**: Implemented to handle the "Zero Probability" problem for words not present in the training set.
* **Step-by-Step Logic**: The notebook provides a detailed mathematical breakdown of how scores are calculated for specific classes (e.g., "Mandrill" vs. other classes).

### 3. Model Training & Evaluation
* **Vectorization**: Uses `CountVectorizer` to convert text data into numerical Bag-of-Words representations.
* **Evaluation Metrics**: Performance is measured using:
    * **Accuracy Score**
    * **Confusion Matrix**: To visualize misclassifications.
    * **Classification Report**: Detailed Precision, Recall, and F1-Score for each category.

## Technologies Used
* **Python** (Jupyter Notebook)
* **NLTK**: For tokenization, stopwords, and lemmatization.
* **Scikit-learn**: For `MultinomialNB`, `train_test_split`, and evaluation metrics.
* **Pandas & NumPy**: For data structures and mathematical operations.
* **Langdetect**: For automatic language identification.
* **Matplotlib & Seaborn**: For visualizing the confusion matrix and data distribution.

## How to Run
1.  Install the required dependencies: `pip install pandas nltk scikit-learn langdetect seaborn`.
2.  Download NLTK resources:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    ```
3.  Ensure your dataset is correctly linked in the notebook.
4.  Execute `zadanie3.ipynb` to see the preprocessing steps, the custom model logic, and the final evaluation.

## Results
The model demonstrates high effectiveness in text classification tasks, with the custom implementation providing results consistent with the `scikit-learn` library, proving the correctness of the underlying mathematical logic.
