## **Movie Genre Classification – CodSoft Internship Task 1**

### **📌 Project Overview**

This project is part of my **Machine Learning Internship at CodSoft**.
The goal is to build a **Movie Genre Classification model** that predicts the genre of a movie based on its plot summary.

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** for feature extraction and experiments with **three machine learning classifiers**:

* **Multinomial Naive Bayes**
* **Logistic Regression**
* **Linear SVM (LinearSVC)**


### **🛠 Tech Stack & Libraries**

* **Anaconda Navigator**
* **Jupyter Notebook**
* **Pandas, NumPy**
* **Scikit-learn**
* **Matplotlib, Seaborn**


### **🚀 Steps Performed**

1. **Data Loading** – Read and clean the dataset (`train_data.txt`).
2. **Text Preprocessing** – Removed null values, extra spaces, and tokenized text.
3. **Feature Extraction** – Used TF-IDF vectorization on the movie plots.
4. **Model Building** – Trained and compared three classifiers:

   * Naive Bayes
   * Logistic Regression
   * Linear SVM (fast and accurate for text classification)
5. **Evaluation** – Calculated accuracy, generated classification reports, and plotted a confusion matrix.


### **📂 Repository Contents**

* **`Task1.ipynb`** 
* **`README.md`** 

---

### **⚡ How to Run**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/movie-genre-classification.git
   ```
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   (or manually install `pandas`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`)
3. **Open Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```
4. **Run `Task1.ipynb` cell by cell.**

---





## 💳 **Credit Card Fraud Detection – CodSoft Internship Task 2**

This project uses machine learning techniques to classify credit card transactions as **fraudulent** or **legitimate**. We explore three different algorithms — Logistic Regression, Decision Tree, and Random Forest — and compare their performance.

## 📁 Dataset

The dataset used for this project contains real transaction data, including features like transaction amount, time, and anonymized numerical values. It has a **high class imbalance**, where fraudulent transactions are rare.

> ⚠️ Note: Due to confidentiality, transaction IDs and other sensitive columns were removed during preprocessing.

## 🧠 Models Used

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**

## 📊 Evaluation Metrics

Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

Given the class imbalance, **recall** for the fraud class (label `1`) is especially important.

## ⚙️ Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn` (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, train_test_split, classification_report, confusion_matrix)

## 📌 Project Highlights

- Handled imbalanced dataset using performance metrics like recall.
- Compared multiple classification models.
- Random Forest performed best overall due to its ensemble nature.

## 🧪 How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/credit-card-fraud-detection.git
    ```
2. Install dependencies (preferably in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```
3. Open the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

## 📸 Sample Output

| Metric        | Logistic Regression | Decision Tree | Random Forest |
|---------------|---------------------|----------------|----------------|
| Accuracy      | 99%                | 99%            | 99%            |
| Recall (Fraud)| 30-35%             | 50-60%         | 66-70%         |

## 📂 Folder Structure

````

├── CreditCardFraudDetection.ipynb
├── dataset.csv
├── README.md

```

## 🔐 Note

This notebook is built purely for educational purposes. Real-world fraud detection requires more sophisticated models, real-time features, and compliance with ethical and legal constraints.


```





---



---

````markdown
## 📱 **Spam SMS Detection – CodSoft Internship Task 3**

This project is part of my **CodSoft Internship**, where I built an AI model to detect whether an SMS message is **Spam** or **Ham** (legitimate). I explored multiple machine learning algorithms to compare their performance on textual classification.

## 📁 Dataset

The dataset used for this project is the [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), containing **5,500+ SMS messages** labeled as `spam` or `ham`.

> ⚠️ Note: Basic preprocessing was applied including lowercasing, punctuation and stopword removal.

## 🧠 Models Used

- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

## 📊 Evaluation Metrics

Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

Due to the binary classification nature, **F1-score and Recall** for spam detection are especially significant.

## ⚙️ Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn` (TfidfVectorizer, train_test_split, MultinomialNB, LogisticRegression, SVC, classification_report, confusion_matrix)

## 📌 Project Highlights

- Preprocessed text data using TF-IDF Vectorization.
- Compared multiple machine learning classifiers.
- Evaluated on multiple metrics to select the most effective model.
- SVM achieved the best overall balance between precision and recall.

## 🧪 How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/spam-sms-detection.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the notebook or script:
    ```bash
    jupyter notebook SpamSMSDetection.ipynb
    ```

## 📸 Sample Output

| Metric        | Naive Bayes | Logistic Regression | SVM         |
|---------------|-------------|----------------------|-------------|
| Accuracy      | 96–97%      | 97–98%               | 98%         |
| Recall (Spam) | 90–94%      | 92–95%               | 96–98%      |

## 📂 Folder Structure

````

├── SpamSMSDetection.ipynb
├── spam.csv
├── README.md

```

## 🔐 Note

This project is a demonstration of basic Natural Language Processing techniques for spam classification. Real-world deployment would require further optimization, pipeline integration, and attention to data privacy.

```

---









### **🙌 Acknowledgment**

This project is completed as part of my **Machine Learning Internship at CodSoft**.

