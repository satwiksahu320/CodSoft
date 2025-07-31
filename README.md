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

```markdown
# 📱 **Spam SMS Detection – CodSoft Internship Task 3**

This project is part of my **CodSoft Internship**, where I built an AI model to detect whether an SMS message is **Spam** or **Ham** (Legitimate).

---

## 🚀 Technologies Used

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn (TF-IDF, Naive Bayes, Logistic Regression, SVM)

---

## 📂 Dataset

- [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- 5,500+ SMS messages labeled as `spam` or `ham`

---

## 🧠 Workflow

1. **Data Preprocessing**
   - Lowercasing
   - Removing punctuation, digits, and whitespaces
   - Label encoding (`ham = 0`, `spam = 1`)

2. **Text Vectorization**
   - Used **TF-IDF Vectorizer** to convert text into numerical features

3. **Model Building**
   - ✅ Naive Bayes
   - ✅ Logistic Regression
   - ✅ Support Vector Machine (SVM)

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - Model comparison bar chart

---

## 📊 Results

All models performed with high accuracy. Here's a summary:

| Model               | Accuracy |
|---------------------|----------|
| Naive Bayes         | ~97.3%   |
| Logistic Regression | ~98.5%   |
| SVM                 | ~98.1%   |

---

## 📌 Future Enhancements

- Add web deployment using Streamlit or Flask
- Include real-time SMS prediction
- Save & load model using `joblib`

---







### **🙌 Acknowledgment**

This project is completed as part of my **Machine Learning Internship at CodSoft**.

