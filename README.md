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

### **🙌 Acknowledgment**

This project is completed as part of my **Machine Learning Internship at CodSoft**.

