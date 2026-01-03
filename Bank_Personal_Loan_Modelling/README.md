# 🏦 Bank Marketing Data Science Project

## 📌 Project Overview

This project is an **end-to-end Data Science and Machine Learning analysis** on a bank marketing dataset.  
The main objective is to **predict whether a bank customer will subscribe to a financial product** based on their personal and campaign-related information.

The project demonstrates the full data science workflow:
- Data understanding  
- Data preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training & evaluation  

---

## 🎯 Problem Statement

Banks invest significant resources in marketing campaigns.  
The goal of this project is to build a **predictive model** that helps the bank:

- Identify customers who are more likely to accept the offered product  
- Reduce marketing costs and improve conversion rates  

---

## 📂 Dataset Description

The dataset contains information about bank customers and marketing campaigns, including:

- **Customer information:** age, job, marital status, education  
- **Financial data:** balance, loans, housing  
- **Campaign data:** contact type, duration, previous outcomes  

**Target variable:**  
- `y` → whether the customer subscribed (`yes` / `no`)

---

## 🛠️ Technologies Used

- **Python**  
- **Pandas** – data manipulation  
- **NumPy** – numerical operations  
- **Matplotlib & Seaborn** – data visualization  
- **Scikit-learn** – machine learning models  

---

## 🔍 Exploratory Data Analysis (EDA)

Key analysis steps include:
- Distribution of customer age and balance  
- Analysis of job, marital status, and education  
- Visualization of subscription rates  
- Correlation analysis between features  

EDA helps uncover important patterns and customer behaviors.

---

## ⚙️ Data Preprocessing

- Handling categorical variables using **One-Hot Encoding**  
- Feature selection  
- Splitting data into training and testing sets  

```python
from sklearn.model_selection import train_test_split
```

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  

Each model was evaluated using:
- Accuracy Score  
- Confusion Matrix  
- Classification Report  

---

## 📊 Model Evaluation

The models were compared to identify the best-performing approach.  
**Random Forest** and **Logistic Regression** achieved the most reliable results, making them suitable for real-world banking applications.

---

## ✅ Results & Insights

- Customer age, balance, and campaign duration have strong influence  
- Certain job categories show higher subscription rates  
- Ensemble models improve prediction accuracy  

---

## 🚀 How to Run the Project

Clone the repository:
```bash
git clone https://github.com/your-username/bank-marketing-project.git
```

Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Open the notebook:
```bash
jupyter notebook Bank_proj.ipynb
```

---

## 📌 Project Level

📈 **Junior to Mid-Level Data Science Project**

Suitable for:
- Academic presentations  
- Portfolio & GitHub showcase  
- Entry-level Data Scientist roles  

---

## ✍️ Author

**Mahdi Shiri**  
Data Science & Machine Learning Enthusiast
