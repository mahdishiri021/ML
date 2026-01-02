
# 🚀 Data Analysis & Linear Regression Project

## 📌 Overview

This project is a **hands-on machine learning and data analysis notebook** developed for ** LineaRegression** of a Data Science course.  
It demonstrates how to transform raw tabular data into a **predictive model** using **Python, Pandas, and Scikit-Learn**.
Se
The notebook walks through the complete pipeline:

> Data → Cleaning → Exploration → Feature Selection → Train/Test Split → Linear Regression → Prediction → Visualization

---

## 🧠 Problem Statement

The goal of this project is to analyze a dataset, understand the relationship between variables, and build a **Linear Regression model** to predict a target variable based on input features.

Applications include:

* Business forecasting
* Price prediction
* Trend analysis
* Decision support systems

---

## 🛠 Technologies & Libraries

* **Python**
* **Pandas** – data manipulation
* **NumPy** – numerical operations
* **Matplotlib** – visualization
* **Scikit-Learn**
  * `train_test_split`
  * `LinearRegression`

---

## 📂 Repository Structure

```
 cardata-LineaRegression/
│
├──  LineaRegression.ipynb   # Main notebook (EDA + ML model)
├── README.md            # Project documentation
```

---

## 🔍 Project Workflow

### 1️⃣ Data Loading
Load the dataset from CSV into a Pandas DataFrame.

### 2️⃣ Exploratory Data Analysis (EDA)
* Inspect dataset structure
* Understand column values
* View statistical summaries
* Detect trends and relationships

### 3️⃣ Data Preparation
* Select input features (`X`)
* Select target variable (`y`)
* Convert data into ML-ready format

### 4️⃣ Train-Test Split
Split dataset into training and testing sets:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Ensures model evaluation on unseen data.

### 5️⃣ Linear Regression Model
Train a Linear Regression model:

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### 6️⃣ Prediction
Use the trained model to predict on test data and compare with actual values.

### 7️⃣ Visualization
Visualize:

* Data points
* Regression line
* Model fit quality using **Matplotlib**

---

## 📈 Results & Insights

The model learns a linear relationship between features and target, confirming:

* Linear Regression is suitable
* Data preprocessing was effective
* Dataset contains meaningful predictive patterns

---

## 🎯 What This Project Shows

✔ Data preprocessing  
✔ Exploratory Data Analysis  
✔ Machine Learning modeling  
✔ Prediction  
✔ Visualization  
✔ End-to-end ML workflow  

Perfect for **junior data scientists and ML students** to showcase practical skills.

---

## 🚀 How to Run

1. Install required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

2. Open the notebook:

```bash
jupyter notebook
```

3. Run:

```
LineaRegression.ipynb
```

---

## 👨‍💻 Author

**Mahdi Shiri**  
Data Science | Machine Learning | Python

---

## ⭐ Support

If you like this project, feel free to ⭐ star the repository or fork it for learning and experimentation.
