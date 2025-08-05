

# Cybersecurity Intrusion Detection with Machine Learning 🛡️

This project demonstrates the use of machine learning to detect network intrusions from traffic data. It implements, compares, and evaluates three different classification algorithms to identify malicious activity.

The primary goal is to provide a clear, end-to-end workflow from data preprocessing to model training, evaluation, and saving the best-performing model.

## ✨ Features

  - **Comprehensive EDA:** Generates and saves a full suite of plots for Exploratory Data Analysis.
  - **Three ML Models:** Implements and compares:
    1.  **Logistic Regression** (as a baseline)
    2.  **K-Nearest Neighbors (KNN)**
    3.  **Random Forest**
  - **Robust Evaluation:** Uses a wide range of metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) to assess performance.
  - **Clear Visualizations:** Creates and saves confusion matrices and ROC curves to visually compare model performance.
  - **Model Persistence:** Automatically saves the best-performing model (`Random Forest`) to a file using `joblib` for future use.

-----

## 🚀 How to Get Started

Follow these steps to run the analysis on your local machine.

### **1. Prerequisites**

Make sure you have Python 3.7+ installed on your system.

### **2. Clone the Repository**

Open your terminal or command prompt and clone this repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

*(Replace `your-username` and `your-repository-name` with your actual GitHub details.)*

### **3. Install Dependencies**

This project uses several Python libraries. You can install all of them at once using the following command:

```bash
pip install pandas scikit-learn matplotlib seaborn joblib
```

### **4. Run the Analysis Script**

Execute the main Python script from your terminal. The script will handle everything from EDA to model training and saving the results.

```bash
python full_cyber_analysis.py
```

-----

## 📂 Project Structure

The repository is organized as follows:

```
├── 📄 Cybersecurity Intrusion Detection.csv    # The raw dataset used for training and testing.
├── 🐍 cyber_analysis.py                   # The main Python script with the complete workflow.
├── 📁 eda_plots/                                # (Generated) Contains all plots from the EDA.
├── 🖼️ model_confusion_matrices.png             # (Generated) Visual comparison of model confusion matrices.
├── 🖼️ model_roc_curves.png                     # (Generated) Visual comparison of model ROC curves.
└── 📦 random_forest_model.joblib               # (Generated) The final, trained Random Forest model pipeline.
```

-----

## 📊 Output

After running the script, you will find:

  - **Console Output:** The terminal will display the performance metrics (Accuracy, Precision, etc.) for each of the three models.
  - **EDA Plots:** A new folder named `eda_plots` will be created, containing detailed visualizations of the dataset's features.
  - **Model Comparison Plots:** Two image files (`model_confusion_matrices.png` and `model_roc_curves.png`) will be saved in the root directory, showing a side-by-side comparison of the models.
  - **Saved Model:** The fully trained `Random Forest` pipeline will be saved as `random_forest_model.joblib`, ready to be loaded and used for future predictions.
