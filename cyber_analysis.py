import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

print("🚀 Starting Full Analysis and Visualization Script...")

# --- 1. Load Data ---
try:
    df = pd.read_csv('Cybersecurity Intrusion Detection.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'Cybersecurity Intrusion Detection.csv' not found.")
    print("Please make sure the CSV file is in the same directory as this script.")
    exit()

# --- 2. Exploratory Data Analysis (EDA) ---
print("\n📊 Starting Exploratory Data Analysis...")

# Create a directory for EDA plots
output_dir = 'eda_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"📁 Created directory '{output_dir}' to save EDA graphs.")

# Define feature types for EDA
numerical_features_eda = ['network_packet_size', 'login_attempts', 'session_duration', 'ip_reputation_score', 'failed_logins']
categorical_features_eda = ['protocol_type', 'encryption_used', 'browser_type', 'unusual_time_access']
target_eda = 'attack_detected'

# EDA Plot 1: Target Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=target_eda, data=df, palette='viridis', hue=target_eda, legend=False)
plt.title('Distribution of Attack Detected', fontsize=16)
plt.xlabel('Attack Detected (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig(os.path.join(output_dir, '1_target_distribution.png'))
plt.close()

# EDA Plot 2: Numerical Histograms
for feature in numerical_features_eda:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True, color='teal')
    plt.title(f'Distribution of {feature.replace("_", " ").title()}', fontsize=16)
    plt.xlabel(feature.replace("_", " ").title(), fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'2_hist_{feature}.png'))
    plt.close()

# EDA Plot 3: Categorical Bar Charts
for feature in categorical_features_eda:
    plt.figure(figsize=(12, 7))
    sns.countplot(y=feature, data=df, order=df[feature].value_counts().index, palette='magma', hue=feature, legend=False)
    plt.title(f'Distribution of {feature.replace("_", " ").title()}', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(feature.replace("_", " ").title(), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'3_countplot_{feature}.png'))
    plt.close()

# EDA Plot 4: Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_features_eda].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_correlation_heatmap.png'))
plt.close()

print(f"✅ EDA graphs have been generated and saved in the '{output_dir}' folder.")

# --- 3. Prepare Data for Modeling ---
print("\n⚙️ Preparing data for machine learning models...")
X = df.drop(['session_id', 'attack_detected'], axis=1, errors='ignore')
y = df['attack_detected']

# Define feature types for preprocessing
categorical_features_model = ['protocol_type', 'encryption_used', 'browser_type']
numerical_features_model = ['network_packet_size', 'login_attempts', 'session_duration', 'ip_reputation_score', 'failed_logins', 'unusual_time_access']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features_model),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_model)])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("✅ Data has been split into training and testing sets.")

# --- 4. Train and Evaluate Machine Learning Models ---
print("\n🤖 Training and evaluating models...")

# Model 1: Logistic Regression
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(random_state=42, max_iter=1000))])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

# Model 2: K-Nearest Neighbors
knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', KNeighborsClassifier())])
knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_pipeline.predict(X_test)
y_prob_knn = knn_pipeline.predict_proba(X_test)[:, 1]

# Model 3: Random Forest
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

# --- 5. Display Model Performance ---
print("\n--- Logistic Regression Evaluation ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_lr):.4f}")

print("\n--- K-Nearest Neighbors Evaluation ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_knn):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_knn):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_knn):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_knn):.4f}")

print("\n--- Random Forest Evaluation ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_rf):.4f}")

# --- 6. Generate and Save Model Comparison Plots ---
print("\n📈 Generating model comparison plots...")
# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle('Model Confusion Matrices', fontsize=20)
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', ax=axes[0]).set_title('Logistic Regression', fontsize=16)
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Oranges', ax=axes[1]).set_title('K-Nearest Neighbors', fontsize=16)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=axes[2]).set_title('Random Forest', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('model_confusion_matrices.png')
plt.close()

# ROC Curves
plt.figure(figsize=(12, 8))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr):.4f})')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_score(y_test, y_prob_knn):.4f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
plt.legend()
plt.grid()
plt.savefig('model_roc_curves.png')
plt.close()

print("✅ Model comparison plots saved.")

# --- 7. Save the Best Model ---
model_filename = 'random_forest_model.joblib'
joblib.dump(rf_pipeline, model_filename)
print(f"\n💾 Best performing model (Random Forest) saved as '{model_filename}'.")

print("\n✨ All tasks complete!")