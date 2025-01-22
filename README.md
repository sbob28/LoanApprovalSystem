# Loan Dataset Analysis and Classification

This project analyzes a loan dataset, cleans and preprocesses the data, visualizes key features, and applies machine learning models to classify loan applicants. Below is a summary of the steps and methods used in the project.

---

### Dataset Overview
The dataset contains information about loan applicants, including demographic details, income, loan amounts, and credit history. Key features include:
- **Gender**
- **Marital Status**
- **Dependents**
- **Applicant and Co-applicant Income**
- **Loan Amount**
- **Credit History**

---

### Data Cleaning and Preprocessing
1. **Handling Missing Values**:
   - Categorical columns (e.g., Gender, Married, Self_Employed) were filled with the mode.
   - Numerical columns (e.g., LoanAmount) were filled with the mean.

2. **Feature Encoding**:
   - Label encoding was applied to transform categorical features into numerical values for model compatibility.

3. **Standardization**:
   - Numerical features were standardized using `StandardScaler` to improve model performance.

---

### Feature Engineering
- **Log Transformations**:
  - Created `loanAmount_log` for the natural logarithm of `LoanAmount` to reduce skewness.
  - Created `TotalIncome` as the sum of `ApplicantIncome` and `CoapplicantIncome`, and its log-transformed version `TotalIncome_log`.

---

### Exploratory Data Analysis (EDA)
Visualizations and statistical summaries provided insights into:
- Gender and loan approval rates.
- Marital status and loan application trends.
- Dependents and their correlation with loan amounts.
- Self-employment and its impact on loan approvals.
- Credit history distribution among applicants.

EDA revealed key trends, such as the significant role of credit history in loan approvals and the dominance of male applicants.

---

### Data Splitting
The dataset was split into training and testing sets:
- **Training Set**: 80%
- **Testing Set**: 20%
- `random_state=0` was used to ensure reproducibility.

---

### Model Training and Testing
Several machine learning models were trained and evaluated:
1. **Random Forest Classifier**
2. **Gaussian Naive Bayes**
3. **Decision Tree Classifier**

#### Steps:
1. Models were trained on the preprocessed training set.
2. Predictions were made on the test set.
3. Accuracy was calculated using `metrics.accuracy_score`.

---

### Results and Accuracy
- **Random Forest Classifier**: Achieved the highest accuracy among models.
- **Gaussian Naive Bayes**: Performed moderately well.
- **Decision Tree Classifier**: Showed lower accuracy, indicating potential overfitting or lack of generalization.

---

### Tools and Libraries
The following tools and libraries were used:
- **Python Libraries**:
  - `numpy`, `pandas`: Data manipulation and analysis.
  - `matplotlib`, `seaborn`: Visualization.
  - `sklearn`: Machine learning models and preprocessing.

---

### How to Run the Code
1. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
2. Load the dataset (`LoanDataset.csv`).
3. Follow the steps in the code to preprocess, visualize, and train models.
4. Evaluate results and accuracy metrics.

---

### Future Improvements
1. **Feature Selection**:
   - Analyze feature importance to improve model performance.
2. **Hyperparameter Tuning**:
   - Optimize model parameters using GridSearchCV.
3. **Model Diversification**:
   - Experiment with other classifiers like Support Vector Machines (SVM) or Gradient Boosting.

For any questions or feedback, please feel free to reach out!

