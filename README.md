HEAD
# Stroke-prediction
Predicting strokes using ensemble models
# Stroke Prediction


## Introduction
Stroke is one of the leading causes of death, disability and dementia worldwide.1 The annual global mortality rate for stroke is 84.7 deaths per 100,000 people.2 According to estimates from the global burden of disease and injuries study, the global prevalence of stroke was 93.8 million and its incidence was 11.9 million in the year 2021 alone.3 People who have stroke experience a wide range of disabilities, including paralysis, sudden falls, speech difficulties, cognitive impairment and dementia.

This project applies machine learning techniques to identify individuals at high risk of stroke based on key health indicators. The model is trained on structured health dataset from kaggle.

##  Data Understanding and Exploring 


The dataset consists of **5,110 records** and **12 columns**. It contains information about individuals that can be used to predict the likelihood of experiencing a stroke. Below is a description of each column:

Below is a summary of the number of unique values and descriptions for each column:

| Column Name          | Unique Values | Description |
|----------------------|----------------|-------------|
| `id`                 | 5,110          | Unique identifier for each individual. |
| `gender`             | 3              | Gender of the individual: Male, Female and Other. |
| `age`                | 104            | Age of the individual (in years). |
| `hypertension`       | 2              | Whether the individual has hypertension (1 = Yes, 0 = No). |
| `heart_disease`      | 2              | Whether the individual has heart disease (1 = Yes, 0 = No). |
| `ever_married`       | 2              | Marital status of the individual (Yes/No). |
| `work_type`          | 5              | Type of employment: 'Private', 'Self-employed', 'Govt_job', 'children' and 'Never_worked'. |
| `Residence_type`     | 2              | The person lives in an Urban or Rural area. |
| `avg_glucose_level`  | 3,979          | Average glucose level in blood. |
| `bmi`                | 418            | Body Mass Index. |
| `smoking_status`     | 4              | Smoking behavior: formerly smoked, never smoked, smokes and Unknown. |
| `stroke`             | 2              | Target variable: 1 = stroke occurred, 0 = no stroke. |


## Data Cleaning

Before modeling, it is essential to clean the dataset to ensure its quality. This dataset contains information about individuals that can be used to predict the likelihood of experiencing a stroke. 


### Missing Values (Before Cleaning)

The following table shows the number of missing values in each column before cleaning:

| Column Name         | Missing Values |
|---------------------|----------------|
| `id`                | 0              |
| `gender`            | 0              |
| `age`               | 0              |
| `hypertension`      | 0              |
| `heart_disease`     | 0              |
| `ever_married`      | 0              |
| `work_type`         | 0              |
| `Residence_type`    | 0              |
| `avg_glucose_level` | 0              |
| `bmi`               | 201            |
| `smoking_status`    | 0              |
| `stroke`            | 0              |

### Handling Missing Values

- The `bmi` column contained 201 missing values.
- These were filled using **mean imputation**:
  ```python
  stroke['bmi'].fillna(stroke['bmi'].mean(), inplace=True)
| Column Name         | Missing Values |
|---------------------|----------------|
| `id`                | 0              |
| `gender`            | 0              |
| `age`               | 0              |
| `hypertension`      | 0              |
| `heart_disease`     | 0              |
| `ever_married`      | 0              |
| `work_type`         | 0              |
| `Residence_type`    | 0              |
| `avg_glucose_level` | 0              |
| `bmi`               | 0            |
| `smoking_status`    | 0              |
| `stroke`            | 0              |

In the dataset, the inconsistent values in the gender column were replaced with "female" to maintain data consistency.

The age column was transformed into categorical age groups to better capture age-related patterns in the data. The age groups were defined as follows:

0 for children (age ≤ 18)

1 for youth (18 < age < 60)

2 for elderly (age ≥ 60)
## Data Analysis
### the correlation between numeric features
![alt text](image-1.png)
### the relationship between age and glucose level
![alt text](image.png)
### Age distribution
![alt text](image-2.png)
### Gender distributiony
![alt text](image-3.png)
### patients had a stroke compared to those who didn’t have it 
![alt text](image-4.png)
### BMI distributed between stroke and non-stroke patients
![alt text](image-5.png)
### Is heart disease a strong indicator of stroke risk?
![alt text](image-6.png)
### Married people more likely to have a stroke than unmarried people
![alt text](image-7.png)
### the stroke rate among people who have both hypertension and heart disease
![alt text](image-8.png)

## Modeling
To evaluate the performance of different classification models, the dataset was split into training and testing sets using an 80/20 ratio. This means that 80% of the data was used to train the models, while 20% was reserved for evaluating their performance on unseen data.
### Logistic Regression
Logistic regression is a linear model used for binary and multi-class classification. It estimates the probability that an instance belongs to a particular class using a logistic (sigmoid) function. It works well when the data is linearly separable and offers the benefit of interpretability.
### Random Forest Classifier
Random Forest is an ensemble learning method that builds multiple decision trees and combines their outputs to improve accuracy and prevent overfitting. Each tree is trained on a random subset of the data and features, which adds diversity and robustness.

## Performance Evaluation
After training both models, their performance was evaluated using a confusion matrix. The confusion matrix shows how many predictions were correctly and incorrectly classified for each class.
### Logistic Regression
![alt text](image-9.png)
### Random Forest Classifier
![alt text](image-10.png)


During model development, I faced the challenge of misclassification, primarily due to class imbalance—stroke cases were significantly underrepresented compared to non-stroke cases. To address this, I applied SMOTE (Synthetic Minority Oversampling Technique) to balance the training data and used feature standardization to ensure all input variables contributed equally to the model. The dataset was split into training and testing sets with stratification to preserve class distribution. I selected XGBoost for its ability to handle imbalanced data and used cross-validation to evaluate its stability and performance using F1-score.

To further improve performance, I optimized the classification threshold based on the Precision-Recall curve, selecting a lower threshold to prioritize recall—critical in identifying stroke risk. After training the model and adjusting the threshold, I evaluated it using accuracy, precision, recall, F1 score, and ROC AUC. Finally, I visualized the confusion matrix to understand the types of errors the model was making, ensuring a more balanced and sensitive prediction, especially for the minority (stroke) class.

### XGBOOST Perfomance
![alt text](image-11.png)
