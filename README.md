# README: Sampling Assignment Project

## Project Overview

This project focuses on applying different sampling techniques to a credit card dataset to determine the most effective sampling technique for achieving high accuracy across various machine learning models. Below is the workflow and methodology used to complete this project.

---

## Steps Followed

### 1. **Dataset**

- The dataset used for this project is `Creditcard_data.csv`, sourced from the following GitHub link: [Creditcard\_data.csv](https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv).

---

### 2. **Balancing the Dataset**

- The dataset was found to have class imbalances, which were addressed using technique such as:
  - **Oversampling**: Duplicating or generating synthetic samples for the minority class using methods like SMOTE (Synthetic Minority Oversampling Technique).

---

### 3. **Creating Samples**

- Five samples were created from the balanced dataset.
- The sample size was determined using a predefined formula discussed during class.

---

### 4. **Sampling Techniques**

Five different sampling techniques were applied to these samples:

1. Sampling1:- Simple Random Sampling
2. Sampling2:- Stratified Sampling
3. Sampling3:- Cross-Validation Sampling
4. **Sampling4:- Systematic Sampling**
5. Sampling5:- Bootstrap Sampling

---

### 5. **Machine Learning Models**

Five machine learning models were used to evaluate the sampling techniques:

1. **M1:- LogisticRegression**
2. **M2:**- DecisionTreeClassifier
3. **M3:-** RandomForestClassifier
4. M4:- KNeighborsClassifier
5. M5:- SupportVectorClassifier 

Each model was trained on datasets created by the respective sampling techniques.

### 6. **Results and Discussion**

- **Key Observations:**

  - Sampling5 performed well on M1 and M4, while Sampling3 showed the highest accuracy on M3.
  - Sampling1 and Sampling2 yielded consistent results across multiple models.
  - Sampling 4 yielded the highest accuracy with M3 out of all the combinations this yielded the highest accuracy 

- **Conclusion:**

  - The effectiveness of sampling techniques depends significantly on the underlying data and the model's characteristics. Further optimization might be necessary for specific use cases.

