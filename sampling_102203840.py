import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/Creditcard_data.csv')
df.head()

class_0 = len(df["Class"][df.Class == 0])
class_1 = len(df["Class"][df.Class == 1])

arr = np.array([class_0, class_1])
labels = ['class_0', 'class_1']

print("Total No. Of 0 Cases :- ", class_0)
print("Total No. Of 1 Cases :- ", class_1)

plt.pie(arr, labels=labels, explode=[0.2, 0.0], shadow=True)
plt.show()

from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random
import math

# Step 2: Balance the Class Data
majority_class = df[df['Class'] == 0]
minority_class = df[df['Class'] == 1]

# Using oversampling to balance the dataset
minority_upsampled = resample(minority_class,
                               replace=True,
                               n_samples=len(majority_class),
                               random_state=42)

balanced_df = pd.concat([majority_class, minority_upsampled])

# Step 3: Calculate the Sample Size
Z = 1.96
p = 0.5
E = 0.05

sample_size = math.ceil((Z**2 * p * (1 - p)) / (E**2))
print(f"Calculated Sample Size: {sample_size}")

# Step 4: Create Five Samples
def create_samples(data, n_samples, sample_size):
    samples = []
    for _ in range(n_samples):
        samples.append(data.sample(n=sample_size, random_state=random.randint(1, 100)))
    return samples

samples = create_samples(balanced_df, 5, sample_size)

# Step 5: Define Sampling Techniques
def apply_sampling_techniques(data, technique):
    if technique == "Sampling1":  # Random Sampling
        return data.sample(frac=0.5, random_state=42)
    elif technique == "Sampling2":  # Stratified Sampling
        return data.groupby('Class').apply(lambda x: x.sample(frac=0.5)).reset_index(drop=True)
    elif technique == "Sampling3":  # Cross-Validation Sampling
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(data):
            return data.iloc[train_index]
    elif technique == "Sampling4":  # Systematic Sampling
        return data.iloc[::2, :]
    elif technique == "Sampling5":  # Bootstrap Sampling
        return resample(data, replace=True, n_samples=int(len(data) * 0.5), random_state=42)
    else:
        raise ValueError("Unknown sampling technique!")

# Step 6: Training ML Models and Evaluate Accuracy
models = {
    "M1": LogisticRegression(max_iter=1000),
    "M2": DecisionTreeClassifier(),
    "M3": RandomForestClassifier(),
    "M4": KNeighborsClassifier(),
    "M5": SVC()
}

results = []

for i, sample in enumerate(samples):
    for technique in ["Sampling1", "Sampling2", "Sampling3", "Sampling4", "Sampling5"]:
        # Applying sampling technique
        sampled_data = apply_sampling_techniques(sample, technique)

        X = sampled_data.drop('Class', axis=1)
        y = sampled_data['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        for model_name, model in models.items():

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)


            results.append((f"Sample {i+1}", technique, model_name, acc))

# Step 7: Saving Results
results_df = pd.DataFrame(results, columns=['Sample', 'Sampling Technique', 'Model', 'Accuracy'])