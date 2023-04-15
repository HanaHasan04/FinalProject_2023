import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import ast


# File paths
train_file = r"C:\Users\USER\Documents\UniversityProjects\PythonProjects\FinalProject\loocv_splits\S1\train.xlsx"
val_file = r"C:\Users\USER\Documents\UniversityProjects\PythonProjects\FinalProject\loocv_splits\S1\val.xlsx"
test_file = r"C:\Users\USER\Documents\UniversityProjects\PythonProjects\FinalProject\loocv_splits\S1\test.xlsx"

# Load train and val files
train_df = pd.read_excel(train_file)
val_df = pd.read_excel(val_file)

# Concatenate train and val dataframes
df = pd.concat([train_df, val_df])


# Extract features (vectors) and labels from the data
X = df['Vector'].tolist()
y = df['LABEL'].tolist()


# Create a Gaussian Naive Bayes classifier
clf = GaussianNB()

# Train the classifier
clf.fit(X, y)

# Load test file
test_df = pd.read_excel('test.xlsx')

# Group images by video
grouped = test_df['Image Name'].str.split('__', expand=True)
test_df['Video'] = grouped[0]
grouped_df = test_df.groupby('Video')

# Iterate over groups and perform majority vote
test_pred = []
for _, group in grouped_df:
    # If group has only one image, directly predict and append to test_pred
    if len(group) == 1:
        pred = clf.predict(group['Vector'].tolist())[0]
        test_pred.append(pred)
    else:
        # Perform majority vote for images in the same video
        video_pred = group['Vector'].tolist()
        majority_vote = max(set(video_pred), key=video_pred.count)
        test_pred.append(majority_vote)

# Convert test predictions to a numpy array
test_pred = pd.Series(test_pred).values

# Calculate metrics
accuracy = accuracy_score(test_df['LABEL'], test_pred)
recall = recall_score(test_df['LABEL'], test_pred, average='weighted')
precision = precision_score(test_df['LABEL'], test_pred, average='weighted')
f1 = f1_score(test_df['LABEL'], test_pred, average='weighted')

# Print metrics
print('Accuracy:', accuracy)
print('Recall:', recall)
print('Precision:', precision)
print('F1 Score:', f1)
