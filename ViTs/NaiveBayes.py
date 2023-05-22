import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

train_data = pd.read_excel('train.xlsx')
val_data = pd.read_excel('val.xlsx')
train_val_data = pd.concat([train_data, val_data])

test_data = pd.read_excel('test.xlsx')
X_train_val = train_val_data.iloc[:, 5:]  # Select all columns starting from the 6th column
y_train_val = train_val_data['Emotion']

# Naive Bayes classifier.
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_val, y_train_val)
X_test = test_data.iloc[:, 5:]      # Features.
test_data['pred'] = nb_classifier.predict(X_test)
test_data.to_excel('test_with_pred.xlsx', index=False)

# majority voting for each video.
video_predictions = test_data.groupby('Video')['pred'].agg(lambda x: x.value_counts().index[0]).reset_index()
video_predictions.columns = ['Video', 'video_pred']
video_predictions['Emotion'] = test_data.groupby('Video')['Emotion'].first().values
video_predictions.to_excel('video_predictions_with_metrics.xlsx', index=False)

# evaluation metrics.
accuracy = accuracy_score(video_predictions['Emotion'], video_predictions['video_pred'])
f1 = f1_score(video_predictions['Emotion'], video_predictions['video_pred'], average='weighted')
precision = precision_score(video_predictions['Emotion'], video_predictions['video_pred'], average='weighted')
recall = recall_score(video_predictions['Emotion'], video_predictions['video_pred'], average='weighted')
conf_matrix = confusion_matrix(video_predictions['Emotion'], video_predictions['video_pred'])

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:")

# plot the confusion matrix.
labels = np.unique(video_predictions[['Emotion', 'video_pred']])
cm = confusion_matrix(video_predictions['Emotion'], video_predictions['video_pred'], labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.show()
