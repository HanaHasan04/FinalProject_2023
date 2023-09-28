import os
import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

i_start = 1
i_end = 30

class_labels = ["Anticipation", "Baseline", "Disappointment", "Frustration"]

all_metadata_path = f'data/metadata'
new_metadata_path = f'data/new_metadata'
metrics_path = f'data/metrics'
os.makedirs(metrics_path, exist_ok=True)

print("begin")
# Concatenate metadata from all 30 horses
metadata_list = []
for j in range(1, 31):
    metadata_path = os.path.join(new_metadata_path, f"metadata_{j}.xlsx")
    metadata = pd.read_excel(metadata_path)
    metadata_list.append(metadata)
all_new_metadata = pd.concat(metadata_list, ignore_index=True)
metadata_list.clear()

print("start")

# leave-one-out cross-validation, subject-wise.
for i in range(i_start, i_end + 1):
    # train set
    train_metadata = all_new_metadata[all_new_metadata['Horse'] != i]
    X_train = train_metadata.iloc[:, 6:]  # all features
    y_train = train_metadata["Emotion"]

    # hold-out test set
    metadata_path = os.path.join(all_metadata_path, f"metadata_{i}.xlsx")
    metadata = pd.read_excel(metadata_path)
    X_test = metadata.iloc[:, 6:]  # all features
    y_test = metadata["Emotion"]

    print("Before SVM")
    s_time = time.time()

    # SVM classifier with class weights.
    class_weights = "balanced"  # assign class weights based on sample count.
    classifier = SVC(class_weight=class_weights, probability=True)
    # train SVM.
    classifier.fit(X_train, y_train)
    # make predictions on the hold-out test set.
    # y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    print("After SVM")
    print(f"Time for horse {i}: ", time.time() - s_time)

    # majority voting on video-level predictions.
    video_labels = {}

    for index, row in metadata.iterrows():
        video = row["Video"]
        label_test = row["Emotion"]
        label_pred = y_pred_proba[index]

        if video in video_labels:
            video_labels[video]["labels_test"].append(label_test)
            video_labels[video]["labels_pred"].append(label_pred)
        else:
            video_labels[video] = {
                "labels_test": [label_test],
                "labels_pred": [label_pred]
            }

    y_vid_test = []
    y_vid_pred = []

    for video, predictions in video_labels.items():
        labels_test = predictions["labels_test"]
        labels_pred = predictions["labels_pred"]

        sum_pred = [sum(items) for items in zip(*labels_pred)]  # sum our confidence levels for each emotion
        pred = sum_pred.index(max(sum_pred))  # pick index of most confident emotion
        pred_class = class_labels[pred]  # translate emotion number into label
        majority_vote_test = max(set(labels_test), key=labels_test.count)  # return the real emotion of the video
        y_vid_test.extend([majority_vote_test])
        y_vid_pred.extend([pred_class])

    y_vid_test = np.array(y_vid_test)
    y_vid_pred = np.array(y_vid_pred)

    # Evaluate video-level predictions.
    accuracy = accuracy_score(y_vid_test, y_vid_pred)
    precision = precision_score(y_vid_test, y_vid_pred, average='macro', zero_division=1)
    recall = recall_score(y_vid_test, y_vid_pred, average='macro')
    f1 = f1_score(y_vid_test, y_vid_pred, average='macro')
    confusion_mat = confusion_matrix(y_vid_test, y_vid_pred)

    # print the metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(confusion_mat)

    # save the metrics to an Excel file.
    metrics_dict = {
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
        "Confusion Matrix": [confusion_mat]
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_path_i = os.path.join(metrics_path, f"metrics_{i}.xlsx")
    metrics_df.to_excel(metrics_path_i, index=False)

    print(f"LOOCV iteration {i}: Files saved successfully.")