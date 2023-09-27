import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class_labels = ["Anticipation", "Baseline", "Disappointment", "Frustration"]

i_start = 1
i_end = 30

k = 100
k_b = 34

all_metadata_path = f'data/metadata_grayST'

metadata_sampling_grayST_k_path = f'data/metadata_grayST_k100'
os.makedirs(metadata_sampling_grayST_k_path, exist_ok=True)

metrics_sampling_grayST_path = 'data/metrics_grayST'
os.makedirs(metrics_sampling_grayST_path, exist_ok=True)

metrics_sampling_grayST_k_path = 'data/metrics_grayST_k100'
os.makedirs(metrics_sampling_grayST_k_path, exist_ok=True)

result_excels_path = 'data/result_excels'
os.makedirs(result_excels_path, exist_ok=True)

save_directory_path = 'data/histograms'
os.makedirs(save_directory_path, exist_ok=True)

def excels_metadata_top_k(new_metadata_path, metadata, k, k_b, i, y_pred_proba):
    # majority voting on video-level predictions.
    video_labels = {}
    top_k_0 = [(-1, 0) for _ in range(k)]
    top_k_1 = [(-1, 0) for _ in range(k_b)]
    top_k_2 = [(-1, 0) for _ in range(k)]
    top_k_3 = [(-1, 0) for _ in range(k)]

    for index, row in metadata.iterrows():
        video = row["Video"]
        label_test = row["Emotion"]
        label_pred = y_pred_proba[index]

        # update to save the best 20 frames
        if label_pred[0] > top_k_0[k - 1][0] and label_test == class_labels[0]:
            top_k_0[k - 1] = (label_pred[0], index)
            top_k_0 = sorted(top_k_0, reverse=True)

        if label_pred[1] > top_k_1[k_b - 1][0] and label_test == class_labels[1]:
            top_k_1[k_b - 1] = (label_pred[1], index)
            top_k_1 = sorted(top_k_1, reverse=True)

        if label_pred[2] > top_k_2[k - 1][0] and label_test == class_labels[2]:
            top_k_2[k - 1] = (label_pred[2], index)
            top_k_2 = sorted(top_k_2, reverse=True)

        if label_pred[3] > top_k_3[k - 1][0] and label_test == class_labels[3]:
            top_k_3[k - 1] = (label_pred[3], index)
            top_k_3 = sorted(top_k_3, reverse=True)

        if video in video_labels:
            video_labels[video]["labels_test"].append(label_test)
            video_labels[video]["labels_pred"].append(label_pred)
        else:
            video_labels[video] = {
                "labels_test": [label_test],
                "labels_pred": [label_pred]
            }

    unique_indexes = []
    indices = [top_k_0[i][1] for i in range(len(top_k_0))] + \
              [top_k_1[i][1] for i in range(len(top_k_1))] + \
              [top_k_2[i][1] for i in range(len(top_k_2))] + \
              [top_k_3[i][1] for i in range(len(top_k_3))]
    unique_indexes = list(set(indices))
    unique_indexes = [index for index in unique_indexes if index != 0]

    new_metadata_df = metadata.iloc[unique_indexes]
    mew_metadata_path_i = os.path.join(new_metadata_path, f"metadata_{i}.xlsx")
    new_metadata_df.to_excel(mew_metadata_path_i, index=False)
    return video_labels

def predict_on_4_predictions(video_labels):
    y_vid_test = []
    y_vid_pred = []

    for video, predictions in video_labels.items():
        labels_test = predictions["labels_test"]
        labels_pred = predictions["labels_pred"]

        one_preds = []
        for i, pred in enumerate(labels_pred):
            one_preds.append(class_labels[max(range(len(pred)), key=pred.__getitem__)])

        majority_vote_pred = max(set(one_preds), key=one_preds.count)
        majority_vote_test = max(set(labels_test), key=labels_test.count)

        y_vid_pred.extend([majority_vote_pred])
        y_vid_test.extend([majority_vote_test])


    y_vid_test = np.array(y_vid_test)
    y_vid_pred = np.array(y_vid_pred)
    return y_vid_test, y_vid_pred

def save_metrics_in_excels(y_vid_test, y_vid_pred, metrics_path):
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
    return

def concatenate_metadata(all_metadata_path):
    metadata_list = []
    for j in range(1, 31):
        print(j)
        metadata_path = os.path.join(all_metadata_path, f"metadata_{j}.xlsx")
        metadata = pd.read_excel(metadata_path)
        metadata_list.append(metadata)
    all_metadata = pd.concat(metadata_list, ignore_index=True)
    metadata_list.clear()
    return all_metadata

def train_set(all_metadata):
    train_metadata = all_metadata[all_metadata['Horse'] != i]
    X_train = train_metadata.iloc[:, 6:]  # all features
    y_train = train_metadata["Emotion"]
    return X_train, y_train


# Concatenate metadata from all 30 horses
all_metadata = concatenate_metadata(all_metadata_path)

# leave-one-out cross-validation, subject-wise.
for i in range(i_start, i_end + 1):
    # train set
    X_train, y_train = train_set(all_metadata)

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
    y_pred_proba = classifier.predict_proba(X_test)
    print("After SVM")
    print(f"Time for horse {i}: ", time.time() - s_time)

    video_labels = excels_metadata_top_k(metadata_sampling_grayST_k_path, metadata, k, k_b, i, y_pred_proba)

    y_vid_test, y_vid_pred = predict_on_4_predictions(video_labels)
    save_metrics_in_excels(y_vid_test, y_vid_pred, metrics_sampling_grayST_path)

    print(f"LOOCV iteration {i}: Files saved successfully.")




################################# retrain
print("-----------------------------------------------")
print("RE-TRAIN ON TOP K-S")
print("-----------------------------------------------")

def train_classifier_with_4_predictions(X_train, y_train):
    classifier = SVC(class_weight=class_weights, probability=True)
    classifier.fit(X_train, y_train)
    y_pred_proba = classifier.predict_proba(X_test)
    return y_pred_proba

def most_common_element_and_count(labels_pred):
    # Count occurrences of each prediction in labels_pred
    counts = Counter(labels_pred)
    # Find the most common element and its count
    most_common_element, count = counts.most_common(1)[0]
    return most_common_element, count

# Concatenate metadata from all 30 horses (for each k)
print("Concatenate metadata for k")
all_metadata_k = concatenate_metadata(metadata_sampling_grayST_k_path)

# leave-one-out cross-validation, subject-wise.
for i in range(i_start, i_end + 1):

    # FOR K:
    # train set
    X_train_k, y_train_k = train_set(all_metadata_k)

    # hold-out test set
    metadata_path = os.path.join(all_metadata_path, f"metadata_{i}.xlsx")
    metadata = pd.read_excel(metadata_path)
    X_test = metadata.iloc[:, 6:]  # all features
    y_test = metadata["Emotion"]

    print("Before SVM")
    s_time = time.time()
    # SVM classifier with class weights
    class_weights = "balanced"  # assign class weights based on sample count

    # FOR K:
    y_pred_proba_k = train_classifier_with_4_predictions(X_train_k, y_train_k)

    print("After SVM")
    print(f"Time for horse {i}: ", time.time() - s_time)

    # majority voting on video-level predictions
    video_labels = {}

    for index, row in metadata.iterrows():
        video = row["Video"]
        label_test = row["Emotion"]
        label_pred = y_pred_proba_k[index]

        if video in video_labels:
            video_labels[video]["labels_test"].append(label_test)
            video_labels[video]["labels_pred_k"].append(label_pred)
        else:
            video_labels[video] = {
                "labels_test": [label_test],
                "labels_pred_k": [label_pred]
            }

    y_vid_test = []
    y_vid_pred_k = []

    for video, predictions in video_labels.items():
        labels_test = predictions["labels_test"]
        labels_pred_k = predictions["labels_pred_k"]

        one_preds = []
        for l, pred in enumerate(labels_pred_k):
            one_preds.append(class_labels[max(range(len(pred)), key=pred.__getitem__)])

        majority_vote_pred_k = max(set(one_preds), key=one_preds.count)
        majority_vote_test = max(set(labels_test), key=labels_test.count)

        y_vid_pred_k.extend([majority_vote_pred_k])
        y_vid_test.extend([majority_vote_test])

    y_vid_test = np.array(y_vid_test)
    y_vid_pred_k = np.array(y_vid_pred_k)

    save_metrics_in_excels(y_vid_test, y_vid_pred_k, metrics_sampling_grayST_k_path)

    save_directory_horse_path = os.path.join(save_directory_path, f"S{i}")
    os.makedirs(save_directory_horse_path, exist_ok=True)

    print(f"LOOCV iteration {i}: Files saved successfully.")