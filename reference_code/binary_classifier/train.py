import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def concatenate_metadata(all_metadata_path):
    metadata_list = []
    for i in range(1, 31):
        metadata_path = os.path.join(all_metadata_path, f"metadata_{i}.xlsx")
        metadata = pd.read_excel(metadata_path)
        metadata_list.append(metadata)
    all_metadata = pd.concat(metadata_list, ignore_index=True)
    metadata_list.clear()
    return all_metadata


def train_set(all_metadata, i):
    train_metadata = all_metadata[all_metadata['Horse'] != i]
    X_train = train_metadata.iloc[:, 6:]  # all features
    y_train = train_metadata["Emotion"]
    return X_train, y_train


def test_set(metadata_dir, i):
    metadata_path = os.path.join(metadata_dir, f"metadata_{i}.xlsx")
    metadata = pd.read_excel(metadata_path)
    X_test = metadata.iloc[:, 6:]  # all features
    y_test = metadata["Emotion"]
    return X_test, y_test, metadata


def train_the_model(metadata_dir, new_metadata_dir, metrics_dir, new_metrics_dir, class_labels, k, i_start, i_end, retrain=False):
    # Concatenate metadata from all 30 horses
    if retrain:
        all_metadata = concatenate_metadata(new_metadata_dir)
    else:
        all_metadata = concatenate_metadata(metadata_dir)

    # leave-one-out cross-validation, subject-wise
    for i in range(i_start, i_end + 1):
        # train set
        X_train, y_train = train_set(all_metadata, i)

        # hold-out test set
        X_test, y_test, metadata = test_set(metadata_dir, i)

        # SVM classifier with class weights
        class_weights = "balanced"  # assign class weights based on sample count
        classifier = SVC(class_weight=class_weights, probability=True)

        # train SVM
        classifier.fit(X_train, y_train)

        # make predictions on the hold-out test set
        y_pred_proba = classifier.predict_proba(X_test)

        video_labels = {}
        if not retrain:
            top_k_0 = [(-1, 0) for _ in range(k)]
            top_k_1 = [(-1, 0) for _ in range(k)]

        for index, row in metadata.iterrows():
            video = row["Video"]
            label_test = row["Emotion"]
            label_pred = y_pred_proba[index]

            if not retrain:
                # save the top k frames
                if label_pred[0] > top_k_0[k - 1][0] and label_test == class_labels[0]:
                    top_k_0[k - 1] = (label_pred[0], index)
                    top_k_0 = sorted(top_k_0, reverse=True)

                if label_pred[1] > top_k_1[k - 1][0] and label_test == class_labels[1]:
                    top_k_1[k - 1] = (label_pred[1], index)
                    top_k_1 = sorted(top_k_1, reverse=True)


            if video in video_labels:
                video_labels[video]["labels_test"].append(label_test)
                video_labels[video]["labels_pred"].append(label_pred)
            else:
                video_labels[video] = {
                    "labels_test": [label_test],
                    "labels_pred": [label_pred]
                }

        if not retrain:
            indices = [top_k_0[i][1] for i in range(len(top_k_0))] + \
                      [top_k_1[i][1] for i in range(len(top_k_1))]
            unique_indexes = list(set(indices))
            unique_indexes = [index for index in unique_indexes if index != 0]

            new_metadata_df = metadata.iloc[unique_indexes]
            mew_metadata_path_i = os.path.join(new_metadata_dir, f"metadata_{i}.xlsx")
            new_metadata_df.to_excel(mew_metadata_path_i, index=False)

        y_vid_test = []
        y_vid_pred = []

        for video, predictions in video_labels.items():
            labels_test = predictions["labels_test"]
            labels_pred = predictions["labels_pred"]

            one_preds = []
            for _, pred in enumerate(labels_pred):
                one_preds.append(class_labels[max(range(len(pred)), key=pred.__getitem__)])

            majority_vote_pred = max(set(one_preds), key=one_preds.count)
            majority_vote_test = max(set(labels_test), key=labels_test.count)

            y_vid_pred.extend([majority_vote_pred])
            y_vid_test.extend([majority_vote_test])

        y_vid_test = np.array(y_vid_test)
        y_vid_pred = np.array(y_vid_pred)

        # Evaluate video-level predictions
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

        # save the metrics to an Excel file
        metrics_dict = {
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1 Score": [f1],
            "Confusion Matrix": [confusion_mat]
        }
        metrics_df = pd.DataFrame(metrics_dict)
        if retrain:
            metrics_path_i = os.path.join(new_metrics_dir, f"metrics_{i}.xlsx")
        else:
            metrics_path_i = os.path.join(metrics_dir, f"metrics_{i}.xlsx")
        metrics_df.to_excel(metrics_path_i, index=False)

        print(f"LOOCV iteration {i}: Files saved successfully.")