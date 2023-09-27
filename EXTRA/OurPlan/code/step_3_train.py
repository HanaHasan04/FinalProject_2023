import os
import pandas as pd
import time
from sklearn.svm import SVC

i_start = 1
i_end = 30
k = 30

class_labels = ["Anticipation", "Baseline", "Disappointment", "Frustration"]

all_metadata_path = f'data/metadata'
new_metadata_path = f'data/new_metadata'
os.makedirs(new_metadata_path, exist_ok=True)

# Concatenate metadata from all 30 horses
metadata_list = []
for j in range(1, 31):
    print(j)
    metadata_path = os.path.join(all_metadata_path, f"metadata_{j}.xlsx")
    metadata = pd.read_excel(metadata_path)
    metadata_list.append(metadata)
all_metadata = pd.concat(metadata_list, ignore_index=True)
metadata_list.clear()

print("start")

# leave-one-out cross-validation, subject-wise.
for i in range(i_start, i_end + 1):
    # train set
    train_metadata = all_metadata[all_metadata['Horse'] != i]
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
    y_pred_proba = classifier.predict_proba(X_test)
    print("After SVM")
    print(f"Time for horse {i}: ", time.time() - s_time)

    # majority voting on video-level predictions.
    video_labels = {}
    top_k_0 = [(-1, 0) for _ in range(k)]
    top_k_1 = [(-1, 0) for _ in range(k)]
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

        if label_pred[1] > top_k_1[k - 1][0] and label_test == class_labels[1]:
            top_k_1[k - 1] = (label_pred[1], index)
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

    print(f"LOOCV iteration {i}: Files saved successfully.")