import os
import pandas as pd


def tree_emotions_classifier(metadata_dir, i_start, i_end):
    for i in range(i_start, i_end + 1):
        metadata_path = os.path.join(metadata_dir, f"metadata_{i}.xlsx")
        metadata = pd.read_excel(metadata_path)
        metadata["Emotion"] = metadata["Emotion"].replace({"Anticipation": "Combined", "Frustration": "Combined"})
        os.remove(metadata_path)
        metadata.to_excel(metadata_path, index=False)