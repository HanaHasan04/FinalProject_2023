## Guide:  
`Video2frames.py`: extract and save video frames as "VideoName__FrameIndex.jpg"  
`LOOCV_splits.py`: create 30 folders, each containing three subfolders: test, train, and val. Test folder includes frames of all videos of a specific horse (i),
val folder includes frames of randomly selected horses (3 horses), and the rest are in the train folder.
`FeatureExtractor.py`: extract features using **ViT-B/8** as the embedding model.  
- Save the features in three separate Excel files: train, val, and test. 
- Each file contains columns for Emotion, Image, Video, Horse, Frame, and Feature_{i+1} (i ranges from 0 to 768, depending on the embed_dim).  
`NaiveBayes.py`: Train a Gaussian Naive Bayes classifier in using the concatenated train and val data, and test it on the test data. 
- Employ majority voting to determine the prediction of a "video" (all its frames as one row).  
- Calculate metrics such as Accuracy, F1 score, Precision, Recall, and Confusion Matrix.  
`PCA_TSNE.py`: for dimensionality reduction. Create a new Excel file with reduced dimensions. By default, PCA is used with 100 components. 
Modify the code (remove #comments) to use t-SNE or change the number of components.
  
  
 steps:
  sun Video2frames.py
  run LOOCV_splits.py
  for all test horse i in [1, 30]:
    run FeatureExtractor.py
    run NaiveBayes.py and write results in report
    
    run PCA_TSNE.py on all excels 
    run NaiveBayes.py and comapare results 
  
    
  
  

