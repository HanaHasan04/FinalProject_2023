## Guide:  
`Video2frames.py`: extract and save video frames as "VideoName__FrameIndex.jpg"  

`LOOCV_splits.py`: create 30 folders, each containing three subfolders: test, train, and val. Test folder includes frames of all videos of a specific horse (i),
val folder includes frames of randomly selected horses (3 horses), and the rest are in the train folder.  

`FeatureExtractor.py`: extract features using **ViT-B/8** as the embedding model.  
Save the features in three separate Excel files: train, val, and test.  
Each file contains columns for Emotion, Image, Video, Horse, Frame, and Feature_{i+1} (i ranges from 0 to 768, depending on the embed_dim).    

`NaiveBayes.py`: train a Gaussian Naive Bayes classifier using the concatenated train and val data, and test it on the test data.  
Employ majority voting to determine the prediction of a "video" (all its frames as one row).  
Calculate metrics such as Accuracy, F1 score, Precision, Recall, and Confusion Matrix.  

`PCA_TSNE.py`: for dimensionality reduction. Create a new Excel file with reduced dimensions. By default, PCA is used with 100 components. 
Modify the code (remove #comments) to use t-SNE or change the number of components.  
Perform Naive Bayes classification on the reduced feature data and compare the results with the original results.  
  
  
**Repeat the following steps for each test horse (i) in the range [1, 30]:**
- Run `FeatureExtractor.py`.
- Run `NaiveBayes.py` and record the results in a report.
- Run `PCA_TSNE.py` on all Excel files.
- Run `NaiveBayes.py` again and compare the results.  
  
**Real-Time Analysis Guide**:  
- Use the `RealTime.py` script for real-time analysis or graphical visualization of our model.  
- Recover the best Naive Bayes classifier using the Excel file with the best results.  
- Load the video.  
- Process the video frame-by-frame:  
- - Apply YOLO object detection to draw a bounding box around the horse's face in each frame.
- - Use the Naive Bayes classifier to predict the emotion (label) for that frame.  
- - Display the predicted emotion alongside the bounding box.  
- - Apply majority voting to determine the current overall prediction for the video, which changes as frames progress.


  
  

