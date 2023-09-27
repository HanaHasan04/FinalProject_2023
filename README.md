# Horse Emotion Classification from Videos 
**Submitting Students:** Hana Hasan, Hallel Weinberg, Tidhar Rettig.  
**Under the supervision of:** Dr. Marcelo Feighelstein and Prof. Anna Zamansky, Tech4Animals Lab.

## Problem Statement
Just like humans, animals' facial expressions are closely connected to their emotional states. However, compared to the human domain, the automatic recognition of emotional states from animals' facial expressions remains largely unexplored. This lack of exploration is primarily due to challenges in collecting data and establishing a reliable ground truth for the emotional states of non-verbal subjects. 

This study aims to bridge this gap by being the first, to the best of our knowledge, to investigate the emotional states of **horses**. The research utilizes a dataset gathered in a controlled experimental environment, which includes videos of 30 horses. These horses were deliberately induced into four emotional states for the experiment: anticipation, frustration, baseline, and disappointment.  

## Our Methods and Techniques
### Overview
We present a machine learning model for emotion classification using the facial area, reaching accuracy of above $76\%$. We also apply techniques of frame selection to better exploit the availability of video data.

![image](https://github.com/HallelWeinberg/Horse-Emotion-Classification/assets/100043559/2c41416b-61ea-48f0-8c99-eaf455a4bef1)

We propose a two-stage pipeline which involves pre-processing and model training. In the pre-processing stage, we automatically detect and crop horses' faces from the videos, sample frames, and apply Grayscale Short-Term Stacking (GrayST) to incorporate temporal information. The model training stage employs a DINO-ViT embedding combined with SVM Classification. We then apply techniques of sophisticated frame selection to better exploit the availability of video data. The intuition here is by this specific manner of undersampling we can remove ‘noisy’ frames caused by the in-the-wild videos containing many low-quality frames, due to
obstruction (bars, horse not facing camera), blurry frames (caused by movement), or the fact that the emotion reflected visually does not always remain on the same fixed level throughout the video. Such removal of ‘noise’ indeed leads to increased
performance of the second model which is trained only on the top (highest confidence) frames.

### Dataset
* A controlled experiment was conducted to acquire data.
* The data consists of short videos of 30 horses.
* Each video has a duration of approximately 3 seconds.
* Emotion labeling was done for each video, with 4 different emotions: anticipation, baseline, disappointment, and frustration.
* Each horse has, on average, 3 videos labeled "anticipation", 1 video labeled "baseline", 3 videos labeled "disappointment", and 3 videos labeled "frustration".
* Every video was recorded using a 60 frames per second encoding.

<table align="center">
  <tr>
    <td><img src="https://github.com/HallelWeinberg/Horse-Emotion-Classification/assets/100043559/b098eb3a-b535-4e95-8316-c5f8c1a7a49a" width="400"></td>
    <td><img src="https://github.com/HallelWeinberg/Horse-Emotion-Classification/assets/100043559/7105adf7-3be9-493a-9fd0-d74c9999edc6" width="400"></td>
  </tr>
</table>

### Source Code
Our model consists of the following code files:
* [main.py](main.py)
* [extract_frames.py](extract_frames.py)
* [yolo.py](yolo.py)
* [grayST.py](grayST.py)
* [embedding.py](embedding.py)
* [train.py](train.py)
* [average.py](average.py)

You can see other codes we used [here](reference_code):
* [sampling.py:](reference_code/sampling.py) a code that sample frames from the dataset.
* [train_with_average_confidence_level.py:](reference_code/train_with_average_confidence_level.py) a code of the training phase with average confidence level instead of majority vote.
* [Binary classifier:](reference_code/binary_classifier) a code of a binary classifier between the two problematic emotions.
* [Three emotions classifier:](reference_code/three_emotions_classifier) a code of a three-category classifier that combines the two problematic emotions into a single category.
  
### Pre-Processing
#### Horse Face Detection and Cropping
The original video frames contain background clutter including the surrounding horse stable, food bowl, horse body, etc. We
aim to focus on the facial expressions of the horses and avoid learning other emotional state predictors (e.g. horse body postures). Hence, we used Yolov5 object detection model to identify horse faces, and used it to crop the facial bounding box from each image.

<table align="center">
  <tr>
    <td><img src="https://github.com/HallelWeinberg/Horse-Emotion-Classification/assets/100043559/3609725c-9532-4d72-8f21-2fbb85a5942a" width="400"></td>
    <td><img src="https://github.com/HallelWeinberg/Horse-Emotion-Classification/assets/100043559/007e0853-7138-427f-9739-2aaf390eb732" width="400"></td>
  </tr>
</table>  

#### Grayscale Short-Term Stacking (GrayST)
We use the Grayscale Short-Term Stacking (GrayST) to incorporate temporal information for video classification without augmenting the computational burden. This sampling
strategy involves substituting the conventional three color channels with three grayscale frames, obtained from three consecutive
time steps. Consequently, the backbone network can capture short-term temporal dependencies while sacrificing the capability
to analyze color.

<div align="center">
  <img src="https://github.com/HallelWeinberg/Horse-Emotion-Classification/assets/100043559/ff990735-96f4-4dec-b4ee-2f26b11782d4" alt="Your Image Description">
</div>

![Horses-GrayST-demo-GIF](https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/e85f22e8-49f8-4f54-b173-079ddd1bc004)  

  
#### After the Pre-Processing
Using GrayST to incorporate temporal information, we combined every three consecutive frames. This results in a sequence of 60 frames, on average, for each video, meaning that each frame encapsulates the essence of 0.05 seconds (a second comprises 60 frames). Consequently, for each horse, the distribution of frames per emotion category is as follows: 180 frames labeled as "anticipation," 60 frames labeled as "baseline," 180 frames labeled as "disappointment," and 180 frames labeled as "frustration".

### Model Training
#### Framework
For our emotion classification task, we adopt a combination of Vision Transformer (ViT) embedding and Support Vector Machine (SVM) classification. 
To achieve this, we employ a self-supervised ViT (DINO-ViT), trained using a self-distillation approach. Specifically, we utilize the ViT-B/8 architecture for encoding images.
In the context of ViT-B/8, "ViT" stands for Vision Transformer, and "B/8" denotes the batch size utilized during the model's training process. This means that the data is divided into batches, with each batch containing 8 samples. The choice of batch size is a crucial parameter in machine learning models as it impacts the efficiency and memory requirements during training.
We extract the output of the final layer as a 768-dimensional embedding vector that will be used for emotion classification.

We utilize a SVM for the classification of the 768-dimensional embedding vectors. In order to emulate a balanced dataset, we adjust the weights assigned to each class in inverse proportion to their frequencies in the input data. 
![image](https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/b1d3b6ba-4393-4de9-8039-e5a40d69aab5)  

#### Leave-One-Out Cross-Validation (LOOCV)
To assess the performance of our models, we rely on standard evaluation metrics, including accuracy, precision, recall, and F1. As part of our validation process, we employ the leave-one-subject-out cross-validation method, ensuring no subject overlap.

Given the relatively small number of horses in our dataset (n=30) and the corresponding number of samples (n=30*4), adhering to a stricter validation approach is more appropriate. Thus, during each iteration of LOOCV, we designate one horse (along with all of its frames) as the hold-out test set. Subsequently, we train the model on the data from the other 29 horses and evaluate its performance using accuracy, precision, recall, and F1.

Upon completion of the LOOCV process, we compute the average values for accuracy, precision, recall, and F1 across all 30 horses. These final results obtained after LOOCV are considered as indicative of the model's performance.

#### From Frame Predictions to Video Predictions
To transition from predicting emotions for individual frames to predicting emotions for entire videos, we propose an aggregation method that yields a prediction-per-video. During the training process, we utilize all frames from the 29 horses in the dataset. These frames are passed through a pre-trained Vision Transformer (ViT) model to obtain 768-dimensional embedding vectors. Subsequently, these vectors are used as input for a Support Vector Machine (SVM) classifier, where the emotion label serves as the target output (y). Once the training is complete, we apply the trained model to each frame from all horses to make predictions.

To obtain predictions per video, we propose the following majority voting aggregation method. For frames belonging to the same video, we identify the most confident label for each frame and then determine the most frequently occurring label among them. In this manner, each video is assigned a single predicted emotion label based on the aggregation of frame-level predictions.

The evaluation metrics are then computed on a video-wise basis. This process allows us to assess the model's performance in predicting emotions for entire videos, providing a more meaningful evaluation than considering individual frames separately.

#### Retraining on Top Frames
After the initial model training using the DINO-ViT embedding combined with SVM Classification, we aimed to further enhance our model's performance by retraining it on a subset of frames with the highest confidence scores. This process, called "frame selection," involved selecting a specific number of frames, denoted as $k$, with the highest confidence for each horse's video and emotion category.

To determine the optimal value of $k$, we experimented with various choices.

The motivation behind this frame selection approach was to address the potential presence of noisy frames in the video data. In-the-wild videos often contain low-quality frames, such as those with obstructions (e.g., bars, horse not facing the camera), blurry frames (caused by movement), or varying levels of visual expression of the emotion throughout the video. By selecting frames with the highest confidence, we aimed to remove noisy frames and improve the overall performance of the model.

In our experiments, for the emotions "anticipation," "disappointment," and "frustration," we selected $k=100$. As for the "baseline" emotion, we chose $k=34$.


### Model Performance

As stated earlier, for evaluating our model's performance, we employed the Leave-One-Horse-Out Cross-Validation (LOOCV) method. This approach ensured that during each iteration of evaluation, one horse's data, along with all its associated frames and emotions, was held out as the test set, while the model was trained on data from the remaining 29 horses. We repeated this process for all 30 horses, ensuring that each horse's data was used once as the test set.

In our evaluation, we utilized standard performance metrics for video-wise assessment, including accuracy, precision, recall, and F1-score. These metrics allowed us to gauge the model's ability to accurately classify emotions for entire videos.

To summarize the model's overall performance, we averaged the evaluation metrics obtained from the 30 iterations of LOOCV, and added up the confusion matrices.

#### Model Performance Metrics

| Method                                 | Accuracy | Precision | Recall | F1    |
| -------------------------------------- | -------- | --------- | ------ | ----- |
| No Preprocessing                       | 0.71     | 0.80      | 0.76   | 0.72  |
| Face Cropping                          | 0.67     | 0.78      | 0.71   | 0.69  |
| Face Cropping + GrayST                 | 0.68     | 0.77      | 0.71   | 0.70  |
| **Face Cropping + GrayST + k=100**    | **0.76** | **0.84**  | **0.79** | **0.77** |

#### Model Performance on Different Frame Selection Techniques

| Method                                 | Accuracy | Precision | Recall | F1    |
| -------------------------------------- | -------- | --------- | ------ | ----- |
| Face Cropping + GrayST + k = 50        | 0.74     | 0.85      | 0.77   | 0.76  |
| Face Cropping + GrayST + k = 100       | 0.76     | 0.84      | 0.79   | 0.77  |
| Face Cropping + GrayST + k = 150       | 0.74     | 0.83      | 0.77   | 0.75  |

![conf_mat](https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/fc7d999a-4707-42d1-93b5-f9f6e4559d01)

#### Confidence Histograms of Frames Classified by Our Model

<img width="433" alt="confidence_levels_all" src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/a3cc8dec-8083-4e5c-82e2-d8474787c81b">

In these histograms, you can observe the confidence level distributions of frames classified by our model, with a substantial number of frames with high confidence levels, followed by a long tail of frames with low confidence levels. An interesting observation arises in the Anticipation and Frustration histogram, reflecting a significant overlap between these two emotions, as also evident in the confusion matrix above. To address this challenge, we explore a dedicated binary classifier for distinguishing between Anticipation and Frustration in [Section](#).

