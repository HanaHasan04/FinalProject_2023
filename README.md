# Horse Emotion Classification from Videos 
**Submitting Students:** Hana Hasan, Hallel Weinberg, Tidhar Rettig.  
**Under the supervision of:** Dr. Marcelo Feighelstein and Prof. Anna Zamansky, Tech4Animals Lab.

## Problem Statement
Just like humans, animals' facial expressions are closely connected to their emotional states. However, compared to the human domain, the automatic recognition of emotional states from animals' facial expressions remains largely unexplored. This lack of exploration is primarily due to challenges in collecting data and establishing a reliable ground truth for the emotional states of non-verbal subjects. 

This study aims to bridge this gap by being the first, to the best of our knowledge, to investigate the emotional states of **horses**. The research utilizes a dataset gathered in a controlled experimental environment, which includes videos of 30 horses. These horses were deliberately induced into four emotional states for the experiment: anticipation, frustration, baseline, and disappointment.  

## Our Methods and Techniques
### Overview
We present a machine learning model for emotion classification using the facial area, reaching accuracy of above $76$%. We also apply techniques of frame selection to better exploit the availability of video data.

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

You can see other code we used [here](reference_code):
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

![image](https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/85b58662-85bf-4381-aaf2-7c6a32e59b70)

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

<img width="960" alt="conusion_matrix_all" src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/b52649f9-3f2a-4f7d-bd1d-ec39157fa401">


#### Confidence Histograms of Frames Classified by Our Model

<img width="433" alt="confidence_levels_all" src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/7f1d37b8-e27d-4c07-9357-8be33f80e984">

In these histograms, you can observe the confidence level distributions of frames classified by our model, with a substantial number of frames with high confidence levels, followed by a long tail of frames with low confidence levels. An interesting observation arises in the Anticipation and Frustration histogram, reflecting a significant overlap between these two emotions, as also evident in the confusion matrix above. To address this challenge, we explore a dedicated binary classifier for distinguishing between Anticipation and Frustration.


## Distinguishing Between Anticipation and Frustration  
### Binary Classifier
For this task, we adapted the pipeline that generated our top-performing model, which encompassed processes such as face cropping, GrayST, training, and retraining on the top 100 frames. The table provides an overview of the model's performance metrics, including accuracy, precision, recall, and F1-score. Notably, compared to the four-emotion classifier, when focusing on the two most "confounding" emotions, the performance of this binary classifier is somewhat reduced, with an accuracy of 0.61 compared to 0.76. One possible explanation for this confusion between Anticipation and Frustration is that both emotions are highly "expressive" in nature, in contrast to the baseline and disappointment, leading the model to occasionally misclassify between these two states.

#### Model Performance of the Binary Classifier: Anticipation vs. Frustration.

| Accuracy | Precision | Recall | F1    |
|----------|-----------|--------|-------|
| 0.61     | 0.65      | 0.60   | 0.56  |


### Three-Category Classifier
Furthermore, we explored the development of a three-category classifier that combines these two emotions into a single category. The table provides an overview of the model's performance metrics, including accuracy, precision, recall, and F1-score. Notably, compared to the four-emotion classifier, when combining the two most "confounding" emotions into one, the performance of this 3-category classifier is increased, with an accuracy of 0.90 compared to 0.76. The reasoning behind this might lie in the fact that the model could actually separate the baseline, disappointment, and the amalgamation of "anticipation and frustration" into a single entity. If we consider frustration as a form of "negative anticipation," it's possible that the horse's expressions of these emotions are not easily distinguishable, contributing to this improved classifier performance.

#### Model Performance of the 3-Category Classifier: Anticipation and Frustration Merged.

| Accuracy | Precision | Recall | F1    |
|----------|-----------|--------|-------|
| 0.90     | 0.93      | 0.90   | 0.90  |  


## On the Power of DINO-ViT Features  
Vision Transformers (ViT) have emerged as powerful backbones for image analysis. When trained using self-distillation techniques like DINO, they produce rich representations that capture both local and global features within images.  
To gain insights into these representations, we employed Principal Component Analysis (PCA) with $n=2$ components to project the features into a 2D space. In this visualization, each point corresponds to the embedding of a single frame, and emotions are differentiated by colors. 
<img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/a9dfcd74-bacd-4c02-b241-fce2090088e6" alt="PCA_S11" width="50%">


To further explore the deep features extracted by the ViT model trained with DINO, 
we visualize the attention map of one of the ViT heads. The ViT discovers the semantic structure of an image in an unsupervised way. In the figures, we present the input image, the attention map extracted from the last layer, the map after applying Gaussian blur for smoothing and normalizing, and lastly, the original image overlaid with the smoothed map, for each one of the emotional classes.

#### Anticipation
<table align="center">
  <tr>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/cb1a56d0-4aa9-4a58-b86d-08182963d96d" width="200"></td>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/a6bfeb85-c19d-476b-90c3-0ae79e1ac485" width="200"></td>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/b6fa56ac-92af-441f-9777-554d8711a2dc" width="200"></td>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/df9f3eee-71fa-4885-9ec5-708c76d1304e" width="200"></td>
  </tr>
</table>

#### Baseline
<table align="center">
  <tr>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/64664ce6-ff93-4e66-a5e7-4316858b9781" width="200"></td>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/f8767c16-3c90-4e8f-b74b-ebaf09397730" width="200"></td>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/ef00a782-0670-4640-9cd5-644dc2662dbd" width="200"></td>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/113b022b-92bd-43ae-bb9e-e1be55618e2f" width="200"></td>
  </tr>
</table>

#### Disappointment
<table align="center">
  <tr>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/7a27de3a-ab2e-4499-b42c-2dc718952053" width="200"></td>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/21cbf129-34d6-4490-b4ce-25adc86a33f8" width="200"></td>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/67b4f5ff-d717-4d18-ac6b-2cbde9c842fb" width="200"></td>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/766be6f5-2a2d-4d81-9680-91802642b109" width="200"></td>
  </tr>
</table>

#### Frustration
<table align="center">
  <tr>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/da66976b-4f3d-4baa-8c23-52c6d8ca64a8" width="200"></td>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/24f4e0a2-c278-42b2-a6fc-5ae3f867b5aa" width="200"></td>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/846b9d44-fbc5-4e72-a6ab-1f003e6b129f" width="200"></td>
    <td><img src="https://github.com/HanaHasan04/FinalProject_2023/assets/100927079/2ac2f3db-3d04-4978-a876-79a4ad4ac939" width="200"></td>
  </tr>
</table>


## Step-By-Step Guide
### Dataset:
	The dataset is organized into four categories of horse behavior videos:
	1. Anticipation.
	2. Baseline.
	3. Disappointment.
	4. Frustration.

	Example video name: the video's name follows the pattern: S*the number of the horse*-*the name of the video*.mp4.
	For example:
	'S1-T1-A1-C1-3.mp4'
	- S1: Horse number 1.
	- T1-A1-C1-3: The name of the video.

### Usage:
	To replicate and use this project, run the main.py file along with the following files:
	1. extract_frames.py
	2. yolo.py
	3. grayST.py
	4. embedding.py
	5. train.py
	6. average.py
	
