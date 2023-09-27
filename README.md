# Horse Emotion Classification from Videos 
**Submitting Students:** Hana Hasan, Hallel Weinberg, Tidhar Rettig  
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

<table align="center">
  <tr>
    <td><img src="https://github.com/HallelWeinberg/Horse-Emotion-Classification/assets/100043559/3609725c-9532-4d72-8f21-2fbb85a5942a" width="400"></td>
    <td><img src="https://github.com/HallelWeinberg/Horse-Emotion-Classification/assets/100043559/e1a59891-00df-45b0-9051-5cceb9a6e648" width="400"></td>
  </tr>
</table> 

#### After the Pre-Processing
Using GrayST to incorporate temporal information, we combined every three consecutive frames. This results in a sequence of 60 frames, on average, for each video, meaning that each frame encapsulates the essence of 0.05 seconds (a second comprises 60 frames). Consequently, for each horse, the distribution of frames per emotion category is as follows: 180 frames labeled as "anticipation," 60 frames labeled as "baseline," 180 frames labeled as "disappointment," and 180 frames labeled as "frustration".

### Model Training
For our emotion classification task, we adopt a combination of Vision Transformer (ViT) embedding and Support Vector Machine (SVM) classification. 
To achieve this, we employ a self-supervised ViT (DINO-ViT), trained using a self-distillation approach. Specifically, we utilize the ViT-B/8 architecture for encoding images.
In the context of ViT-B/8, "ViT" stands for Vision Transformer, and "B/8" denotes the batch size utilized during the model's training process. This means that the data is divided into batches, with each batch containing 8 samples. The choice of batch size is a crucial parameter in machine learning models as it impacts the efficiency and memory requirements during training.
We extract the output of the final layer as a 768-dimensional embedding vector that will be used for emotion classification.

We utilize a SVM for the classification of the 768-dimensional embedding vectors. In order to emulate a balanced dataset, we adjust the weights assigned to each class in inverse proportion to their frequencies in the input data. 
