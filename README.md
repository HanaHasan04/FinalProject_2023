# Horse Emotion Classification from Videos  

## **Students:**  
- Hana Hasan  
- Hallel Weinberg  
- Tidhar Rettig  

## **Mentors:**  
- Anna Zamansky  
- Marcelo Feighelstein  
  
  
## **Achievements and  Plans**:  
- We learned the basics of OpenCV  
- We are learning Deep Learning and Neural Networks  
- We extracted and saved the video frames (we got ~179 frames per each ~3 sec video)  
** We have 4 files: Anticipation, Baseline, Disappointment, Frustration. Each containing images of the emotion.  
** The name of the image contains its corresponding video name so a WHOLE video (all of its frames) will be in the train/test set to avoid leakage.  
- We cropped the horses faces, but we're not sure yet if to train the model on cropped images or not  
- We will use ResNet  
- Classification will be done per frame independently. We started a discussion on Facebook: https://m.facebook.com/groups/MDLI1/permalink/2362483173915717/  
- In case we do not get good results, we might try using LSTM  

## **Progress Report:**
- **January:** We learned the theory and practical tools: the basics of OpenCV, Deep Learning and Neural Networks.
- **February:** We prepared the data: We extracted and saved the video frames. We cropped the horses faces. We started to check which model to work with.
- **March:** We started to train a model that performs classification using tensorflow.
