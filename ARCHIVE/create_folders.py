import os

path = 'C:/Users/USER/Documents/UniversityProjects/PythonProjects/FinalProject/loocv_splits'
for i in range(2, 31):
  folder_name = "S" + str(i)
  folder_path = os.path.join(path, folder_name)
  os.makedirs(folder_path)
  subsets = ["train", "test", "val"]
  for subset in subsets:
      susbset_path = os.path.join(folder_path, subset)
      os.makedirs(susbset_path)

      emotions = ["Anticipation", "Baseline", "Disappointment", "Frustration"]
      for emotion in emotions:
          subfolder_path = os.path.join(susbset_path, emotion)
          os.makedirs(subfolder_path)