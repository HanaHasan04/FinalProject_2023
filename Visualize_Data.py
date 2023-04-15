import os
import matplotlib.pyplot as plt

path = 'C:/Users/USER/Documents/UniversityProjects/PythonProjects/FinalProject/Videos'

# Create a dictionary to store the number of videos in each folder
num_videos = {}
for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        num_videos[folder] = len(os.listdir(folder_path))

# Plot a bar chart of the number of videos in each folder
plt.bar(num_videos.keys(), num_videos.values())
plt.xlabel('Emotion')
plt.ylabel('Number of Videos')
plt.title('Distribution of videos per emotion')
plt.show()
