# Imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm


# Data Preparation
data_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

train_dataset = torchvision.datasets.ImageFolder(root='C:/Users/USER/Documents/UniversityProjects/PythonProjects/FinalProject/loocv_splits/S1/train',
                                                 transform=data_transform)
test_dataset = torchvision.datasets.ImageFolder(root='C:/Users/USER/Documents/UniversityProjects/PythonProjects/FinalProject/loocv_splits/S1/test',
                                                transform=data_transform)
val_dataset = torchvision.datasets.ImageFolder(root='C:/Users/USER/Documents/UniversityProjects/PythonProjects/FinalProject/loocv_splits/S1/val',
                                               transform=data_transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0)

# Define the model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4) # 4 number of classes

# Define the Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

# Train the model
num_epochs= 10
for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader, desc="Epoch {:1d}".format(epoch+1)), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
    print('Epoch {} - Training loss: {}'.format(epoch + 1, running_loss / len(trainloader)))

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

#Calculate Precision, Recall and F1-Score
y_true = []
y_pred = []

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.append(labels.numpy())
        y_pred.append(predicted.numpy())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

target_names = ['Anticipation', 'Baseline', 'Disappointment', 'Frustration']
print(classification_report(y_true, y_pred, target_names=target_names))