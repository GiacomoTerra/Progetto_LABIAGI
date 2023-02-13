#import packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DisneyDataset import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
Image.LOAD_TRUNCATED_IMAGES = True

#Define a transform to preprocess the data		
data_transform = transforms.Compose([
	transforms.Resize(64),
	transforms.CenterCrop(64),
	transforms.ToTensor(),
	transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

#Split the data into training and validation sets
train_dataset = DisneyDataset(root = 'cartoon', split = 'train', transform = data_transform)
test_dataset = DisneyDataset(root = 'cartoon', split = 'test', transform = data_transform)
print(len(train_dataset))
print(len(test_dataset))

#Create a data loader to load the data in batches during training
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = False)


#Definisco la rete neurale
class DisneyNet(nn.Module):
	def __init__(self):
		super(DisneyNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)
		self.norm = nn.BatchNorm2d(4)
		self.relu = nn.ReLU(inplace = True)
		self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
		self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)		
		self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)		
		self.fc1 = nn.Linear(in_features = 4*16*16, out_features = 120)
		self.fc2 = nn.Linear(in_features = 120, out_features = 84)
		self.fc3 = nn.Linear(in_features = 84, out_features = 10)
	def forward(self, x):
		x = self.conv1(x)
		x = self.norm(x)
		x = self.pool(F.relu((x)))
		x = self.conv2(x)
		x = self.norm(x)		
		x = self.pool2(F.relu((x)))
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

#Creo un'istanza della rete neurale
my_nn = DisneyNet().to(device)
print(my_nn)

#Definisco la funzione di perdita e l'ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(my_nn.parameters(), lr = 0.01)
#optimizer = optim.SGD(my_nn.parameters(), lr = 0.001, momentum = 0.9)
if torch.cuda.is_available():
	criterion = criterion.cuda()
	optimizer = optimizer.cuda()

#loop over the dataset multiple times
for epoch in range(10):
	running_loss = 0.0
	for idx, data in enumerate(train_loader, 0):
		#get the inputs; data is a list of [inputs, labels]
		inputs, labels = data
		#zero the parameter gradients
		optimizer.zero_grad()
		#forward + backward + optimize
		outputs = my_nn(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		#print statistics
		running_loss += loss.item()
	else:
		print("Epoch {} - Training Loss: {}".format(epoch + 1, running_loss/len(train_loader)))
		#print evvery 2000 mini-batches
		#if idx % 2000 == 1999:
			#print(f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 2000:.3f}')
			#running_loss = 0.0
print('Finished Training')

#salvo il modulo addestrato
path = './disney_net.pth'
torch.save(my_nn.state_dict(), path)

#switch the model to evaluation mode
my_nn.eval()

correct_count = 0.0
total_count = 0.0
for images, labels in test_loader:	
	for i in range(len(labels)):
		with torch.no_grad():
			outputs = my_nn(images)
		ps = torch.exp(outputs)
		probability = list(ps.cpu()[i])
		pred_label = probability.index(max(probability))
		true_label = labels.cpu()[i]
		#total += labels.size(0)
		if (true_label == pred_label):
			correct_count += 1
		total_count += 1
print("Numero Immagini Testate: ", total_count)
print("\nModel Accuracy: ", (correct_count / total_count))
	
























