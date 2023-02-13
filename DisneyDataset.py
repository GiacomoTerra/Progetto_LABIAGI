#import packages
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import os
import cv2
from PIL import Image


#ignore warnings
import warnings
warnings.filterwarnings("ignore")
Image.LOAD_TRUNCATED_IMAGES = True

#define the class for the custom dataset
class DisneyDataset(Dataset):
	
	def __init__(self, root, split, transform = None):
		self.root = os.path.join(root, split)
		self.split = split
		self.transform = transform
		self.images = []
		self.labels = []
		self.classes = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
		#loop over the characters folders
		for character_id, character_name in enumerate(self.classes):
			class_folder = os.path.join(self.root, character_name)
			for image_name in os.listdir(class_folder):
				if (image_name.endswith('.jpg') or image_name.endswith('.jpeg') or image_name.endswith('.png') or image_name.endswith('.gif')):
					self.images.append(os.path.join(class_folder, image_name))
					self.labels.append(character_id)
								
	def __getitem__(self, index):
		image = Image.open(self.images[index]).convert('RGB')
		try:
			image = Image.open(self.images[index]).convert('RGB')
		except Exception as e:
			print(index, e)
		label = self.labels[index]
		if self.transform:
			image = self.transform(image)
		return image, label
	
	def __len__(self):
		return len(self.images)

#Define a transform to preprocess the data		
data_transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

#Load the custom datset
#dataset = DisneyDataset(root = 'cartoon', transform = data_transform)

#Split the data into training and validation sets
train_dataset = DisneyDataset(root = 'cartoon', split = 'train', transform = data_transform)
test_dataset = DisneyDataset(root = 'cartoon', split = 'test', transform = data_transform)
print(len(train_dataset))
print(len(test_dataset))

#Create a data loader to load the data in batches during training
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = False)



