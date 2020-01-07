import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import timeit

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
from torchsummary import summary




#Load the best trained model
arch = 'best'

model_file = '%s_places365.pth.tar' % arch
model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.cuda()


#Print out the model summary
summary(model, (3, 224, 224))



#Setting up the training set and the testing test
batch_size = 1
lr = 0.01
momentum = 0.9
weight_decay = 10**(-4)
workers = 0

folder_path =r'C:\Users\duke\Desktop\CS6670_Final_Project\places365standard_easyformat\places365_standard'
traindir = os.path.join(folder_path, 'train')
valdir = os.path.join(folder_path, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,])),
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True)

#Define loss function (criterion) and pptimizer
criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)



#Function to calculate the accuracy
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


#Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



#Define the model, which is to take the CNN part of the model
class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x

res50_conv = ResNet50Bottom(model)


#Compute the feature vectors for training data
print_freq = 100
training_X = []
res50_conv.eval()



for i, (image_input, target) in enumerate(train_loader):
    image_input = image_input.cuda()
    label = target.numpy()[0]
    training_Y.append(label)
    target = target.cuda()
    image_input_var = torch.autograd.Variable(image_input)
    target_var = torch.autograd.Variable(target)
    
    #Compute the output
    output = res50_conv(image_input_var)
    outputs_temp = output.data.cpu().numpy()
    training_X.append(outputs_temp[0, :, 0 , 0])
    print(i)
    


#Store the training feature vectors in txt format
count = 0
invalid_count = 0
filename = r'C:\Users\duke\Desktop\CS6670_Final_Project\CNN_1200_100_train.txt'
with open(filename, "a") as txt_file:
    for i in range(len(training_X)):
        fd = np.array([training_X[i]])
        np.savetxt(txt_file, fd, fmt="%s")
        print(i)

           
print('done')            
txt_file.close()   





#Compute the feature vectors for testing data
print_freq = 100
training_Y = []
res50_conv.eval()



for i, (image_input, target) in enumerate(val_loader):
    #image_input = image_input.to('cuda:0')
    image_input = image_input.cuda()
    label = target.numpy()[0]
    training_Y.append(label)
    target = target.cuda()
    image_input_var = torch.autograd.Variable(image_input)
    target_var = torch.autograd.Variable(target)
    
    #Compute the output
    output = res50_conv(image_input_var)
    outputs_temp = output.data.cpu().numpy()
    training_X.append(outputs_temp[0, :, 0 , 0])
    print(i)




#Store the testing feature vectors in txt format
count = 0
invalid_count = 0
filename = r'C:\Users\duke\Desktop\CS6670_Final_Project\CNN_1200_100_val.txt'
with open(filename, "a") as txt_file:
    for i in range(len(training_X)):
        fd = np.array([training_X[i]])
        np.savetxt(txt_file, fd, fmt="%s")
        print(i)

           
print('done')            
txt_file.close()   




#Create labels for training set
training_Y = np.zeros(438000)
category_idx = 0


for i in range(438000):
    if i%1200 == 0 and i != 0.0:
        category_idx = category_idx + 1
    training_Y[i] = category_idx



#Create labels for testing set
testing_Y = np.zeros(36500)
category_idx = 0


for i in range(36500):
    if i%100 == 0 and i != 0.0:
        category_idx = category_idx + 1
    testing_Y[i] = category_idx


#Fitting SVM
svm_model = OneVsRestClassifier(LinearSVC(random_state=0))
svm_model.fit(training_X, training_Y)        


#Predicting labels for testing data
pred_y = svm_model.predict(testing_X)




#Calculating confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(testing_Y, pred_y)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)


#Calculate precision and recall
precision = np.diag(cm) / np.sum(cm, axis = 0)
np.nanmean(precision)

recall = np.diag(cm) / np.sum(cm, axis = 1)
np.mean(recall)

























