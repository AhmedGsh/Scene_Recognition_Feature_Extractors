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
import torch.utils.data as utils



#Get the GPU device
cuda0 = torch.device('cuda:0')
torch.cuda.get_device_name(0)


#Read the txt file of training data
filename = r'C:\Users\yunxi\Desktop\CS6670\CNN_1200_100_train.txt'
with open(filename, 'r') as f:
    x = f.readlines()

f.close()



#Read the feature vectors of training data
training_X = []
c_disk = 'C:'

for i in range(len(x)):
    print(i)
    ppp = x[i].replace('\n',' ')
    ppp = ppp.replace('\\' ,' ')
    ppp = ppp.split(' ')
    ppp = ppp[:-1]
    if len(ppp) == 2048:
        ppp = np.array(ppp)
        ppp = ppp.astype(np.float)
        training_X.append(ppp)



#Create labels for training set
training_Y = np.zeros(438000)
category_idx = 0


for i in range(438000):
    if i%1200 == 0 and i != 0.0:
        category_idx = category_idx + 1
    training_Y[i] = category_idx


#Create the dataloader for training data
tensor_x = torch.stack([torch.Tensor(i) for i in training_X])
tensor_y = torch.stack([torch.Tensor(np.array(i)) for i in training_Y])

my_dataset = utils.TensorDataset(tensor_x,tensor_y) 
train_loader = utils.DataLoader(my_dataset, batch_size=32, shuffle=True) 




#Read the txt file of testing data
filename = r'C:\Users\Ahmed Almostafa\Desktop\Cornell\fall 2019\computer vision\project\CNN_1200_100_val.txt'
with open(filename, 'r') as f:
    x = f.readlines()

f.close()



#Read the feature vectors of testing data
testing_X = []
c_disk = 'C:'

for i in range(len(x)):
    print(i)
    ppp = x[i].replace('\n',' ')
    ppp = ppp.replace('\\' ,' ')
    ppp = ppp.split(' ')
    ppp = ppp[:-1]
    if len(ppp) == 2048:
        ppp = np.array(ppp)
        ppp = ppp.astype(np.float)
        testing_X.append(ppp)



#Create labels for testing set
testing_Y = np.zeros(36500)
category_idx = 0


for i in range(36500):
    if i%100 == 0 and i != 0.0:
        category_idx = category_idx + 1
    testing_Y[i] = category_idx


#Create the dataloader for testing data
tensor_x = torch.stack([torch.Tensor(i) for i in testing_X]) 
tensor_y = torch.stack([torch.Tensor(np.array(i)) for i in testing_Y])

my_dataset = utils.TensorDataset(tensor_x,tensor_y) 
val_loader = utils.DataLoader(my_dataset, batch_size=32, shuffle=True) 



#Define the MLP model
class MLP(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
            
            
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            return output

model = MLP(2048, 1024,365)
model.cuda(cuda0)


#Define the loss and optimizier
lr = 0.1
momentum = 0.9
weight_decay = 10**(-4)
workers = 0

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



#Save the model
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


#Sets the learning rate to the initial LR decayed by 10 every 30 epochs
def adjust_learning_rate(optimizer, epoch, learning_rate):
    learning_rate = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate



#Train the model
import timeit
import shutil
arch = 'CNN_MLP_'

num_epoch = 120

print_freq = 100
best_prec1 = 0

runinng_time_vector = []
training_loss_vector = []
validation_loss_vector = []
training_top1_accuracy_vector = []
validation_top1_accuracy_vector = []
training_top5_accuracy_vector = []
validation_top5_accuracy_vector = []


for epoch in range(num_epoch):
    
    start = timeit.default_timer()
    adjust_learning_rate(optimizer, epoch, lr)
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    
    
    for i, (image_input, target) in enumerate(train_loader):
        image_input = image_input.cuda(cuda0)
        target = target.cuda(cuda0)
        image_input_var = torch.autograd.Variable(image_input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(image_input_var)
        loss = criterion(output, target_var.long())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, image_input.size(0))
        top1.update(prec1, image_input.size(0))
        top5.update(prec5, image_input.size(0))


        #Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i % print_freq) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))
            
    training_loss_vector.append(losses.avg)
    training_top1_accuracy_vector.append(top1.avg)
    training_top5_accuracy_vector.append(top5.avg)
            
    
   
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    model.eval()
    
            
    for i, (image_input, target) in enumerate(val_loader):
        
        image_input = image_input.cuda(cuda0)
        target = target.cuda(cuda0)
        image_input_var = torch.autograd.Variable(image_input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(image_input_var)
        loss = criterion(output, target_var.long())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, image_input.size(0))

        top1.update(prec1, image_input.size(0))
        top5.update(prec5, image_input.size(0))

        if (i % print_freq == 0):
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader),  loss=losses,
                top1=top1, top5=top5))
    
    prec1 = top1.avg
    validation_loss_vector.append(losses.avg)
    validation_top1_accuracy_vector.append(prec1)
    validation_top5_accuracy_vector.append(top5.avg)
    
    is_best = prec1 > best_prec1
    if prec1 > best_prec1:
        best_prec1 = prec1
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, arch.lower())
    
    stop = timeit.default_timer()
    running_time = stop - start
    runinng_time_vector.append(running_time)
    print(running_time)
    print('epoch ', epoch)





#Create the dataset for testing data again for label predicting
tensor_x = torch.stack([torch.Tensor(i) for i in testing_X]) 
tensor_y = torch.stack([torch.Tensor(np.array(i)) for i in testing_Y])

my_dataset = utils.TensorDataset(tensor_x,tensor_y) 
val_loader = utils.DataLoader(my_dataset, batch_size=1, shuffle=False) 





#Use the trained MLP to predict labels
print_freq = 100
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
labels = []


model.eval()


for i, (image_input, target) in enumerate(val_loader):
    image_input = image_input.cuda(cuda0)
    target = target.cuda(cuda0)
    image_input_var = torch.autograd.Variable(image_input)
    target_var = torch.autograd.Variable(target)

    #Compute the output
    output = model(image_input_var)
    _, index = torch.max(output, 1)
    label = index[0].cpu().numpy()
    labels.append(label)
    
    loss = criterion(output, target_var.long())

    #Measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    print(prec1)
    losses.update(loss.data, image_input.size(0))
    top1.update(prec1, image_input.size(0))
    top5.update(prec5, image_input.size(0))


    if (i % print_freq == 0):
        print('Test: [{0}/{1}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            i, len(val_loader),  loss=losses,
            top1=top1, top5=top5))

print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))



testing_Y = np.zeros(36500)
category_idx = 0



#Create labels for testing set
for i in range(36500):
    if i%100 == 0 and i != 0.0:
        category_idx = category_idx + 1
    testing_Y[i] = category_idx



#Calculating confusion matrix
from sklearn.metrics import confusion_matrix


cm = confusion_matrix(testing_Y, labels)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)


#Calculate precision and recall
precision = np.diag(cm) / np.sum(cm, axis = 0)
np.nanmean(precision)

recall = np.diag(cm) / np.sum(cm, axis = 1)
np.mean(recall)













