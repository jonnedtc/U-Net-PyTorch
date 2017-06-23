import os
import time
import networks
import numpy as np
from subprocess import call
from loss import dice as dice_loss
from data_load import LiverDataSet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


### variables ###

model_name = '25D'

augment = True
dropout = True

# using dice loss or cross-entropy loss
dice = True

# how many slices of context (2.5D)
context = 2

# learning rate, batch size, samples per epoch, epoch where to lower learning rate and total number of epochs
lr = 1e-2
batch_size = 10
num_samples = 1000
low_lr_epoch = 80
epochs = 100

#################


train_folder = 'data/train' 
val_folder = 'data/val'

print model_name
print "augment="+str(augment)+" dropout="+str(dropout)
print str(epochs) + " epochs - lr: " + str(lr) + " - batch size: " + str(batch_size)

# GPU enabled
cuda = torch.cuda.is_available()

# cross-entropy loss: weighting of negative vs positive pixels and NLL loss layer
loss_weight = torch.FloatTensor([0.01, 0.99])
if cuda: loss_weight = loss_weight.cuda()
criterion = nn.NLLLoss2d(weight=loss_weight)

# network and optimizer
net = networks.VNet_Xtra(dice=dice, dropout=dropout, context=context)
if cuda: net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).cuda()
optimizer = optim.Adam(net.parameters(), lr=lr)

# train data loader
train = LiverDataSet(directory=train_folder, augment=augment, context=context)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=train.getWeights(), num_samples=num_samples)
train_data = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, sampler=train_sampler, num_workers=2)

# validation data loader (per patient)
val = LiverDataSet(directory=val_folder, context=context)
val_data_list = []
patients = val.getPatients()
for key in patients.keys():
    samples = patients[key]
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(samples)
    val_data = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=2)
    val_data_list.append(val_data)

# train loop

print 'Start training...'

for epoch in range(epochs):

    running_loss = 0.0

    # lower learning rate
    if epoch == low_lr_epoch:
        for param_group in optimizer.param_groups:
            lr = lr / 10
            param_group['lr'] = lr

    # switch to train mode
    net.train()
    
    for i, data in enumerate(train_data):

        # wrap data in Variables
        inputs, labels = data
        if cuda: inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        # forward pass and loss calculation
        outputs = net(inputs)

        # get either dice loss or cross-entropy
        if dice:
            outputs = outputs[:,1,:,:].unsqueeze(dim=1)
            loss = dice_loss(outputs, labels)
        else:
            labels = labels.squeeze(dim=1)
            loss = criterion(outputs, labels)

        # empty gradients, perform backward pass and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # save and print statistics
        running_loss += loss.data[0]
    
    # print statistics
    if dice:
        print('  [epoch %d] - train dice loss: %.3f' % (epoch + 1, running_loss/(i+1)))
    else:
        print('  [epoch %d] - train cross-entropy loss: %.3f' % (epoch + 1, running_loss/(i+1)))
    
    # switch to eval mode
    net.eval()

    all_dice = []
    all_accuracy = []

    # only validate every 10 epochs
    if (epoch+1)%10 != 0: continue

    # loop through patients
    for val_data in val_data_list:

        accuracy = 0.0
        intersect = 0.0
        union = 0.0

        for i, data in enumerate(val_data):

            # wrap data in Variable
            inputs, labels = data
            if cuda: inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
            
            # inference
            outputs = net(inputs)
            
            # log softmax into softmax
            if not dice: outputs = outputs.exp()

            # round outputs to either 0 or 1
            outputs = outputs[:, 1, :, :].unsqueeze(dim=1).round()

            # accuracy
            outputs, labels = outputs.data.cpu().numpy(), labels.data.cpu().numpy()
            accuracy += (outputs == labels).sum() / float(outputs.size)

            # dice
            intersect += (outputs+labels==2).sum()
            union += np.sum(outputs) + np.sum(labels)

        all_accuracy.append(accuracy / float(i+1))
        all_dice.append(1 - (2 * intersect + 1e-5) / (union + 1e-5))

    print('    val dice loss: %.9f - val accuracy: %.8f' % (np.mean(all_dice), np.mean(all_accuracy)))
    
# save weights

torch.save(net, "model_"+str(model_name)+".pht")

print 'Finished training...'
