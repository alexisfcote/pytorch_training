#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.tensor as tt
from tqdm import tqdm as tqdm

import matplotlib.pyplot as plt
import time
import os
import copy
import re

from pathlib import Path

from IPython.core.debugger import set_trace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloaders = {'train':dl_train, 'val':dl_val}


from collections import defaultdict

def train_model(model, criterion, optimizer, scheduler, recorder, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    epoch_tqdm = tqdm(total=num_epochs)
    epoch_tqdm.set_description('epoch')
    train_tqdm = tqdm(total=len(dl_train))
        

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                recorder[phase]['lr'].append(scheduler.get_lr())
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            train_tqdm.n = 0
            train_tqdm.total = len(dataloaders[phase])
            train_tqdm.last_print_n = 0
            train_tqdm.start_t = time.time()
            train_tqdm.last_print_t = time.time()
            train_tqdm.set_description(phase)
            train_tqdm.refresh()
            
            for inputs, labels in dataloaders[phase]:
                train_tqdm.update()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            recorder[phase]['loss'].append(epoch_loss)
            recorder[phase]['acc'].append(epoch_acc)            
            if phase == 'val':
                train_tqdm.set_postfix_str(
                    'train Loss: {:.4f} Acc: {:.4f} | valid Loss: {:.4f} Acc: {:.4f}'.format(
                            recorder['train']['loss'][-1], recorder['train']['acc'][-1],
                            recorder['val']['loss'][-1], recorder['val']['acc'][-1],
                        )
                )

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
        epoch_tqdm.update()

    time_elapsed = time.time() - since
    epoch_tqdm.close()
    train_tqdm.close()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.005, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

cosine_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_conv, T_max=25//2, eta_min=0.0005, last_epoch=25//2)

recorder = defaultdict(lambda:defaultdict(list))
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         cosine_lr_scheduler, recorder, num_epochs=25)



torch.save(model_conv.state_dict(), 'model_conv.pth')
torch.save(recorder, 'model_conv_recorder.pth')
