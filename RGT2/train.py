import copy, time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from model import CNN
from settings import *
from dataset import DSDDataset

ds = DSDDataset(analysis_path)

n = len(ds)
n_val = int(n * val_size)
n_train = n - n_val
indices = list(range(n))
np.random.shuffle(indices)
train_idx, valid_idx = indices[n_val:], indices[:n_val]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(ds, batch_size = batch_size,
               num_workers = 4, sampler = train_sampler)

val_loader = DataLoader(ds, batch_size = batch_size,
               num_workers = 4, sampler = val_sampler)

model = CNN()
best_val_loss = np.inf

if torch.cuda.is_available():
	model.cuda()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = torch.nn.NLLLoss()


for epoch in range(n_epochs):
    print("Epoch ",epoch," of ",n_epochs,":")
    start = time.time()
    model.train()
    epoch_loss = 0
    val_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
           data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data[0]

    for v_batch_idx, (data, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        loss = criterion(model(data), target)
        val_loss += loss.data[0]

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        current_patience = patience
        best_model = copy.deepcopy(model.state_dict())
        torch.save(best_model, "cnn.pickle")
    else:
        current_patience -= 1
    print('Train Loss: {:.10f}\nVal Loss: {:.10f}\nPatience:  {:2d}\nElapsed: {:.5f}'.format(
        epoch_loss / batch_idx, val_loss / v_batch_idx, current_patience, time.time() - start)
    )

    if current_patience <=0:
        print("Patience over, stopping")
        break
