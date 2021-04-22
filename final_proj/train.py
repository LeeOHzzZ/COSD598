import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import timeit

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        # if batch_idx == 0:
        #     print('train data batch size: {}'.format(data.size()))
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    # print('Train last batch idx:', batch_idx)
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    eval_cntr = 0
    with torch.no_grad():
        # t_1 = 0
        # t_2 = 0
        # t_3 = 0
        for data, target in dataloader:
            # if total == 0:
            #     print('eval data batch size: {}'.format(data.size()))
            start_time = timeit.default_timer()
            data, target = data.to(device), target.to(device)
            # time_1 = timeit.default_timer()
            output = model(data)
            # time_2 = timeit.default_timer()
            total += loss(output, target).item() * data.size(0)
            # time_3 = timeit.default_timer()
            # t_1 += time_1 - start_time
            # t_2 += time_2 - time_1
            # t_3 += time_3 - time_2
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
            eval_cntr += 1
        # print('eval verbose', t_1, t_2, t_3)
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    # print('eval cntr: ', eval_cntr)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5


def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose,
                    train_sampler = None):
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5]]
    # print('batch number of train_loader:', len(train_loader), 'batch_size:',train_loader.batch_size)
    # print('batch number of test_loader:', len(test_loader), 'batch_size:', test_loader.batch_size)
    train_time = 0
    eval_time = 0
    for epoch in tqdm(range(epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        start_timer = timeit.default_timer()
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        train_timer = timeit.default_timer()
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        eval_timer = timeit.default_timer()
        # print('Train_Eval_Loop: training time :: testing time is {}s::{}s'.format(
        #     train_timer-start_timer, eval_timer-train_timer
        # ))
        train_time += train_timer - start_timer
        eval_time += eval_timer - train_timer

        row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        rows.append(row)
    if epochs>0:
        print('Train_Eval_Loop: total_train_time::total_eval_time is {:04f}s::{:04f}s'.format(
            train_time/epochs, eval_time/epochs
        ))
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    return pd.DataFrame(rows, columns=columns)


