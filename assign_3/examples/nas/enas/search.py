# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

import datasets
from macro import GeneralNetwork
from micro import MicroNetwork
from nni.algorithms.nas.pytorch import enas
from nni.nas.pytorch.callbacks import (ArchitectureCheckpoint,
                                       LRSchedulerCallback)
from utils import accuracy, reward_accuracy

logger = logging.getLogger('nni')


if __name__ == "__main__":
    parser = ArgumentParser("enas")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--search-for", choices=["macro", "micro"], default="macro")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs (default: macro 310, micro 150)")
    parser.add_argument("--visualization", default=False, action="store_true")
    parser.add_argument("--v1", default=False, action="store_true")
    parser.add_argument("--gpu", default=2, type=int)
    args = parser.parse_args()

    dataset_train, dataset_valid = datasets.get_dataset("cifar10")
    mutator = None
    ctrl_kwargs = {}
    if args.search_for == "macro":
        model = GeneralNetwork()
        num_epochs = args.epochs or 310
    elif args.search_for == "micro":
        model = MicroNetwork(num_layers=6, out_channels=20, num_nodes=5, dropout_rate=0.1, use_aux_heads=False)
        num_epochs = args.epochs or 150
        if args.v1:
            mutator = enas.EnasMutator(model, tanh_constant=1.1, cell_exit_extra_step=True)
        else:
            ctrl_kwargs = {"tanh_constant": 1.1}
    else:
        raise AssertionError

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)

    if args.v1:
        trainer = enas.EnasTrainer(model,
                                   device=torch.device('cuda:{}'.format(args.gpu)),
                                   loss=criterion,
                                   metrics=accuracy,
                                   reward_function=reward_accuracy,
                                   optimizer=optimizer,
                                   callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")],
                                   batch_size=args.batch_size,
                                   num_epochs=num_epochs,
                                   dataset_train=dataset_train,
                                   dataset_valid=dataset_valid,
                                   log_frequency=args.log_frequency,
                                   mutator=mutator)
        if args.visualization:
            trainer.enable_visualization()
        trainer.train()
    else:
        from nni.retiarii.oneshot.pytorch.enas import EnasTrainer
        trainer = EnasTrainer(model,
                              loss=criterion,
                              metrics=accuracy,
                              reward_function=reward_accuracy,
                              optimizer=optimizer,
                              batch_size=args.batch_size,
                              num_epochs=num_epochs,
                              dataset=dataset_train,
                              log_frequency=args.log_frequency,
                              ctrl_kwargs=ctrl_kwargs)
        trainer.fit()
