import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *

import timeit
import os

# module for distributed data parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as ddp

# this file is migrated from the orginal singleshot function to support DDP

def run(gpu_id, args):

    ## parameters for multi-processing
    print('using gpu', gpu_id)
    dist.init_process_group(
      backend = 'nccl',
      init_method = 'env://',
      world_size = args.world_size,
      rank = gpu_id
    )

    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    # device = load.device(args.gpu)
    device = torch.device(gpu_id)

    args.gpu_id = gpu_id

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset)

    ## need to change the workers for loading the data
    args.workers = int((args.workers + 4 - 1)/4)
    print('Adjusted dataloader worker number is ', args.workers)

    # prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers,
    #                 args.prune_dataset_ratio * num_classes, world_size=args.world_size, rank=gpu_id)
    prune_loader, _ = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers,
                    args.prune_dataset_ratio * num_classes)

    ## need to divide the training batch size for each GPU
    args.train_batch_size = int(args.train_batch_size/args.gpu_count)
    train_loader, train_sampler = load.dataloader(args.dataset, args.train_batch_size, True, args.workers, args=args)
    # args.test_batch_size = int(args.test_batch_size/args.gpu_count)
    test_loader, _ = load.dataloader(args.dataset, args.test_batch_size, False, args.workers)
    
    print("data loader batch size (prune::train::test) is {}::{}::{}".format(
        prune_loader.batch_size,
        train_loader.batch_size,
        test_loader.batch_size
    ))

    log_filename = '{}/{}'.format(args.result_dir, 'result.log')
    fout = open(log_filename, 'w')
    fout.write('start!\n')

    if args.compression_list == []:
        args.compression_list.append(args.compression)
    if args.pruner_list == []:
        args.pruner_list.append(args.pruner)
    
    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    
    # load the pre-defined model from the utils
    model = load.model(args.model, args.model_class)(input_shape, 
                                                    num_classes, 
                                                    args.dense_classifier, 
                                                    args.pretrained)  
    
    ## wrap model with distributed dataparallel module
    torch.cuda.set_device(gpu_id)
    # model = model.to(device)
    model.cuda(gpu_id)
    model = ddp(model, device_ids = [gpu_id])

    ## don't need to move the loss to the GPU as it contains no parameters
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

    ## Pre-Train ##
    print('Pre-Train for {} epochs.'.format(args.pre_epochs))
    pre_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                test_loader, device, args.pre_epochs, args.verbose, train_sampler=train_sampler)
    print('Pre-Train finished!')
    
    ## Save Original ##
    torch.save(model.state_dict(),"{}/pre_train_model_{}.pt".format(args.result_dir, gpu_id))
    torch.save(optimizer.state_dict(),"{}/pre_train_optimizer_{}.pt".format(args.result_dir, gpu_id))
    torch.save(scheduler.state_dict(),"{}/pre_train_scheduler_{}.pt".format(args.result_dir, gpu_id))

    if not args.unpruned:
        for compression in args.compression_list:
            for p in args.pruner_list:
                # Reset Model, Optimizer, and Scheduler
                print('compression ratio: {} ::: pruner: {}'.format(compression, p))
                model.load_state_dict(torch.load("{}/pre_train_model_{}.pt".format(args.result_dir, gpu_id), map_location=device))
                optimizer.load_state_dict(torch.load("{}/pre_train_optimizer_{}.pt".format(args.result_dir, gpu_id), map_location=device))
                scheduler.load_state_dict(torch.load("{}/pre_train_scheduler_{}.pt".format(args.result_dir, gpu_id), map_location=device))

                ## Prune ##
                print('Pruning with {} for {} epochs.'.format(p, args.prune_epochs))
                pruner = load.pruner(p)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
                sparsity = 10**(-float(compression))
                prune_loop(model, loss, pruner, prune_loader, device, sparsity, 
                        args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)

                
                ## Post-Train ##
                print('Post-Training for {} epochs.'.format(args.post_epochs))
                post_train_start_time = timeit.default_timer()
                post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                            test_loader, device, args.post_epochs, args.verbose, train_sampler=train_sampler) 
                post_train_end_time = timeit.default_timer()
                print("Post Training time: {:.4f}s".format(post_train_end_time - post_train_start_time))

                ## Display Results ##
                frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
                train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
                prune_result = metrics.summary(model, 
                                            pruner.scores,
                                            metrics.flop(model, input_shape, device),
                                            lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
                total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
                possible_params = prune_result['size'].sum()
                total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
                possible_flops = prune_result['flops'].sum()
                print("Train results:\n", train_result)
                # print("Prune results:\n", prune_result)
                # print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
                # print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))
                
                ## recording testing time for task 2 ##
                # evaluating the model, including some data gathering overhead
                # eval(model, loss, test_loader, device, args.verbose)
                model.eval()
                start_time = timeit.default_timer()
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        temp_eval_out = model(data)
                end_time = timeit.default_timer()
                print("Testing time: {:.4f}s".format(end_time - start_time))

                fout.write('compression ratio: {} ::: pruner: {}'.format(compression, p))
                fout.write('Train results:\n {}\n'.format(train_result))
                fout.write('Prune results:\n {}\n'.format(prune_result))
                fout.write('Parameter Sparsity: {}/{} ({:.4f})\n'.format(total_params, possible_params, total_params / possible_params))
                fout.write("FLOP Sparsity: {}/{} ({:.4f})\n".format(total_flops, possible_flops, total_flops / possible_flops))
                fout.write("Testing time: {}s\n".format(end_time - start_time))
                fout.write("remaining weights: \n{}\n".format((prune_result['sparsity'] * prune_result['size'])))
                fout.write('flop each layer: {}\n'.format((prune_result['sparsity'] * prune_result['flops']).values.tolist()))
                ## Save Results and Model ##
                if args.save:
                    print('Saving results.')
                    if not os.path.exists('{}/{}'.format(args.result_dir, compression)):
                        os.makedirs('{}/{}'.format(args.result_dir, compression))
                    # pre_result.to_pickle("{}/{}/pre-train.pkl".format(args.result_dir, compression))
                    # post_result.to_pickle("{}/{}/post-train.pkl".format(args.result_dir, compression))
                    # prune_result.to_pickle("{}/{}/compression.pkl".format(args.result_dir, compression))
                    # torch.save(model.state_dict(), "{}/{}/model.pt".format(args.result_dir, compression))
                    # torch.save(optimizer.state_dict(),
                    #         "{}/{}/optimizer.pt".format(args.result_dir, compression))
                    # torch.save(scheduler.state_dict(),
                    #         "{}/{}/scheduler.pt".format(args.result_dir, compression))

    else:
        print('Staring Unpruned NN training')
        print('Training for {} epochs.'.format(args.post_epochs))
        model.load_state_dict(torch.load("{}/pre_train_model.pt".format(args.result_dir), map_location=device))
        optimizer.load_state_dict(torch.load("{}/pre_train_optimizer.pt".format(args.result_dir), map_location=device))
        scheduler.load_state_dict(torch.load("{}/pre_train_scheduler.pt".format(args.result_dir), map_location=device))

        train_start_time = timeit.default_timer()
        result = train_eval_loop(model, loss, optimizer, scheduler, train_loader,
                                 test_loader, device, args.post_epochs, args.verbose, train_sampler=train_sampler)
        train_end_time = timeit.default_timer()
        frames = [result.head(1), result.tail(1)]
        train_result = pd.concat(frames, keys=['Init.', 'Final'])
        print('Train results:\n', train_result)

    fout.close()

    dist.destroy_process_group()
