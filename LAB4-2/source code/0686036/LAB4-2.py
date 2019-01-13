from hyperopt import hp, fmin, tpe, rand
import math
import numpy as np
import argparse
import time
import datetime
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
try:
    import distiller
except ImportError:
    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
    sys.path.append(module_path)
    import distiller
import apputils
from models import ALL_MODEL_NAMES, create_model

def float_range(val_str):
    val = float(val_str)
    if val < 0 or val >= 1:
        raise argparse.ArgumentTypeError('Must be >= 0 and < 1 (received {0})'.format(val_str))
    return val

parser = argparse.ArgumentParser(description='Distiller image classification model compression')
parser.add_argument('--data', metavar='DIR', default='../../../data.cifar10/', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56_cifar',
                    choices=ALL_MODEL_NAMES,
                    help='model architecture: ' +
                    ' | '.join(ALL_MODEL_NAMES) +
                    ' (default: resnet56_cifar)')
parser.add_argument('-r', '--rounds', default=10, type=int,
                    metavar='R', help='max rounds (default: 10)')
parser.add_argument('--epochs', default=120, type=int,
                    metavar='E', help='epochs (default: 120)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                    help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--validation-size', '--vs', type=float_range, default=0.1,
                    help='Portion of training dataset to set aside for validation')
parser.add_argument('--deterministic', '--det', action='store_true',
                    help='Ensure deterministic execution for re-producible results.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Manual setting hyperparameters 
args = parser.parse_args()
args.dataset = 'cifar10'
if args.gpus is not None:
    try:
        args.gpus = [int(s) for s in args.gpus.split(',')]
    except ValueError:
        exit(1)
    available_gpus = torch.cuda.device_count()
    for dev_id in args.gpus:
        if dev_id >= available_gpus:
            exit(1)
    # Set default device in case the first one on the list != 0
    torch.cuda.set_device(args.gpus[0])

model = create_model(False, args.dataset, args.arch, device_ids=args.gpus) # Get arch state_dict
train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_size, args.deterministic)

count = 0

model_best = create_model(False, args.dataset, args.arch, device_ids=args.gpus)
best_dict = {'trial':0, 'score':100, 'tr_acc':0., 'v_acc':0., 'te_acc':0., 'sparsity':0., 'model_best':model_best}

def objective(space):
    global model
    global count
    global best_dict
    
    #Explore new model
    model = create_model(False, args.dataset, args.arch, device_ids=args.gpus)
    if args.resume:
        model, _, _ = apputils.load_checkpoint(
            model, chkpt_file=args.resume)
    
    count += 1
    print('{} trial starting...'.format(count))
    # Objective function: F(Acc, Lat) = (1 - Acc.) + (alpha * Sparsity)
    accuracy = 0
    #alpha = 0.2 # Super-parameter: the importance of inference time
    alpha = 1.0 # Super-parameter: the importance of inference time
    sparsity = 0.0
    # Training hyperparameter
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    """
    distiller/distiller/config.py
        # Element-wise sparsity
        sparsity_levels = {net_param: sparsity_level}
        pruner = distiller.pruning.SparsityLevelParameterPruner(name='sensitivity', levels=sparsity_levels)
        policy = distiller.PruningPolicy(pruner, pruner_args=None)
        scheduler = distiller.CompressionScheduler(model)
        scheduler.add_policy(policy, epochs=[0, 2, 4])
        # Local search 
        add multiple pruner for each layer
    """
    sparsity_levels = {}
    for key, value in space.items():
        sparsity_levels[key] = value
    pruner = distiller.pruning.SparsityLevelParameterPruner(name='sensitivity', levels=sparsity_levels)
    policy = distiller.PruningPolicy(pruner, pruner_args=None)
    lrpolicy = distiller.LRPolicy(torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1))
    compression_scheduler = distiller.CompressionScheduler(model)
    #compression_scheduler.add_policy(policy, epochs=[90])
    compression_scheduler.add_policy(policy, epochs=[0])
    compression_scheduler.add_policy(lrpolicy, starting_epoch=0, ending_epoch=90, frequency=1)
    """
    distiller/example/classifier_compression/compress_classifier.py
    For each epoch:
        compression_scheduler.on_epoch_begin(epoch)
        train()
        save_checkpoint()
        compression_scheduler.on_epoch_end(epoch)

    train():
        For each training step:
            compression_scheduler.on_minibatch_begin(epoch)
            output = model(input)
            loss = criterion(output, target)
            compression_scheduler.before_backward_pass(epoch)
            loss.backward()
            optimizer.step()
            compression_scheduler.on_minibatch_end(epoch)
    """
    for i in range(args.epochs):
        compression_scheduler.on_epoch_begin(i)
        train_accuracy = train(i,criterion, optimizer, compression_scheduler)
        val_accuracy = validate() # Validate hyperparameter setting
        t, sparsity = distiller.weights_sparsity_tbl_summary(model, return_total_sparsity=True)
        compression_scheduler.on_epoch_end(i, optimizer)
        apputils.save_checkpoint(i, args.arch, model, optimizer, compression_scheduler, train_accuracy, False,
                                         'hyperopt', './')
        print('{} epochs => train acc:{:.2f}%,  val acc:{:.2f}%'.format(i, train_accuracy, val_accuracy))
        
    test_accuracy = validate(test_loader) # Validate hyperparameter setting
    #score = (1-(val_accuracy/100.)) + (alpha * (1-sparsity/100.)) # objective funtion here
    
    # objective funtion here
    # accuracy: 98~90%, sparsity: 80%~50%
    score = -((val_accuracy/100.)**2-0.9**2 + alpha * ((sparsity/100.)**2-0.5**2)) 
    print('{} trials: score: {:.2f}\ttrain acc:{:.2f}%\tval acc:{:.2f}%\ttest acc:{:.2f}%\tsparsity:{:.2f}%'.format(count, 
                                      score, 
                                      train_accuracy, 
                                      val_accuracy, 
                                      test_accuracy,
                                      sparsity))
    if score < best_dict['score']:
        best_dict['trial'] = count
        best_dict['score'] = score
        best_dict['tr_acc'] = train_accuracy        
        best_dict['v_acc'] = val_accuracy
        best_dict['te_acc'] = test_accuracy
        best_dict['sparsity'] = sparsity
        best_dict['model_best'] = copy.deepcopy(model)

    return score


def train(epoch, criterion, optimizer, compression_scheduler):
    correct = 0
    total = 0
    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    for train_step, (inputs, targets) in enumerate(train_loader):
        compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.numpy()
        loss = criterion(outputs, targets)
        compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                   optimizer=optimizer, return_loss_components=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)
    accuracy = 100. * correct / total    
    return accuracy


def validate(dataloader = val_loader):
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad():
        for test_step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().data.numpy()
    accuracy = 100. * correct / total    
    return accuracy
    

def get_space(min_v, max_v):
    space = {}
    for name, parameter in model.named_parameters():
        if 'conv' in name and 'weight' in name or 'fc' in name and 'weight' in name:
            space[name] = hp.uniform(name, min_v, max_v)
    return space

def main(min_v=0.01, max_v=0.99, algo=tpe.suggest):
#if __name__ == '__main__':
    global count
    count = 0
    global model_best
    model_best = create_model(False, args.dataset, args.arch, device_ids=args.gpus)
    global best_dict
    best_dict = {'trial':0, 'score':100, 'tr_acc':0., 'v_acc':0., 'te_acc':0., 'sparsity':0., 'model_best':model_best}

    space = get_space(min_v, max_v)
    best = fmin(objective, space, algo=algo, max_evals=args.rounds)
    print('===========================================')
    print('The best accuracy and sparsity is occurred by:')
    print('    trials    :{}'.format(best_dict['trial']))
    print('    score     :{:.4f}'.format(best_dict['score'])) 
    print('    train acc :{:.2f}%'.format(best_dict['tr_acc'])) 
    print('    val acc   :{:.2f}%'.format(best_dict['v_acc']))
    print('    test acc  :{:.2f}%'.format(best_dict['te_acc']))
    print('    sparsity  :{:.2f}%'.format(best_dict['sparsity']))
    for element in best:
        print('        {:30}: {:.4f}'.format(element, best[element]))
    
    

        
if __name__ == '__main__':
    #Tpe + constraint
    main(0.7, 0.99, tpe.suggest)
    
    #Random + constraint
    main(0.7, 0.99, rand.suggest)
    
    #Tpe + no constraint
    main(0.1, 0.99, tpe.suggest)
    
    #Random + no constraint
    main(0.1, 0.99, rand.suggest)
    
    