import hyperopt
from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
import math
import numpy as np
import argparse
import time
import datetime
import os
import sys
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from collections import OrderedDict

script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
try:
    import distiller
except ImportError:
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
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20_cifar',
                    choices=ALL_MODEL_NAMES,
                    help='model architecture: ' +
                    ' | '.join(ALL_MODEL_NAMES) +
                    ' (default: resnet20_cifar)')
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
parser.add_argument('--method', choices=['random','tpe'],
                      default='random', type=str,
                    help='search method, random or tpe (default: random)')
parser.add_argument('--pruner-constraint', '--con', action='store_true',
                    help='speedup to meet the constraint')

# Manual setting hyperparameters 
args = parser.parse_args()
print(args)

args.dataset = 'cifar10' if 'cifar' in args.arch else 'imagenet'
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

if(args.pruner_constraint == True):
    PrunerConstraint = True
else:
    PrunerConstraint = False

print('PrunerConstraint = {}'.format(PrunerConstraint))

PrunerEpoch = 0
Expected_Sparsity_Level_High = 72.5
Expected_Sparsity_Level_Low = 67.5
count = 0
global_min_score = 10.

def objective(space):
    global model
    global count
    global global_min_score
    
    #Explore new model
    model = create_model(False, args.dataset, args.arch, device_ids=args.gpus)
    count += 1
    # Objective function: F(Acc, Lat) = (1 - Acc.) + (alpha * Sparsity)
    accuracy = 0
    alpha = 0.3 # Super-parameter: the importance of inference time
    latency = 0.0
    sparsity = 0.0
    # Training hyperparameter

    if args.resume:
        model, compression_scheduler, start_epoch = apputils.load_checkpoint(
            model, chkpt_file=args.resume)
        print('resume mode: {}'.format(args.resume))

    print(global_min_score)
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
    #print(sparsity_levels)

    pruner = distiller.pruning.SparsityLevelParameterPruner(name='sensitivity', levels=sparsity_levels) # for SparsityLevelParameterPruner
    # pruner = distiller.pruning.SensitivityPruner(name='sensitivity', sensitivities=sparsity_levels) # for SensitivityPruner
    policy = distiller.PruningPolicy(pruner, pruner_args=None)
    lrpolicy = distiller.LRPolicy(torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1))
    compression_scheduler = distiller.CompressionScheduler(model)
    compression_scheduler.add_policy(policy, epochs=[PrunerEpoch])
    # compression_scheduler.add_policy(policy, starting_epoch=0, ending_epoch=38, frequency=2)
    compression_scheduler.add_policy(lrpolicy, starting_epoch=0, ending_epoch=50, frequency=1)
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
    
    local_min_score = 2.
    for i in range(args.epochs):
        compression_scheduler.on_epoch_begin(i)
        train_accuracy = train(i,criterion, optimizer, compression_scheduler)
        val_accuracy = validate() # Validate hyperparameter setting
        t, sparsity = distiller.weights_sparsity_tbl_summary(model, return_total_sparsity=True)
        compression_scheduler.on_epoch_end(i, optimizer)
        apputils.save_checkpoint(i, args.arch, model, optimizer, compression_scheduler, train_accuracy, False,
                                         'hyperopt', './')
        print('Epoch: {}, train_acc: {:.4f}, val_acc: {:.4f}, sparsity: {:.4f}'.format(i, train_accuracy, val_accuracy, sparsity))
        
        score = (1-(val_accuracy/100.)) + (alpha * (1-sparsity/100.)) # objective funtion here
        if(score < global_min_score):
            global_min_score = score
            apputils.save_checkpoint(i, args.arch, model, optimizer, compression_scheduler, train_accuracy, True, 'best', './')

        if(score < local_min_score):
            local_min_score = score

        if (PrunerConstraint == True and i >= PrunerEpoch and (sparsity < Expected_Sparsity_Level_Low or sparsity > Expected_Sparsity_Level_High)):
            break 

    test_accuracy = test() # Validate hyperparameter setting

    print('{} trials: score: {:.4f}, train_acc:{:.4f}, val_acc:{:.4f}, test_acc:{:.4f}, sparsity:{:.4f}'.format(count, 
                                      local_min_score, 
                                      train_accuracy, 
                                      val_accuracy, 
                                      test_accuracy,
                                      sparsity))

    return local_min_score

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

def validate():
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
    
def test():
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad():
        for test_step, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().data.numpy()
    accuracy = 100. * correct / total    
    return accuracy

def get_space():
    space = {}
    for name, parameter in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            if (PrunerConstraint == True):
                space[name] = hp.uniform(name, 0.55, 0.85)
            else:
                space[name] = hp.uniform(name, 0.30, 0.99)

    return space

def Test_Best_Model(fname):
    global model
    model = create_model(False, args.dataset, args.arch, device_ids=args.gpus)
    model, compression_scheduler, start_epoch = apputils.load_checkpoint(
            model, chkpt_file=fname)
    test_accuracy = test()
    print('test acc: {:.4f}'.format(test_accuracy))

def main():
    #print('TPE Constraint')
    #Test_Best_Model('./tpe_constraint/best_best.pth.tar')
    #print('Random Constraint')
    #Test_Best_Model('./rnd_constraint/best_best.pth.tar')
    #print('Random No Constraint')
    #Test_Best_Model('./rnd_no_con/best_best.pth.tar')
    #Test_Best_Model('./tpe_no_con/best_best.pth.tar')
    #exit(0)

    space = get_space()
    if(args.method == 'random'):
        print('random')
        best = fmin(objective, space, algo=hyperopt.rand.suggest, max_evals=args.rounds)
    elif(args.method == 'tpe'):
        print('tpe')
        best = fmin(objective, space, algo=tpe.suggest, max_evals=args.rounds)
    else:
        print('Error parameter on method: only \'random\' or \'tpe\' is allowed')

    print(best)

if __name__ == '__main__':
    main()

