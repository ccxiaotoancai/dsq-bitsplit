import argparse
import os
import numpy
import random
import shutil
import time
import warnings
import numpy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models



from torch.utils.tensorboard import SummaryWriter

from PyTransformer.transformers.torchTransformer import TorchTransformer
from PyTransformer.transformers.quantize import QConv2d, QuantConv2d, QLinear, ReLUQuant
#from DSQConv_split import DSQConv_split
from DSQLinear import DSQLinear
from DSQConv_int8_int4 import DSQConv_int8_int4


os.environ['TORCH_HOME'] = 'models'
#writer = None
#writer=SummaryWriter("log")
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',default="/dataset/public/ImageNetOrigin/",
                   help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')#在验证集上
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')#节点数量
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:54078', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=True ,#,action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--log_path', default='log', type=str,
                    help='log for tensorboardX.')#改为了tensorboard

parser.add_argument('-q', '--quantize', default="DSQ", type=str,
                    help='quantization type.')

parser.add_argument('--quantize_input', dest='quantize_input', action='store_true',                    
                    help='quantization input.')

# parser.add_argument('--quan_bit', default=8, type=int,
#                     help='quantization bit num.')#原代码中默认w和inputbit数一样

"分别设置w和input比特数"
parser.add_argument('--quan_bit_w', default=4, type=int,
                    help='quantization bit_w num.')
parser.add_argument('--quan_bit_input', default=4, type=int,
                     help='quantization bit_input num.')

#判断是在每个block中哪一个conv的input进行split，set_quanbit_input函数中的a参数
# parser.add_argument('--split_conv_input', default=8, type=int,
#                      help='which conv_input of each  block in resnet')


best_acc1 = 0

def main():
    args = parser.parse_args()

    

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    #global writer
    


    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.quantize is not None:
        assert args.quantize in ["uniform", "DSQ"]

    ngpus_per_node = torch.cuda.device_count()#
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu




    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # https://blog.csdn.net/lbling123/article/details/117286080
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()#载入pytorch的model_zoo

    
    if args.quantize is not None:        
        transformer = TorchTransformer()
        if args.quantize == "uniform":		    
            print("Using Uniform Quantization...")     
            print(args.quantize_input)       
            if args.quantize_input:
                print("Quaintization Input !!!")
                transformer.register(nn.Conv2d, QuantConv2d)	    
            else:
                print("No Quaintization Input !!!")
                transformer.register(nn.Conv2d, QConv2d)	               
            
        else:
            print("Using  DSQConv_int8_int4 ...")
            transformer.register(nn.Conv2d, DSQConv_int8_int4) #改变注册层的名字
            # transformer.register(nn.Linear, DSQLinear)#将线性结构转变为DSQLinear
        # transformer.register(nn.ReLU, ReLUQuant)
        model = transformer.trans_layers(model)#改变层结构
        
        # set quan bit
        # current use num_bit   
        print("Setting target quanBit_w to {} bit".format(args.quan_bit_w))
        #model = set_quanbit(model, args.quan_bit)#设置比特数
        model=set_quanbit_w(model, args.quan_bit_w)#设置比特数
        #model,_=set_quanbit_input(model, args.quan_bit_input)
        print("Setting Quantization Input : {} ".format(args.quantize_input))
        model = set_quanInput(model, args.quantize_input)
        if args.quantize_input:

            model, _ = set_quanbit_input(model, args.quan_bit_input)

    
    # log_alpha(model)
    # print(model)
    # sys.exit()    
    

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:#没有多线程，只指定gpu则只跑一个gpu
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:#多线程和gpu都不指定则用dataparallel
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')

    # files_train=os.listdir(traindir)
    # for filename in files_train:
    #     portion= os.path.splitext(filename)
    #     if portion[1]==".JPEG":
    #         newname=portion[0]+".jpg"
    #         os.rename(filename,newname)

    valdir = os.path.join(args.data, 'val2')
    #files_val= os.listdir(traindir)

    # for filename in files_val:
    #     portion= os.path.splitext(filename)
    #     if portion[1]==".JPEG":
    #         newname=portion[0]+".jpg"
    #         os.rename(filename,newname)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir,
            transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args, None ,0)
        return   
    if args.rank % ngpus_per_node == 0 :
        writer = SummaryWriter(args.log_path)
    else:
        writer=None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)  #每过30epoch学习率下降0.1倍

        # train for one epoch
        #if args.multiprocessing_distributed and args.rank % ngpus_per_node == 0 :
           # torch.distributed.barrier()
        train(train_loader, model, criterion, optimizer, epoch, args,writer)
        #if args.multiprocessing_distribute  and args.rank % ngpus_per_node == 0 :
            #torch.distributed.barrier()#使各个进程同步
        # evaluate on validation set
        #这是验证集，所以这里需要每一代都验证一下
        acc1 = validate(val_loader, model, criterion, args, epoch,writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)       #测试数据中最好的，防止过拟合
        if writer is not None:
            writer.add_scalar("Best val Acc1", best_acc1, epoch) #把每个 epoch 数值中最好的准确率作为这个epoch的准确率

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):  #在local_rank=0的地方进行模型的保存
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, save_path=args.log_path)

def set_quanbit(model, quan_bit = 8):   #设置比特数
    
    for module_name in model._modules:
        if len(model._modules[module_name]._modules) > 0:#递归至基础模型
            set_quanbit(model._modules[module_name], quan_bit)
        else:
            if hasattr(model._modules[module_name], "num_bit"):#对基础模型进行赋值
                setattr(model._modules[module_name], "num_bit", quan_bit) 
    return model

"*********************************设置input和weight不一致的情况**************************"
def set_quanbit_w(model, quan_bit_w=8):  # 设置比特数

    for module_name in model._modules:
        if len(model._modules[module_name]._modules) > 0:  # 递归至基础模型
            set_quanbit_w(model._modules[module_name], quan_bit_w)
        else:
            if hasattr(model._modules[module_name], "num_bit_w"):  # 对基础模型进行赋值
                setattr(model._modules[module_name], "num_bit_w", quan_bit_w)
    return model

#a=0 后面可以放到argparses中进行
def set_quanbit_input(model, quan_bit_input=8,a=0):  #当a设置为0时其实就是对每一个block的第一个卷积的输出做比特拆分,当a设置为1时就是对每一个block第二个卷积的输出做bit拆分

    for module_name in model._modules:
        if len(model._modules[module_name]._modules) > 0:  # 递归至基础模型

            _,a=set_quanbit_input(model._modules[module_name], quan_bit_input,a)
        else:
            if hasattr(model._modules[module_name], "num_bit_input"):  # 对基础模型进行赋值
                #setattr(model._modules[module_name], "num_bit_input", quan_bit_input)
                #这里就是设置第二个卷积层的输入(第一个卷积层的)为2n bit的
                a+=1#判断这是第几个输出
                if a%2==0:
                    print("set bit_chage {}".format(a))
                    setattr(model._modules[module_name], "num_bit_input",quan_bit_input*2)
                    setattr(model._modules[module_name], "flag_split", True)
                else:
                    setattr(model._modules[module_name], "num_bit_input", quan_bit_input)
    return model,a
"****************************************************************************************"


#设置是否需要对input进行量化
def set_quanInput(model, quan_input = True):

    for module_name in model._modules:        
        if len(model._modules[module_name]._modules) > 0:
            set_quanInput(model._modules[module_name], quan_input)
        else:
            # for DSQ# 递归至了基础模块如conv2d
            if hasattr(model._modules[module_name], "quan_input"):
                setattr(model._modules[module_name], "quan_input", quan_input)

    return model

def log_alpha(model, epoch, writer,index = 0):#画图不管
    for module_name in model._modules: #会按层次输出层直到输出最到向nn.conv2d这种基础层（因为_modules中有sequential）
        if len(model._modules[module_name]._modules) > 0:
            log_alpha(model._modules[module_name], index,writer)
        else:           
            
            # DSQ          
            if hasattr(model._modules[module_name], "alphaW"):
                if writer is not None:
                    writer.add_scalar("{}_{}_weight".format(module_name, index), getattr(model._modules[module_name], "alphaW"), epoch)
                    print("{}_{}_weight : {}".format(module_name, index, getattr(model._modules[module_name], "alphaW")))

            if hasattr(model._modules[module_name], "alphaA"):
                if writer is not None:
                    writer.add_scalar("{}_{}_activation".format(module_name, index), getattr(model._modules[module_name], "alphaA"), epoch)
                    print("{}_{}_activation : {}".format(module_name, index, getattr(model._modules[module_name], "alphaA")))
            
            index = index +1
            
    return index
                

def train(train_loader, model, criterion, optimizer, epoch, args,writer):
    #if writer is not None:
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)#处理数据的时间

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #if writer is not None:
        if i % args.print_freq == 0:
            progress.display(i)
            remain_epoch = args.epochs - epoch
            remain_iters = remain_epoch * len(train_loader) + (len(train_loader) - i)
            remain_seconds = remain_iters * batch_time.get_avg()
            seconds = (remain_seconds//1) % 60
            minutes = (remain_seconds//(1*60)) % 60
            hours = (remain_seconds//(1*60*60)) % 24
            days = (remain_seconds//(1*60*60*24))
            time_stamp = ""
            if (days > 0): time_stamp += "{} days, ".format(days)
            if (hours > 0) : time_stamp += "{} hr, ".format(hours)
            if (minutes > 0) : time_stamp += "{} min, ".format(minutes)
            if (seconds > 0) : time_stamp += "{} sec, ".format(seconds)
            print(">>>>>>>>>>>> Remaining Times: {}  <<<<<<<<<<<<<<<<<".format(time_stamp) )
        # if i % 100 == 0:
        #     print("PPPPPPPPPPPPPPPPPPPPP")
        #     with torch.no_grad():
        #         log_alpha(model)
            #writer.add_scalar("Train Loss", loss.item(), epoch)
    if writer is not None:
        writer.add_scalar("Train Loss", loss.item(), epoch)
        writer.add_scalar("Train Acc1", top1.avg, epoch)
        writer.add_scalar("Train Acc5", top5.avg, epoch)


def validate(val_loader, model, criterion, args, epoch,writer):
    # if  writer is not None:#rank0打印
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        # log alpha
        log_alpha(model,epoch,writer)
        
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        # if writer is not None:#rank 0 打印
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
        if writer is not None:
            writer.add_scalar("Val Loss", loss.item(), epoch)
            writer.add_scalar("Val Acc1", top1.avg, epoch)
            writer.add_scalar("Val Acc5", top5.avg, epoch)

        

    return top1.avg


def save_checkpoint(state, is_best, save_path):
    filename = os.path.join(save_path ,'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def get_avg(self):
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
