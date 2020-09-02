import argparse
import os
import sys
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from dataloaders.kitti_loader import load_calib, oheight, owidth, input_options, KittiDepth
from model import DepthCompletionNet
from metrics import AverageMeter, Result
import criteria
import glob
import helper
from inverse_warp import Intrinsics, homography_from
sys.path.insert(1, os.path.join(sys.path[0], '..'))  # be able to import module from parent directory
from aux_functions import depth_write

# arguments parsing (everything is optional, because required!=True)
parser = argparse.ArgumentParser(description='Sparse-to-Dense')  # will hold all the information necessary to parse the command line into Python data types

parser.add_argument(  # fill with information about program arguments (how to take the strings on the command line and turn them into objects)
                    '-w', '--workers',  # linux convention, gives a possibility to use both: - a shortcut, -- the full name (a must)
                    default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=11, type=int, metavar='N', help='number of total epochs to run (default: 11)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-c', '--criterion', metavar='LOSS', default='l2', choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) + ' (default: l2)')
parser.add_argument('-b', '--batch-size', default=1, type=int, help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, metavar='LR', help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--data-folder', default='../data', type=str, metavar='PATH', help='data folder (default: none)')
parser.add_argument('-i', '--input', type=str, default='gd', choices=input_options, help='input: | '.join(input_options))
parser.add_argument('-l', '--layers', type=int, default=34, help='use 16 for sparse_conv; use 18 or 34 for resnet')
parser.add_argument('--pretrained', action="store_true", help='use ImageNet pre-trained weights')
parser.add_argument('--val', type=str, default="select", choices=["select", "full"], help='full or select validation set')
parser.add_argument('--jitter', type=float, default=0.1, help='color jitter for images')
parser.add_argument('--rank-metric', type=str, default='rmse', choices=[m for m in dir(Result()) if not m.startswith('_')],
                    help='metrics for which best result is sbatch_datacted')
parser.add_argument('-m', '--train-mode', type=str, default="dense", choices=["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
                    help='dense | sparse | photo | sparse+photo | dense+photo')
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
parser.add_argument('--cpu', action="store_true", help='run on cpu')
parser.add_argument('--partial-train', default="no", type=str, help='size of the whole train dataset (default: all), else, need to provide a txt file with the relevant drives')
parser.add_argument('--test', default="no", type=str, help='will use the test folder (default: no) during eval mode, instead of the val-full / select')
parser.add_argument('--save-images', action="store_true", help='if used, will save the depth images during inference. else, will just show error metrics.')


args = parser.parse_args()  # convert argument strings to objects and assign them as attributes of the namespace. Return the populated namespace

args.use_pose = ("photo" in args.train_mode)  # add attributes (more arguments), regardless the initial input
# args.pretrained = not args.no_pretrained

args.result = os.path.join('..', 'outputs/var_final_NN')
args.use_rgb = ('rgb' in args.input) or args.use_pose
args.use_d = 'd' in args.input  # sparse depth input (pixels without measured depth are set to zero)
args.use_g = 'g' in args.input  # input: gray images (instead of RGB)
if args.use_pose:  # relevant for self-supervised training framework
    args.w1, args.w2 = 0.1, 0.1  # for the photometric & smooth losses
else:
    args.w1, args.w2 = 0, 0

# handling GPU/CPU
cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("\n" + "=> using '{}' for computation.".format(device))

# define loss functions
depth_criterion = criteria.MaskedMSELoss() if (args.criterion == 'l2') else criteria.MaskedL1Loss()
photometric_criterion = criteria.PhotometricLoss()
smoothness_criterion = criteria.SmoothnessLoss()

if args.use_pose:
    # hard-coded KITTI camera intrinsics
    K = load_calib()
    fu, fv = float(K[0, 0]), float(K[1, 1])
    cu, cv = float(K[0, 2]), float(K[1, 2])
    kitti_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv)
    if cuda:
        kitti_intrinsics = kitti_intrinsics.cuda()


def iterate(mode, args, loader, model, optimizer, logger, epoch):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, epoch)
    else:
        model.eval()  # batchnorm or dropout layers will work in eval mode instead of training mode
        lr = 0

    for i, batch_data in enumerate(loader):  # batch_data keys: 'd' (depth), 'gt' (ground truth), 'g' (gray)
        start = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }

        gt = batch_data['gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - start

        start = time.time()

        pred = model(batch_data)
        if args.save_images: # save depth predictions
            pred_out_dir = max(glob.glob('../outputs/var_final_NN/var.test*'), key=os.path.getmtime) + '/dense_depth_images'
            pred1 = pred.cpu().detach().numpy()[:, 0, :, :]
            for im_idx, pred_im in enumerate(pred1):
                pred_out_dir1 = os.path.abspath(pred_out_dir)
                cur_path = os.path.abspath((loader.dataset.paths)['d'][i])
                basename = os.path.basename(cur_path)
                cur_dir = os.path.abspath(os.path.dirname(cur_path))
                cur_dir = cur_dir.split('var_final_NN/')[1]
                new_dir = os.path.abspath(pred_out_dir1 + '/' + cur_dir)
                new_path = os.path.abspath(new_dir + '/' + basename)
                if os.path.isdir(new_dir) == False:
                    os.makedirs(new_dir)

                depth_write(new_path, pred_im)

        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None
        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if 'sparse' in args.train_mode:
                depth_loss = depth_criterion(pred, batch_data['d'])
                mask = (batch_data['d'] < 1e-3).float()
            elif 'dense' in args.train_mode:
                depth_loss = depth_criterion(pred, gt)
                mask = (gt < 1e-3).float()

            # Loss 2: the self-supervised photometric loss
            if args.use_pose:
                # create multi-scale pyramids
                pred_array = helper.multiscale(pred)
                rgb_curr_array = helper.multiscale(batch_data['rgb'])
                rgb_near_array = helper.multiscale(batch_data['rgb_near'])
                if mask is not None:
                    mask_array = helper.multiscale(mask)
                num_scales = len(pred_array)

                # compute photometric loss at multiple scales
                for scale in range(len(pred_array)):
                    pred_ = pred_array[scale]
                    rgb_curr_ = rgb_curr_array[scale]
                    rgb_near_ = rgb_near_array[scale]
                    mask_ = None
                    if mask is not None:
                        mask_ = mask_array[scale]

                    # compute the corresponding intrinsic parameters
                    height_, width_ = pred_.size(2), pred_.size(3)
                    intrinsics_ = kitti_intrinsics.scale(height_, width_)

                    # inverse warp from a nearby frame to the current frame
                    warped_ = homography_from(rgb_near_, pred_, batch_data['r_mat'], batch_data['t_vec'], intrinsics_)
                    photometric_loss += photometric_criterion(rgb_curr_, warped_, mask_) * (2**(scale - num_scales))

            # Loss 3: the depth smoothness loss
            smooth_loss = smoothness_criterion(pred) if args.w2 > 0 else 0

            # backprop
            loss = depth_loss + args.w1 * photometric_loss + args.w2 * smooth_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        gpu_time = time.time() - start

        # measure accuracy and record loss of each batch
        with torch.no_grad():  # impacts the autograd engine and deactivate it (will reduce memory usage and speed up computations)
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()  # metrics
            if mode != 'test_prediction' and mode != 'test_completion':
                result.evaluate(pred.data, gt.data, photometric_loss)
            [
                m.update(result, gpu_time, data_time, mini_batch_size)
                for m in meters
            ]
            logger.conditional_print(mode, i, epoch, args.epochs, lr, len(loader), block_average_meter, average_meter)
            logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
            logger.conditional_save_pred(mode, i, pred, epoch)
        del pred

    avg = logger.conditional_save_info(mode, average_meter, epoch)  # take the avg of all the batches, to get the epoch metrics
    is_best = logger.rank_conditional_save_best(mode, avg, epoch, args.epochs)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best


def main():
    global args
    if args.partial_train == 'yes':  # train on a part of the whole train set
        print("Can't use partial train here. It is used only for test check. Exit...")
        return

    if args.test != "yes":
        print("This main should use only for testing, but test=yes wat not given. Exit...")
        return

    print("Evaluating test set with main_test:")
    whole_ts = time.time()
    checkpoint = None
    is_eval = False
    if args.evaluate:  # test a finished model
        args_new = args  # copies
        if os.path.isfile(args.evaluate):  # path is an existing regular file
            print("=> loading finished model from '{}' ... ".format(args.evaluate), end='')  # "end=''" disables the newline
            checkpoint = torch.load(args.evaluate, map_location=device)
            args = checkpoint['args']
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            args.save_images = args_new.save_images
            args.result = args_new.result
            is_eval = True
            print("Completed.")
        else:
            print("No model found at '{}'".format(args.evaluate))
            return
    elif args.resume:  # resume from a checkpoint
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint from '{}' ... ".format(args.resume), end='')
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer ... ", end='')
    model = DepthCompletionNet(args).to(device)
    model_named_params = [p
                          for _, p in model.named_parameters()  # "_, p" is a direct analogy to an assignment statement k, _ = (0, 1). Unpack a tuple object
                          if p.requires_grad]
    optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay)
    print("completed.")
    [f'{k:<20}: {v}' for k, v in model.__dict__.items()]

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    model = torch.nn.DataParallel(model)  # make the model run parallelly: splits your data automatically and sends job orders to multiple models on several GPUs.
                                          # After each model finishes their job, DataParallel collects and merges the results before returning it to you

    # data loading code
    print("=> creating data loaders ... ")
    if not is_eval:  # we're not evaluating
        train_dataset = KittiDepth('train', args)  # get the paths for the files
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, sampler=None)  # load them
        print("\t==> train_loader size:{}".format(len(train_loader)))
    

    if args_new.test == "yes":  # will take the data from the "test" folders
        val_dataset = KittiDepth('test', args)
        is_test = 'yes'
    else:
        val_dataset = KittiDepth('val', args)
        is_test = 'no'
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    # create backups and results folder
    logger = helper.logger(args, is_test)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")  # logger records sequential data to a log file

    # main code - run the NN
    if is_eval:
        print("=> starting model evaluation ...")
        result, is_best = iterate("val", args, val_loader, model, None, logger, checkpoint['epoch'])
        return

    print("=> starting model training ...")
    for epoch in range(args.start_epoch, args.epochs):
        print("=> start training epoch {}".format(epoch) + "/{}..".format(args.epochs))
        train_ts = time.time()
        iterate("train", args, train_loader, model, optimizer, logger, epoch)  # train for one epoch
        result, is_best = iterate("val", args, val_loader, model, None, logger, epoch)  # evaluate on validation set
        helper.save_checkpoint({  # save checkpoint
            'epoch': epoch,
            'model': model.module.state_dict(),
            'best_result': logger.best_result,
            'optimizer': optimizer.state_dict(),
            'args': args,
        }, is_best, epoch, logger.output_directory)
        print("finish training epoch {}, time elapsed {:.2f} hours, \n".format(epoch, (time.time() - train_ts) / 3600))
    last_checkpoint = os.path.join(logger.output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')  # delete last checkpoint because we have the best_model and we dont need it
    os.remove(last_checkpoint)
    print("finished model training, time elapsed {0:.2f} hours, \n".format((time.time() - whole_ts) / 3600))


if __name__ == '__main__':
    main()
