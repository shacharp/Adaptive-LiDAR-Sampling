import argparse
import os
import time
import torch.utils.data
import glob
import sys
import shutil
from pathlib import Path
import globals
from dataloaders.kitti_loader import KittiDepth
from model import DepthCompletionNet
from aux_functions import create_dir_tree_skeleton, copy_partial_train_val, create_custom_depth_maps, copy_partial_val_selection_cropped, copypath, divide_drives_to_mini_sets, \
    depth_write, apply_empty_d_maps_to_data, read_weights

# parsing arguments
parser = argparse.ArgumentParser(description='## Adaptive-LiDAR-sampling ##')  # will hold all the information necessary to parse the command line into Python data types
parser.add_argument('-B', '--budget', default=1024, type=int, help='Total samples to be generated for the final sparse depth scheme. Default: 1024.')
parser.add_argument('-K', '--phases', default=4, type=int, help='Number of phases. Default: 4.')
parser.add_argument('-M', '--predictors', default=5, type=int, help='Number of predictors (the ensemble). Default: 5.')
parser.add_argument('--epochs', default=7, type=int, help='Number of training epochs. Default: 7.')
parser.add_argument('--layers', type=int, default=18, choices=[18, 34, 50, 101, 152], help='For resnet (we tested only 18 and 34). Default: 18.')
parser.add_argument('--train-bs', default=4, type=int, help='Training batch size. Default: 4.')
parser.add_argument('--pred-bs', default=4, type=int, help='Predicting batch size during the phases. Better be higher as possible. Default: 4.')
parser.add_argument('--miniset-size', default=2000, type=int, help='Will try dividing the train set to M minisets of equal (+-) size. This number will be the minimum size of each mini-set, '
                                                                   'so the total train set will be ~(M*miniset-size). Default: 2000.')
parser.add_argument('--big-portion', default="False", type=str, choices=["True", "False"], help='Choose True for the whole train set/a big portion of it, otherwise False for small mini-sets. '
                                                                                                  'This argument relates to the miniset-size argument. Default: False.')
parser.add_argument('--val-size', default=203, type=int, help='Number of val_select images (out of 1000). Default: 203.')
parser.add_argument('--samp-method', type=str, default="pm", choices=["pm", "greedy"], help='Desired sample strategy from the variance map. Probability matching or MAX. Default: pm.')
parser.add_argument('--just-d', action="store_true", help='Use only d as input. For the first phase, using empty depth maps. Default: None.')
parser.add_argument('--allrgbd', action="store_true", help='Use d with empty depth maps (in addition to rgb) for the first phase. Else, only rgb for the first one. Default: None.')
parser.add_argument('--inference', type=str, default="", help='Full path to the text file with the weights in the right format. Default: None.')
parser.add_argument('--rank-metric', type=str, default="rmse", choices=["rmse", "mae"], help='Will save the best weights after each training epoch, based on this parameter. Default: rmse.')

arguments = parser.parse_args()

# handling GPU/CPU
cuda = torch.cuda.is_available()
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("\n" + "=> using '{}' for computation.".format(device) + "\n")

# save predictions
def predict(curr_args, loader, curr_model):
    curr_model.eval()

    for i, batch_data in enumerate(loader):
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }

        pred = curr_model(batch_data)
        pred_out_dir = curr_args.pred_dir

        if pred_out_dir is not '':
            pred1 = pred.cpu().detach().numpy()[:, 0, :, :]
            for im_idx, pred_im in enumerate(pred1):
                data_folder1 = os.path.abspath(curr_args.data_folder)
                pred_out_dir1 = os.path.abspath(pred_out_dir)
                cur_path = os.path.abspath((loader.dataset.paths)['d'][(curr_args.batch_size * i) + im_idx])
                basename = os.path.basename(cur_path)
                cur_dir = os.path.abspath(os.path.dirname(cur_path))
                cur_dir = cur_dir[len(data_folder1):]
                new_dir = os.path.abspath(pred_out_dir1 + '/' + cur_dir)
                new_path = os.path.abspath(new_dir + '/' + basename)
                if os.path.isdir(new_dir) == False:
                    os.makedirs(new_dir)

                depth_write(new_path, pred_im)

            del pred  # helps run with a bigger batch size
            continue


if __name__ == '__main__':
    global args

    whole_run = time.time()
    if arguments.inference == "":
        print("\n" + "#@#@# Full algorithm run #@#@#\n")
    else:
        print("\n" + "#@#@# Inferencing test set, based on given weights #@#@#\n")

    print("chosen hyper-parameters for this run:")
    print(arguments)
    print("\n")

    # hyper parameters #
    M = arguments.predictors  # number of mini-sets / NNs
    globals.B = arguments.budget
    globals.K = arguments.phases  # there are K phases to accumulate the desired 'budget' + a final train with 1 NN (var_final_nn) on the whole dataset with the 'budget'
    val_select_desired_images = arguments.val_size
    epochs = arguments.epochs
    train_batch_size = arguments.train_bs
    predict_batch_size = arguments.pred_bs
    rank_metric = arguments.rank_metric
    criterion = 'l1' if rank_metric == 'mae' else 'l2'
    layers = arguments.layers
    sample_method = arguments.samp_method
    samples_between_phases = round(globals.B / globals.K)
    d_only = arguments.just_d
    first_phase_rgbd = arguments.allrgbd or d_only  # if true, first phase training will be rgb + d with black (empty) sample maps, and not rgb only
    existing_weights = read_weights(arguments.inference, globals.K, M)

    assert M > 2, "M <= 2, which means no var analysis is possible"
    if globals.B % globals.K != 0:
        print("Budget/phases is {:.2f} which isn't an int, rounding sampling points per phase to be {}. Meaning, we will have effective total budget of {} instead of {}"
                                                                                    .format(globals.B/globals.K, round(globals.B/globals.K), round(globals.B/globals.K)*globals.K, globals.B))

    # initial phase: divide the training & val data into M mini sets #
    print("@@@ DIVIDING train set into {} mini-sets & copy the val_select set {}@@@ \n".format(M, "" if len(existing_weights) == 0 else "- WON'T be used "))
    divide_time = time.time()
    data_mini_sets = list(range(1, M + 1))

    old_data_folders = glob.glob('../data_new/phase*')  # remove older data so we could make the new one (otherwise it will fail)
    for f in old_data_folders:
        shutil.rmtree(f)
    if glob.glob('../data_new/var_final_NN') != []:  # remove older data
        shutil.rmtree('../data_new/var_final_NN')

    for m in range(1, M+1):
        create_dir_tree_skeleton('../data_new/phase_1/mini_set_' + str(m), '../data')

    (sets, sizes, _) = divide_drives_to_mini_sets(M, arguments.miniset_size, '../data/', 'train', arguments.big_portion == "True")
    assert len(sets) >= M, "There are not enough train drives to make M mini-sets"
    print("Total train size is: {}".format(sizes.sum()))  # based on data_depth_annotated folder (we'll have a little bit more images in rgb_data but we aren't using them)

    with open('chosen_train_mini_sets.txt', 'w') as f:  # save the mini-sets names
        for _list in sets:
            for _set in _list:
                f.write(_set + '\n')

    # create desired train & val_select & test sets #
    for m in range(1, M+1):  # fill train folders (inside the mini-sets) - copy relevant rgb, gt, vel
        copy_partial_train_val('../data/', '../data_new/phase_1/mini_set_' + str(m), sets[m-1], 'train')
        print("Train folder {}/{} contains {} images.".format(m, M, len(list(Path('../data_new/phase_1/mini_set_' + str(m) + '/data_depth_annotated').rglob('*.png')))))

    copy_partial_val_selection_cropped('../data/', '../data_new/phase_1/mini_set_1/', [0, val_select_desired_images])  # fill val_select folder - will be only in mini-set 1 because shared to all
    print("Val_select folder contains {} images.".format(len(os.listdir('../data_new/phase_1/mini_set_1/depth_selection/val_selection_cropped/velodyne_raw/'))))

    copypath('../data/data_rgb/test/', '../data_new/phase_1/mini_set_1/data_rgb/test/', True)  # fill test folder - will be only in mini-set 1 because shared to all
    copypath('../data/data_depth_annotated/test/', '../data_new/phase_1/mini_set_1/data_depth_annotated/test/', True)
    copypath('../data/data_depth_velodyne/test/', '../data_new/phase_1/mini_set_1/data_depth_velodyne/test/', True)
    print("Test folder contains {} images.".format(len(list(Path('../data_new/phase_1/mini_set_1/data_depth_annotated/test/').rglob('*.png')))))
    print("Preparing sets finished, time elapsed {:.2f} minutes \n".format((time.time() - divide_time) / 60))

    if first_phase_rgbd == True:
        print("Preparing black (empty) depth maps tor first phase input")
        for m in range(1, M + 1):
            apply_empty_d_maps_to_data('../data_new/phase_1/mini_set_' + str(m) + '/')  # overwrite the velodyne_raw with black images

    # (a) Generating an Adaptive Sampling Pattern in K Phases #
    for phase in range(1, globals.K + 1):
        print("@@@ START PHASE: {}/{} @@@ \n".format(phase, globals.K))
        phase_time = time.time()
        torch.cuda.empty_cache()

        if d_only:  # d all the way
            input_type = 'd'
        elif phase == 1:
            if first_phase_rgbd == True:
                input_type = 'rgbd'
            else:
                input_type = 'rgb'
        else:  # rgbd in phases > 1
            input_type = 'rgbd'

        # train the different M NNs #
        if len(existing_weights) == 0:  # not only inferencing
            whole_train = time.time()
            for m in range(1, M+1):
                print("@@@ TRAIN NN {}/{} on mini-set {} @@@".format(m, M, m))
                train_ts = time.time()
                set_num = data_mini_sets[m - 1]
                data_folder = '../data_new/phase_' + str(phase) + '/mini_set_' + str(set_num)
                result = os.path.join('../outputs/NN' + str(set_num) + '/phase_' + str(phase))

                exit_code = os.system('python black_box_main.py --train-mode dense --layers ' + str(layers) + ' -b ' + str(train_batch_size) + ' --input ' + input_type + ' --pretrained --epochs ' +
                                str(epochs) + ' --data-folder ' + data_folder + ' --result ' + result + ' --sample_method ' + sample_method + ' --samples_per_samples_batch ' +
                                str(samples_between_phases) + ' --budget ' + str(globals.B) + ' --criterion ' + criterion + ' --rank-metric ' + rank_metric)  # the "Black-Box" algorithm
                if exit_code > 0:
                    print("Error {} while running 'python black_box_main.py'. Exiting...".format(exit_code), file=sys.stderr)
                    exit(exit_code)

                print("=> phase {0}/{1}: training model {2}/{3} has finished, time elapsed {4:.2f} hours \n".format(phase, globals.K, m, M, (time.time() - train_ts) / 3600))
            print("Training finished, time elapsed {:.2f} hours \n".format((time.time() - whole_train) / 3600))

        # predict & create samples the data #
        print("@@@ START GENERATING samples patterns @@@\n")
        generating_samples = time.time()

        NNs_weights = []
        for m in range(1, M + 1):  # the NNs we just trained
            if len(existing_weights) != 0:  # inferencing based on given weights
                NNs_weights.append(existing_weights[phase - 1][m - 1])
            else:
                latest_nn_full_name = max(glob.glob(os.path.join('../outputs/NN' + str(m) + '/phase_' + str(phase), '*/')), key=os.path.getmtime)
                NNs_weights.append(latest_nn_full_name + 'model_best.pth.tar')

        NN_arguments = []
        models = []
        for i, current_NN in enumerate(NNs_weights):  # relevant code from the 'Black-Box' models (to use the NNs for the prediction)
            checkpoint = None
            if os.path.isfile(current_NN):
                print("=> loading checkpoint '{}' ... ".format(current_NN), end='')
                checkpoint = torch.load(current_NN, map_location=device)
                args = checkpoint['args']
                is_eval = True
                print("Completed.")
            else:
                assert False, ("No model found at '{}'".format(current_NN))

            model = DepthCompletionNet(args).to(device)
            model_named_params = [p for _, p in model.named_parameters() if p.requires_grad]
            model.load_state_dict(checkpoint['model'])
            model = torch.nn.DataParallel(model)

            NN_arguments.append(args)
            models.append(model)

        # predict & create samples for the training sets #
        if len(existing_weights) == 0:  # not only inferencing
            for set_num in range(1, M+1):
                print("\nSTART PREDICTING train set num: {} for next phase\n".format(set_num))
                pred_samp_train_set_time = time.time()
                create_dir_tree_skeleton('../data_new/phase_' + str(phase+1) + '/mini_set_' + str(set_num), '../data')

                # get next phase's samples map by accumulating samples #
                sample_batch_time = time.time()
                for i, current_NN in enumerate(NNs_weights):  # get predictions of M-1 NNs on one mini-set. Every prediction is based on all of the last phase's sample points
                    if (i+1) == set_num:  # we don't want to predict a mini-set we were trained on
                        continue
                    else:  # predicting on the current phase (using RGB + last phase d), saving in the next phase tmp-s folders (to create the next velodyne from it)
                        print("\npredicting mini-set {} with NN{}".format(set_num, i+1))
                        input_t = input_type
                        data_in = '../data_new/phase_' + str(phase) + '/mini_set_' + str(set_num)
                        pred_dir = '../data_new/phase_' + str(phase + 1) + '/mini_set_' + str(set_num) + '/predictions_tmp/NN' + str(i + 1)

                        NN_arguments[i].data_folder = data_in
                        NN_arguments[i].pred_dir = pred_dir
                        NN_arguments[i].val = 'full'
                        NN_arguments[i].use_d = 'd' in input_t
                        NN_arguments[i].batch_size = predict_batch_size

                        train_dataset = KittiDepth('val', NN_arguments[i])  # we adjusted 'val-full' option for predicting on the train data
                        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=NN_arguments[i].batch_size, shuffle=False, num_workers=2, pin_memory=True)
                        print("\t==> train_loader size:{}".format(len(train_loader)))
                        print("=> starting predictions with args:\n {}".format(NN_arguments[i]))
                        predict(NN_arguments[i], train_loader, models[i])
                        print("finished predictions\n")

                print("Predictions for the creation of the new samples has finished, time elapsed {:.2f} hours \n".format((time.time() - sample_batch_time) / 3600))

                # create the probability & accumulated sample maps #
                depth_map_time = time.time()
                print("START CREATING new depth maps")
                create_custom_depth_maps(pred_dir.split("/NN")[0], data_in, M, phase, samples_between_phases, sample_method, rank_metric, is_test=False, first_phase_rgbd=first_phase_rgbd)
                print("Creating depth map with {} new samples has finished, time elapsed {:.2f} hours \n".format(samples_between_phases, (time.time() - depth_map_time) / 3600))

                print("Generating samples for train set {} has finished, time elapsed {:.2f} hours".format(set_num, (time.time() - pred_samp_train_set_time) / 3600))

        # predict & create samples for the validation & test sets #
        print("\n@@@ START GENERATING pattern for {} @@@ \n".format("val_cropped & test sets" if len(existing_weights) == 0 else "test set"))

        pred_val_test_time = time.time()
        for i, current_NN in enumerate(NNs_weights):  # get predictions of all M NNs
            input_t = input_type
            val_s_data_in = '../data_new/phase_' + str(phase) + '/mini_set_1'
            val_s_pred_dir = '../data_new/phase_' + str(phase + 1) + '/mini_set_1/predictions_tmp_val_select/NN' + str(i + 1)
            test_data_in = '../data_new/phase_' + str(phase) + '/mini_set_1'
            test_pred_dir = '../data_new/phase_' + str(phase + 1) + '/mini_set_1/predictions_tmp/NN' + str(i + 1)

            NN_arguments[i].data_folder = val_s_data_in
            NN_arguments[i].pred_dir = val_s_pred_dir
            NN_arguments[i].val = 'select'
            NN_arguments[i].use_d = 'd' in input_t
            NN_arguments[i].batch_size = predict_batch_size

            if len(existing_weights) == 0:  # not only inferencing
                val_dataset = KittiDepth('val', NN_arguments[i])
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=NN_arguments[i].batch_size, shuffle=False, num_workers=2, pin_memory=True)
                print("\t==> val_loader size:{}".format(len(val_loader)))
                print("predicting val_crop with NN{}".format(i + 1))
                print("args: {}\n".format(NN_arguments[i]))
                predict(NN_arguments[i], val_loader, models[i])

            NN_arguments[i].data_folder = test_data_in
            NN_arguments[i].pred_dir = test_pred_dir
            NN_arguments[i].val = 'select'
            NN_arguments[i].use_d = 'd' in input_t
            NN_arguments[i].batch_size = predict_batch_size

            test_dataset = KittiDepth('test', NN_arguments[i])
            val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=NN_arguments[i].batch_size, shuffle=False, num_workers=2, pin_memory=True)
            print("\t==> test_loader size:{}".format(len(val_loader)))
            print("predicting test with NN{}".format(i + 1))
            print("args: {}\n".format(NN_arguments[i]))
            predict(NN_arguments[i], val_loader, models[i])

        print("Predictions for the creation of the new samples has finished, time elapsed {:.2f} hours \n".format((time.time() - pred_val_test_time) / 3600))

        # create the probability & accumulated sample maps #
        depth_map_time = time.time()
        print("START CREATING new depth maps")  # create the next phase velodyne depth (the NNs next d input)
        if len(existing_weights) == 0:  # not only inferencing
            create_custom_depth_maps(val_s_pred_dir.split('/NN')[0], val_s_data_in, M, phase, samples_between_phases, sample_method, rank_metric, first_phase_rgbd=first_phase_rgbd)
        create_custom_depth_maps(test_pred_dir.split('/NN')[0], test_data_in, M, phase, samples_between_phases, sample_method, rank_metric, is_test=True, first_phase_rgbd=first_phase_rgbd)
        print("Generating samples for {} has finished, time elapsed {:.2f} hours \n".format("val_select & test sets" if len(existing_weights) == 0 else "test set", (time.time() - depth_map_time) / 3600))
        print("Generating all samples has finished, time elapsed {:.2f} hours \n".format((time.time() - generating_samples) / 3600))

        exit_code = os.system("find " + '../data_new/phase_' + str(phase) + " -type d -empty -delete")  # for folder-tree readability only, can be deleted
        if exit_code > 0:
            print("Error {} while trying delete empty folders in phase_{}. Exiting...".format(exit_code, phase), file=sys.stderr)
            exit(exit_code)

        print("TOTAL ACTIONS IN PHASE {} were finished, time elapsed {:.2f} hours \n".format(phase, (time.time() - phase_time) / 3600))

    exit_code = os.system("find " + '../data_new/phase_' + str(phase+1) + " -type d -empty -delete")  # for folder-tree readability only, can be deleted
    if exit_code > 0:
        print("Error {} while trying delete empty folders in phase_{}. Exiting...".format(exit_code, phase+1), file=sys.stderr)
        exit(exit_code)
    print("@@@ FINISHED ALL PHASES @@@")

    # (b) Predicting Dense Depth #
    print("@@@ START FINAL NN - predicting dense depth with the entire budget @@@ \n")
    create_dir_tree_skeleton('../data_new/var_final_NN', '../data')
    if len(existing_weights) == 0:  # not only inferencing
        for m in range(1, M+1):  # create a union of the whole mini-sets of the last phase into one train folder
            copy_partial_train_val('../data_new/phase_1/mini_set_' + str(m) + '/', '../data_new/var_final_NN', sets[m-1], 'train')  # copy all train (GT, vel, RGB) from phase 1
            copy_partial_train_val('../data_new/phase_' + str(phase + 1) + '/mini_set_' + str(m) + '/', '../data_new/var_final_NN', sets[m-1], 'train', ['data_depth_velodyne'])  # overrun vel with phase 5 vel's
        (_, sizes, _) = divide_drives_to_mini_sets(1, 100000, '../data_new/var_final_NN', 'train')  # some number > ~85K (full kitti dataset)
        print("Total train size is: {}".format(sizes.sum()))

        copy_partial_val_selection_cropped('../data/', '../data_new/var_final_NN', [0, val_select_desired_images])  # copy all
        copypath('../data_new/phase_' + str(phase + 1) + '/mini_set_1/depth_selection/val_selection_cropped/velodyne_raw/',
                                    '../data_new/var_final_NN/depth_selection/val_selection_cropped/velodyne_raw/', True)  # overwrite only new velodyne
        print("Val_select folder contains {} images.".format(len(os.listdir('../data_new/var_final_NN/depth_selection/val_selection_cropped/velodyne_raw/'))))

    copypath('../data_new/phase_1/mini_set_1/data_rgb/test/', '../data_new/var_final_NN/data_rgb/test/', True)
    copypath('../data_new/phase_1/mini_set_1/data_depth_annotated/test/', '../data_new/var_final_NN/data_depth_annotated/test/', True)
    copypath('../data_new/phase_' + str(phase + 1) + '/mini_set_1/data_depth_velodyne/test/', '../data_new/var_final_NN/data_depth_velodyne/test/', True)
    print("Test folder contains {} images.".format(len(list(Path('../data_new/var_final_NN/data_depth_annotated/test/').rglob('*.png')))))

    torch.cuda.empty_cache()
    # train the final NN #
    if len(existing_weights) == 0:  # not only inferencing
        print("@@@ START TRAINING final NN @@@ \n")
        final_train_ts = time.time()

        data_folder = '../data_new/var_final_NN'
        result = os.path.join('../outputs/var_final_NN')

        exit_code = os.system('python black_box_main.py --train-mode dense --layers ' + str(layers) + ' -b ' + str(train_batch_size) + ' --input ' + input_type + ' --pretrained --epochs ' + str(epochs) +
            ' --data-folder ' + data_folder + ' --result ' + result + ' --criterion ' + criterion + ' --rank-metric ' + rank_metric + ' --budget ' + str(globals.B))  # the Black-Box algorithm
        if exit_code > 0:
            print("Error {} while running 'python black_box_main.py'. Exiting...".format(exit_code), file=sys.stderr)
            exit(exit_code)

        print("Training final model has finished, time elapsed {:.2f} hours \n".format((time.time() - final_train_ts) / 3600))
        print("@@@ FINISH TRAINING FINAL NN @@@ \n")

    # testing #
    print("@@@ START TESTING FINAL NN @@@ \n")
    test_ts = time.time()
    if len(existing_weights) == 0:  # not only inferencing
        final_NN_weights = max(glob.glob('../outputs/var_final_NN/var.mode*'), key=os.path.getmtime)
        final_NN_weights = final_NN_weights + '/model_best.pth.tar'
    else:
        final_NN_weights = existing_weights[-1][0]
        print("Using given final NN weights (no training was done)")
    exit_code = os.system('python varNN-test/main_test.py --evaluate ' + final_NN_weights + ' --test yes --save-images')  # if --save-images: will save outputs as images')
    if exit_code > 0:
        print("Error {} while running 'python main_test.py'. Exiting...".format(exit_code), file=sys.stderr)
        exit(exit_code)
    print("Testing final model has finished, time elapsed {:.2f} hours \n".format((time.time() - test_ts) / 3600))

    print("@@@ FINISH TESTING FINAL NN @@@ \n")
    print("Whole run has finished, time elapsed {:.2f} hours \n".format((time.time() - whole_run) / 3600))
