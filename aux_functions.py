import sys
from PIL import Image
from pathlib import Path
import os
import shutil
import glob
import time
from dataloaders.kitti_loader import rgb_read
import cv2
import matplotlib.pyplot as plt
import numpy as np


"""
Choose samples from the gt (and not from vel, so we could have more options to samples from) and accumulate it.
Input:
    current_pred_dir: the NNs prediction on the current phase images
    depth_dir_path: velodyne from the current phase (based on the last phase predictions)
    num_of_NN: the number of predictors
    phase: current phase
    budget: number of desired new samples
    samp_method: PM or greedy (MAX)
    metric: rmse or mae
    is_test: True if we create maps for test set, otherwise False
    save_probs_and_sample_maps: True if we want to save the probability maps (made from the variance map of all the predictions) & the new (only) sample maps - for debug & presentation
    first_phase_rgbd: True if we are in K=1 & using rgbd input (the default is rgb only)
Outputs: None, beside saving the new velodyne_raw files for the next phase
"""
def create_custom_depth_maps(current_pred_dir, current_depth_dir, num_of_NN, phase, budget, samp_method, metric, is_test=False, save_probs_and_sample_maps=False, first_phase_rgbd=False):
    if 'val_select' in current_pred_dir:  # dealing with val_select
        glob_d = "".join(current_depth_dir + '/depth_selection/val_selection_cropped/velodyne_raw/*.png')  # velodyne from the current phase (based on last phase's predictions)
        paths_d = sorted(glob.glob(glob_d))

        glob_gt = '../data_new/phase_1/mini_set_1/depth_selection/val_selection_cropped/groundtruth_depth/*.png'  # the gt aren't changing
        paths_gt = sorted(glob.glob(glob_gt))

        glob_d_pred = "".join(current_pred_dir + '/NN*/depth_selection/val_selection_cropped/velodyne_raw/*.png')  # the NNs prediction on the current phase's images
        paths_d_pred = sorted(glob.glob(glob_d_pred), key=lambda x: x.split('velodyne_raw/')[1])  # the sort makes sure same images in different NN* are arranged together

        predictions = num_of_NN
        num_different_images = int(len(paths_d_pred) / num_of_NN)
        assert len(paths_d_pred) == len(paths_gt) * num_of_NN, "create_custom_depth_maps: there are not enough predictions per image for val_select"
    elif is_test == False:  # dealing with train
        glob_d = "".join(current_depth_dir + '/data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')
        paths_d = sorted(glob.glob(glob_d))

        glob_gt = "".join('../data_new/phase_1/mini_set_' + current_depth_dir.split('mini_set_')[1][0] + '/data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png')
        paths_gt = sorted(glob.glob(glob_gt))

        glob_d_pred = "".join(current_pred_dir + '/NN*/data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')
        paths_d_pred = sorted(glob.glob(glob_d_pred), key=lambda x: x.split('train/')[1])

        predictions = num_of_NN - 1
        num_different_images = int(len(paths_d_pred) / (num_of_NN - 1))
        assert len(paths_d_pred) == len(paths_gt) * (num_of_NN - 1), "create_custom_depth_maps: there are not enough predictions per image for train"
    else:  # dealing with test
        glob_d = "".join(current_depth_dir + '/data_depth_velodyne/test/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')
        paths_d = sorted(glob.glob(glob_d))

        glob_gt = "".join('../data_new/phase_1/mini_set_1/data_depth_annotated/test/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png')
        paths_gt = sorted(glob.glob(glob_gt))

        glob_d_pred = "".join(current_pred_dir + '/NN*/data_depth_velodyne/test/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')
        paths_d_pred = sorted(glob.glob(glob_d_pred), key=lambda x: x.split('test/')[1])

        predictions = num_of_NN
        num_different_images = int(len(paths_d_pred) / num_of_NN)
        assert len(paths_d_pred) == len(paths_gt) * num_of_NN, "create_custom_depth_maps: there are not enough predictions per image for test"

    paths_d_pred_itr = iter(paths_d_pred)
    for image in range(num_different_images):
        predictions_of_same_image = []  # will be: [num_of_images][row][col]
        for i in range(predictions):
            predictions_of_same_image.append(depth_read(next(paths_d_pred_itr)))

        # creating inputs for lidar_choose function
        h, w = predictions_of_same_image[0].shape  # should be 352x1216
        reshaped_predictions_of_same_image = np.asarray([np.reshape(predictions_of_same_image[j], (h * w, 1)) for j in range(predictions)])
        reshaped_predictions_of_same_image = (reshaped_predictions_of_same_image.squeeze(axis=2)).transpose()  # shape should be: (h*w, predictions)

        current_gt = depth_read(paths_gt[image])  # should be bigger than 352x1216 if train, else exactly 352x1216 (validation & test sets came from val_select_cropped)
        m = current_gt.shape[0] - 352
        n = int(round((current_gt.shape[1] - 1216) / 2.))
        current_gt = current_gt[m:(m + 352), n:(n + 1216)]

        if (phase == 1) and (not first_phase_rgbd):  # no d inputs
            valid_mask = np.reshape(current_gt.copy(), (h * w, 1))  # the whole gt samples are valid, changing to a vector
            current_velodyne = np.zeros(current_gt.shape)  # we don't have samples yet
        else:  # cropping, make sure we don't give option to sample pixels that we've already sampled, changing to a vector
            current_velodyne = depth_read(paths_d[image])  # should be 352x1216 because our predictions are already cropped and we made the last velodyne to be the same shape as them
            if first_phase_rgbd and current_velodyne.shape[0] != 352:  # if first phase is RGBD so we just need to be sure the first d-maps (with 0 samples) are in the right shape
                current_velodyne = np.zeros(current_gt.shape)
                if phase > 1:
                    print("Create map: should not enter here after first phase")
                    exit()
            assert current_velodyne.shape == current_gt.shape, "'create_custom_depth_maps' function: current_velodyne & current_gt shapes are different"
            valid_mask = current_gt.copy()
            valid_mask[current_velodyne > 0] = 0  # can be 0 or -1, lidar_choose doesn't care. if we have valid samples in velodyne we don't want to choose them again
            valid_mask = np.reshape(valid_mask, (h * w, 1))
            current_velodyne[current_velodyne <= 0] = 0  # keep only valid samples (pixels that are >=0)

        if metric == 'rmse':
            inds_to_sample_next, prob_only_valid_for_next_sampling, x = lidar_choose(reshaped_predictions_of_same_image, valid_mask.squeeze(), budget, 'var', samp_method)
        elif metric == 'mae':
            inds_to_sample_next, prob_only_valid_for_next_sampling, x = lidar_choose(reshaped_predictions_of_same_image, valid_mask.squeeze(), budget, 'median', samp_method)
        else:
            print("Wrong criterion, exiting...")
            exit()

        if inds_to_sample_next.size != 0:  # we have samples
            next_velodyne = np.reshape(np.zeros((h, w)), (h * w, 1))  # vector of zeros (non valid depth)
            inds_to_sample_next = np.expand_dims(inds_to_sample_next, 0).transpose()
            next_velodyne[inds_to_sample_next] = np.reshape(current_gt, (h * w, 1))[inds_to_sample_next]
            next_velodyne = np.reshape(next_velodyne, (h, w))
            next_velodyne = next_velodyne + current_velodyne  # we want to add the previous samples
        else:  # we don't have (already took all valid samples), take current velodyne instead
            next_velodyne = current_velodyne

        if 'val_select' in current_pred_dir:
            filename = current_pred_dir.split('/predictions_tmp_val_select')[0] + '/depth_selection' + paths_d_pred[((predictions) * image) + i].split('/depth_selection')[1]
        else:
            filename = current_pred_dir.split('/predictions_tmp')[0] + '/data_depth_velodyne' + paths_d_pred[((predictions) * image) + i].split('/data_depth_velodyne')[1]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        depth_write(filename, next_velodyne)  # if we've opened with depth_read, we need to save with depth_write. This will be cropped

        if save_probs_and_sample_maps == True:  # to debug or look at the semi-results (only on mini-set 1)
            if 'val_select' in current_pred_dir:
                print("create_custom_depth_maps: save probs and sample maps of val_select is not available yet. Need to make some modifications. Continue without saving...")
                continue
            elif filename.split('set_')[1].split('/')[0] != '1':  # we are saving only things in miniset-1
                continue

            next_velodyne[next_velodyne > 0] = 1
            ps = paths_gt[image].split('/')
            if is_test:
                rgb_path = '/'.join(ps[:-7] + ['data_rgb'] + ps[-6:-4] + ps[-2:-1] + ['data'] + [ps[-1:][0].replace('groundtruth_depth', 'image')])
            else:
                rgb_path = '/'.join(ps[:-7] + ['data_rgb'] + ps[-6:-4] + ps[-2:-1] + ['data'] + ps[-1:])
            x_only_gt_valid = np.reshape(x.copy(), (h, w))  # will be only valid
            x_only_gt_valid[current_gt < 0] = 0
            show_prob_map_and_sampling_indices_on_image(phase, rgb_path, next_velodyne, np.reshape(x / np.sum(x), (h, w)),
                                                        x_only_gt_valid, show_results=False, save_results=True)  # if want the variance map, take x instead of x / np.sum(x)

"""
Chooses the next samples to be fed to the NN during train phase.
Input:
    scores: shape - depth_samples_col_vec x num_sources depth predictions. the user should turn each depth samples image (the input to the NN) to a column vector
    valid_mask: mask image of all relevant pixels to be chosen, as a vector. We don't want to choose points where either the value is not known or that were chosen already. >0 means valid (to choose), <=0 invalid.
    budget: how many points to sample this time
    method: two options: 'var', 'median'
    pm_greedy - choose use the probability matching (sample based on the probability map), or the greedy (MAX - highest probabilities)
Output:
    inds: Chosen indices.
    prob: A map of the pixels probability (to be chosen).
    x: the values related to the chosen method.
Example:
    for total of 5 NN, meaning we compare 4 outputs to a single one: inds,prob,x = lidar_choose(scores, mask, budget=1024, method='var',
                                                            quantile_discard=[0.25,0.75], pm_greedy='pm')
"""
def lidar_choose(scores, valid_mask, budget, method, pm_greedy):
    assert method in ['var', 'median'], "unknown method:" + method
    assert pm_greedy in ['pm', 'greedy'], "unknown pm_greedy:" + pm_greedy
    assert scores.shape[0] == valid_mask.shape[0], "scores.shape[0] != valid_mask.shape[0]"
    assert np.all(scores >= 0.), "non-positive predictions found!"

    n, k = scores.shape
    scores = np.sort(scores, axis=1).astype('float32')  # sort each cell (depth predictions of a pixel) from small to large value

    if method == 'var':  # relates to L2
        x = np.var(scores, axis=1).astype('float32')
    elif method == 'quantiles':
        x = (scores[:, -1] - scores[:, 0]) ** 2
    elif method == 'median':  # relates to L1
        med = np.median(scores, axis=1)  # median of even array is a non existing element (the middle)
        medians = np.transpose(np.tile(med, (scores.shape[1], 1)))
        x = np.mean(np.absolute(scores - medians), axis=1)
    else:
        assert (1)

    prob_only_valid = x.copy()  # will contain cues where it's best to sample (based on the predictors)

    # give zero prob to invalid points
    bad_inds, = np.where(valid_mask <= 0)
    good_inds = scores.shape[0] - bad_inds.shape[0]  # something to sample from
    prob_only_valid[bad_inds] = 0.
    s = np.sum(prob_only_valid)

    if budget > good_inds:  # we want to sample more then we can
        print("Warning: lidar_choose: taking {} instead of {} samples, because not enough valid pixels to sample from gt".format(good_inds, budget))
        budget = good_inds
        if budget == 0:  # nothing to sample
            prob_only_valid = np.array([])  # nothing is valid. We don't have valid prob map
            return (np.array([]), prob_only_valid, x)

    if budget > prob_only_valid[prob_only_valid > 0].shape[0]:  # not enough pixels in prob_map (not much difference between predictors). We'll have to take some random samples because
        budget_prob = prob_only_valid[prob_only_valid > 0].shape[0]
        budget_rand = budget - budget_prob  # prob map == 0 for them
        print("Warning: lidar_choose: only {} different valid pixels to sample from (probability map is zero elsewhere). {} remaining samples will be taken randomly uniform from {} "
                                                                                                                "valid pixels".format(budget_prob, budget_rand, good_inds-budget_prob))
        if budget_prob != 0:  # samples based on the probability map
            prob_only_valid = prob_only_valid / s
            if pm_greedy == 'pm':
                inds1 = np.random.choice(n, budget_prob, False, prob_only_valid)  # len(inds) = budget. Choose budget numbers from n array (np.arange(n)),
                                                                                  # based on the probabilities prob. No replacements
            else:  # greedy-MAX
                inds1 = np.argpartition(prob_only_valid, -budget_prob)[-budget_prob:]  # get 'budget' max indices of 'prob'
            valid_mask[inds1] = 0  # remove indices we took already
        else:  # we'll take only random (everything valid is zeros)
            prob_only_valid = np.array([])  # we don't have valid prob map
            inds1 = np.array([]).astype('int')

        # sample randomly - no matter PM or greedy (all have the same weights - uniform)
        valid_mask[valid_mask > 0] = 1
        valid_mask[valid_mask < 0] = 0
        valid_mask = valid_mask / valid_mask.sum()  # uniform
        inds2 = np.random.choice(n, budget_rand, False, valid_mask)
        return (np.concatenate([inds1, inds2]), prob_only_valid, x)

    else:  # enough pixels in prob_map
        prob_only_valid = prob_only_valid / s
        if pm_greedy == 'pm':
            inds = np.random.choice(n, budget, False, prob_only_valid)
        else:  # greedy-MAX
            inds = np.argpartition(prob_only_valid, -budget)[-budget:]

    return (inds, prob_only_valid, x)

"""
Saves depth image as uint16
"""
def depth_write(filename, img):
    img[img < 0] = 0  # negative depth is like 0 depth
    img = img * 256
    if np.max(img) >= 2 ** 16:
        print('Warning: {} pixels in {} have depth >= 2**16 (max is: {}).Truncating before saving.'.format(img[img >= 2**16].shape[0], "/".join(filename.split('/')[-5:]), np.max(img)))
        img = np.minimum(img, 2 ** 16 - 1)

    img = img.astype('uint16')
    cv2.imwrite(filename, img)

"""
Loads depth map D from png file and returns it as a numpy array,
"""
def depth_read(filename):
    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    if np.max(depth_png) > 65536:  # not relevant for debug (when NNs don't predict good depth), leading to 0.9m in all of the image, resulting this error OR when we insert black image to the NN
        print("warning: max depth {} in while reading image{} in depth_read".format(np.max(depth_png), filename))

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth

"""
Given a kitti dir, and a type ('train'/'val'), return a list of drives and a map from drive name to the number of images in the drive. The number of images is 2* the number in the image_02 dir
"""
def get_train_val_drive_list(src_path, dir_type):
    assert (dir_type in train_and_val())
    dirname = src_path + '/' + get_train_val_type_dirs()[0] + '/' + dir_type  # takes data_depth_annotated
    drives = os.listdir(dirname)
    img_count = {}
    for drive in drives:
        g = Path(dirname + '/' + drive).rglob('*.png')
        img_count[drive] = len(list(g))

    drives = list(img_count.keys())
    return (drives, img_count)

"""
Copy some of the drives in the train or val
Inputs:
    src_path - for example, the full kitti
    dest_path - a partial kitti being built
    drive_list - a list of drive directories (e.g. '2011_09_26_drive_0001_sync'). the list is unique.
    set_type - 'train' or 'val'
    dirs - if None, all the 3 top dirs (get_train_val_type_dirs()). otherwise, a subset of those.
    
IMPORTANT: dirs = None is intended for building a new directory from scratch, and will not allow
stepping on existing content. dirs != None is intended for such stepping and will allow it.
The user is then responsible for keeping directory valid (same drives in all)
if drives do not exist, a warning will be printed.
"""
def copy_partial_train_val(src_path, dest_path, drive_list, set_type, dirs=None):
    assert (set_type in train_and_val())

    if dirs is None:
        dirs = get_train_val_type_dirs()
        dirs_exist_ok = False
    else:
        assert (len(dirs) > 0)
        for dir in dirs:
            assert (dir in get_train_val_type_dirs())

        dirs_exist_ok = True

    # make list of unique values
    drive_list = list(set(drive_list))

    for drive in drive_list:
        success = True
        for dir in dirs:
            cur_src = os.path.join(src_path, dir, set_type, drive)
            cur_dest = os.path.join(dest_path, dir, set_type, drive)
            if os.path.isdir(cur_src) == False:
                success = False
            else:
                # make sure we are replacing an existing directory
                if dirs_exist_ok == True:
                    assert (os.path.isdir(cur_dest) == True)
                    shutil.rmtree(cur_dest)

                shutil.copytree(cur_src, cur_dest)

        if success == False:
            print('Warning: invalid drive ' + drive)

"""
Create the kitti skeleton - all dirs (without the drives), no data. root_path string must be new (it will create the folder itself)
"""
def create_dir_tree_skeleton(root_path, example_path):
    if os.path.exists(root_path) == True:
        print('path ' + root_path + ' already exists. Exiting.')
        return

    if os.path.isdir(example_path) == False:
        print(example_path + 'is not a directory or does not exist. Exiting.')
        return

    for dir in get_train_val_type_dirs():
        for subdir in train_and_val_and_test():
            dirname = root_path + '/' + dir + '/' + subdir
            os.makedirs(dirname)

    for root, dirs, files in os.walk(example_path + '/depth_selection'):
        for name in dirs:
            dirname = root_path + '/' + root[len(example_path) + 1:] + '/' + name
            os.makedirs(dirname)

def train_and_val():
    return ['train', 'val']

def train_and_val_and_test():
    return ['train', 'val', 'test']

"""
Divide drives to minisets. itereate on drives in descending number of images, and allocate drive to the miniset (with the demand of a minimum size of a set). 
Continue until all sets have at least min_imgs_per_set for the first time (return success/failure). When a set is reaching min_imgs - no more will be added 
(in addition, the count is for the ground truth - 10 fewer than the rgb...).
Inputs:
    src_path, dir_type - as in get_train_val_drive_list
    add_descending - True is good for the whole set/big portion of it. If False, good for creating small mini sets (for small runs)
"""
def divide_drives_to_mini_sets(n_sets, min_imgs_per_set, src_path, dir_type, add_descending=True):
    drives, img_count = get_train_val_drive_list(src_path, dir_type)
    keys = list(img_count.keys())
    values = list(img_count.values())
    order = np.argsort(np.array(list(values)))

    drives_sorted_by_nums = []
    nums_sorted = sorted(values)
    success = False

    for i in range(order.shape[0]):
        drives_sorted_by_nums.append(keys[order[i]])

    sets = [[] for i in range(n_sets)]
    sizes = np.zeros(n_sets, dtype='int')

    indices = np.arange(0, len(nums_sorted))
    if add_descending == True:  # bigger syncs first
        indices = indices[::-1]

    for i in indices:
        smallest = np.argmin(sizes)
        if sizes[smallest] >= min_imgs_per_set:
            success = True
            break

        sets[smallest].append(drives_sorted_by_nums[i])
        sizes[smallest] += nums_sorted[i]

    return (sets, sizes, success)

"""
return the names of the directories with train/val subdirectories
"""
def get_train_val_type_dirs():
    return ['data_depth_annotated', 'data_depth_velodyne', 'data_rgb']

"""
Given a directory name (user's responsibility that it is a directory), return the elements in a-b order.
"""
def get_sorted_ls(src_path):
    return sorted(os.listdir(src_path))

"""
Copy files in val_selection_cropped folder
Inputs:
    src_path - for example, the full kitti
    dest_path - a partial kitti being built
    subset_params - either a range or a drive list. if range, get a list [start,end] and copy files start:end in alphabetical order, where the range must be valid 
        (from 0 to n, where n is # of total files). otherwise expects a list of drives (e.g. '2011_09_26_drive_0001_sync'), and takes files from these drives. the list is uniqued. 
        if drives do not exist, a warning will be printed.
"""
def copy_partial_val_selection_cropped(src_path, dest_path, subset_params):
    vsc_suf = os.path.join('depth_selection', 'val_selection_cropped')

    # find if params specify range or drive list
    if len(subset_params) > 0:
        dirnames = os.listdir(
            os.path.join(src_path, vsc_suf))  # ['velodyne_raw', 'image', 'groundtruth_depth', 'intrinsics']
        if type(subset_params[0]) != type(str()):  # images range
            n = len(os.listdir(os.path.join(src_path, vsc_suf, dirnames[0])))  # number of files inside 'velodyne_raw' folder. Should be 1000
            start, end = subset_params
            if start < 0 or end > n:
                sys.exit('copy_partial_val_selection_cropped: invalid range')  # probably there are not enough images to copy

            for dirname in dirnames:
                for filename in get_sorted_ls(os.path.join(src_path, vsc_suf, dirname))[start:end]:
                    shutil.copyfile(os.path.join(src_path, vsc_suf, dirname, filename),
                                    os.path.join(dest_path, vsc_suf, dirname, filename))
        else:  # drive list
            drive_list = list(set(subset_params))  # make unique
            for drive in drive_list:
                found = False
                for filename in Path(os.path.join(src_path, vsc_suf)).rglob(drive + '*'):
                    found = True
                    strname = str(filename)
                    shutil.copyfile(strname, os.path.join(dest_path, strname[len(src_path) + 1:]))
                if found == False:
                    print('Warning: invalid drive: ' + drive)
    else:
        assert 1, 'invalid params'

"""
Copy files and dirs (recursive). If dest exists, will succeed for files but not for dirs. If fails, calls sys.exit().
"""
def copypath(src, dest, overwrite_dest_dir=False):
    if os.path.isdir(src):
        try:
            if overwrite_dest_dir and os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        except:
            sys.exit('FAILED copytree. Exiting.')
    else:
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(src, dest)
        except:
            sys.exit('FAILED copyfile. Exiting.')

"""
Replace (overwrite) existing depth images with empty (zeros) ones - on KITTI folders tree
"""
def apply_empty_d_maps_to_data(src_path_to_overwrite):
    start_time = time.time()
    apply_empty_d_maps_to_directory(os.path.join(src_path_to_overwrite, 'data_depth_velodyne'))
    apply_empty_d_maps_to_directory(os.path.join(src_path_to_overwrite, 'depth_selection', 'val_selection_cropped', 'velodyne_raw'))
    print("Finished after {:.2f} hours".format((time.time() - start_time) / 3600))

def apply_empty_d_maps_to_directory(dirpath):
    print("number of images in folder {}: ".format(dirpath), flush=True)
    os.system('find ' + dirpath + ' -name \'*.png\' | wc -l')  # we'll get 0 when trying to deal with val_select that is not in mini-set 1 (because they should not be there - so it's ok)
    empty_map = np.zeros((352, 1216), dtype=int)

    for filename in Path(dirpath).rglob('*.png'):
        strname = str(filename)
        depth_write(strname, empty_map)

"""
Can show or save current probability maps (based on the variance map from the predictions) & the current samples on the rgb image
"""
def show_prob_map_and_sampling_indices_on_image(phase, rgb_path, samples_map, prob_map, x_gt_valid, show_results=False, save_results=False):
    rgb = rgb_read(rgb_path)
    m = rgb.shape[0] - 352
    n = int(round((rgb.shape[1] - 1216) / 2.))
    rgb = rgb[m:(m + 352), n:(n + 1216), :]
    samples_on_image = show_img_effect(rgb, samples_map, enlarge_samples=True)
    cmap = plt.cm.get_cmap("jet")  # color map name. Red is higher, blue is lower

    if save_results == True:
        path = '../data_new/phase_' + str(phase + 1) + '/' + rgb_path.split('/')[3] + '/prob_and_sampling_maps_results' + rgb_path.split('data_rgb')[1]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        samples_on_image = (255 * samples_on_image).astype('uint8')
        im = Image.fromarray(samples_on_image)
        im.save(path.split('.png')[0] + '_samples.png')

        prob_gt_valid = (x_gt_valid - x_gt_valid.min()) / (x_gt_valid.max() - x_gt_valid.min())
        # depth_write(path.split('.png')[0] + '_prob_gt_valid.png', prob_gt_valid)  # real values (rather than visualization - in the next lines)
        colored_prob_valid_map = cmap(prob_gt_valid)  # apply the colormap, will result 4-channel image (R,G,B,A) in float [0,1]
        im = Image.fromarray((colored_prob_valid_map[:, :, :3] * 255).astype(np.uint8))  # convert to RGB in uint8
        im.save(path.split('.png')[0] + '_prob_gt_valid.png')

        # np.save(path.split('.png')[0] + '_prob.npy', prob_map)  # real values (rather than visualization - in the next lines)
        prob_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min())
        # depth_write(path.split('.png')[0] + '_prob.png', prob_map)  # real values (rather than visualization - in the next lines)
        colored_prob_map = cmap(prob_map)  # apply the colormap, will result 4-channel image (R,G,B,A) in float [0,1]
        im = Image.fromarray((colored_prob_map[:, :, :3] * 255).astype(np.uint8))  # convert to RGB in uint8
        im.save(path.split('.png')[0] + '_prob.png')

    if show_results == True:
        plt.close()
        plt.figure(1)
        plt.imshow(samples_on_image)
        plt.figure(2)
        plt.imshow(prob_map)

"""
Combine the original image and an effect of interest into a single image.
Inputs:
    rgb: original image HxWx3
    effect: image HxW
    enlarge_samples: if the effect is a sample map - will enlarge each pixel by 4 different directions: up, down, left, right (for better view later on).
                     is it only for display purposes because it's changing the data
    gs: gray scale image. when given - what 'rgb' argument is non relevant
    jet_mode: relevant only for gs. will shot the image in JET colormap, and the effect will show on top of it. Use it only for data that you don't
              care for the values of the pixels, just if there is data or not (binary). Excellent for sample map as effect
    previous_effect: relevant only when the effect is samples. We'll use the previous_effect to determine what are the new samples
Outputs:
    img - returns the effected image.
"""
def show_img_effect(rgb, effect, enlarge_samples=False, gs=None, jet_mode=False, previous_effect=None):
    # normalize to [0, 1]
    if effect.min() == effect.max():  # no effect at all
        effect = np.zeros(effect.shape)
    else:
        effect = (effect - np.min(effect)) / (np.max(effect) - np.min(effect))

    if previous_effect is not None:
        if previous_effect.min() == previous_effect.max():  # no effect at all
            previous_effect = np.zeros(previous_effect.shape)
        else:
            previous_effect = (previous_effect - np.min(previous_effect)) / (np.max(previous_effect) - np.min(previous_effect))

    # enlarge a pixel to a '+' sign (on top of the image) - to make it clearer to the viewer
    if enlarge_samples == True:
        if previous_effect is not None:
            effect = effect - previous_effect  # only what's new
            add_to_right = np.roll(previous_effect, 1, axis=1)  # cyclic shift 1 column to the right
            add_to_right[:, 0] = 0  # ignore leftmost col (due to cyclic)
            add_to_left = np.roll(previous_effect, -1, axis=1)  # 1 column to the left
            add_to_left[:, -1] = 0  # ignore rightmost col

            add_to_up = np.roll(previous_effect, -1, axis=0)  # 1 row up
            add_to_up[-1, :] = 0  # ignore lower row
            add_to_down = np.roll(previous_effect, 1, axis=0)  # 1 row down
            add_to_down[0, :] = 0  # ignore upper row

            previous_effect = previous_effect + add_to_right + add_to_left + add_to_up + add_to_down

        add_to_right = np.roll(effect, 1, axis=1)  # cyclic shift 1 column to the right
        add_to_right[:, 0] = 0  # ignore leftmost col (due to cyclic)
        add_to_left = np.roll(effect, -1, axis=1)  # 1 column to the left
        add_to_left[:, -1] = 0  # ignore rightmost col

        add_to_up = np.roll(effect, -1, axis=0)  # 1 row up
        add_to_up[-1, :] = 0  # ignore lower row
        add_to_down = np.roll(effect, 1, axis=0)  # 1 row down
        add_to_down[0, :] = 0  # ignore upper row

        effect = effect + add_to_right + add_to_left + add_to_up + add_to_down

    if gs is None:  # rgb image
        assert (rgb.shape[0:2] == effect.shape)
        if np.max(rgb) > 1.:
            rgb = rgb / 255.  # to [0, 1]

        gs = np.dot(rgb, [0.299, 0.587, 0.114])  # the ratios between channels. need to sum up to 1 in order to get a true gray scale
        img = np.zeros(rgb.shape)
    else:  # gray scale image
        img = np.zeros((gs.shape[0], gs.shape[1], 3))
        gs = (gs - np.min(gs)) / (np.max(gs) - np.min(gs))  # to [0, 1]

        if jet_mode == True:  # only possible in with gs
            cmap = plt.cm.get_cmap("jet")
            colored_gs = cmap(gs)[:, :, :3]  # apply the colormap + take only RGB in float [0,1]
            if previous_effect is not None:
                (colored_gs[:, :, 0])[previous_effect > 0] = 0.45  # the old sample points will be gray-ish. this won't show value, but a binary - there is a data, or not
                (colored_gs[:, :, 1])[previous_effect > 0] = 0.45
                (colored_gs[:, :, 2])[previous_effect > 0] = 0.45

            (colored_gs[:, :, 0])[effect > 0] = 1  # the new sample points will be white. this won't show value, but a binary - there is a data, or not
            (colored_gs[:, :, 1])[effect > 0] = 1
            (colored_gs[:, :, 2])[effect > 0] = 1


            return colored_gs

    img[:, :, 0] = effect
    img[:, :, 1] = gs
    return img


"""
Get the name of the NN weights, from an output file.
"""
def read_weights(file_path, phases, predictors):
    if file_path == "":  # we are not inferencing
        return []

    weights = []
    NN_in_stage = []

    with open(file_path) as f:
        lines = f.read().splitlines()

    for line in lines:
        if line != "":  # as long as we don't encounter empty line (means a new stage)
            NN_in_stage.append(line)
        else:
            weights.append(NN_in_stage)
            NN_in_stage = []

    weights.append(NN_in_stage)  # final net

    for phase in range(len(weights) - 2):
        assert len(weights[phase]) == len(weights[phase+1]), "read_weights: number of weights isn't match between phases"
    assert len(weights[-1]) == 1, "read_weights: there should be 1 final net weights"
    assert phases == len(weights)-1, "read_weights: given argument K isn't match to the number of given phases in .txt"
    assert len(weights[0]) == predictors, "read_weights: given argument M isn't match to the number of given weights inside each phase in .txt"

    print("Detects {} weights in each phase, total {} phases, in addition to {} final net weights\n".format(len(weights[0]), len(weights)-1, len(weights[-1])))
    return weights