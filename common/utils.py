import zipfile
from torchmetrics import AveragePrecision, Specificity, Accuracy

import torch
import numpy as np
import hashlib

from einops import rearrange
from torch.autograd import Variable
import os

actions_h36m = ["Directions", "Discussion", "Eating", "Greeting",
                "Phoning", "Photo", "Posing", "Purchases",
                "Sitting", "SittingDown", "Smoking", "Waiting",
                "WalkDog", "Walking", "WalkTogether"]

actions_unrealcv = ['Basketball', 'Dance', 'Dance2', 'Martial', 'Soccer', 'Boxing', 'Exercise']


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def confusion_per_action(prediction, target):
    confusion_vector = prediction / target

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def recall(predicted, target, threshold):
    predicted = predicted > threshold
    tp, fp, tn, fn = confusion_per_action(predicted, target)
    return tp / (tp + fn)

def precision(predicted, target, threshold):
    predicted = predicted > threshold
    tp, fp, tn, fn = confusion_per_action(predicted, target)
    return tp / (tp + fp)

def AP_per_action_calc(predicted, target):
    assert predicted.shape == target.shape
    # predicted = rearrange(predicted, 'b f j c -> (b f) j c', )
    # target = rearrange(target, 'b f j c -> (b f) j c', )
    return AveragePrecision(task='binary', thresholds=11).cuda()(predicted, target.int())

def NPV_per_action_calc(predicted, target):
    assert predicted.shape == target.shape
    predicted = rearrange(predicted, 'b f j c -> (b f) j c', )
    target = rearrange(target, 'b f j c -> (b f) j c', )

    predicted = predicted > 0.5
    tp, fp, tn, fn = confusion_per_action(predicted, target)
    npv = tn / (tn + fn)
    return npv


def TNR_per_action_calc(predicted, target):
    assert predicted.shape == target.shape
    # predicted = rearrange(predicted, 'b f j c -> (b f) j c', )
    # target = rearrange(target, 'b f j c -> (b f) j c', )
    return Specificity(task='binary')(predicted.detach().cpu(), target.int().detach().cpu())


def TPR_per_action_calc(predicted, target):
    assert predicted.shape == target.shape
    predicted = rearrange(predicted, 'b f j c -> (b f) j c', )
    target = rearrange(target, 'b f j c -> (b f) j c', )

    predicted = predicted > 0.5
    tp, fp, tn, fn = confusion_per_action(predicted, target)
    tpr = tp / (tp + fn)
    return tpr


def confusion(prediction, target):
    confusion_vector = prediction / target

    true_positives = torch.sum(confusion_vector == 1, dim=1)
    false_positives = torch.sum(confusion_vector == float('inf'), dim=1)
    true_negatives = torch.sum(torch.isnan(confusion_vector), dim=1)
    false_negatives = torch.sum(confusion_vector == 0, dim=1)

    return true_positives, false_positives, true_negatives, false_negatives


def acc_per_sample_calc(predicted, target, mask):
    assert predicted.shape == target.shape
    b, f = predicted.shape[0], predicted.shape[1]

    predicted = predicted > 0.5
    if mask is not None:
        res = torch.zeros(b)
        for i in range(b):
            # result = (target[i] == predicted[i])[mask[i]].float().mean()
            result = torch.masked_select(target[i] == predicted[i], mask[i].bool()).float().mean()
            res[i] = -1 if torch.isnan(result) else result
    else:
        tp, fp, tn, fn = confusion(predicted, target)
        res = (tp + tn) / (tp + tn + fp + fn)
    return res


def mpjpe_by_joint_p1(predicted, target, action, joint_error_sum, head_dist):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    f = predicted.size(1)
    assert f == 1
    J = predicted.size(2)
    dist = torch.norm(predicted - target, dim=len(target.shape) - 1).squeeze()
    # dist = head_dist.squeeze()

    for i in range(num):
        end_index = action[i].find(' ')
        if end_index != -1:
            action_name = action[i][:end_index]
        else:
            action_name = action[i]
        for j in range(J):
            joint_error_sum[action_name][j].update(dist[i, j].item() * f, f)
    return joint_error_sum



def test_calculation_joints_mpjpe(predicted, target, action, error_sum, dist):
    error_sum = mpjpe_by_joint_p1(predicted, target, action, error_sum, dist)
    return error_sum


def test_calculation_mpjpe(predicted, target, action, error_sum, dataset='h36m'):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum, dataset)
    error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum, dataset)

    return error_sum


def test_calculation_binary_class_metrics(predicted, target, action, error_sum, dataset):
    error_sum = binary_class_metrics_by_action(predicted, target, action, error_sum, dataset)
    return error_sum


def test_calculation_acc(predicted, target, action, error_sum, dataset, mask=None):
    error_sum = acc_by_action(predicted, target, action, error_sum, dataset, mask)
    return error_sum


def acc_by_action(predicted, target, action, action_error_sum, dataset, mask=None):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    acc = acc_per_sample_calc(predicted, target, mask)

    if len(set(list(action))) == 1:
        if dataset == 'unrealcv':
            action_name = get_action_from_idx(action[0], dataset)
        else:
            end_index = action[0].find(' ')
            if end_index != -1:
                action_name = action[0][:end_index]
            else:
                action_name = action[0]

        if mask is not None:
            N = len(torch.where(acc != -1)[0])
            if N > 0:
                action_error_sum[action_name]['acc'].update(torch.mean(acc[acc != -1]) * N, N)
        else:
            action_error_sum[action_name]['acc'].update(torch.mean(acc).item() * num, num)
    else:
        for i in range(num):
            if dataset == 'unrealcv':
                action_name = get_action_from_idx(action[i], dataset)
            else:
                end_index = action[i].find(' ')
                if end_index != -1:
                    action_name = action[i][:end_index]
                else:
                    action_name = action[i]

            if mask is not None:
                if acc[i] != -1:
                    action_error_sum[action_name]['acc'].update(acc[i].item(), 1)
            else:
                action_error_sum[action_name]['acc'].update(torch.mean(acc[i]).item(), 1)
    return action_error_sum


def binary_class_metrics_by_action(predicted, target, action, action_error_sum, dataset):
    assert predicted.shape == target.shape
    num = predicted.size(0)

    if len(set(list(action))) == 1:
        if dataset == 'unrealcv':
            action_name = get_action_from_idx(action[0], dataset)
        else:
            end_index = action[0].find(' ')
            if end_index != -1:
                action_name = action[0][:end_index]
            else:
                action_name = action[0]

        action_error_sum[action_name]['pred'].append(predicted)
        action_error_sum[action_name]['target'].append(target)
    else:
        for i in range(num):
            if dataset == 'unrealcv':
                action_name = get_action_from_idx(action[i], dataset)
            else:
                end_index = action[i].find(' ')
                if end_index != -1:
                    action_name = action[i][:end_index]
                else:
                    action_name = action[i]

            action_error_sum[action_name]['pred'].append(predicted)
            action_error_sum[action_name]['target'].append(target)

    return action_error_sum


def mpjpe_by_action_p1(predicted, target, action, action_error_sum, dataset):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    f = predicted.size(1)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        if dataset == 'unrealcv':
            action_name = get_action_from_idx(action[0], dataset)
        else:
            end_index = action[0].find(' ')
            if end_index != -1:
                action_name = action[0][:end_index]
            else:
                action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item() * f * num, f * num)
    else:
        for i in range(num):
            if dataset == 'unrealcv':
                action_name = get_action_from_idx(action[i], dataset)
            else:
                end_index = action[i].find(' ')
                if end_index != -1:
                    action_name = action[i][:end_index]
                else:
                    action_name = action[i]

            if f == 1:
                action_error_sum[action_name]['p1'].update(dist[i].item(), 1)
            else:
                action_error_sum[action_name]['p1'].update(torch.mean(dist[i]).item() * f, f)

    return action_error_sum


def mpjpe_by_action_p2(predicted, target, action, action_error_sum, dataset):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)

    if len(set(list(action))) == 1:
        if dataset == 'unrealcv':
            action_name = get_action_from_idx(action[0], dataset)
        else:
            end_index = action[0].find(' ')
            if end_index != -1:
                action_name = action[0][:end_index]
            else:
                action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            if dataset == 'unrealcv':
                action_name = get_action_from_idx(action[i], dataset)
            else:
                end_index = action[i].find(' ')
                if end_index != -1:
                    action_name = action[i][:end_index]
                else:
                    action_name = action[i]
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)

    return action_error_sum


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)


def get_action_from_idx(action_idx, dataset):
    actions = actions_unrealcv if dataset == 'unrealcv' else actions_h36m
    return actions[action_idx]


def define_actions(action, dataset='h36m'):
    actions = actions_unrealcv if dataset == 'unrealcv' else actions_h36m

    if action == "All" or action == "all" or action == '*':
        return actions

    if action not in actions:
        raise (ValueError, "Unrecognized action: %s" % action)

    return [action]


def define_error_joints_mpjpe_list(opt, actions):
    error_sum = {}
    for i in range(len(actions)):
        error_sum[actions[i]] = {}
        for j in range(opt.n_joints):
            error_sum[actions[i]][j] = AccumLoss()
    return error_sum

def define_error_mpjpe_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: {'p1': AccumLoss(), 'p2': AccumLoss()} for i in range(len(actions))})
    return error_sum


def define_binary_class_metrics_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: {'pred': [], 'target': []} for i in range(len(actions))})
    return error_sum


def define_acc_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: {'acc': AccumLoss()} for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def get_variable(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


def print_error_mpjpe(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = print_error_mpjpe_action(action_error_sum, is_train)
    return mean_error_p1, mean_error_p2


def print_acc(data_type, action_error_sum, is_train):
    mean_acc = print_acc_action(action_error_sum, is_train)
    return mean_acc


def print_binary_class_metrics(data_type, action_error_sum, is_train):
    mean_ap = print_AP_action(action_error_sum, is_train)
    mean_npv = print_NPV_action(action_error_sum, is_train)
    mean_tnr = print_TNR_action(action_error_sum, is_train)
    mean_tpr = print_TPR_action(action_error_sum, is_train)
    return mean_ap, mean_npv, mean_tnr, mean_tpr


def print_error_mpjpe_joint(joint_error_sum):
    mean_error_each_action = define_error_mpjpe_list(list(joint_error_sum.keys()))
    mean_error_each_joint = {}
    mean_error_each_joint.update({i: {'p1': AccumLoss()} for i in range(17)})
    mean_error_all = {'p1': AccumLoss()}
    mean_error_each = {'p1': AccumLoss()}

    print("{0:=^12} {1:=^12} {2:=^10}".format("Action", "Joint", "p#1 mm"))

    for action in joint_error_sum.keys():
        for joint, value in joint_error_sum[action].items():
            mean_error_each['p1'] = joint_error_sum[action][joint].avg * 1000.0
            mean_error_each_action[action]['p1'].update(mean_error_each['p1'], 1)
            mean_error_each_joint[joint]['p1'].update(mean_error_each['p1'], 1)
            print("{0:<12} ".format(str(action)), end="")
            print("{0:<12} ".format(str(joint)), end="")

            print("{0:>6.4f}".format(mean_error_each['p1']))

        mean_error_each['p1'] = mean_error_each_action[action]['p1'].avg
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        print("{0:<12} ".format(str(action)), end="")
        print("{0:>6.4f}".format(mean_error_each['p1']))

    for joint in range(17):
        mean_error_each['p1'] = mean_error_each_joint[joint]['p1'].avg
        print("{0:<12} ".format(str(joint)), end="")
        print("{0:>6.4f}".format(mean_error_each['p1']))

    print("{0:<12} {1:>6.4f}".format("Average", mean_error_all['p1'].avg))

    return mean_error_all['p1'].avg


def print_error_mpjpe_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
                                                    mean_error_all['p2'].avg))

    return mean_error_all['p1'].avg, mean_error_all['p2'].avg


def print_acc_action(action_error_sum, is_train):
    mean_acc_all = AccumLoss()

    if is_train == 0:
        print("{0:=^12} {1:=^10}".format("Action", "Accuracy %"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        mean_acc_each = action_error_sum[action]['acc'].avg * 100
        mean_acc_all.update(mean_acc_each, 1)

        if is_train == 0:
            print("{0:>6.4f}".format(mean_acc_each))

    if is_train == 0:
        print("{0:<12} {1:>6.4f}".format("Average", mean_acc_all.avg))

    return mean_acc_all.avg


def print_NPV_action(action_error_sum, is_train):
    mean_NPV_all = AccumLoss()

    if is_train == 0:
        print("{0:=^12} {1:=^10}".format("Action", "Negative Predictive Value"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        predicted = torch.cat(action_error_sum[action]['pred'], dim=0)
        target = torch.cat(action_error_sum[action]['target'], dim=0)
        mean_NPV_each = NPV_per_action_calc(predicted, target)
        mean_NPV_all.update(mean_NPV_each, 1)

        if is_train == 0:
            print("{0:>6.4f}".format(mean_NPV_each))

    if is_train == 0:
        print("{0:<12} {1:>6.4f}".format("Average", mean_NPV_all.avg))

    return mean_NPV_all.avg


def print_AP_action(action_error_sum, is_train):
    mean_AP_all = AccumLoss()

    if is_train == 0:
        print("{0:=^12} {1:=^10}".format("Action", "Average Precision"))

    for action, value in action_error_sum.items():
        mean_AP_each = AccumLoss()
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        for s in range(len(action_error_sum[action]['pred'])):
            predicted = action_error_sum[action]['pred'][s]
            target = action_error_sum[action]['target'][s]
            mean_AP_each.update(AP_per_action_calc(predicted, target) * predicted.shape[0], predicted.shape[0])
        mean_AP_all.update(mean_AP_each.avg, 1)

        if is_train == 0:
            print("{0:>6.4f}".format(mean_AP_each.avg))

    if is_train == 0:
        print("{0:<12} {1:>6.4f}".format("Average", mean_AP_all.avg))

    return mean_AP_all.avg


def print_TPR_action(action_error_sum, is_train):
    mean_TPR_all = AccumLoss()

    if is_train == 0:
        print("{0:=^12} {1:=^10}".format("Action", "True Positive Rate"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        predicted = torch.cat(action_error_sum[action]['pred'], dim=0)
        target = torch.cat(action_error_sum[action]['target'], dim=0)
        mean_TPR_each = TPR_per_action_calc(predicted, target)
        mean_TPR_all.update(mean_TPR_each, 1)

        if is_train == 0:
            print("{0:>6.4f}".format(mean_TPR_each))

    if is_train == 0:
        print("{0:<12} {1:>6.4f}".format("Average", mean_TPR_all.avg))

    return mean_TPR_all.avg


def print_TNR_action(action_error_sum, is_train):
    mean_TNR_all = AccumLoss()

    if is_train == 0:
        print("{0:=^12} {1:=^10}".format("Action", "True Negative Rate"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        predicted = torch.cat(action_error_sum[action]['pred'], dim=0)
        target = torch.cat(action_error_sum[action]['target'], dim=0)
        mean_TNR_each = TNR_per_action_calc(predicted, target)
        mean_TNR_all.update(mean_TNR_each, 1)

        if is_train == 0:
            print("{0:>6.4f}".format(mean_TNR_each))

    if is_train == 0:
        print("{0:<12} {1:>6.4f}".format("Average", mean_TNR_all.avg))

    return mean_TNR_all.avg


def save_model(previous_name, save_dir, epoch, data_threshold, optimizer, model, model_name, extra=None):
    if previous_name is not None and os.path.exists(previous_name):
        os.remove(previous_name)

    chk_path = '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100)

    torch.save({
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'model_pos': model.state_dict(),
        'extra': extra,
    }, chk_path)

    return chk_path


def lr_decay(step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    return lr


def generate_heatmaps(opt, joints):
    target = torch.zeros(opt.n_joints, opt.heatmap_res_out, opt.heatmap_res_out, dtype=torch.float32).cuda()

    tmp_size = opt.heatmap_sigma * 3

    for joint_id in range(opt.n_joints):
        feat_stride = opt.cropped_image_size / opt.heatmap_res_out
        mu_x = int(joints[joint_id][0] / feat_stride + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride + 0.5)
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= opt.heatmap_res_out or ul[1] >= opt.heatmap_res_out or br[0] < 0 or br[1] < 0:
            continue

        size = 2 * tmp_size + 1
        x = torch.arange(0, size, 1, dtype=torch.float32).cuda()
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * opt.heatmap_sigma ** 2)).cuda()

        g_x = max(0, -ul[0]), min(br[0], opt.heatmap_res_out) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], opt.heatmap_res_out) - ul[1]
        img_x = max(0, ul[0]), min(br[0], opt.heatmap_res_out)
        img_y = max(0, ul[1]), min(br[1], opt.heatmap_res_out)

        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target


def heatmap_loss(criterion, opt, predictions, target_skeletons):
    b, f = predictions.shape[0], predictions.shape[1]
    assert f == 1
    predictions = rearrange(predictions, 'b f j d1 d2  -> (b f) j d1 d2', )
    target_skeletons = rearrange(target_skeletons, 'b f j c  -> (b f) j c', )

    target_volume = torch.zeros(predictions.shape).cuda()

    for idx in range(b * f):
        target_volume[idx] = generate_heatmaps(opt, target_skeletons[idx])

    per_joint_heatmap_losses = []
    for joint_idx in range(opt.n_joints):
        per_voxel_loss = criterion(predictions[:, [joint_idx]], target_volume[:, [joint_idx]])
        per_joint_heatmap_losses.append(torch.mean(per_voxel_loss))

    loss_heatmap = torch.stack(per_joint_heatmap_losses).mean()
    return loss_heatmap


def extract_archive(from_path, to_path=None, remove_finished=False):
    """
    Extract a given archive
    :param from_path: path to archive which should be extracted
    :param to_path: path to which archive should be extracted
        default: parent directory of from_path
    :param remove_finished: if set to True, delete archive after extraction
    """
    if not os.path.exists(from_path):
        print('Path does not exists')
        return

    if to_path is None:
        to_path = os.path.dirname(from_path)

    try:
        with zipfile.ZipFile(from_path, 'r') as zip_:
            zip_.extractall(to_path)
    except ValueError:
        print("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)

    return to_path


# Source: Soubarna Banik
def calc_dists(preds, target, normalize):
    """
    calculate Eucledian distance per joint
    :param preds:
    :param target:
    :param normalize:
    :return: num_of_joints x num_batches
    """
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)

    dists = np.zeros((preds.shape[1], preds.shape[0]))

    for _batch in range(preds.shape[0]):
        for _joint in range(preds.shape[1]):
            if all(target[_batch, _joint, :] > 1):
                normed_preds = preds[_batch, _joint, :] / normalize[_batch]
                normed_targets = target[_batch, _joint, :] / normalize[_batch]
                dists[_joint, _batch] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[_joint, _batch] = -1
    return dists


def get_head_size(gt):
    head_size = np.linalg.norm(gt[:, 10, :] - gt[:, 9, :], axis=1)
    for n in range(gt.shape[0]):
        if gt[n, 10, 0] < 0 or gt[n, 9, 0] < 0:
            head_size[n] = 0

    return head_size


def get_torso_size(gt):
    torso_size = np.linalg.norm(gt[:, 0, :] - gt[:, 8, :], axis=1)
    for n in range(gt.shape[0]):
        if gt[n, 0, 0] < 0 or gt[n, 8, 0] < 0:
            torso_size[n] = 0

    return torso_size


def get_normalized_distance(gt, prediction, head_size):
    num_images = prediction.shape[0]
    num_keypoints = prediction.shape[1]
    distances = np.zeros([num_images, num_keypoints])
    for img_id in range(num_images):
        current_head_size = head_size[img_id]
        if current_head_size == 0:
            distances[img_id, :] = -1
        else:
            for kpt_id in range(num_keypoints):
                if all(gt[img_id, kpt_id, :] > 1):
                    distances[img_id, kpt_id] = np.linalg.norm(
                        gt[img_id, kpt_id, :] - prediction[img_id, kpt_id, :]) / current_head_size
                else:
                    distances[img_id, kpt_id] = -1
    return distances
