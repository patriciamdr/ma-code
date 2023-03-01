import os
import cv2
import numpy as np
import torch
import sys

sys.path.append(os.getcwd())

from demo.lib.sort.sort import Sort
from demo.lib.hrnet.lib.utils.utilitys import PreProcess
from demo.lib.preprocess import h36m_coco_format
from demo.lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
from demo.lib.hrnet.lib.utils.inference import get_final_preds

from common.unrealcv_dataset import UnrealCvDataset
from scripts.keypoint_generation.models.pose_cpn import get_cpn


def cpn_pose(model, video, j, gt_bbox=None, num_person=1):
    people_sort = Sort(min_hits=0)

    video_length = len(video)

    kpts_result = []
    scores_result = []

    data_shape = (288, 384)
    for ii in range(video_length):
        frame = video[ii]
        bbox = gt_bbox[ii]

        # Using Sort to track people
        people_track = people_sort.update(bbox)

        # Track the first two people in the video and remove the ID
        if people_track.shape[0] == 1:
            people_track_ = people_track[-1, :-1].reshape(1, 4)
        elif people_track.shape[0] >= 2:
            people_track_ = people_track[-num_person:, :-1].reshape(num_person, 4)
            people_track_ = people_track_[::-1]
        else:
            continue

        track_bboxs = []
        for people_track_bbox in people_track_:
            track_bbox = [round(i, 2) for i in list(people_track_bbox)]
            track_bboxs.append(track_bbox)

        with torch.no_grad():
            inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, data_shape, num_person)
            inputs = inputs[:, [2, 1, 0]]

            global_pred, refine_pred = model(inputs)

            preds, maxvals = get_final_preds(True, refine_pred.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

            kpts = np.zeros((num_person, 17, 2), dtype=np.float32)
            scores = np.zeros((num_person, 17), dtype=np.float32)
            for i, kpt in enumerate(preds):
                kpts[i] = kpt

            for i, score in enumerate(maxvals):
                scores[i] = score.squeeze()

        kpts_result.append(kpts)
        scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

    return keypoints, scores


def get_pose2D_cpn(model, video, gt_bbox, j):
    print('\nGenerating 2D pose...')
    keypoints, scores = cpn_pose(model, video, j, gt_bbox=gt_bbox)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    print('Generating 2D pose successful!')
    return keypoints, scores


def get_pose2D_hrnet(video_path, bbox, j, use_gt_bbox):
    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, j, det_dim=416, num_peroson=1, gen_output=True, gt_bboxs=bbox, use_gt_bbox=use_gt_bbox)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    print('Generating 2D pose successful!')
    return keypoints, scores


if __name__ == "__main__":
    keypoint_detector = 'CPN'
    use_gt_bbox = True
    suffix = '_gt-bbox' if keypoint_detector == 'HRNet' and use_gt_bbox else ''

    model = get_cpn()
    model.eval()

    dataset_dir = '/home/patricia/dev/datasets/occlusion_person/unrealcv'

    print('==> Loading train dataset...')
    dataset = UnrealCvDataset(dataset_dir, train=True, load_images=True)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                                   shuffle=False, num_workers=8,
                                                   pin_memory=True)
    train_keypoints = dict()
    train_scores = dict()
    print(len(train_dataloader))

    for j, (filename, bbox) in enumerate(train_dataloader):
        images = []
        files = []
        bboxes = []
        for i, f in enumerate(filename):
            # print(f)
            image_file = os.path.join(dataset.archive_path, f)
            img = cv2.imread(image_file, cv2.IMREAD_COLOR)
            b = bbox[i]

            images.append(img)
            bboxes.append(b)
            files.append(f)
        if keypoint_detector == 'CPN':
            keypoints, scores = get_pose2D_cpn(model, images, bboxes, j)
        elif keypoint_detector == 'HRNet':
            keypoints, scores = get_pose2D_hrnet(images, bboxes, j, use_gt_bbox)
        else:
            raise KeyError('Invalid keypoint detector')

        train_keypoints.update(zip(files, keypoints.squeeze()))
        train_scores.update(zip(files, scores.squeeze()))

    output_dir = f'/home/patricia/dev/StridedTransformer-Pose3D/dataset/unrealcv/train_keypoints_scores_{keypoint_detector.lower()}{suffix}.npz'
    np.savez_compressed(output_dir, train_keypoints=train_keypoints, train_scores=train_scores)

    #########################

    print('==> Loading test dataset...')
    dataset = UnrealCvDataset(dataset_dir, load_images=True)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                                  shuffle=False, num_workers=8,
                                                  pin_memory=True)
    test_keypoints = dict()
    test_scores = dict()
    print(len(test_dataloader))

    for j, (filename, bbox) in enumerate(test_dataloader):
        images = []
        files = []
        bboxes = []
        for i, f in enumerate(filename):
            print(f)
            image_file = os.path.join(dataset.archive_path, f)
            img = cv2.imread(image_file)
            b = bbox[i]

            images.append(img)
            bboxes.append(b)
            files.append(f)

        if keypoint_detector == 'CPN':
            keypoints, scores = get_pose2D_cpn(model, images, bboxes, j)
        elif keypoint_detector == 'HRNet':
            keypoints, scores = get_pose2D_hrnet(images, bboxes, j, use_gt_bbox)
        else:
            raise KeyError('Invalid keypoint detector')

        test_keypoints.update(zip(files, keypoints.squeeze()))
        test_scores.update(zip(files, scores.squeeze()))

    output_dir = f'/home/patricia/dev/StridedTransformer-Pose3D/dataset/unrealcv/test_keypoints_scores_{keypoint_detector.lower()}{suffix}.npz'
    np.savez_compressed(output_dir, test_keypoints=test_keypoints, test_scores=test_scores)
