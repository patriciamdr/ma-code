import copy

import cv2
import numpy as np
import torch
import os
import sys
sys.path.append(os.getcwd())

from scripts.keypoint_generation.models.pose_cpn import get_cpn
from common.h36m_dataset import Human36mDatasetImages
# Loading human detector model
from demo.lib.yolov3.human_detector import load_model as yolo_model
from demo.lib.yolov3.human_detector import yolo_human_det as yolo_det
from demo.lib.sort.sort import Sort
from demo.lib.preprocess import h36m_coco_format
from demo.lib.hrnet.lib.utils.utilitys import PreProcess
from demo.lib.hrnet.lib.utils.inference import get_final_preds
from demo.lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose, parse_args, reset_config


def cpn_pose(model, video, j, det_dim=416, num_person=1):
    # Updating configuration
    args = parse_args()
    reset_config(args)

    cap = cv2.VideoCapture(video)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loading detector and pose model, initialize sort for track
    human_model = yolo_model(inp_dim=det_dim)
    people_sort = Sort(min_hits=0)

    kpts_result = []
    scores_result = []

    data_shape = (288, 384)
    for ii in range(video_length):
        ret, frame = cap.read()
        if not ret:
            continue

        bboxs, scores = yolo_det(frame, human_model, reso=det_dim, confidence=args.thred_score)

        if bboxs is None or not bboxs.any():
            print('No person detected!')
            bboxs = bboxs_pre
            scores = scores_pre
        else:
            bboxs_pre = copy.deepcopy(bboxs)
            scores_pre = copy.deepcopy(scores)

        # Using Sort to track people
        people_track = people_sort.update(bboxs)

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


def get_pose2D_cpn(model, video_path, j=None):
    print('\nGenerating 2D pose...')
    keypoints, scores = cpn_pose(model, video_path, j)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    print('Generating 2D pose successful!')
    return keypoints, scores


def get_pose2D_hrnet(video_path, j=None):
    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, j, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    print('Generating 2D pose successful!')
    return keypoints, scores

if __name__ == "__main__":
    keypoint_detector = 'CPN'

    model = get_cpn()
    model.eval()

    print('==> Loading dataset...')
    subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

    for subject in subjects:
        dataset = Human36mDatasetImages('/home/patricia/dev/datasets/human3.6m/orig/pose', subjects=[subject])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 shuffle=False, num_workers=8, pin_memory=True)

        keypoints = dict()
        scores_dict = dict()
        print(len(dataloader))

        for filename in dataloader:
            print(filename)
            action = filename[0].split('/')[-1].split('.')[0]
            print('Action: ' + action)
            cam_idx = filename[0].split('/')[-1].split('.')[1]
            print('Cam: ' + cam_idx)

            if keypoint_detector == 'CPN':
                kps, scores = get_pose2D_cpn(model, filename[0])
            elif keypoint_detector == 'HRNet':
                kps, scores = get_pose2D_hrnet(filename[0])
            else:
                raise KeyError('Invalid keypoint detector')

            if subject not in keypoints:
                keypoints[subject] = {action: {cam_idx: kps.squeeze()}}
                scores_dict[subject] = {action: {cam_idx: scores.squeeze()}}
            elif action not in keypoints[subject]:
                keypoints[subject][action] = {cam_idx: kps.squeeze()}
                scores_dict[subject][action] = {cam_idx: scores.squeeze()}
            elif cam_idx not in keypoints[subject][action]:
                keypoints[subject][action][cam_idx] = kps.squeeze()
                scores_dict[subject][action][cam_idx] = scores.squeeze()

        output_dir = f'/home/patricia/dev/StridedTransformer-Pose3D/dataset/h36m/{subject}_keypoints_scores_{keypoint_detector.lower()}.npz'
        np.savez_compressed(output_dir, keypoints=keypoints, scores=scores_dict)
