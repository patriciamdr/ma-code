import argparse
import os
import pickle
import numpy as np
import sys
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='unrealcv')
parser.add_argument('--keypoints', type=str, default='CPN')
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--data_path', type=str, default='/home/patricia/dev/StridedTransformer-Pose3D/dataset')
args = parser.parse_args()


def load_db(dataset_file):
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
        return dataset


if args.dataset == 'unrealcv':
    data_dir = os.path.join(args.data_path, args.dataset)

    acc_total = np.zeros(2)
    vis_total = np.zeros(2)
    score_total = np.zeros(2)
    for j, spec in enumerate(['train', 'test']):
        file = f'{spec}_keypoints_scores_{args.keypoints.lower()}'
        keypoints_data = np.load(os.path.join(data_dir, file + '.npz'), allow_pickle=True)
        kps_scores = keypoints_data[f'{spec}_scores'].item()

        anno_file = os.path.join(data_dir, 'annot', f'unrealcv_{spec}.pkl')

        db = load_db(anno_file)

        dataloader_len = len(db)
        acc_all = np.zeros(dataloader_len)
        vis_all = np.zeros(dataloader_len)
        score_all = np.zeros(dataloader_len)
        print('Dataloader length: ' + str(dataloader_len))

        for i, data in enumerate(db):
            score = kps_scores[data['image']]

            # TODO: Use model visibility prediction for head and neck, instead of removing them
            if args.dataset == 'unrealcv':
                # Ignore neck and head from model prediction because no ground truth available
                score = np.hstack([score[:9], score[11:]])

            vis = data['joints_vis_2d'].squeeze()
            assert vis.shape == score.shape

            vis_all[i] = vis.mean()

            score_threshold = score > args.threshold
            score_all[i] = score_threshold.mean()

            acc = (np.equal(vis, score_threshold)).astype(float).mean()
            acc_all[i] = acc

        acc_avg = acc_all.mean()
        acc_total[j] = acc_avg
        print(f'Average {spec} accuracy: ' + str(acc_avg))

        percentage_joints_vis = vis_all.mean()
        vis_total[j] = percentage_joints_vis
        percentage_joints_score = score_all.mean()
        score_total[j] = percentage_joints_score

        print('Percentage of visible joints based on ground truth: ' + str(percentage_joints_vis))
        print('Percentage of visible joints based on score: ' + str(percentage_joints_score))

    acc_total_avg = acc_total.mean()
    print('Average accuracy: ' + str(acc_total_avg))
    percentage_joints_vis_total = vis_total.mean()
    percentage_joints_score_total = score_total.mean()

    print('Average percentage of visible joints based on ground truth: ' + str(percentage_joints_vis_total))
    print('Average percentage of visible joints based on score: ' + str(percentage_joints_score_total))
