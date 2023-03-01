import numpy as np
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

keypoints_all = dict()
scores_all = dict()

for subject in subjects:
    print(subject)

    file = f'{subject}_keypoints_scores_cpn.npz'
    keypoints_data = np.load('/home/patricia/dev/StridedTransformer-Pose3D/dataset/h36m/' + file, allow_pickle=True)
    kps_coordinates = keypoints_data['keypoints'].item()
    kps_scores = keypoints_data['scores'].item()

    for action in kps_coordinates[subject]:
        assert action in kps_scores[subject]

        def sort_dict(d):
            return dict(sorted(d.items()))

        kps = list(sort_dict(kps_coordinates[subject][action]).values())
        scores = list(sort_dict(kps_scores[subject][action]).values())

        if subject not in keypoints_all:
            keypoints_all[subject] = {action: kps}
            scores_all[subject] = {action: scores}
        elif action not in keypoints_all[subject]:
            keypoints_all[subject][action] = kps
            scores_all[subject][action] = scores

print(keypoints_all.keys())
print(scores_all.keys())

output_dir = '/home/patricia/dev/StridedTransformer-Pose3D/dataset/h36m/keypoints_scores_cpn.npz'
np.savez_compressed(output_dir, keypoints=keypoints_all, scores=scores_all)

