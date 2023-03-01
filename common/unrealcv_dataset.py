import os
import pickle
import numpy as np
import torch

from common.skeleton import Skeleton
from common.utils import extract_archive

unrealcv_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 10, 8, 12, 13],
                             joints_left=[4, 5, 6, 9, 10, 11], joints_right=[1, 2, 3, 12, 13, 14])


class UnrealCvDataset:
    def __init__(self, path, train=False, extract=False, load_images=False):
        self._skeleton = unrealcv_skeleton

        self.orig_image_size = np.array((1280, 720))
        self.num_joints = 15
        self.actual_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'thorax',
            9: 'lsho',
            10: 'lelb',
            11: 'lwri',
            12: 'rsho',
            13: 'relb',
            14: 'rwri'
        }

        self.load_images = load_images
        self.train = train
        self.path = path

        file = 'train' if self.train else 'test'
        if not extract:
            self.archive_path = os.path.join(self.path, 'images')
        else:
            self.archive_path = extract_archive(os.path.join(self.path, 'images.zip'))

        anno_file = os.path.join(self.path, 'annot', f'unrealcv_{file}.pkl')

        self.db = self.load_db(anno_file)
        for datum in self.db:
            joints_vis_3d = datum['joints_vis']
            joints_vis_2d = datum['joints_vis_2d']
            # joints_vis_new = np.concatenate([joints_vis_2d], axis=1)
            datum['joints_vis_3d'] = joints_vis_3d
            datum['joints_vis'] = joints_vis_2d

    def __getitem__(self, key):
        data = self.db[key]
        if self.load_images:
            return data['image'], torch.tensor((data['box'][0], data['box'][1], data['box'][2], data['box'][3])).view(1, 4)
        else:
            return data

    def __len__(self):
        return len(self.db)

    def skeleton(self):
        return self._skeleton

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset