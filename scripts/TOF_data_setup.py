import argparse
import copy
import os
import numpy as np
from glob import glob
import sys
sys.path.append(os.getcwd())

from common.skeleton import Skeleton
from common.camera import world_to_camera, wrap, project_to_2d, normalize_screen_coordinates, image_coordinates
from common.h36m_dataset import h36m_cameras_extrinsic_params, h36m_cameras_intrinsic_params
from demo.vis import show2Dpose

output_dir = '/home/patricia/dev/StridedTransformer-Pose3D/dataset/h36m/'
output_filename_TOF = 'data_3d_h36m_TOF.npz'
# subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
subjects = ['S1', 'S6']

res_w = 176
res_h = 144
# TODO: Fix camera center
center = np.array([102, 73])
focal_length = np.array([220.0156959, 231.165386])
# focal_length = np.array([223.4010148, 235.5013885])
# focal_length = np.array([248, 248])

h36m_tof_intrinsic_params = np.concatenate((focal_length,
                                            center,
                                            [0, 0, 0],  # radial distortion
                                            [0, 0]))  # tangential distortion



joints_to_remove = [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]
h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                  16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                         joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                         joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])


_cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
for cameras in _cameras.values():
    for i, cam in enumerate(cameras):
        cam.update(h36m_cameras_intrinsic_params[i])
        for k, v in cam.items():
            if k not in ['id', 'res_w', 'res_h']:
                cam[k] = np.array(v, dtype='float32')

        # cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype(
        #     'float32')
        # cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2

        if 'translation' in cam:
            # Add RGB camera height as offset to TOF sensor translation
            cam['translation'] = ((cam['translation'] + [0, 29, 0])/ 1000).astype(np.float32)

        cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                           cam['center'],
                                           cam['radial_distortion'],
                                           cam['tangential_distortion']))

error_thr = 0.3  # meters

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human3.6M dataset downloader/converter')

    # Convert dataset from original source, using original .cdf files (the Human3.6M dataset path must be specified manually)
    # This option does not require MATLAB, but the Python library cdflib must be installed
    parser.add_argument('--from-source-cdf', default='', type=str, metavar='PATH', help='convert original dataset')

    args = parser.parse_args()

    if args.from_source_cdf:
        print('Converting original Human3.6M dataset from', args.from_source_cdf, '(CDF files)')
        output = {}

        import cdflib

        data = np.load(output_dir + 'data_3d_h36m.npz', allow_pickle=True)['positions_3d'].item()

        for subject in subjects:
            output[subject] = {}
            file_list = glob(args.from_source_cdf + '/' + subject + '/TOF/*.cdf')
            assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
            for f in file_list:
                action = os.path.splitext(os.path.basename(f))[0]

                if subject == 'S11' and action == 'Directions':
                    continue  # Discard corrupted video

                # Use consistent naming convention
                action = action.replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog')

                hf = cdflib.CDF(f)

                depth_data = {}

                # TODO: Unit conversion?
                ranges = hf['RangeFrames'].squeeze()

                skeleton = copy.deepcopy(h36m_skeleton)
                kept_joints = skeleton.remove_joints(joints_to_remove)
                pos_3d_world = data[subject][action][:, kept_joints]

                skeleton._parents[11] = 8
                skeleton._parents[14] = 8

                filtered_pos_3d_world = pos_3d_world[hf['Indicator'].squeeze().astype(bool)]  # Filter poses where depth not available due to different Hz
                mocap_length = filtered_pos_3d_world.shape[0]
                assert ranges.shape[2] >= mocap_length

                if ranges.shape[2] > mocap_length:
                    print(subject)
                    print(action)
                    start_offset = ranges.shape[2] - mocap_length
                    print('Start-offset: ' + str(start_offset))
                    ranges = ranges[:, :, start_offset:mocap_length+start_offset]

                cam = _cameras[subject][1]  # TOF sensor is above camera 2
                filtered_pos_3d_cam = world_to_camera(filtered_pos_3d_world, R=cam['orientation'], t=cam['translation'])

                filtered_pos_2d_pixel = wrap(project_to_2d, filtered_pos_3d_cam, h36m_tof_intrinsic_params, unsqueeze=True)
                # filtered_pos_2d_pixel = image_coordinates(filtered_pos_2d_image, w=res_w, h=res_h)

                from einops import rearrange
                filtered_pos_3d_cam = rearrange(filtered_pos_3d_cam, 'f j c -> j c f', )
                filtered_pos_2d_pixel = rearrange(filtered_pos_2d_pixel, 'f j c -> j c f', )

                # minimum = np.ones(mocap_length)
                # maximum = np.zeros(mocap_length)

                # MM = ranges.max(axis=0).max(axis=0)
                # maximum = np.where(maximum < MM, MM, maximum)

                # mm = ranges.min(axis=0).min(axis=0)
                # minimum = np.where(minimum > mm, mm, minimum)

                # f = (maximum-minimum)

                # ranges = ranges/f

                vis_seq = np.zeros((filtered_pos_2d_pixel.shape[0], mocap_length))

                for i in range(mocap_length):
                    r = ranges[:, :, i]
                    p = (filtered_pos_2d_pixel[:, :, i]).astype(int)
                    P = filtered_pos_3d_cam[:, :, i]

                    # Create artificial visibility target
                    vis = np.zeros(p.shape[0])
                    for j in range(p.shape[0]):
                        try:
                            depth = r[p[j][1], p[j][0]]
                            z_coordinate = P[j][2]
                            vis[j] = np.less(z_coordinate - depth, error_thr)[..., None]
                        except IndexError:
                            # Joint outside image view
                            vis[j] = 1

                    # if (vis == 0).any():
                    #     print(np.where(vis == 0))
                    #     import cv2
                    #     img = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    #     img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    #     img_pose = show2Dpose(p, img)
                    #     cv2.imwrite('/home/patricia/dev/StridedTransformer-Pose3D/'
                    #                 + subject + '_' + action + '_' + str(('%04d'% i)) + '.png', img_pose)

                    vis_seq[:, i] = vis

                output[subject][action] = vis_seq

        print('Saving...')
        np.savez_compressed(output_dir + output_filename_TOF, depth_data=output)
        print('Done.')

    else:
        print('Please specify the dataset source')
        exit(0)
