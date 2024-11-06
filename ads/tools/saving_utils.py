import os
import pickle
import numpy as np
import pandas as pd
import moviepy.video.io.ImageSequenceClip

from tqdm import tqdm
from loguru import logger
from PIL import Image

from .dataset_utils import preprocess_birdview_and_routemap, binary_to_integer

from .view_recoder import DisplayInterface


class DataWriter:

    def __init__(
            self,
            dir_path,
            run_info=None,
            render=True
    ):
        self._dir_path = dir_path
        if run_info is None:
            self.run_info = {
            }
        else:
            self.run_info = run_info  # save episode basic info

        self.run_info['dir_path'] = self._dir_path
        self._data_list = []
        self.render = render

        if self.render:
            self.dis_renderer = DisplayInterface()

    def write(
            self,
            step,
            action,  # control
            speed,  # speed
            features,
            gps,
            imu,
            target_point,
            bev_masks,
            view_img,
            bev_img,
            meta_data,
            shift_distance,
            repaired_action: str,  # repaired_log
            prev_action
    ):
        if self.render:
            action_str = np.array2string(action, precision=2, separator=',', suppress_small=True)
            if repaired_action is None:
                repaired_action_str = 'None'
            else:
                repaired_action_str = repaired_action
            render_txt = {
                "basic_info": f"(Basic) speed: {speed:.2f}",
                "control": f"(Action) {action_str}",
                "repair": f"(Repair) {repaired_action_str}",
            }
            view_img = self.dis_renderer.run_interface(view_img, render_txt)

        data_dict = {
            'supervision':
                {
                    'step': step,
                    'action': action,
                    'speed': speed,
                    'features': features,
                    'gps': gps,
                    'imu': imu,
                    'target_point': target_point,
                    'prev_action': prev_action
                },
            'repair': {
                'shift': shift_distance,  # score
                'repaired_action': repaired_action,  # load in this
                    },
            'bev_masks': bev_masks,
            'view_img': view_img,
            'bev_img': bev_img,
            'meta_data': meta_data
        }
        self._data_list.append(data_dict)

    def close(self):
        # clean up data
        logger.info(f'Episode finished, len={len(self._data_list)}')
        self.save_files()

    def save_files(self):
        # os.makedirs(os.path.join(self._dir_path, 'view_image'), exist_ok=True)  # camera view
        # os.makedirs(os.path.join(self._dir_path, 'bev_image'), exist_ok=True)  # camera view
        # os.makedirs(os.path.join(self._dir_path, 'birdview'), exist_ok=True)  # bev masks
        # os.makedirs(os.path.join(self._dir_path, 'routemap'), exist_ok=True)  # route maps
        os.makedirs(os.path.join(self._dir_path, 'meta_data'), exist_ok=True)  # meta data folder

        dict_dataframe = {
            'action': [],
            'speed': [],
            'features': [],
            'gps': [],
            'imu': [],
            'target_point': [],
            'view_image_path': [],
            'bev_image_path': [],
            'birdview_path': [],
            'routemap_path': [],
            'meta_data_path': [],
            'n_classes': [],  # Number of classes in the bev
            'shift': [],
            'repaired_action': [],
            'prev_action': []
        }

        # add meta
        for k in self.run_info.keys():
            dict_dataframe[k] = []

        logger.info(f'Saving {self._dir_path}, data_len={len(self._data_list)}, saving ...')
        video_frames = []
        for i, data in enumerate(self._data_list):

            supervision = data['supervision']
            for k, v in supervision.items():
                if k not in dict_dataframe.keys():
                    dict_dataframe[k] = []
                dict_dataframe[k].append(v)

            repair = data['repair']
            for k, v in repair.items():
                if k not in dict_dataframe.keys():
                    dict_dataframe[k] = []
                dict_dataframe[k].append(v)

            # Add run information
            for k, v in self.run_info.items():
                dict_dataframe[k].append(v)

            # bev_image = data['bev_img']
            # view_image = data['view_img']
            meta_data = data['meta_data']

            # Process birdview and save as png
            # birdview, route_map = preprocess_birdview_and_routemap(data['bev_masks'])
            # birdview, route_map = birdview.numpy(), route_map.numpy()
            # n_bits, h, w = birdview.shape
            # birdview = birdview.reshape(n_bits, -1)
            # birdview = birdview.transpose((1, 0))
            # # Convert bits to integer for storage
            # birdview = binary_to_integer(birdview, n_bits).reshape(h, w)

            # view_image_path = os.path.join(f'view_image', f'image_{i:09d}.png')
            # bev_image_path = os.path.join(f'bev_image', f'image_{i:09d}.png')
            # birdview_path = os.path.join(f'birdview', f'birdview_{i:09d}.png')
            # routemap_path = os.path.join(f'routemap', f'routemap_{i:09d}.png')
            meta_data_path = os.path.join(f'meta_data', f'meta_data_{i:09d}.pkl')
            # dict_dataframe['view_image_path'].append(view_image_path)
            # dict_dataframe['bev_image_path'].append(bev_image_path)
            # dict_dataframe['birdview_path'].append(birdview_path)
            # dict_dataframe['routemap_path'].append(routemap_path)
            dict_dataframe['meta_data_path'].append(meta_data_path)
            # dict_dataframe['n_classes'].append(n_bits)
            # # Save RGB images
            # video_frames.append(view_image)
            # view_image = Image.fromarray(view_image)
            # # Image.fromarray(view_image).save(os.path.join(self._dir_path, view_image_path))
            # # Get the original size
            # view_image_size = view_image.size  # (width, height)
            # # Calculate the new size (half of the original size)
            # new_size = (view_image_size[0] // 2, view_image_size[1] // 2)
            # # Resize the image
            # view_image = view_image.resize(new_size, Image.Resampling.LANCZOS)
            # view_image.save(os.path.join(self._dir_path, view_image_path))
            # Image.fromarray(bev_image).save(os.path.join(self._dir_path, bev_image_path))
            # Image.fromarray(birdview, mode='I').save(os.path.join(self._dir_path, birdview_path))
            # Image.fromarray(route_map, mode='L').save(os.path.join(self._dir_path, routemap_path))
            with open(os.path.join(self._dir_path, meta_data_path), 'wb') as f:
                pickle.dump(meta_data, f)

        # pd_dataframe = pd.DataFrame(dict_dataframe)
        # pd_dataframe.to_pickle(os.path.join(self._dir_path, 'pd_dataframe.pkl'))

        # logger.info('Start convert to video...')
        # fps = 20
        # save_file = os.path.join(self._dir_path, 'view_video.mp4')
        # if len(video_frames) > 0:
        #     clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(video_frames, fps=fps, with_mask=False)
        #     clip.write_videofile(save_file, preset="fast", logger=None, audio=False)
        # logger.info('Success convert to video...')

        self._data_list.clear()
