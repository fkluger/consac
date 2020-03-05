import numpy as np
from torch.utils.data import Dataset
from datasets.nyu_vp.nyu import NYUVP


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


class NYUVPDataset(Dataset):

    def __init__(self, data_dir_path, max_num_segments, max_num_vps, split='train', keep_in_mem=True,
                 mat_file_path=None, permute_lines=True, calib=False, return_images=False):
        self.nyuvp = NYUVP(data_dir_path, split, keep_in_mem, mat_file_path, normalise_coordinates=True,
                           remove_borders=True)
        self.max_num_segments = max_num_segments
        self.max_num_vps = max_num_vps
        self.permute_lines = permute_lines
        self.calib = calib
        self.Kinv = np.array(self.nyuvp.Kinv).astype(np.float32)
        self.return_images = return_images

    def __len__(self):
        return len(self.nyuvp)

    def __getitem__(self, key):
        datum = self.nyuvp[key]

        if self.max_num_segments is None:
            max_num_segments = datum['line_segments'].shape[0]
        else:
            max_num_segments = self.max_num_segments

        if self.permute_lines:
            np.random.shuffle(datum['line_segments'])

        line_segments = np.zeros((max_num_segments, 15)).astype(np.float32)
        vps = np.zeros((self.max_num_vps, 3)).astype(np.float32)
        mask = np.zeros((max_num_segments, )).astype(np.int)

        num_actual_line_segments = np.minimum(datum['line_segments'].shape[0], max_num_segments)
        line_segments[0:num_actual_line_segments, :] = datum['line_segments'][0:num_actual_line_segments, :].copy()

        mask[0:num_actual_line_segments] = 1

        if num_actual_line_segments < max_num_segments:
            rest = max_num_segments-num_actual_line_segments
            line_segments[num_actual_line_segments:num_actual_line_segments+rest, :] = line_segments[0:rest, :].copy()

        vp_key = 'VDs' if self.calib else 'VPs'

        num_actual_vps = np.minimum(datum[vp_key].shape[0], self.max_num_vps)
        vps[0:num_actual_vps,:] = datum[vp_key][0:num_actual_vps]
        if self.return_images:
            return line_segments, vps, num_actual_line_segments, num_actual_vps, mask, datum['image']
        else:
            return line_segments, vps, num_actual_line_segments, num_actual_vps, mask
