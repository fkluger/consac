import numpy as np
from torch.utils.data import Dataset
from datasets.yud_plus.yud import YUDVP


class YUDVPDataset(Dataset):

    def __init__(self, data_dir_path, max_num_segments, max_num_vps, split='train', keep_in_mem=True,
                 permute_lines=True, return_images=False, yudplus=False):
        self.dataset = YUDVP(data_dir_path, split, keep_in_mem, normalize_coords=True, return_images=return_images,
                             yudplus=yudplus)
        self.max_num_segments = max_num_segments
        self.max_num_vps = max_num_vps
        self.permute_lines = permute_lines
        self.return_images = return_images

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        datum = self.dataset[key]

        if self.max_num_segments is None:
            max_num_segments = datum['line_segments'].shape[0]
        else:
            max_num_segments = self.max_num_segments

        line_segments = np.zeros((max_num_segments, 12)).astype(np.float32)
        vps = np.zeros((self.max_num_vps, 3)).astype(np.float32)
        mask = np.zeros((max_num_segments,)).astype(np.int)

        num_actual_line_segments = np.minimum(datum['line_segments'].shape[0], max_num_segments)
        if self.permute_lines:
            np.random.shuffle(line_segments)
        line_segments[0:num_actual_line_segments, :] = datum['line_segments'][0:num_actual_line_segments, :]

        mask[0:num_actual_line_segments] = 1

        num_actual_vps = np.minimum(datum['VPs'].shape[0], self.max_num_vps)
        vps[0:num_actual_vps, :] = datum['VPs'][0:num_actual_vps]
        if self.return_images:
            return line_segments, vps, num_actual_line_segments, num_actual_vps, mask, datum['image']
        else:
            return line_segments, vps, num_actual_line_segments, num_actual_vps, mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from util import sampling

    dataset = YUDVP("./yud_plus/data", split='all', normalize_coords=False, return_images=True, yudplus=True)

    max_num_vp = 0
    max_num_ls = 0
    all_distances_smallest = []
    all_distances_second = []
    all_num_vps = []
    for idx in range(len(dataset)):
        vps = dataset[idx]['VPs']
        num_vps = vps.shape[0]
        print("%d vp: " % idx, num_vps)
        all_num_vps += [num_vps]
        if num_vps > max_num_vp: max_num_vp = num_vps
        num_ls = dataset[idx]['line_segments'].shape[0]
        if num_ls > max_num_ls: max_num_ls = num_ls

        ls = dataset[idx]['line_segments']
        vp = dataset[idx]['VPs']

        distances_per_img = []

        for vi in range(vp.shape[0]):
            distances = sampling.vp_consistency_measure_angle_np(vp[vi], ls)
            distances_per_img += [distances]
        distances_per_img = np.sort(np.vstack(distances_per_img), axis=0)

        smallest = distances_per_img[0, :]
        all_distances_smallest += [smallest]

    print("max vps: ", max_num_vp)
    print(np.unique(all_num_vps, return_counts=True))

    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(9, 3))
    values, bins, patches = plt.hist(all_num_vps, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    print(values)
    print(bins)
    plt.show()
