import glob
import os
import numpy as np
from torch.utils.data import Dataset


class HomographyData:

    train_sequences = [
        "st_peters_square/train_data",
        "sacre_coeur/test_data",
        "buckingham_palace/test_data",
        "notre_dame_front_facade/test_data",
        "reichstag/test_data",
        "fountain/test_data",
        "herzjesu/test_data"
    ]

    indoor_sequences = [
        "brown_bm_3---brown_bm_3-maxpairs-10000-random---skip-10-dilate-25/train_data",
        "harvard_c4---hv_c4_1---skip-10-dilate-25/test_data",
        "mit_w85g---g_0---skip-10-dilate-25/test_data",
        "mit_46_6conf---bcs_floor6_conf_1---skip-10-dilate-25/test_data",
        "brown_cs_7---brown_cs7---skip-10-dilate-25/test_data",
        "harvard_robotics_lab---hv_s1_2---skip-10-dilate-25/test_data",
        "mit_46_6lounge---bcs_floor6_long---skip-10-dilate-25/test_data",
        "mit_32_g725---g725_1---skip-10-dilate-25/test_data",
        "mit_w85h---h2_1---skip-10-dilate-25/test_data",
        "harvard_corridor_lounge---hv_lounge1_2---skip-10-dilate-25/test_data",
        "harvard_c10---hv_c10_2---skip-10-dilate-25/test_data",
        "brown_cs_3---brown_cs3---skip-10-dilate-25/test_data",
        "brown_cogsci_2---brown_cogsci_2---skip-10-dilate-25/test_data",
        "brown_cogsci_6---brown_cogsci_6---skip-10-dilate-25/test_data",
        "hotel_florence_jx---florence_hotel_stair_room_all---skip-10-dilate-25/test_data"
    ]

    val_sequences = [
        "reichstag/val_data",
    ]

    def __init__(self, data_dir, split='', keep_in_mem=True, normalize_coords=False, max_score_ratio=1.,
                 use_indoor=False, fair_sampling=False, repeat_scenes=100, augmentation=False):

        self.data_dir = data_dir
        self.keep_in_mem = keep_in_mem
        self.normalize_coords = normalize_coords
        self.max_score_ratio = max_score_ratio
        self.fair_sampling = fair_sampling
        self.augmentation = augmentation

        self.dataset_files = []
        self.dataset_files_per_scene = []

        if use_indoor:
            train_sequences = self.train_sequences + self.indoor_sequences
        else:
            train_sequences = self.train_sequences

        if split is not None:
            if split == "train":
                for seq in train_sequences:
                    seq_ = os.path.join(self.data_dir, seq)
                    files = glob.glob(os.path.join(seq_, "*.npy"))
                    files.sort()
                    self.dataset_files += files
                    self.dataset_files_per_scene += [files]
            elif split == "val":
                for seq in self.val_sequences:
                    seq_ = os.path.join(self.data_dir, seq)
                    files = glob.glob(os.path.join(seq_, "*.npy"))
                    files.sort()
                    self.dataset_files += files
                    self.dataset_files_per_scene += [files]
            elif split == "all":
                for seq in (self.val_sequences + train_sequences):
                    seq_ = os.path.join(self.data_dir, seq)
                    files = glob.glob(os.path.join(seq_, "*.npy"))
                    files.sort()
                    print("%s: %d files" % (seq, len(files)))
                    self.dataset_files += files
                    self.dataset_files_per_scene += [files]
            else:
                assert False, "invalid split: %s " % split

        self.dataset_files_per_scene *= repeat_scenes

        if fair_sampling:
            self.dataset = [[None for _ in d] for d in self.dataset_files_per_scene]
        else:
            self.dataset = [None for _ in self.dataset_files]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):

        if self.fair_sampling:
            scene = self.dataset_files_per_scene[key]
            file_idx = np.random.randint(0, len(scene))
            filename = scene[file_idx]
            datum = self.dataset[key][file_idx]
        else:
            filename = self.dataset_files[key]
            datum = self.dataset[key]

        if datum is None:

            data = np.load(filename, allow_pickle=True)

            pts1 = data[0].squeeze()
            pts2 = data[1].squeeze()
            sideinfo = data[2].squeeze()
            img1size = data[3]
            img2size = data[4]

            if self.max_score_ratio < 1.:
                pts1_list = []
                pts2_list = []
                sinf_list = []
                for di in range(pts1.shape[0]):
                    if sideinfo[di] < self.max_score_ratio:
                        pts1_list += [pts1[di]]
                        pts2_list += [pts2[di]]
                        sinf_list += [sideinfo[di]]

                pts1 = np.vstack(pts1_list)
                pts2 = np.vstack(pts2_list)
                sideinfo = np.vstack(sinf_list).squeeze()

            if self.normalize_coords:
                scale1 = np.max(img1size)
                scale2 = np.max(img2size)

                pts1[:,0] -= img1size[1]/2.
                pts2[:,0] -= img2size[1]/2.
                pts1[:,1] -= img1size[0]/2.
                pts2[:,1] -= img2size[0]/2.
                pts1 /= (scale1/2.)
                pts2 /= (scale2/2.)

                if self.augmentation:
                    if np.random.uniform(0, 1) < 0.5:
                        pts1[:,0] *= -1
                        pts2[:,0] *= -1
                    if np.random.uniform(0, 1) < 0.5:
                        pts1[:,1] *= -1
                        pts2[:,1] *= -1

                    scale_x = np.random.uniform(0.9, 1.1)
                    scale_y = np.random.uniform(0.9, 1.1)
                    pts1[:,0] *= scale_x
                    pts2[:,0] *= scale_x
                    pts1[:,1] *= scale_y
                    pts2[:,1] *= scale_y

                    shift_x = np.random.uniform(-.1, .1)
                    shift_y = np.random.uniform(-.1, .1)
                    pts1[:,0] += shift_x
                    pts2[:,0] += shift_x
                    pts1[:,1] += shift_y
                    pts2[:,1] += shift_y

            datum = {'points1': pts1, 'points2': pts2, 'sideinfo': sideinfo, 'img1size': img1size, 'img2size': img2size,}

            if self.keep_in_mem:

                if self.fair_sampling:
                    self.dataset[key][file_idx] = datum
                else:
                    self.dataset[key] = datum

        return datum


class HomographyDataset(Dataset):

    def __init__(self, data_dir_path, max_num_points, split='train', keep_in_mem=True, augmentation=False,
                 permute_points=True, max_score_ratio=1., use_indoor=False, fair_sampling=False):
        self.homdata = HomographyData(data_dir_path, split, keep_in_mem, normalize_coords=True,
                                      max_score_ratio=max_score_ratio, use_indoor=use_indoor,
                                      fair_sampling=fair_sampling, augmentation=augmentation)
        self.max_num_points = max_num_points
        self.permute_points = permute_points

    def __len__(self):
        return len(self.homdata)

    def __getitem__(self, key):
        datum = self.homdata[key]

        if self.max_num_points is None:
            max_num_points = datum['points1'].shape[0]
        else:
            max_num_points = self.max_num_points

        if self.permute_points:
            perm = np.random.permutation(datum['points1'].shape[0])
            datum['points1'] = datum['points1'][perm]
            datum['points2'] = datum['points2'][perm]
            datum['sideinfo'] = datum['sideinfo'][perm]

        points = np.zeros((max_num_points, 5)).astype(np.float32)
        mask = np.zeros((max_num_points, )).astype(np.int)

        num_actual_points = np.minimum(datum['points1'].shape[0], max_num_points)
        points[0:num_actual_points, 0:2] = datum['points1'][0:num_actual_points, :].copy()
        points[0:num_actual_points, 2:4] = datum['points2'][0:num_actual_points, :].copy()
        points[0:num_actual_points, 4] = datum['sideinfo'][0:num_actual_points].copy()

        mask[0:num_actual_points] = 1

        if num_actual_points < max_num_points:
            rest = max_num_points-num_actual_points
            points[num_actual_points:num_actual_points+rest, :] = points[0:rest, :].copy()

        return points, num_actual_points, mask


if __name__ == '__main__':

    dataset = HomographyData("/data/kluger/datasets/ngransac_data", split='all', normalize_coords=False,
                             max_score_ratio=0.9, use_indoor=True)
    num_matches = []

    for idx in range(len(dataset)):
        data = dataset[idx]

        pts1 = data['points1']
        pts2 = data['points2']
        img1size = data['img1size']

        print("%05d / %05d - matches: " % (idx, len(dataset)), pts1.shape[0])

        num_matches += [pts1.shape[0]]

    print(np.mean(num_matches), np.min(num_matches), np.max(num_matches), np.median(num_matches))

