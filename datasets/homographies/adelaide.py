import os.path
import scipy.io
from torch.utils.data import Dataset
from util.sampling import *

class AdelaideRMF:

    homography_sequences = [
        "barrsmith.mat",
        "bonhall.mat",
        "bonython.mat",
        "elderhalla.mat",
        "elderhallb.mat",
        "hartley.mat",
        "johnsona.mat",
        "johnsonb.mat",
        "ladysymon.mat",
        "library.mat",
        "napiera.mat",
        "napierb.mat",
        "neem.mat",
        "nese.mat",
        "oldclassicswing.mat",
        "physics.mat",
        "sene.mat",
        "unihouse.mat",
        "unionhouse.mat",
    ]

    fundamental_sequences = [
        "cube.mat",
        "book.mat",
        "biscuit.mat",
        "game.mat",
        "biscuitbook.mat",
        "breadcube.mat",
        "breadtoy.mat",
        "cubechips.mat",
        "cubetoy.mat",
        "gamebiscuit.mat",
        "breadtoycar.mat",
        "carchipscube.mat",
        "toycubecar.mat",
        "breadcubechips.mat",
        "biscuitbookbox.mat",
        "cubebreadtoychips.mat",
        "breadcartoychips.mat",
        "dinobooks.mat",
        "boardgame.mat"
    ]

    def __init__(self, data_dir, keep_in_mem=True, normalize_coords=False, return_images=False):

        self.data_dir = data_dir
        self.keep_in_mem = keep_in_mem
        self.normalize_coords = normalize_coords
        self.return_images = return_images

        self.dataset_files = []

        sequences = self.homography_sequences

        for seq in sequences:
            seq_ = os.path.join(self.data_dir, seq)
            self.dataset_files += [seq_]

        self.dataset = [None for _ in self.dataset_files]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):

        filename = self.dataset_files[key]
        datum = self.dataset[key]

        if datum is None:

            data_mat = scipy.io.loadmat(filename, variable_names=["data", "label", "score", "img1", "img2"])
            pts1 = np.transpose(data_mat["data"][0:2,:])
            pts2 = np.transpose(data_mat["data"][3:5,:])
            sideinfo = data_mat["score"].squeeze()
            gt_label = data_mat["label"].squeeze()
            img1size = data_mat["img1"].shape[0:2]
            img2size = data_mat["img2"].shape[0:2]

            if self.normalize_coords:
                scale1 = np.max(img1size)
                scale2 = np.max(img2size)

                pts1[:,0] -= img1size[1]/2.
                pts2[:,0] -= img2size[1]/2.
                pts1[:,1] -= img1size[0]/2.
                pts2[:,1] -= img2size[0]/2.
                pts1 /= (scale1/2.)
                pts2 /= (scale2/2.)

            datum = {'points1': pts1, 'points2': pts2, 'sideinfo': sideinfo, 'img1size': img1size, 'img2size': img2size,
                     'labels': gt_label}

            if self.return_images:
                datum["img1"] = data_mat["img1"]
                datum["img2"] = data_mat["img2"]

            if self.keep_in_mem:
                self.dataset[key] = datum

        return datum


class AdelaideRMFDataset(Dataset):

    def __init__(self, data_dir_path, max_num_points, keep_in_mem=True,
                 permute_points=True, return_images=False, return_labels=True):
        self.homdata = AdelaideRMF(data_dir_path, keep_in_mem, normalize_coords=True, return_images=return_images)
        self.max_num_points = max_num_points
        self.permute_points = permute_points
        self.return_images = return_images
        self.return_labels = return_labels

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
            datum['labels'] = datum['labels'][perm]

        points = np.zeros((max_num_points, 5)).astype(np.float32)
        mask = np.zeros((max_num_points, )).astype(np.int)
        labels = np.zeros((max_num_points, )).astype(np.int)

        num_actual_points = np.minimum(datum['points1'].shape[0], max_num_points)
        points[0:num_actual_points, 0:2] = datum['points1'][0:num_actual_points, :].copy()
        points[0:num_actual_points, 2:4] = datum['points2'][0:num_actual_points, :].copy()
        points[0:num_actual_points, 4] = datum['sideinfo'][0:num_actual_points].copy()
        labels[0:num_actual_points] = datum['labels'][0:num_actual_points].copy()

        mask[0:num_actual_points] = 1

        if num_actual_points < max_num_points:
            rest = max_num_points-num_actual_points
            points[num_actual_points:num_actual_points+rest, :] = points[0:rest, :].copy()
            labels[num_actual_points:num_actual_points+rest] = labels[0:rest].copy()
        if self.return_labels:
            if self.return_images:
                return points, num_actual_points, mask, labels, (datum['img1'], datum['img2'])
            else:
                return points, num_actual_points, mask, labels
        else:
            return points, num_actual_points, mask
