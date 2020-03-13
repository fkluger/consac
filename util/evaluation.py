import numpy as np
import scipy.optimize
import torch
from util import sampling


def single_eval_nyu(true_vps, estm_vps, separate_errors=True, normalised_coords=True, missing_vp_penalty=90.):

    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02

    S = np.matrix([[1. / 320., 0, -1.], [0, 1. / 320., -.75], [0, 0, 1]])
    K = np.matrix([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])
    SK = S * K
    Kinv = K.I
    SKinv = SK.I

    invmat = SKinv if normalised_coords else Kinv

    true_num_vps = true_vps.shape[0]
    true_vds = (invmat * np.matrix(true_vps).T).T
    for vi in range(true_vds.shape[0]):
        true_vds[vi,:] /= np.maximum(np.linalg.norm(true_vds[vi,:]), 1e-16)

    estm_num_vps = estm_vps.shape[0]
    num_vp_penalty = np.maximum(true_num_vps-estm_num_vps, 0)

    missing_vps = -estm_num_vps+true_num_vps

    estm_vds = (invmat * np.matrix(estm_vps).T).T
    for vi in range(estm_vds.shape[0]):
        estm_vds[vi,:] /= np.maximum(np.linalg.norm(estm_vds[vi,:]), 1e-16)

    cost_matrix = np.arccos(np.abs(np.array(true_vds * estm_vds.T))) * 180. / np.pi

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    loss = cost_matrix[row_ind, col_ind].sum() + num_vp_penalty * missing_vp_penalty

    errors = []
    for ri, ci in zip(row_ind, col_ind):
        errors += [cost_matrix[ri,ci]]
    if missing_vp_penalty > 0:
        errors += [missing_vp_penalty for _ in range(num_vp_penalty)]

    if separate_errors:
        return errors, missing_vps, row_ind, col_ind
    else:
        return loss, missing_vps, row_ind, col_ind


def single_eval_yud(invmat, true_vps, estm_vps, separate_errors=True, missing_vp_penalty=90.):

    true_num_vps = true_vps.shape[0]
    true_vds = (invmat * np.matrix(true_vps).T).T
    for vi in range(true_vds.shape[0]):
        true_vds[vi,:] /= np.maximum(np.linalg.norm(true_vds[vi,:]), 1e-16)

    estm_num_vps = estm_vps.shape[0]
    num_vp_penalty = np.maximum(true_num_vps-estm_num_vps, 0)

    estm_vds = (invmat * np.matrix(estm_vps).T).T
    for vi in range(estm_vds.shape[0]):
        estm_vds[vi,:] /= np.maximum(np.linalg.norm(estm_vds[vi,:]), 1e-16)

    cost_matrix = np.arccos(np.abs(np.array(true_vds * estm_vds.T))) * 180. / np.pi

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    loss = cost_matrix[row_ind, col_ind].sum() + num_vp_penalty * missing_vp_penalty

    errors = []
    for ri, ci in zip(row_ind, col_ind):
        errors += [cost_matrix[ri,ci]]
    if missing_vp_penalty > 0:
        errors += [missing_vp_penalty for _ in range(num_vp_penalty)]

    if separate_errors:
        return errors, row_ind, col_ind
    else:
        return loss, row_ind, col_ind


def calc_labels_and_misclassification_error(data, num_estm_instances, estm_models, threshold, gt_labels):

    is_inlier = np.zeros((data.size(0),), dtype=np.int)
    estm_label = np.zeros((data.size(0),), dtype=np.int)
    min_dists = np.ones((data.size(0),)) * 100000
    transformed_points = []

    for mi in range(num_estm_instances):
        h = torch.from_numpy(estm_models[mi])
        distances, HX1 = sampling.homography_consistency_measure(h, data, None, return_transformed=True)
        distances = distances.cpu().numpy()
        HX1 = HX1.cpu().numpy()
        transformed_points += [HX1]
        for di in range(distances.shape[0]):
            if distances[di] < threshold * 100 and distances[di] < min_dists[di]:
                estm_label[di] = mi
                is_inlier[di] = 1
            min_dists[di] = distances[di] if distances[di] < min_dists[di] else min_dists[di]

    true_labels = gt_labels.numpy().astype(np.int)
    estm_labels = estm_label + 1
    estm_labels *= is_inlier
    num_estm_labels = np.max(estm_labels) + 1
    num_true_labels = np.max(true_labels) + 1
    true_clusters = [[] for _ in range(num_true_labels)]
    estm_clusters = [[] for _ in range(num_estm_labels)]
    for li in range(num_true_labels):
        for di in range(true_labels.shape[0]):
            if true_labels[di] == li:
                true_clusters[li] += [di]
    for li in range(num_estm_labels):
        for di in range(estm_labels.shape[0]):
            if estm_labels[di] == li:
                estm_clusters[li] += [di]

    iou_matrix = np.zeros((num_estm_labels, num_true_labels))

    for ti in range(num_true_labels):
        for ei in range(num_estm_labels):
            true_cluster = true_clusters[ti]
            estm_cluster = estm_clusters[ei]
            inter = len(set(true_cluster).intersection(set(estm_cluster)))
            union = len(set(true_cluster).union(set(estm_cluster)))
            iou = 1. * inter / union
            iou_matrix[ei, ti] = iou
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(1 - iou_matrix)
    assigned_labels = -1 * np.ones_like(estm_labels)
    for idx, ei in enumerate(row_ind):
        ai = col_ind[idx]
        for di in range(estm_labels.shape[0]):
            if estm_labels[di] == ei:
                assigned_labels[di] = ai
    num_correct = 0
    num_false = 0
    for di in range(assigned_labels.shape[0]):
        if assigned_labels[di] == true_labels[di]:
            num_correct += 1
        else:
            num_false += 1

    miss_rate = num_false * 1. / (num_false + num_correct)

    return assigned_labels, miss_rate


def calc_labels(data, num_estm_instances, estm_models, threshold):

    is_inlier = np.zeros((data.size(0),), dtype=np.int)
    estm_label = np.zeros((data.size(0),), dtype=np.int)
    min_dists = np.ones((data.size(0),)) * 100000
    transformed_points = []

    for mi in range(num_estm_instances):
        h = torch.from_numpy(estm_models[mi])
        distances, HX1 = sampling.homography_consistency_measure(h, data, None, return_transformed=True)
        distances = distances.cpu().numpy()
        HX1 = HX1.cpu().numpy()
        transformed_points += [HX1]
        for di in range(distances.shape[0]):
            if distances[di] < threshold and distances[di] < min_dists[di]:
                estm_label[di] = mi
                is_inlier[di] = 1
            min_dists[di] = distances[di] if distances[di] < min_dists[di] else min_dists[di]

    estm_labels = estm_label + 1
    estm_labels *= is_inlier

    return estm_labels