import torch
import numpy as np
import time

def sigmoid(x):
    return 1. / (1. + torch.exp(-x))


def soft_inlier_fun_gen(beta, tau):
    def f(d):
        return 1 - sigmoid(beta * d - beta * tau)
    return f


def soft_inlier_fun(d, beta, tau):
    return 1 - sigmoid(beta * d - beta * tau)


def hard_inlier_fun(d, thresh):
    return torch.where(d < thresh, torch.ones_like(d), torch.zeros_like(d))


def sample_model_pool(data, num_data, num_hypotheses, cardinality, inlier_fun, model_gen_fun, consistency_fun,
                      probs=None, device=None, replacement=False, model_size=3, min_prob=0.):
    all_models = torch.zeros((num_hypotheses, model_size), device=device)
    all_inliers = torch.zeros((num_hypotheses, data.shape[0]), device=device)
    all_distances = torch.zeros((num_hypotheses, data.shape[0]), device=device)
    all_choice_vec = torch.zeros((num_hypotheses, data.shape[0]), device=device)

    for hi in range(num_hypotheses):
        cur_probs = None if probs is None else probs

        model, choice_vec, inliers, inlier_count, distances = sample_model(data, num_data, inlier_fun, cardinality,
                                                                           cur_probs, model_gen_fun, consistency_fun,
                                                                           device, replacement=replacement,
                                                                           min_prob=min_prob)
        all_inliers[hi, 0:num_data] = inliers
        all_distances[hi, 0:num_data] = distances
        all_choice_vec[hi] = choice_vec
        all_models[hi] = model

    return all_models, all_inliers, all_choice_vec, all_distances


def sample_model_pool_multiple(data, num_data, num_hypotheses, cardinality, inlier_fun, model_gen_fun, consistency_fun,
                               probs=None, device=None, replacement=False, model_size=3, sample_count=1, min_prob=0.):
    all_models = torch.zeros((num_hypotheses, sample_count, model_size), device=device)
    all_inliers = torch.zeros((num_hypotheses, sample_count, data.shape[0]), device=device)
    all_distances = torch.zeros((num_hypotheses, sample_count, data.shape[0]), device=device)
    all_choice_vec = torch.zeros((num_hypotheses, sample_count, data.shape[0]), device=device)

    for hi in range(num_hypotheses):
        cur_probs = None if probs is None else probs

        model, choice_vec, inliers, inlier_count, distances = sample_models(data, num_data, inlier_fun, cardinality,
                                                                            sample_count, cur_probs, model_gen_fun,
                                                                            consistency_fun, device,
                                                                            replacement=replacement, min_prob=min_prob)
        all_inliers[hi, :, 0:num_data] = inliers
        all_distances[hi, :, 0:num_data] = distances
        all_choice_vec[hi] = choice_vec
        all_models[hi] = model

    return all_models, all_inliers, all_choice_vec, all_distances


def sample_model_pool_multiple_parallel(data, num_data, cardinality, inlier_fun, model_gen_fun, consistency_fun,
                               probs=None, device=None, replacement=False, model_size=3, sample_count=1, min_prob=0.):

    Y = data.size(0)
    P = probs.size(0)
    S = probs.size(1)

    cur_probs = None if probs is None else probs

    models, choice_vec, inliers, inlier_count, distances = sample_models_parallel(data, num_data, inlier_fun, cardinality,
                                                                            sample_count, cur_probs, model_gen_fun,
                                                                            consistency_fun, device,
                                                                            replacement=replacement, min_prob=min_prob)
    return models, inliers, choice_vec, distances



def sample_model(data, num_data, inlier_fun, cardinality, probs, model_gen_fun, consistency_fun, device=None,
                 replacement=False, min_prob=0.):
    if probs is None:
        choice_weights = torch.ones(num_data, device=device)
        choice = torch.multinomial(choice_weights, cardinality, replacement=replacement)
    else:
        choice_weights = probs[0:num_data] + min_prob
        choice = torch.multinomial(choice_weights, cardinality, replacement=replacement)

    choice_vec = torch.zeros((data.shape[0],), device=device)
    choice_vec[choice] = 1

    model = model_gen_fun(data, choice, device)

    inliers, inlier_count, distances = count_inliers(data[0:num_data, :], model, inlier_fun, consistency_fun, device)

    return model, choice_vec, inliers, inlier_count, distances


def sample_models(data, num_data, inlier_fun, cardinality, sample_count, probs, model_gen_fun, consistency_fun,
                  device=None, replacement=False, min_prob=0.):
    if probs is None:
        choice_weights = torch.ones((sample_count, num_data), device=device)
        choice = torch.multinomial(choice_weights, cardinality, replacement=replacement)
    else:
        choice_weights = probs[:, 0:num_data] + min_prob
        choice = torch.multinomial(choice_weights, cardinality, replacement=replacement)

    choice_vec = torch.zeros((sample_count, data.shape[0],), device=device)
    for si in range(sample_count):
        choice_vec[si, choice[si]] = 1

    models = model_gen_fun(data, choice, device, sample_count=sample_count)

    inliers, inlier_count, distances = count_inliers(data[0:num_data, :], models, inlier_fun, consistency_fun, device)

    return models, choice_vec, inliers, inlier_count, distances


def sample_models_parallel(data, num_data, inlier_fun, cardinality, sample_count, probs, model_gen_fun, consistency_fun,
                  device=None, replacement=False, min_prob=0.):
    P = probs.size(0)
    S = probs.size(1)
    Y = probs.size(2)
    choice_vec = torch.zeros((P, S, Y), device=device)
    choices = torch.zeros((P, S, cardinality), device=device, dtype=torch.long)
    for pi in range(P):
        choice_weights = probs[pi, :, 0:num_data] + min_prob
        choice = torch.multinomial(choice_weights, cardinality, replacement=replacement)
        choices[pi] = choice
        choice_vec[pi, :, choice] = 1

    models = model_gen_fun(data, choices, device, sample_count=sample_count)

    all_inliers, all_counts, all_distances = \
        count_inliers(data[0:num_data, :], models, inlier_fun, consistency_fun, device)

    return models, choice_vec, all_inliers, all_counts, all_distances


def count_inliers(data, model, inlier_fun, consistency_fun, device=None):
    distances = consistency_fun(model, data, device)
    inliers = inlier_fun(distances)
    inlier_count = inliers.sum(-1)

    return inliers, inlier_count, distances


def vp_consistency_measure_angle(vp, data, device):
    centroids = data[:, 9:12]
    lines = data[:, 6:9]

    vp_ex = vp.expand(centroids.size())
    constrained_lines = torch.cross(centroids, vp_ex)
    con_line_norms = torch.norm(constrained_lines[:, 0:2], dim=1).unsqueeze_(-1).expand(constrained_lines.size())
    constrained_lines /= (con_line_norms + 1e-8)
    line_norms = torch.norm(lines[:, 0:2], dim=1).unsqueeze_(-1).expand(lines.size())
    lines /= (line_norms + 1e-8)
    distances = 1 - torch.abs(torch.mul(lines[:, 0:2], constrained_lines[:, 0:2]).sum(dim=1))

    return distances


def vp_consistency_measure_angle_np(vp, data):
    centroids = data[:, 9:12]
    lines = data[:, 6:9]

    constrained_lines = np.zeros((data.shape[0], 3))
    for di in range(data.shape[0]):
        constrained_lines[di] = np.cross(centroids[di], vp)
        line_norm = np.linalg.norm(constrained_lines[di, 0:2])
        constrained_lines[di] /= line_norm
        line_norm = np.linalg.norm(lines[di, 0:2])
        lines[di] /= line_norm

    distances = 1 - np.abs((lines[:, 0:2] * constrained_lines[:, 0:2]).sum(axis=1))

    return distances


def vp_from_lines(data, choice, device):
    datum1 = data[choice[0], 6:9]
    datum2 = data[choice[1], 6:9]

    vp = torch.cross(datum1, datum2)

    vp_norm = torch.norm(vp, keepdim=True) + 1e-8
    vp = vp / vp_norm

    return vp


def homography_from_points(data, choice, device, n_matches=4):
    A = torch.zeros((2 * n_matches + 1, 9), device=None)
    for i in range(n_matches):
        A[2 * i, 3:5] = data[choice[i], 0:2] * (-1)
        A[2 * i, 5] = -1
        A[2 * i, 6:8] = data[choice[i], 0:2] * data[choice[i], 3]
        A[2 * i, 8] = data[choice[i], 3]
        A[2 * i + 1, 0:2] = data[choice[i], 0:2]
        A[2 * i + 1, 2] = 1
        A[2 * i + 1, 6:8] = data[choice[i], 0:2] * data[choice[i], 2] * (-1)
        A[2 * i + 1, 8] = -data[choice[i], 2]

    u, s, v = torch.svd(A)
    h = v[:, -1].to(device)
    return h


def homographies_from_points(data, choice, device, n_matches=4, sample_count=1):
    A = torch.zeros((sample_count, 2 * n_matches + 1, 9), device=device)
    for si in range(sample_count):
        for i in range(n_matches):
            A[si, 2 * i, 3:5] = data[choice[si, i], 0:2] * (-1)
            A[si, 2 * i, 5] = -1
            A[si, 2 * i, 6:8] = data[choice[si, i], 0:2] * data[choice[si, i], 3]
            A[si, 2 * i, 8] = data[choice[si, i], 3]
            A[si, 2 * i + 1, 0:2] = data[choice[si, i], 0:2]
            A[si, 2 * i + 1, 2] = 1
            A[si, 2 * i + 1, 6:8] = data[choice[si, i], 0:2] * data[choice[si, i], 2] * (-1)
            A[si, 2 * i + 1, 8] = -data[choice[si, i], 2]

    u, s, v = torch.svd(A.cpu())
    h = v[:, :, -1].to(device)

    return h


def multiple_svds(matrix):
    u, s, v = torch.svd(matrix)
    h = v[:, :, -1]
    return h


def homographies_from_points_parallel(data, choice, device, n_matches=4, sample_count=1):
    P = choice.size(0)
    S = choice.size(1)

    A = torch.zeros((P, S, 2 * n_matches + 1, 9), device=device)
    for i in range(n_matches):
        A[:, :, 2 * i, 5] = -1
        A[:, :, 2 * i + 1, 2] = 1
        A[:, :, 2 * i, 3:5] = data[choice[:, :, i], 0:2] * (-1)
        A[:, :, 2 * i, 6:8] = data[choice[:, :, i], 0:2]
        A[:, :, 2 * i, 6] *= data[choice[:, :, i], 3]
        A[:, :, 2 * i, 7] *= data[choice[:, :, i], 3]
        A[:, :, 2 * i, 8] = data[choice[:, :, i], 3]
        A[:, :, 2 * i + 1, 0:2] = data[choice[:, :, i], 0:2]
        A[:, :, 2 * i + 1, 6:8] = data[choice[:, :, i], 0:2] * (-1)
        A[:, :, 2 * i + 1, 6] *= data[choice[:, :, i], 2]
        A[:, :, 2 * i + 1, 7] *= data[choice[:, :, i], 2]
        A[:, :, 2 * i + 1, 8] = -data[choice[:, :, i], 2]

    B = A.cpu()
    u, s, v = torch.svd(B)
    h = v[:, :, :, -1].to(device)

    return h


def homography_consistency_measure(h, data, device, return_transformed=False):
    H = h.view(3, 3)
    if torch.det(H) < 1e-12:
        Hinv = H
    else:
        Hinv = torch.inverse(H)
    x1 = data[:, 0:2]
    x2 = data[:, 2:4]
    X1 = torch.ones((data.size(0), 3, 1), device=device)
    X2 = torch.ones((data.size(0), 3, 1), device=device)
    X1[:, 0:2, 0] = x1
    X2[:, 0:2, 0] = x2

    HX1 = torch.matmul(H, X1)
    HX1[:, 0, 0] /= torch.clamp(HX1[:, 2, 0], min=1e-8)
    HX1[:, 1, 0] /= torch.clamp(HX1[:, 2, 0], min=1e-8)
    HX1[:, 2, 0] /= torch.clamp(HX1[:, 2, 0], min=1e-8)
    HX2 = torch.matmul(Hinv, X2)
    HX2[:, 0, 0] /= torch.clamp(HX2[:, 2, 0], min=1e-8)
    HX2[:, 1, 0] /= torch.clamp(HX2[:, 2, 0], min=1e-8)
    HX2[:, 2, 0] /= torch.clamp(HX2[:, 2, 0], min=1e-8)

    signed_distances_1 = X2 - HX1
    distances_1 = (signed_distances_1 * signed_distances_1).sum(dim=1).squeeze()
    signed_distances_2 = X1 - HX2
    distances_2 = (signed_distances_2 * signed_distances_2).sum(dim=1).squeeze()

    distances = distances_1 + distances_2

    if return_transformed:
        return distances, HX1
    else:
        return distances


def homographies_consistency_measure(h, data, device):
    H = h.view(-1, 1, 3, 3)
    I = torch.eye(3, device=device).view(1, 1, 3, 3).expand(H.size(0), H.size(1), 3, 3)
    det = torch.det(H).view(-1, 1, 1, 1)
    H_ = torch.where(det < 1e-12, I, H)
    Hinv = torch.inverse(H_)
    x1 = data[:, 0:2]
    x2 = data[:, 2:4]
    X1 = torch.ones((1, data.size(0), 3, 1), device=device)
    X2 = torch.ones((1, data.size(0), 3, 1), device=device)
    X1[0, :, 0:2, 0] = x1
    X2[0, :, 0:2, 0] = x2

    HX1 = torch.matmul(H, X1)
    HX1[:, :, 0, 0] /= torch.clamp(HX1[:, :, 2, 0], min=1e-8)
    HX1[:, :, 1, 0] /= torch.clamp(HX1[:, :, 2, 0], min=1e-8)
    HX1[:, :, 2, 0] /= torch.clamp(HX1[:, :, 2, 0], min=1e-8)
    HX2 = torch.matmul(Hinv, X2)
    HX2[:, :, 0, 0] /= torch.clamp(HX2[:, :, 2, 0], min=1e-8)
    HX2[:, :, 1, 0] /= torch.clamp(HX2[:, :, 2, 0], min=1e-8)
    HX2[:, :, 2, 0] /= torch.clamp(HX2[:, :, 2, 0], min=1e-8)

    signed_distances_1 = HX1 - X2
    distances_1 = (signed_distances_1 * signed_distances_1).sum(dim=2).squeeze()
    signed_distances_2 = HX2 - X1
    distances_2 = (signed_distances_2 * signed_distances_2).sum(dim=2).squeeze()

    distances = distances_1 + distances_2

    return distances


def homographies_consistency_measure_parallel(h, data, device):
    P = h.size(0)
    S = h.size(1)

    H = h.view(P, S, 1, 3, 3)
    I = torch.eye(3, device=device).view(1, 1, 1, 3, 3).expand(H.size(0), H.size(1), H.size(2), 3, 3)
    det = torch.det(H).view(P, S, 1, 1, 1)
    H_ = torch.where(det < 1e-12, I, H)
    Hinv = torch.inverse(H_)
    x1 = data[:, 0:2]
    x2 = data[:, 2:4]
    X1 = torch.ones((1, 1, data.size(0), 3, 1), device=device)
    X2 = torch.ones((1, 1, data.size(0), 3, 1), device=device)
    X1[0, 0, :, 0:2, 0] = x1
    X2[0, 0, :, 0:2, 0] = x2

    HX1 = torch.matmul(H, X1)
    HX1[:, :, :, 0, 0] /= torch.clamp(HX1[:, :, :, 2, 0], min=1e-8)
    HX1[:, :, :, 1, 0] /= torch.clamp(HX1[:, :, :, 2, 0], min=1e-8)
    HX1[:, :, :, 2, 0] /= torch.clamp(HX1[:, :, :, 2, 0], min=1e-8)
    HX2 = torch.matmul(Hinv, X2)
    HX2[:, :, :, 0, 0] /= torch.clamp(HX2[:, :, :, 2, 0], min=1e-8)
    HX2[:, :, :, 1, 0] /= torch.clamp(HX2[:, :, :, 2, 0], min=1e-8)
    HX2[:, :, :, 2, 0] /= torch.clamp(HX2[:, :, :, 2, 0], min=1e-8)

    signed_distances_1 = HX1 - X2
    distances_1 = (signed_distances_1 * signed_distances_1).sum(dim=3).squeeze()
    signed_distances_2 = HX2 - X1
    distances_2 = (signed_distances_2 * signed_distances_2).sum(dim=3).squeeze()

    distances = distances_1 + distances_2

    return distances


def line_consistency_measure(line, data, device):
    model_ex = line.expand(data.size())
    distances = torch.abs(torch.mul(data, model_ex).sum(dim=1))

    return distances


def lines_consistency_measure(lines, data, device):
    data_ex = data.unsqueeze(0)
    model_ex = lines.unsqueeze(1)
    distances = torch.abs(torch.mul(data_ex, model_ex).sum(dim=-1))

    return distances


def line_from_points(data, choice, device):
    datum1 = data[choice[0], :]
    datum2 = data[choice[1], :]

    line = torch.cross(datum1, datum2)
    norm = torch.norm(line[0:2], keepdim=False)
    line = line / torch.clamp(norm, min=1e-9)

    return line


def lines_from_points(data, choice, device, sample_count=1):
    lines = torch.zeros(sample_count, 3, device=device)
    for si in range(sample_count):
        datum1 = data[choice[si, 0], :]
        datum2 = data[choice[si, 1], :]

        line = torch.cross(datum1, datum2)
        norm = torch.norm(line[0:2], keepdim=True)
        line = line / torch.clamp(norm, min=1e-9)
        lines[si] = line

    return lines
