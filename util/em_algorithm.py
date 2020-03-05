import torch
import math

def e_step(models, data, distance_fun, priors, variances, data_weights, device=None):
    # data: B x N x C
    # data_weights: B x N
    # models: B x M x D
    # priors: B x M
    # variances: B x M
    # distances: B x N x M

    distances = distance_fun(models, data, device=device)

    if priors is None:
        priors = torch.ones((models.size(0), models.size(1)), device=device)
    if isinstance(variances, (int, float)):
        variances = variances * torch.ones((models.size(0), models.size(1)), device=device)

    variances_ = variances.unsqueeze(1).expand(-1, distances.size(1), -1)
    data_weights_ = data_weights.unsqueeze(-1).expand(-1, -1, distances.size(2))
    # likelihood: B x N x M
    likelihood = 1. / torch.sqrt(2 * math.pi * variances_) * torch.exp(
        -distances * distances / (2 * variances_)) * data_weights_

    # marginal: B x N
    marginal = torch.max(torch.matmul(likelihood, priors.unsqueeze(-1)), 1e-8 * torch.ones(1, device=device))

    # posterior: B x N x M
    posterior = likelihood * priors.unsqueeze(-1).transpose(1, 2) / marginal

    return posterior


def m_step_homographies_svd(data, weights, distance_fun, device=None):
    # data: B x N x C
    # weights: B x N x M

    B = weights.size(0)
    N = weights.size(1)
    M = weights.size(2)

    A = torch.zeros((B, M, 2 * N, 9), device=device)
    for bi in range(B):
        A[bi, :, ::2, 3:5] = data[bi, :, 0:2]
        A[bi, :, ::2, 6] = data[bi, :, 0] * data[bi, :, 3]
        A[bi, :, ::2, 7] = data[bi, :, 1] * data[bi, :, 3]
        A[bi, :, ::2, 8] = data[bi, :, 3]
        A[bi, :, 1::2, 0:2] = data[bi, :, 0:2]
        A[bi, :, 1::2, 6] = data[bi, :, 0] * data[bi, :, 2] * (-1)
        A[bi, :, 1::2, 7] = data[bi, :, 1] * data[bi, :, 2] * (-1)
        A[bi, :, 1::2, 8] = -data[bi, :, 2]

    for bi in range(B):
        for mi in range(M):
            A[bi, mi, ::2, 3] *= (-1) * weights[bi, :, mi]
            A[bi, mi, ::2, 4] *= (-1) * weights[bi, :, mi]
            A[bi, mi, ::2, 5] = -1 * weights[bi, :, mi]
            A[bi, mi, ::2, 6] *= weights[bi, :, mi]
            A[bi, mi, ::2, 7] *= weights[bi, :, mi]
            A[bi, mi, ::2, 8] *= weights[bi, :, mi]
            A[bi, mi, 1::2, 0] *= weights[bi, :, mi]
            A[bi, mi, 1::2, 1] *= weights[bi, :, mi]
            A[bi, mi, 1::2, 2] = 1 * weights[bi, :, mi]
            A[bi, mi, 1::2, 6] *= weights[bi, :, mi]
            A[bi, mi, 1::2, 7] *= weights[bi, :, mi]
            A[bi, mi, 1::2, 8] *= weights[bi, :, mi]

    u, s, v = torch.svd(A.cpu())
    h = v[:, :, :, -1].to(device)
    h *= torch.sign(h[:, :, -1].unsqueeze(-1))

    distances = distance_fun(h, data, device=device)

    distances_sq = distances * distances
    distances_sq_weighted = distances_sq * weights
    distances_sqwt_sum = distances_sq_weighted.sum(1)
    weights_sum = torch.max(weights.sum(1), 1e-8 * torch.ones(1, device=device))

    variances = distances_sqwt_sum / weights_sum

    return h, variances, distances


def m_step_vp_svd(data, weights, distance_fun, device=None):
    # data: B x N x C
    # weights: B x N x M
    M = weights.size(2)

    # lines: B x N x 3
    lines = data[:, :, 6:9]

    # lines_repeated: B x N x M x 3
    lines_repeated = lines.unsqueeze(2).expand(-1, -1, M, -1)

    # weights_repeated: B x N x M x 3
    weights_repeated = weights.unsqueeze(-1).expand(-1, -1, -1, 3)

    # Mat: B x M x N x 3
    Mat = (lines_repeated * weights_repeated).transpose(1, 2)
    # MatSq: B x M x 3 x 3
    MatSq = torch.matmul(Mat.transpose(2, 3), Mat)
    # V: BM x 3 x 3
    U, S, V = torch.svd(MatSq)

    # vps: B x M x 3
    vps = V[:, :, :, 2]

    # distances: B x N x M
    distances = distance_fun(vps, data)

    distances_sq = distances * distances
    distances_sq_weighted = distances_sq * weights
    distances_sqwt_sum = distances_sq_weighted.sum(1)
    weights_sum = torch.max(weights.sum(1), 1e-8 * torch.ones(1, device=device))

    variances = distances_sqwt_sum / weights_sum

    return vps, variances, distances


def m_step_vp_grad(data, weights, distance_fun, vps, device=None):
    # data: B x N x C
    # weights: B x N x M
    M = weights.size(2)

    distances = distance_fun(vps, data)

    for i in range(3):
        weighted_distances = weights * distances
        weighted_error = (weighted_distances * weighted_distances).sum(1)
        weighted_error.backward(torch.ones_like(weighted_error, device=device), retain_graph=True)

        vps = vps.detach() - vps.grad
        vps.requires_grad_(True)

        distances = distance_fun(vps, data)

    distances_sq = distances * distances
    distances_sq_weighted = distances_sq * weights
    distances_sqwt_sum = distances_sq_weighted.sum(1)
    weights_sum = torch.max(weights.sum(1), 1e-8 * torch.ones(1, device=device))

    variances = distances_sqwt_sum / weights_sum

    return vps, variances, distances


def vp_consistency_measure_batched(vps, data, device=None):
    # data: B x N x C
    # vps: B x M x 3

    centroids = data[:, :, 9:12]
    endpoints = data[:, :, 0:3]

    # B x M x N x 3
    vps_ex = vps.unsqueeze(2).expand(-1, -1, centroids.size(1), -1)
    centroids_ex = centroids.unsqueeze(1).expand(-1, vps.size(1), -1, -1)
    endpoints_ex = endpoints.unsqueeze(1).expand(-1, vps.size(1), -1, -1)
    constrained_lines = torch.cross(centroids_ex, vps_ex, dim=3)

    line_norms = torch.norm(constrained_lines[:, :, :, 0:2], dim=-1).unsqueeze(-1).expand(constrained_lines.size())
    constrained_lines = constrained_lines / (line_norms + 1e-8)
    distances = torch.abs(torch.mul(endpoints_ex, constrained_lines).sum(dim=-1)).transpose(1, 2)

    return distances


def vp_consistency_measure_angle_batched(vps, data, device=None):
    # data: B x N x C
    # vps: B x M x 3

    centroids = data[:, :, 9:12]
    lines = data[:, :, 6:9]

    # B x M x N x 3
    vps_ex = vps.unsqueeze(2).expand(-1, -1, centroids.size(1), -1)
    centroids_ex = centroids.unsqueeze(1).expand(-1, vps.size(1), -1, -1)
    lines_ex = lines.unsqueeze(1).expand(-1, vps.size(1), -1, -1)
    constrained_lines = torch.cross(centroids_ex, vps_ex, dim=3)

    con_line_norms = torch.norm(constrained_lines[:, :, :, 0:2], dim=-1).unsqueeze(-1).expand(constrained_lines.size())
    constrained_lines = constrained_lines / (con_line_norms + 1e-8)
    line_norms = torch.norm(lines_ex[:, :, :, 0:2], dim=-1).unsqueeze(-1).expand(constrained_lines.size())
    lines_ex = lines_ex / (line_norms + 1e-8)

    distances = 1 - torch.abs(torch.mul(lines_ex[:, :, :, 0:2], constrained_lines[:, :, :, 0:2]).sum(dim=-1)).transpose(
        1, 2)

    return distances


def homographies_consistency_measure(h, data, device=None, return_transformed=False):
    # data: B x N x C
    # h: B x M x 9
    B = data.size(0)
    N = data.size(1)
    M = h.size(1)
    H = h.view(B, 1, M, 3, 3)
    I = torch.eye(3, device=device).view(1, 1, 1, 3, 3).expand(H.size(0), 1, H.size(2), 3, 3)
    det = torch.det(H).view(B, 1, M, 1, 1).expand(H.size(0), 1, H.size(2), 3, 3)
    H_ = torch.where(det < 1e-8, I, H)
    H__ = torch.where(torch.isnan(det), I, H_)
    Hinv = torch.inverse(H__)
    x1 = data[:, :, 0:2]
    x2 = data[:, :, 2:4]
    X1 = torch.ones((B, N, 1, 3, 1), device=device)
    X2 = torch.ones((B, N, 1, 3, 1), device=device)
    X1[:, :, 0, 0:2, 0] = x1
    X2[:, :, 0, 0:2, 0] = x2

    HX1 = torch.matmul(H, X1)
    HX1[:, :, :, 0, 0] /= torch.clamp(HX1[:, :, :, 2, 0], min=1e-8)
    HX1[:, :, :, 1, 0] /= torch.clamp(HX1[:, :, :, 2, 0], min=1e-8)
    HX1[:, :, :, 2, 0] /= torch.clamp(HX1[:, :, :, 2, 0], min=1e-8)
    HX2 = torch.matmul(Hinv, X2)
    HX2[:, :, :, 0, 0] /= torch.clamp(HX2[:, :, :, 2, 0], min=1e-8)
    HX2[:, :, :, 1, 0] /= torch.clamp(HX2[:, :, :, 2, 0], min=1e-8)
    HX2[:, :, :, 2, 0] /= torch.clamp(HX2[:, :, :, 2, 0], min=1e-8)

    signed_distances_1 = HX1 - X2
    distances_1 = (signed_distances_1 * signed_distances_1).sum(dim=3).squeeze(-1)
    signed_distances_2 = HX2 - X1
    distances_2 = (signed_distances_2 * signed_distances_2).sum(dim=3).squeeze(-1)

    distances = distances_1 + distances_2

    return distances


def em_for_vp(data, vps, data_weights, init_variance=1e-6, iterations=1, device=None):
    variances = init_variance

    all_vps = [vps]

    posterior = None

    for i in range(iterations):
        posterior = e_step(vps, data, vp_consistency_measure_angle_batched, None, variances, data_weights,
                           device=device)
        vps, _, distances = m_step_vp_svd(data, posterior, vp_consistency_measure_batched, device=device)
        all_vps += [vps]

    all_vps = torch.stack(all_vps, dim=1)
    return vps, posterior, variances, all_vps


def em_for_homographies(data, homographies, data_weights, init_variance=1e-6, iterations=1, device=None):
    variances = init_variance

    all_homs = [homographies]

    for i in range(iterations):
        posterior = e_step(homographies, data, homographies_consistency_measure, None, variances, data_weights,
                           device=device)
        homographies, _, distances = m_step_homographies_svd(data, posterior, homographies_consistency_measure,
                                                             device=device)
        all_homs += [homographies]

    all_homs = torch.stack(all_homs, dim=1)
    return homographies, posterior, variances, all_homs
