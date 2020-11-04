from util.misc import *
from util.cn_net import CNNet
from util import sampling
from util.em_algorithm import em_for_homographies, em_for_vp
from util.evaluation import calc_labels
import datasets.nyu_vp.lsd.lsd as lsd
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
import random
import cv2


torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(
    description='',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--problem",
    choices=[
        "homography",
        "vp"
    ],
    # default="vp",
    default="homography",
    help="Choose either homography or vanishing point (VP) estimation."
)
parser.add_argument('--defaults', dest='defaults', action='store_true', help='use default parameters for the problem', default=False)
parser.add_argument('--img1', default='demo/img1.jpg', help='left image or image for VP estimation')
parser.add_argument('--img2', default='demo/img2.jpg', help='right image (for homography estimation)')
parser.add_argument('--ckpt', default='models/consac-s_homography.net', help='path to NN weights')
parser.add_argument('--threshold', '-t', type=float, default=0.0001, help='tau - inlier threshold')
parser.add_argument('--threshold2', type=float, default=0.0002, help='theta - inlier threshold')
parser.add_argument('--inlier_cutoff', type=float, default=50, help='Theta - inlier cutoff')
parser.add_argument('--hyps', '-hyps', type=int, default=100, help='S - inner hypotheses (single instance hypotheses)')
parser.add_argument('--outerhyps', type=int, default=100, help='P - outer hypotheses (multi-hypotheses)')
parser.add_argument('--instances', type=int, default=5, help='Max. number of instances')
parser.add_argument('--em', type=int, default=10, help='Number of EM iterations')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--uniform', dest='uniform', action='store_true', help='disable guided sampling', default=False)
parser.add_argument('--cpu', dest='cpu', action='store_true', help='Run CPU only', default=False)
parser.add_argument('--unconditional', dest='unconditional', action='store_true', help='disable conditional sampling',
                    default=False)

opt = parser.parse_args()

if opt.defaults:
    if opt.problem == "vp":
        opt.ckpt = 'models/consac_vp.net'
        opt.img1 = 'demo/nord_lb.jpg'
        opt.threshold = 0.001
        opt.threshold2 = 0.001
        opt.hyps = 32
        opt.outerhyps = 32
        opt.instances = 7
        opt.em = 10
    elif opt.problem == "homography":
        opt.ckpt = 'models/consac-s_homography.net'
        opt.img1 = 'demo/img1.jpg'
        opt.img2 = 'demo/img2.jpg'
        opt.threshold = 0.0001
        opt.threshold2 = 0.0002
        opt.hyps = 100
        opt.outerhyps = 100
        opt.instances = 5
        opt.em = 10


def get_homography_features_for_demo(opt):
    img1 = cv2.imread(opt.img1)
    img2 = cv2.imread(opt.img2)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])

    M = max([img1.shape[0], img2.shape[0]])
    N = img1.shape[1] + img2.shape[1]

    data = np.zeros((len(good), 4), dtype=np.float32)
    data_unscaled = np.zeros((len(good), 4))
    img_width = img1.shape[1]
    img_height = img1.shape[0]
    scale = np.maximum(img_width, img_height)/2
    for idx, match in enumerate(good):
        idx1 = match[0].queryIdx
        idx2 = match[0].trainIdx
        p1 = [(kp1[idx1].pt[0]-img_width/2)/scale,
              (kp1[idx1].pt[1]-img_height/2)/scale]
        p2 = [(kp2[idx2].pt[0]-img_width/2)/scale,
              (kp2[idx2].pt[1]-img_height/2)/scale]

        data[idx, 0] = p1[0]
        data[idx, 1] = p1[1]
        data[idx, 2] = p2[0]
        data[idx, 3] = p2[1]

        p1 = [(kp1[idx1].pt[0]),
              (kp1[idx1].pt[1])]
        p2 = [(kp2[idx2].pt[0]),
              (kp2[idx2].pt[1])]

        data_unscaled[idx, 0] = p1[0]
        data_unscaled[idx, 1] = p1[1]
        data_unscaled[idx, 2] = p2[0]
        data_unscaled[idx, 3] = p2[1]

    return data, data_unscaled, img1, img2


def get_vp_features_for_demo(opt):
    image_rgb = cv2.imread(opt.img1)
    image = rgb2gray(image_rgb)

    height = image.shape[0]
    width = image.shape[1]

    lsd_line_segments = lsd.detect_line_segments(image)

    lsd_line_segments_scaled = lsd_line_segments.copy()
    lsd_line_segments_scaled[:, 0] -= width/2
    lsd_line_segments_scaled[:, 2] -= width/2
    lsd_line_segments_scaled[:, 1] -= height/2
    lsd_line_segments_scaled[:, 3] -= height/2
    lsd_line_segments_scaled[:, 0:4] /= width/2

    line_segments = lsd_line_segments[:, 0:4]

    line_segments_scaled = np.zeros((lsd_line_segments_scaled.shape[0], 12), dtype=np.float32)
    for li in range(line_segments.shape[0]):
        p1 = np.array([lsd_line_segments_scaled[li, 0], lsd_line_segments_scaled[li, 1], 1])
        p2 = np.array([lsd_line_segments_scaled[li, 2], lsd_line_segments_scaled[li, 3], 1])
        centroid = 0.5 * (p1 + p2)
        line = np.cross(p1, p2)
        line /= np.linalg.norm(line[0:2])
        line_segments_scaled[li, 0:3] = p1
        line_segments_scaled[li, 3:6] = p2
        line_segments_scaled[li, 6:9] = line
        line_segments_scaled[li, 9:12] = centroid

    return line_segments_scaled, line_segments, image_rgb


torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

if opt.problem == "homography":
    data, data_unscaled, img1, img2 = get_homography_features_for_demo(opt)
    model_dim = 9
    ddim = 5
    batch_norm = False
    model_gen_fun = sampling.homographies_from_points_parallel
    consistency_fun = sampling.homographies_consistency_measure_parallel_2dim
    min_set_size = 4
    em_fun = em_for_homographies
elif opt.problem == "vp":
    data, data_unscaled, img1 = get_vp_features_for_demo(opt)
    model_dim = 3
    ddim = 9
    batch_norm = True
    model_gen_fun = sampling.vp_from_lines_batched
    consistency_fun = sampling.vp_consistency_measure_angle_batched
    min_set_size = 2
    em_fun = em_for_vp
else:
    assert False, "unknown problem type %s" % opt.problem

device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu', 0)

print_options(opt)

model = CNNet(6, ddim, batch_norm=batch_norm)
model = model.to(device)

inlier_fun1 = sampling.soft_inlier_fun_gen(5. / opt.threshold, opt.threshold)

if not opt.uniform:
    checkpoint = torch.load(opt.ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint, strict=False)

all_losses = []

uniform_sampling = opt.uniform


data = torch.from_numpy(data).unsqueeze(0)
B = data.size(0)
Y = data.size(1)
masks = torch.ones((B,Y), device=device)
data = data.to(device)
masks = masks.to(device)

M = opt.instances
P = opt.outerhyps
S = opt.hyps

with torch.no_grad():
    all_inliers = torch.zeros((P, M, S, Y), device=device)
    all_probs = torch.zeros((P, M, Y), device=device)
    all_best_inliers = torch.zeros((P, M, Y), device=device)
    all_best_inlier_counts = torch.zeros(P, device=device)
    all_best_hypos = torch.zeros((P, M,), device=device, dtype=torch.long)
    all_models = torch.zeros((P, M, S, model_dim), device=device)
    all_best_models = torch.zeros((P, M, model_dim), device=device)

    data_and_state = torch.zeros((P, Y, ddim),
                                 device=device)
    for oh in range(P):
        data_and_state[oh, :, 0:(ddim - 1)] = data[0, :, 0:(ddim - 1)]

    uniform_probs = torch.ones((P, Y), device=device)

    inliers_so_far = torch.zeros((P, Y), device=device)

    for mi in range(M):

        if uniform_sampling:
            probs = uniform_probs
        else:
            data = data.to(device)
            log_probs = model(data_and_state)
            probs = torch.softmax(log_probs[:, 0, 0:Y, 0], dim=-1)

        all_probs[:, mi, :Y] = probs

        cur_probs = probs.view(P, 1, 1, Y).expand((P, S, 1, Y))
        models, inliers, choices, distances = \
            sampling.sample_model_pool_multiple_parallel(data[0], Y, min_set_size, inlier_fun1, model_gen_fun,
                                                         consistency_fun, cur_probs, device=device,
                                                         model_size=model_dim, sample_count=S)

        all_inliers[:, mi, :, 0:Y] = inliers.squeeze()[:, 0:Y]
        all_models[:, mi, :] = models.squeeze()

        for oh in range(P):

            all_inliers_so_far = torch.max(all_inliers[oh, mi], inliers_so_far[oh])
            all_inlier_counts_so_far = torch.sum(all_inliers_so_far, dim=-1)

            best_hypo = torch.argmax(all_inlier_counts_so_far)
            all_best_hypos[oh, mi] = best_hypo
            all_best_models[oh, mi] = models[oh].squeeze()[best_hypo]

            if not opt.unconditional:
                data_and_state[oh, 0:Y, (ddim - 1)] = torch.max(
                    all_inliers[oh, mi, best_hypo, 0:Y],
                    data_and_state[oh, 0:Y, (ddim - 1)])
            else:
                uniform_probs[oh] = torch.min(uniform_probs[oh], 1 - all_inliers[oh, mi, best_hypo, 0:Y])

            inliers_so_far[oh] = all_inliers_so_far[best_hypo]

    for oh in range(P):
        inlier_list = []
        for mi in range(M):
            best_hypo = all_best_hypos[oh, mi]
            inliers = all_inliers[oh, mi, best_hypo]
            inlier_list += [inliers]

        best_inliers = torch.stack(inlier_list, dim=0)

        joint_inliers = torch.zeros(best_inliers.size(), device=device)
        joint_inliers[0] = best_inliers[0]
        for mi in range(1, M):
            joint_inliers[mi] = torch.max(joint_inliers[mi - 1], best_inliers[mi])
        cumulative_inlier_counts = torch.sum(joint_inliers, dim=-1)
        average_cumulative_inlier_count = torch.mean(cumulative_inlier_counts)

        all_best_inliers[oh] = best_inliers
        best_inliers, best_inlier_idx = torch.max(best_inliers, dim=0)
        best_inlier_count = torch.sum(best_inliers)
        all_best_inlier_counts[oh] = average_cumulative_inlier_count

    best_outer_hypo = torch.argmax(all_best_inlier_counts)

best_models = all_best_models[best_outer_hypo]

all_probs_np = all_probs.cpu().numpy()

if opt.em > 0:
    best_models_ = best_models.view(1, best_models.size(0), best_models.size(1))

    refined_models, posterior, variances, _ = em_fun(data, best_models_, masks.to(torch.float),
                                                                  iterations=opt.em, init_variance=1e-9,
                                                                  device=device)
    refined_models.squeeze_(0)
    posterior.squeeze_(0)

else:
    refined_models = best_models

refined_inliers = torch.zeros((M, Y))
refined_distances = torch.zeros((M, Y))

if opt.problem == "homography":
    for mi in range(M):
        inliers = all_best_inliers[best_outer_hypo, mi]
        inlier_indices = torch.nonzero(inliers)
        if opt.em:
            new_model = refined_models[mi]
            new_model *= torch.sign(new_model[-1])
        else:
            new_model = best_models[mi]

        refined_models[mi] = new_model
        new_distances = sampling.homography_consistency_measure(new_model, data[0], device)
        new_inliers = sampling.soft_inlier_fun(new_distances, 5. / (opt.threshold2), opt.threshold2)
        refined_inliers[mi] = new_inliers

    last_inlier_count = 0
    selected_instances = 0
    joint_inliers = torch.zeros((Y,))
    for mi in range(M):
        joint_inliers = torch.max(joint_inliers, refined_inliers[mi, :])
        inlier_count = torch.sum(joint_inliers, dim=-1)
        new_inliers = inlier_count - last_inlier_count
        last_inlier_count = inlier_count
        print("instance %d: %.2f additional inliers" % (mi, new_inliers))
        if new_inliers < opt.inlier_cutoff:
            break
        selected_instances += 1
    estm_models = []
    for mi in range(selected_instances):
        estm_models += [refined_models[mi].cpu().numpy()]
    estm_models = np.vstack(estm_models)
    estm_labels = calc_labels(data[0], selected_instances, estm_models, opt.threshold2)

elif opt.problem == "vp":
    for mi in range(refined_models.shape[0]):
        distances = sampling.vp_consistency_measure_angle(refined_models[mi], data[0], device=device)

        refined_distances[mi] = distances
        refined_inliers[mi] = sampling.soft_inlier_fun(distances, 5. / opt.threshold, opt.threshold)

    refined_model_sort = []
    refined_inliers_ = refined_inliers.clone().detach()
    for mi in range(opt.instances):
        inlier_counts = torch.sum(refined_inliers_, dim=-1)
        best_idx = torch.argmax(inlier_counts)
        refined_inliers_ = torch.max(refined_inliers_[best_idx], refined_inliers_[:])
        refined_inliers_[best_idx] = 0
        refined_model_sort += [best_idx.cpu().numpy()]

    refined_models = refined_models.cpu().detach().numpy()[refined_model_sort, :]

    refined_inliers = refined_inliers.detach().cpu().numpy()
    refined_inliers = refined_inliers[refined_model_sort, :]
    refined_inliers = torch.from_numpy(refined_inliers).to(device)

    refined_distances = refined_distances.detach().cpu().numpy()
    refined_distances = refined_distances[refined_model_sort, :]
    refined_distances = torch.from_numpy(refined_distances).to(device)

    estm_inliers_max, estm_inlier_idx = torch.max(refined_inliers, dim=0)
    estm_distances_min, _ = torch.min(refined_distances, dim=0)
    estm_inlier_idx += 1
    estm_inlier_idx[torch.where(estm_distances_min > opt.threshold * 2)] = 0

    refined_inliers = refined_inliers.detach().cpu().numpy()
else:
    assert False

if opt.problem == "homography":

    colours = ['#e6194b', '#4363d8', '#aaffc3', '#911eb4', '#46f0f0', '#f58231', '#3cb44b', '#f032e6',
               '#008080', '#bcf60c', '#fabebe', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3']

    wrong_min_dists = []

    best_probs = all_probs[best_outer_hypo]

    img1 = img1[:,:,(2,1,0)]
    img2 = img2[:,:,(2,1,0)]
    pts1 = data_unscaled[:, 0:2]
    pts2 = data_unscaled[:, 2:4]

    plt.figure(figsize=(22, 10))
    fontsize = 20

    ax1 = plt.subplot2grid((3, 4), (0, 0))
    ax2 = plt.subplot2grid((3, 4), (0, 1))
    ax3 = plt.subplot2grid((3, 4), (0, 2))
    ax4 = plt.subplot2grid((3, 4), (0, 3))
    ax1.imshow(img1)
    ax1.set_title('left image', fontweight="normal", fontsize=fontsize)
    ax2.imshow(img2)
    ax2.set_title('right image', fontweight="normal", fontsize=fontsize)
    ax3.imshow(img1)
    ax3.set_title('left image\nw/ estimate', fontweight="normal", fontsize=fontsize)
    ax4.imshow(img2)
    ax4.set_title('right image\nw/ estimate', fontweight="normal", fontsize=fontsize)

    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    ax4.set_xticks([])
    ax4.set_yticks([])

    ms = 3

    for di in range(pts1.shape[0]):
        label = estm_labels[di]
        c = (['k'] + colours)[label]
        ax3.plot(pts1[di, 0], pts1[di, 1], 'o', c=c, ms=ms)
        ax4.plot(pts2[di, 0], pts2[di, 1], 'o', c=c, ms=ms)

    for mi in range(M):

        cmap = plt.get_cmap('GnBu')
        ax = plt.subplot2grid((3, M), (1, mi))
        ax.imshow(rgb2gray(img1), cmap='Greys_r', vmax=500)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        if mi == 0:
            ax.set_ylabel('sampling\nweights', fontsize=fontsize)

            ax.set_title('instance 1', fontweight="normal", fontsize=fontsize)
        else:
            ax.set_title('%d' % (mi + 1), fontweight="normal", fontsize=fontsize)

        probs = best_probs[mi].cpu().numpy()
        probs /= np.max(probs)
        probsort = np.argsort(probs)
        for di_ in range(pts1.shape[0]):
            di = probsort[di_]
            prob = probs[di]
            marker = 'o'
            c = cmap(1 - prob)
            ax.plot(pts1[di, 0], pts1[di, 1], marker, c=c, ms=ms)

        ax = plt.subplot2grid((3, M), (2, mi))
        ax.imshow(rgb2gray(img1), cmap='Greys_r', vmax=500)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        if mi == 0:
            ax.set_ylabel('state', fontsize=fontsize)
        if mi > 0:
            inliers = np.max(all_best_inliers[best_outer_hypo, 0:(mi + 1)].cpu().numpy(), axis=0)
        else:
            inliers = all_best_inliers[best_outer_hypo, mi].cpu().numpy()

        inliers /= np.max(inliers)
        probsort = np.argsort(inliers)
        for di_ in range(pts1.shape[0]):
            di = probsort[di_]
            inlier = inliers[di]
            marker = 'o'
            c = cmap(1 - inlier)
            ax.plot(pts1[di, 0], pts1[di, 1], marker, c=c, ms=ms)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.01)
    plt.show()

elif opt.problem == 'vp':

    colours = ['#e6194b', '#4363d8', '#aaffc3', '#911eb4', '#46f0f0', '#f58231', '#3cb44b', '#f032e6',
               '#008080', '#bcf60c', '#fabebe', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3']

    wrong_min_dists = []

    best_probs = all_probs[best_outer_hypo]

    image = img1[:, :, (2, 1, 0)]
    pts1 = data_unscaled[:, 0:4]

    plt.figure(figsize=(22, 10))
    fontsize = 20

    ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 2), (0, 1), rowspan=2)
    ax1.imshow(image)
    ax1.set_title('image', fontweight="normal", fontsize=fontsize)
    ax2.imshow(image)
    ax2.set_title('image\nw/ estimate', fontweight="normal", fontsize=fontsize)

    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    ms = 3

    for di in range(pts1.shape[0]):
        label = estm_inlier_idx[di]
        c = (['k'] + colours)[label]
        ax2.plot([pts1[di, 0], pts1[di, 2]], [pts1[di, 1], pts1[di, 3]], '-', c=c, ms=ms)

    for mi in range(M):

        cmap = plt.get_cmap('GnBu')
        ax = plt.subplot2grid((4, M), (2, mi))
        ax.imshow(rgb2gray(img1), cmap='Greys_r', vmax=500)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        if mi == 0:
            ax.set_ylabel('sampling\nweights', fontsize=fontsize)

            ax.set_title('instance 1', fontweight="normal", fontsize=fontsize)
        else:
            ax.set_title('%d' % (mi + 1), fontweight="normal", fontsize=fontsize)

        probs = best_probs[mi].cpu().numpy()
        probs /= np.max(probs)
        probsort = np.argsort(probs)
        for di_ in range(pts1.shape[0]):
            di = probsort[di_]
            prob = probs[di]
            marker = '-'
            c = cmap(1 - prob)
            ax.plot([pts1[di, 0], pts1[di, 2]], [pts1[di, 1], pts1[di, 3]], marker, c=c, ms=ms)

        ax = plt.subplot2grid((4, M), (3, mi))
        ax.imshow(rgb2gray(img1), cmap='Greys_r', vmax=500)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        if mi == 0:
            ax.set_ylabel('state', fontsize=fontsize)
        if mi > 0:
            inliers = np.max(all_best_inliers[best_outer_hypo, 0:(mi + 1)].cpu().numpy(), axis=0)
        else:
            inliers = all_best_inliers[best_outer_hypo, mi].cpu().numpy()

        inliers /= np.max(inliers)
        probsort = np.argsort(inliers)
        for di_ in range(pts1.shape[0]):
            di = probsort[di_]
            inlier = inliers[di]
            marker = '-'
            c = cmap(1 - inlier)
            ax.plot([pts1[di, 0], pts1[di, 2]], [pts1[di, 1], pts1[di, 3]], marker, c=c, ms=ms)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.01)
    plt.show()