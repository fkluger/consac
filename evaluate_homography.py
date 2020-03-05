import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from util.cn_net import CNNet
from datasets.homographies import adelaide
from util import sampling
from util.em_algorithm import em_for_homographies
import random
from util.evaluation import calc_labels_and_misclassification_error
import time


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(
    description='',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset_path', default="/data/scene_understanding/AdelaideRMF/adelaidermf",
                    help='Dataset directory')
parser.add_argument('--ckpt', default='models/consac-s_homography.net', help='path to NN weights')
parser.add_argument('--threshold', '-t', type=float, default=0.0001, help='tau - inlier threshold')
parser.add_argument('--threshold2', type=float, default=0.003, help='theta - inlier threshold')
parser.add_argument('--inlier_cutoff', type=float, default=6, help='Theta - inlier cutoff')
parser.add_argument('--hyps', '-hyps', type=int, default=100, help='S - inner hypotheses (single instance hypotheses)')
parser.add_argument('--outerhyps', type=int, default=100, help='P - outer hypotheses (multi-hypotheses)')
parser.add_argument('--runcount', type=int, default=1, help='Number of runs')
parser.add_argument('--resblocks', '-rb', type=int, default=6, help='CNN residual blocks')
parser.add_argument('--instances', type=int, default=6, help='Max. number of instances')
parser.add_argument('--em', type=int, default=10, help='Number of EM iterations')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--visualise', dest='visualise', action='store_true', help='Visualise each result', default=False)
parser.add_argument('--plot_recall', dest='plot_recall', action='store_true', help='Plot recall curve', default=False)
parser.add_argument('--uniform', dest='uniform', action='store_true', help='disable guided sampling', default=False)
parser.add_argument('--cpu', dest='cpu', action='store_true', help='Run CPU only', default=False)
parser.add_argument('--unconditional', dest='unconditional', action='store_true', help='disable conditional sampling',
                    default=False)
parser.add_argument('--resultfile', default=None)

opt = parser.parse_args()

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

valset = adelaide.AdelaideRMFDataset(opt.dataset_path, None, return_images=True)
valset_loader = torch.utils.data.DataLoader(valset, shuffle=False, num_workers=6, batch_size=1)

device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu', 0)
print(opt)

ddim = 5

model = CNNet(opt.resblocks, ddim, batch_norm=False)
model = model.to(device)

if not opt.uniform:
    checkpoint = torch.load(opt.ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint, strict=True)

all_losses = []

uniform_sampling = opt.uniform

image_idx = 0

all_miss_rates = []
miss_rates_per_image = []

elapsed_time_total = 0

avg_forward_pass_times = []
avg_sampling_times = []
em_times = []

for data, num_data, masks, labels, images in valset_loader:

    print("idx: ", image_idx)

    bi = 0

    data = data.to(device)
    num_data = num_data.to(device)
    masks = masks.to(device)
    np_data = data.cpu().numpy()

    miss_rates = []

    start_time = time.time()

    for si in range(opt.runcount):

        sampling_times = []
        forward_pass_times = []

        with torch.no_grad():
            all_inliers = torch.zeros((opt.outerhyps, opt.instances, opt.hyps, data.size(1)), device=device)
            all_probs = torch.zeros((opt.outerhyps, opt.instances, data.size(1)), device=device)
            all_choices = torch.zeros((opt.outerhyps, opt.instances, opt.hyps, data.size(1)), device=device)
            all_best_choices = torch.zeros((opt.outerhyps, opt.instances, data.size(1)), device=device)
            all_best_inliers = torch.zeros((opt.outerhyps, opt.instances, data.size(1)), device=device)
            all_best_inlier_counts_per_model = torch.zeros((opt.outerhyps, opt.instances), device=device)
            all_best_inlier_counts = torch.zeros(opt.outerhyps, device=device)
            all_best_hypos = torch.zeros((opt.outerhyps, opt.instances,), device=device, dtype=torch.int)
            all_models = torch.zeros((opt.outerhyps, opt.instances, opt.hyps, 9), device=device)
            all_best_models = torch.zeros((opt.outerhyps, opt.instances, 9), device=device)

            for oh in range(opt.outerhyps):

                data_and_state = torch.zeros((data.size(0), data.size(1), ddim),
                                             device=device)
                data_and_state[:, :, 0:(ddim - 1)] = data[:, :, 0:(ddim - 1)]

                uniform_probs = torch.ones(num_data[bi], device=device)

                inliers_so_far = torch.zeros((data.size(1),), device=device)

                for mi in range(opt.instances):

                    forward_pass_start = time.time()
                    if uniform_sampling:
                        probs = uniform_probs
                    else:
                        data = data.to(device)
                        log_probs = model(data_and_state)
                        probs = torch.softmax(log_probs[bi, 0, 0:num_data[bi], 0], dim=0)
                        probs /= torch.max(probs)
                    forward_pass_end = time.time()
                    forward_pass_time = forward_pass_end - forward_pass_start
                    forward_pass_times += [forward_pass_time]

                    sampling_start = time.time()

                    all_probs[oh, mi, :num_data[bi]] = probs

                    cur_probs = probs.expand((opt.hyps, probs.size(0)))

                    models, _, choices, distances = \
                        sampling.sample_model_pool_multiple(data[bi], num_data[bi], 1, 4, opt.threshold,
                                                            sampling.homographies_from_points,
                                                            sampling.homographies_consistency_measure, cur_probs,
                                                            device=device, model_size=9, sample_count=opt.hyps)

                    inliers = sampling.soft_inlier_fun(distances.squeeze(), 5. / opt.threshold, opt.threshold)

                    all_choices[oh, mi, :] = choices.squeeze()
                    all_inliers[oh, mi, :, 0:num_data[bi]] = inliers.squeeze()[:, 0:num_data[bi]]
                    all_models[oh, mi, :] = models.squeeze()

                    all_inliers_so_far = torch.max(all_inliers[oh, mi], inliers_so_far)
                    all_inlier_counts_so_far = torch.sum(all_inliers_so_far, dim=-1)

                    best_hypo = torch.argmax(all_inlier_counts_so_far)
                    all_best_hypos[oh, mi] = best_hypo
                    all_best_models[oh, mi] = models.squeeze()[best_hypo]

                    if not opt.unconditional:
                        data_and_state[bi, 0:num_data[bi], (ddim - 1)] = torch.max(
                            all_inliers[oh, mi, best_hypo, 0:num_data[bi]],
                            data_and_state[bi, 0:num_data[bi], (ddim - 1)])

                    all_best_choices[oh, mi] = all_choices[oh, mi, best_hypo]

                    inliers_so_far = all_inliers_so_far[best_hypo]

                    uniform_probs = torch.min(uniform_probs, 1 - all_inliers[oh, mi, best_hypo, 0:num_data[bi]])

                    sampling_end = time.time()
                    sampling_time = sampling_end - sampling_start
                    sampling_times += [sampling_time]

                inlier_list = []
                for mi in range(opt.instances):
                    best_hypo = all_best_hypos[oh, mi]
                    inliers = all_inliers[oh, mi, best_hypo]
                    inlier_list += [inliers]

                best_inliers = torch.stack(inlier_list, dim=0)

                joint_inliers = torch.zeros(best_inliers.size(), device=device)
                joint_inliers[0] = best_inliers[0]
                for mi in range(1, opt.instances):
                    joint_inliers[mi] = torch.max(joint_inliers[mi - 1], best_inliers[mi])
                cumulative_inlier_counts = torch.sum(joint_inliers, dim=-1)
                average_cumulative_inlier_count = torch.mean(cumulative_inlier_counts)

                all_best_inliers[oh] = best_inliers
                best_inliers, best_inlier_idx = torch.max(best_inliers, dim=0)
                best_inlier_count = torch.sum(best_inliers)
                all_best_inlier_counts[oh] = average_cumulative_inlier_count
                for li in range(best_inliers.size(0)):
                    mi = best_inlier_idx[li]
                    inl = best_inliers[li]
                    all_best_inlier_counts_per_model[oh, mi] += inl

            best_outer_hypo = torch.argmax(all_best_inlier_counts)

        best_choices = all_best_choices[best_outer_hypo]
        best_models = all_best_models[best_outer_hypo]

        all_probs_np = all_probs.cpu().numpy()

        em_start = time.time()
        if opt.em > 0:
            best_models_ = best_models.view(1, best_models.size(0), best_models.size(1))
            refined_models, posterior, variances, _ = em_for_homographies(data, best_models_, masks.to(torch.float),
                                                                          iterations=opt.em, init_variance=1e-9,
                                                                          device=device)
            refined_models.squeeze_(0)
            posterior.squeeze_(0)
        else:
            refined_models = torch.zeros((opt.instances, 9))
        em_end = time.time()
        em_time = em_end-em_start

        refined_inliers = torch.zeros((opt.instances, data.size(1)))
        for mi in range(opt.instances):
            inliers = all_best_inliers[best_outer_hypo, mi]
            inlier_indices = torch.nonzero(inliers)
            if opt.em:
                new_model = refined_models[mi]
                new_model *= torch.sign(new_model[-1])
            else:
                new_model = best_models[mi]

            refined_models[mi] = new_model
            new_distances = sampling.homography_consistency_measure(new_model, data[bi], device)
            new_inliers = sampling.soft_inlier_fun(new_distances, 5. / (opt.threshold2), opt.threshold2)
            refined_inliers[mi] = new_inliers

        last_inlier_count = 0
        selected_instances = 0
        joint_inliers = torch.zeros((data.size(1),))
        for mi in range(opt.instances):
            joint_inliers = torch.max(joint_inliers, refined_inliers[mi, :])
            inlier_count = torch.sum(joint_inliers, dim=-1)
            new_inliers = inlier_count - last_inlier_count
            last_inlier_count = inlier_count
            if new_inliers < opt.inlier_cutoff:
                break
            selected_instances += 1
        estm_models = []
        for mi in range(selected_instances):
            estm_models += [refined_models[mi].cpu().numpy()]

        end_time = time.time()
        
        estm_models = np.vstack(estm_models)
        estm_labels, miss_rate = calc_labels_and_misclassification_error(data[bi], selected_instances, estm_models,
                                                                         opt.threshold, labels[bi])

        print("miss. rate: %.2f" % (miss_rate * 100))
        all_miss_rates += [miss_rate * 100.]

        miss_rates += [miss_rate * 100.]

        print("time elapsed: %.3f seconds" % (end_time-start_time))

        num_forward_passes = len(forward_pass_times)
        num_sampling = len(sampling_times)
        avg_forward_pass_time = np.mean(forward_pass_times)
        avg_sampling_time = np.mean(sampling_times)
        print("%d forward passes (%.4f seconds)" % (num_forward_passes, avg_forward_pass_time))
        print("%d sampling passes (%.4f seconds)" % (num_sampling, avg_sampling_time))
        print("EM time: %.4f seconds" % em_time)

        avg_forward_pass_times += [avg_forward_pass_time]
        avg_sampling_times += [avg_sampling_time]
        em_times += [em_time]

        elapsed_time_total += (end_time-start_time)

        if opt.visualise:

            colours = ['#e6194b', '#4363d8', '#aaffc3', '#911eb4', '#46f0f0', '#f58231', '#3cb44b', '#f032e6',
                       '#008080', '#bcf60c', '#fabebe', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3']

            wrong_min_dists = []

            best_probs = all_probs[best_outer_hypo]

            img1 = images[0].cpu().numpy().squeeze()
            img2 = images[1].cpu().numpy().squeeze()
            pts1 = data[bi, :, 0:2].cpu().numpy()
            pts2 = data[bi, :, 2:4].cpu().numpy()

            img1size = img1.shape[0:2]
            img2size = img2.shape[0:2]

            scale1 = np.max(img1size)
            scale2 = np.max(img2size)

            pts1 *= (scale1 / 2.)
            pts2 *= (scale2 / 2.)

            pts1[:, 0] += img1size[1] / 2.
            pts2[:, 0] += img2size[1] / 2.
            pts1[:, 1] += img1size[0] / 2.
            pts2[:, 1] += img2size[0] / 2.

            plt.figure(figsize=(22, 10))
            fontsize = 26

            ax1 = plt.subplot2grid((3, 6), (0, 0))
            ax2 = plt.subplot2grid((3, 6), (0, 1))
            ax3 = plt.subplot2grid((3, 6), (0, 2))
            ax4 = plt.subplot2grid((3, 6), (0, 3))
            ax5 = plt.subplot2grid((3, 6), (0, 4))
            ax6 = plt.subplot2grid((3, 6), (0, 5))
            ax1.imshow(img1)
            ax1.set_title('left image', fontweight="normal", fontsize=fontsize)
            ax2.imshow(img2)
            ax2.set_title('right image', fontweight="normal", fontsize=fontsize)
            ax3.imshow(img1)
            ax3.set_title('left image\nw/ GT', fontweight="normal", fontsize=fontsize)
            ax4.imshow(img2)
            ax4.set_title('right image\nw/ GT', fontweight="normal", fontsize=fontsize)
            ax5.imshow(img1)
            ax5.set_title('left image\n w/ estimate', fontweight="bold", fontsize=fontsize)

            ax6.text(0.5, 0.5, 'ME: %.2f%%' % (miss_rate * 100.),
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax6.transAxes, fontsize=fontsize, fontweight='bold')
            ax6.set_axis_off()

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
            ax5.set_yticklabels([])
            ax5.set_xticklabels([])
            ax5.set_xticks([])
            ax5.set_yticks([])

            ms = 6

            gt_label = labels[bi].cpu().numpy()

            for di in range(pts1.shape[0]):
                c = (['k'] + colours)[gt_label[di]]
                ax3.plot(pts1[di, 0], pts1[di, 1], 'o', c=c, ms=ms)
                ax4.plot(pts2[di, 0], pts2[di, 1], 'o', c=c, ms=ms)
                label = estm_labels[di]
                c = (['k'] + colours)[label]
                ax5.plot(pts1[di, 0], pts1[di, 1], 'o', c=c, ms=ms)

            if len(wrong_min_dists) > 0:
                print(np.max(wrong_min_dists))

            for mi in range(opt.instances):

                cmap = plt.get_cmap('GnBu')
                ax = plt.subplot2grid((3, 6), (1, mi))
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

                ax = plt.subplot2grid((3, 6), (2, mi))
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

    image_idx += 1
    miss_rates_per_image += [miss_rates]

print("miss rates:")
print(all_miss_rates)
print("avg: %.3f" % np.mean(all_miss_rates))
print("std: %.3f" % np.std(all_miss_rates))
print("med: %.3f" % np.median(all_miss_rates))
avg_miss_rates = []
std_miss_rates = []
for idx, miss_rates in enumerate(miss_rates_per_image):
    print("%02d : %.2f (%.2f) -- %s" % (idx, np.mean(miss_rates), np.std(miss_rates),
                                        adelaide.AdelaideRMF.homography_sequences[idx]))
    avg_miss_rates += [np.mean(miss_rates)]
    std_miss_rates += [np.std(miss_rates)]
print("avg: %.3f" % np.mean(avg_miss_rates))
print("std: %.3f" % np.mean(std_miss_rates))

print("total time: %.3f seconds" % elapsed_time_total)

print("Avg. times forward pass / sampling / EM: %.4f / %.4f / %.4f" % (np.mean(avg_forward_pass_times), np.mean(avg_sampling_times), np.mean(em_times)))
