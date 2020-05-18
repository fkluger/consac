# -*- coding: utf-8 -*-
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from util.evaluation import single_eval_nyu
from util.evaluation import single_eval_yud
from util.cn_net import CNNet
from datasets.vanishing_points import nyu, yud
from util.auc import calc_auc
from util import sampling
from util.em_algorithm import em_for_vp
import random

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(
    description='',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', '-ds', default='NYU', help='Which dataset to use: NYU, YUD, YUD+')
parser.add_argument('--dataset_path', default="./datasets/nyu_vp/data", help='Dataset directory')
parser.add_argument('--nyu_mat_path', default='/home/kluger/tmp/nyu_depth_v2_labeled.matv7.mat',
                    help='Path to NYU dataset .mat file')
parser.add_argument('--ckpt', default='models/consac_vp.net', help='path to NN weights')
parser.add_argument('--threshold', '-t', type=float, default=0.001, help='tau - inlier threshold')
parser.add_argument('--hyps', '-hyps', type=int, default=32, help='S - inner hypotheses (single instance hypotheses)')
parser.add_argument('--outerhyps', type=int, default=32, help='P - outer hypotheses (multi-hypotheses)')
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

if opt.dataset == "YUD":
    dataset = yud.YUDVPDataset(opt.dataset_path, None, 3, split='test', return_images=True, yudplus=False)
elif opt.dataset == "YUD+":
    dataset = yud.YUDVPDataset(opt.dataset_path, None, 8, split='test', return_images=True, yudplus=True)
elif opt.dataset == "NYU":
    dataset = nyu.NYUVPDataset(opt.dataset_path, None, 8, split='test', mat_file_path=opt.nyu_mat_path,
                               return_images=True)
else:
    assert False, "unknown dataset " + opt.dataset

dataset_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=6, batch_size=1)

device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu', 0)
print(opt)

ddim = 9

model = CNNet(opt.resblocks, ddim)
model = model.to(device)

checkpoint = torch.load(opt.ckpt, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint, strict=False)

all_losses = []

uniform_sampling = opt.uniform

inlier_fun = sampling.soft_inlier_fun_gen(5. / opt.threshold, opt.threshold)

idx = 0

all_aucs = []
all_aucs_em = []
all_plotpoints = []
all_plotpoints_em = []

results = []

for ri in range(opt.runcount):

    all_errors = []
    all_errors_em = []
    all_missing_vps = []

    for data, gt_models, num_data, num_models, masks, images in dataset_loader:

        bi = 0

        print("idx: ", idx)

        data = data.to(device)
        num_models = num_models.to(device)
        num_data = num_data.to(device)
        masks = masks.to(device)
        gt_models = gt_models.to(device)

        gt_models = gt_models[bi, :num_models[bi]]
        gt_models_mat = np.matrix(gt_models.cpu())

        with torch.no_grad():
            all_inliers = torch.zeros((opt.outerhyps, opt.instances, opt.hyps, data.size(1)), device=device)
            all_probs = torch.zeros((opt.outerhyps, opt.instances, data.size(1)), device=device)
            all_choices = torch.zeros((opt.outerhyps, opt.instances, opt.hyps, data.size(1)), device=device)
            all_inlier_counts = torch.zeros((opt.outerhyps, opt.instances, opt.hyps), device=device)
            all_best_inliers = torch.zeros((opt.outerhyps, opt.instances, data.size(1)), device=device)
            all_best_inlier_counts_per_model = torch.zeros((opt.outerhyps, opt.instances), device=device)
            all_best_inlier_counts = torch.zeros(opt.outerhyps, device=device)
            all_best_hypos = torch.zeros((opt.outerhyps, opt.instances,), device=device, dtype=torch.int)
            all_models = torch.zeros((opt.outerhyps, opt.instances, opt.hyps, 3), device=device)

            for oh in range(opt.outerhyps):

                data_and_state = torch.zeros((data.size(0), data.size(1), ddim),
                                             device=device)
                data_and_state[:, :, 0:(ddim - 1)] = data[:, :, 0:(ddim - 1)]

                uniform_probs = torch.ones(num_data[bi], device=device)

                for mi in range(opt.instances):

                    if uniform_sampling:
                        probs = uniform_probs
                    else:
                        data = data.to(device)
                        log_probs = model(data_and_state)
                        probs = torch.softmax(log_probs[bi, 0, 0:num_data[bi], 0], dim=0)
                        probs /= torch.max(probs)

                    all_probs[oh, mi, :num_data[bi]] = probs

                    for hi in range(opt.hyps):
                        estm_line, choice_vec, _, inlier_count, distances = \
                            sampling.sample_model(
                                data[bi], num_data[bi], inlier_fun, 2, probs, sampling.vp_from_lines,
                                sampling.vp_consistency_measure_angle, device=device)

                        inliers = sampling.soft_inlier_fun(distances, 5. / opt.threshold, opt.threshold)

                        all_choices[oh, mi, hi] = choice_vec
                        all_inliers[oh, mi, hi, 0:num_data[bi]] = inliers
                        all_inlier_counts[oh, mi, hi] = inlier_count
                        all_models[oh, mi, hi] = estm_line

                    best_hypo = torch.argmax(all_inlier_counts[oh, mi])
                    all_best_hypos[oh, mi] = best_hypo

                    if not opt.unconditional:
                        data_and_state[bi, 0:num_data[bi], ddim - 1] = torch.max(
                            all_inliers[oh, mi, best_hypo, 0:num_data[bi]],
                            data_and_state[bi, 0:num_data[bi], ddim - 1])

                    uniform_probs = torch.min(uniform_probs, 1 - all_inliers[oh, mi, best_hypo, 0:num_data[bi]])

                inlier_list = []
                for mi in range(opt.instances):
                    best_hypo = all_best_hypos[oh, mi]
                    inliers = all_inliers[oh, mi, best_hypo]
                    inlier_list += [inliers]

                best_inliers = torch.stack(inlier_list, dim=0)
                all_best_inliers[oh] = best_inliers
                best_inliers_max, best_inlier_idx = torch.max(best_inliers, dim=0)
                best_inlier_count = torch.sum(best_inliers_max)
                all_best_inlier_counts[oh] = best_inlier_count
                for li in range(best_inliers.size(0)):
                    mi = best_inlier_idx[li]
                    inl = best_inliers_max[li]
                    all_best_inlier_counts_per_model[oh, mi] += inl

            best_outer_hypo = torch.argmax(all_best_inlier_counts)

        estm_models = []
        for mi in range(opt.instances):
            estm_models += [all_models[best_outer_hypo, mi, all_best_hypos[best_outer_hypo, mi]].cpu().numpy()]
        estm_models = np.vstack(estm_models)

        estm_model_sort = []
        all_best_inliers_ = all_best_inliers.clone().detach()
        for mi in range(opt.instances):
            inlier_counts = torch.sum(all_best_inliers_[best_outer_hypo, :], dim=-1)
            best_idx = torch.argmax(inlier_counts)
            all_best_inliers_[best_outer_hypo, :] = torch.max(all_best_inliers_[best_outer_hypo, best_idx],
                                                              all_best_inliers_[best_outer_hypo, :])
            all_best_inliers_[best_outer_hypo, best_idx] = 0
            estm_model_sort += [best_idx.cpu().numpy()]

        estm_models = estm_models[estm_model_sort, :]

        estm_models_torch = torch.from_numpy(estm_models).unsqueeze_(0).to(device)
        estm_models_torch.requires_grad_(True)
        refined_models, posterior, variances, all_vps = em_for_vp(data, estm_models_torch, masks.to(torch.float),
                                                                  iterations=opt.em, init_variance=1e-8, device=device)

        refined_inliers = torch.zeros((opt.instances, data.size(1)), device=device)
        refined_distances = torch.zeros((opt.instances, data.size(1)), device=device)
        for mi in range(refined_models.shape[1]):
            distances = sampling.vp_consistency_measure_angle(refined_models[bi, mi], data[bi], device=device)
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

        refined_models = refined_models.cpu().detach().numpy()[0, refined_model_sort, :]

        refined_models = refined_models[:gt_models_mat.shape[0], :]
        estm_models = estm_models[:gt_models_mat.shape[0], :]

        if "YUD" in opt.dataset:
            errors, _, _ = single_eval_yud(dataset.dataset.K_inv, gt_models_mat, estm_models, missing_vp_penalty=90.)
            errors_em, row_ind, col_ind = single_eval_yud(dataset.dataset.K_inv, gt_models_mat, refined_models,
                                                          missing_vp_penalty=90.)
        else:
            errors, missing_vps, _, _ = single_eval_nyu(gt_models_mat, estm_models, missing_vp_penalty=90.)
            errors_em, _, row_ind, col_ind = single_eval_nyu(gt_models_mat, refined_models, missing_vp_penalty=90.)
        all_errors += errors
        all_errors_em += errors_em

        boh = best_outer_hypo.cpu().numpy()
        all_probs = all_probs.cpu().detach().numpy()[boh]
        all_choices = all_choices.cpu().detach().numpy()[boh]
        all_best_hypos = all_best_hypos.cpu().detach().numpy()[boh]

        errors_em = np.array(errors_em)[col_ind]
        print("errors per VP: ", errors_em)

        if opt.visualise:

            refined_inliers = refined_inliers.detach().cpu().numpy()
            refined_inliers = refined_inliers[refined_model_sort, :]
            refined_inliers = torch.from_numpy(refined_inliers).to(device)

            estm_inliers_max, estm_inlier_idx = torch.max(refined_inliers, dim=0)
            estm_distances_min, _ = torch.min(refined_distances, dim=0)
            estm_inlier_idx[torch.where(estm_distances_min > opt.threshold * 2)] = -1

            refined_inliers = refined_inliers.detach().cpu().numpy()

            estm_inlier_idx_ = estm_inlier_idx.cpu().numpy().copy()
            for ei in range(estm_inlier_idx.shape[0]):
                if estm_inlier_idx_[ei] >= 0:
                    estm_inlier_idx_[ei] += num_models[0]
                for ci in range(col_ind.shape[0]):
                    if estm_inlier_idx[ei] == col_ind[ci]:
                        estm_inlier_idx_[ei] = row_ind[ci]

            np_data = data.cpu().numpy()
            np_data[:, :, 0:2] *= 320
            np_data[:, :, 3:5] *= 320
            np_data[:, :, 0] += 320
            np_data[:, :, 3] += 320
            np_data[:, :, 1] += 240
            np_data[:, :, 4] += 240

            colours = ['#e6194b', '#4363d8', '#aaffc3', '#911eb4', '#46f0f0', '#f58231', '#3cb44b', '#f032e6',
                       '#008080', '#bcf60c', '#fabebe', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3']

            lw = 3.2
            plt.figure(figsize=(22, 10))

            ax = plt.subplot2grid((3, opt.instances), (0, 0))
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(images[bi])
            ax.set_title('image', fontweight="normal", fontsize=32)

            ax = plt.subplot2grid((3, opt.instances), (0, 1))
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

            for di in range(num_data[bi]):
                colour = 'k'
                ax.plot([np_data[bi, di, 0], np_data[bi, di, 3]], [-np_data[bi, di, 1], -np_data[bi, di, 4]], '-',
                        c=colour, lw=lw)
            ax.set_aspect('equal')
            ax.set_title('line segments', fontweight="normal", fontsize=32)

            ax = plt.subplot2grid((3, opt.instances), (0, 2))
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            gt_inliers = torch.zeros((num_models[bi], data.size(1)), device=device)
            gt_distances = torch.zeros((num_models[bi], data.size(1)), device=device)
            for mi in range(num_models[bi]):
                distances = sampling.vp_consistency_measure_angle(gt_models[mi], data[bi], device=device)
                gt_distances[mi] = distances
                gt_inliers[mi] = sampling.soft_inlier_fun(distances, 5. / opt.threshold, opt.threshold)
            gt_inliers_max, gt_inlier_idx = torch.max(gt_inliers, dim=0)
            gt_distances_min, _ = torch.min(gt_distances, dim=0)
            gt_inlier_idx[torch.where(gt_distances_min > opt.threshold * 2)] = -1

            for di in range(num_data[bi]):

                label = gt_inlier_idx[di]
                if label >= 0:
                    colour = colours[label]
                else:
                    colour = 'k'

                ax.plot([np_data[bi, di, 0], np_data[bi, di, 3]], [-np_data[bi, di, 1], -np_data[bi, di, 4]], '-',
                        c=colour, lw=lw)
            ax.set_aspect('equal')
            ax.set_title('ground truth', fontweight="normal", fontsize=32)

            ax = plt.subplot2grid((3, opt.instances), (0, 3))
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

            for di in range(num_data[bi]):

                label = estm_inlier_idx_[di]
                if label >= 0:
                    colour = colours[label]
                else:
                    colour = 'k'

                ax.plot([np_data[bi, di, 0], np_data[bi, di, 3]], [-np_data[bi, di, 1], -np_data[bi, di, 4]], '-',
                        c=colour, lw=lw)
            ax.set_aspect('equal')
            ax.set_title('estimate', fontweight="bold", fontsize=32)

            for mi in range(opt.instances):

                ax = plt.subplot2grid((3, opt.instances), (1, mi))
                ax.imshow(nyu.rgb2gray(images[bi]), cmap='gray', vmax=1000)
                plt.autoscale(False)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                if mi == 0:
                    ax.set_ylabel('sampling\nweights', fontsize=32)

                    ax.set_title('instance 1', fontweight="normal", fontsize=32)
                else:
                    ax.set_title('%d' % (mi + 1), fontweight="normal", fontsize=32)

                cmap = plt.get_cmap('GnBu')
                probs = all_probs[mi, :]
                probs /= np.max(probs)

                prob_sort = np.argsort(probs)

                for di_ in range(num_data[bi]):
                    di = prob_sort[di_]
                    colour = cmap(1 - probs[di])
                    choices = [all_choices[mi, all_best_hypos[mi], di] for mi in range(opt.instances)]
                    marker = '-'
                    ax.plot([np_data[bi, di, 0], np_data[bi, di, 3]], [np_data[bi, di, 1], np_data[bi, di, 4]], marker,
                            c=colour, lw=lw, ms=15)

                ax = plt.subplot2grid((3, opt.instances), (2, mi))
                ax.imshow(nyu.rgb2gray(images[bi]), cmap='gray', vmax=1000)
                plt.autoscale(False)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                if mi == 0:
                    ax.set_ylabel('state', fontsize=32)
                if mi == 0:
                    ax.set_title('VP error: %.2f' % errors_em[0], fontweight="normal", fontsize=24)
                elif mi < errors_em.size:
                    ax.set_title('%.2f°' % errors_em[mi], fontweight="normal", fontsize=24)
                else:
                    ax.set_title('---', fontweight="normal", fontsize=24)

                inliers = np.max(refined_inliers[0:(mi + 1)], axis=0)
                inliers /= np.max(inliers)
                for di in range(num_data[bi]):
                    inlier = inliers[di]
                    colour = cmap(1 - inlier)
                    marker = '-'
                    ax.plot([np_data[bi, di, 0], np_data[bi, di, 3]], [np_data[bi, di, 1], np_data[bi, di, 4]], marker,
                            c=colour, lw=lw, ms=15)

            plt.tight_layout()
            plt.subplots_adjust(hspace=0, wspace=0.01)
            plt.show()

        idx += 1

    auc, plot_points = calc_auc(np.array(all_errors), cutoff=10)
    auc_em, plot_points_em = calc_auc(np.array(all_errors_em), cutoff=10)

    print("AUC: ", auc)
    print("AUC w/ em: ", auc_em)
    all_aucs += [auc]
    all_aucs_em += [auc_em]
    all_plotpoints += [plot_points]
    all_plotpoints_em += [plot_points_em]

    results += [{'AUC': auc_em, 'plot_points': plot_points_em, 'AUC_noem': auc, 'plot_points_noem': plot_points}]

    if opt.plot_recall:
        plt.figure()
        plt.plot(plot_points_em[:, 0], plot_points_em[:, 1], 'b-', lw=3, label='AUC: %.3f ' % (auc_em * 100.))
        axes = plt.gca()
        axes.set_xlim([0, 10])
        axes.set_ylim([0, 1])
        plt.xlabel('error threshold', fontsize=14)
        plt.ylabel('recall', fontsize=14)
        plt.legend()
        plt.show()

print("w/o EM: ", np.mean(all_aucs), np.std(all_aucs), np.median(all_aucs))
print("w/  EM: ", np.mean(all_aucs_em), np.std(all_aucs_em), np.median(all_aucs_em))

if opt.resultfile is not None:
    import pickle

    filename = opt.resultfile
    pickle.dump(results, open(filename, "wb"))
