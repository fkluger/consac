from util import sampling
from util.cn_net import CNNet
from datasets.vanishing_points.nyu import NYUVPDataset

import numpy as np
import argparse
import os
import glob
import platform
from util.tee import Tee
import random
import torch
import torch.optim as optim
import scipy.optimize
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(
    description='',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--ckpt_dir', default='./tmp/checkpoints/vp')
parser.add_argument('--data_path', default='./datasets/nyu_vp/data',
                    help='path to training dataset')
parser.add_argument('--nyu_mat_path', default='./nyu_depth_v2_labeled.v7.mat',
                    help='Path to NYU dataset .mat file')
parser.add_argument('--threshold', '-t', type=float, default=0.001, help='tau - inlier threshold')
parser.add_argument('--hyps', type=int, default=2, help='S - inner hypotheses (single instance hypotheses) ')
parser.add_argument('--outerhyps', type=int, default=2, help='P - outer hypotheses (multi-hypotheses)')
parser.add_argument('--batch', '-bs', type=int, default=16, help='B - batch size')
parser.add_argument('--instances', type=int, default=3, help='M - models (max. number of instances)')
parser.add_argument('--samplecount', '-ss', type=int, default=4, help='K - sample count')
parser.add_argument('--learningrate', '-lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--max_num_data', type=int, default=256, help='max. number of data points')
parser.add_argument('--resblocks', '-rb', type=int, default=6, help='CNN residual blocks')
parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
parser.add_argument('--eval_freq', type=int, default=1, help='eval on validation set every n epochs')
parser.add_argument('--val_iter', type=int, default=32, help='number of eval runs on validation set')
parser.add_argument('--calr', dest='calr', action='store_true', help='use cosine annealing LR schedule')
parser.add_argument('--loss_clamp', type=float, default=0.3, help='clamp absolute value of losses')
parser.add_argument('--selfsupervised', dest='selfsupervised', action='store_true', help='')
parser.add_argument('--unconditional', dest='unconditional', action='store_true', help='sample minimal sets unconditionally')
parser.add_argument('--min_prob', type=float, default=1e-8, help='min sampling weight to avoid degenerate distributions')
parser.add_argument('--max_prob_loss', type=float, default=0., help='kappa - inlier masking regularisation')
parser.add_argument('--max_prob_loss_only', dest='max_prob_loss_only', action='store_true', help='')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID to use')
parser.add_argument('--load', default=None, type=str, help='load pretrained NN weights from file')
parser.add_argument('--seed', default=1, type=int, help='random seed')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

trainset = NYUVPDataset(opt.data_path, opt.max_num_data, opt.instances, split='train', mat_file_path=opt.nyu_mat_path)
trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6, batch_size=opt.batch)

valset = NYUVPDataset(opt.data_path, opt.max_num_data, opt.instances, split='val', mat_file_path=opt.nyu_mat_path)
valset_loader = torch.utils.data.DataLoader(valset, shuffle=False, num_workers=6, batch_size=opt.batch)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', 0)

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

if os.path.isdir(opt.ckpt_dir):
    ckpt_dirs = glob.glob(os.path.join(opt.ckpt_dir, "session_*"))
    ckpt_dirs.sort()
    last_ckpt_dir = os.path.split(ckpt_dirs[-1])[1]
    last_session_id = int(last_ckpt_dir[8:11])
    session_id = last_session_id + 1
else:
    session_id = 0
ckpt_dir = os.path.join(opt.ckpt_dir, "session_%03d_bs-%d_ih-%d_oh-%d_sc-%d_lr-%f" %
                        (session_id, opt.batch, opt.hyps, opt.outerhyps, opt.samplecount, opt.learningrate))
os.makedirs(ckpt_dir)

log_file = os.path.join(ckpt_dir, "output.log")
log = Tee(log_file, "w", file_only=False)

hostname = platform.node()
print("host: ", hostname)
print("checkpoint directory: ", ckpt_dir)
print("settings:\n", opt)

data_dim = 9
model_dim = 3
minimal_set_size = 2

model = CNNet(opt.resblocks, data_dim)
if opt.load is not None:
    model.load_state_dict(torch.load(opt.load))
model = model.cuda()

inlier_fun = sampling.soft_inlier_fun_gen(5. / opt.threshold, opt.threshold)

epochs = opt.epochs

optimizer = optim.Adam(model.parameters(), lr=opt.learningrate, eps=1e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(epochs), eta_min=opt.learningrate*0.01)


tensorboard_directory = ckpt_dir + "/tensorboard/"
if not os.path.exists(tensorboard_directory):
    os.makedirs(tensorboard_directory)
tensorboard_writer = SummaryWriter(tensorboard_directory)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

iteration = 0
for epoch in range(0, epochs):

    print("Epoch ", epoch)

    if epoch % 2 == 0 or epoch == epochs-1:
        torch.save(model.state_dict(), '%s/weights_%06d.net' % (ckpt_dir, epoch))

    if epoch % 5 == 0 or epoch == epochs-1:

        # ============================================================================

        print("Evaluation.")
        model.eval()

        val_losses = []
        entropies = []

        for vi in range(opt.val_iter):

            for data, gt_models, num_data, num_models, masks in valset_loader:
                data = data.to(device)
                gt_models = gt_models.to(device)
                masks = masks.to(device)

                data_and_state = torch.zeros((opt.outerhyps, opt.instances, data.size(0), data.size(1), data_dim), device=device)
                all_grads = torch.zeros((opt.outerhyps, opt.hyps, opt.instances, data.size(0), data.size(1)), device=device)
                all_inliers = torch.zeros((opt.outerhyps, opt.hyps, opt.instances, data.size(0), data.size(1)), device=device)
                all_best_inliers = torch.zeros((opt.outerhyps, opt.instances, data.size(0), data.size(1)), device=device)
                all_models = torch.zeros((opt.outerhyps, opt.hyps, opt.instances, data.size(0), model_dim), device=device)
                best_models = torch.zeros((opt.outerhyps, opt.instances, data.size(0), model_dim), device=device)
                all_best_hypos = torch.zeros((opt.outerhyps, opt.instances,), device=device, dtype=torch.int)
                all_log_probs = torch.zeros((opt.hyps, opt.instances, data.size(0), data.size(1)), device=device)
                all_entropies = torch.zeros((opt.outerhyps,opt.instances, data.size(0)), device=device)

                for oh in range(opt.outerhyps):

                    for mi in range(opt.instances):

                        data_and_state[oh, mi, :, :, 0:(data_dim - 1)] = data[:, :, 0:(data_dim - 1)]
                        log_probs = model(data_and_state[oh, mi])

                        for bi in range(0, data.size(0)):

                            log_prob_grads = []
                            losses = []

                            cur_probs = torch.softmax(log_probs[bi, :, 0:num_data[bi]].squeeze(), dim=-1)

                            models, _, choices, distances = sampling.sample_model_pool(
                                data[bi], num_data[bi], opt.hyps, minimal_set_size, inlier_fun,
                                sampling.vp_from_lines, sampling.vp_consistency_measure_angle, cur_probs, device=device)

                            all_grads[oh, :,mi, bi] = choices
                            inliers = sampling.soft_inlier_fun(distances, 5. / opt.threshold, opt.threshold)

                            all_inliers[oh, :, mi, bi] = inliers
                            all_models[oh, :, mi, bi, :] = models

                            inlier_counts = torch.sum(inliers, dim=-1)
                            best_hypo = torch.argmax(inlier_counts)
                            best_inliers = inliers[best_hypo]
                            all_best_hypos[oh,mi] = best_hypo
                            all_best_inliers[oh, mi, bi] = best_inliers

                            best_models[oh, mi, bi] = models[best_hypo]

                            entropy = torch.distributions.categorical.Categorical(
                                probs=cur_probs).entropy()
                            all_entropies[oh, mi, bi] = entropy

                            if not opt.unconditional and mi+1 < opt.instances:
                                data_and_state[oh, mi + 1, bi, :, data_dim - 1] = torch.max(data_and_state[oh, mi, bi, :, data_dim - 1], best_inliers)
                                   
                exclusive_inliers, _ = torch.max(all_best_inliers, dim=1)
                inlier_counts = torch.sum(exclusive_inliers, dim=-1)
                best_hypo = torch.argmax(inlier_counts, dim=0)
                
                sampled_models = torch.zeros((data.size(0), opt.instances, model_dim), device=device)

                for bi in range(0, data.size(0)):
                    sampled_models[bi] = best_models[best_hypo[bi], :, bi]

                if opt.selfsupervised:
                    for bi in range(0, data.size(0)):
                        inlier_count = 0
                        for mi in range(opt.instances):
                            exclusive_inliers, _ = torch.max(all_best_inliers[best_hypo[bi], 0:mi + 1, bi], dim=0)
                            inlier_count += torch.sum(exclusive_inliers, dim=-1)

                        loss = -(inlier_count * 1. / opt.max_num_data * 1. / opt.instances)

                        val_losses += [loss.cpu().numpy()]
                else:
                    for bi in range(0, data.size(0)):

                        models = sampled_models[bi]
                        tp_models = torch.transpose(models[:num_models[bi]], 0, 1)
                        cost_matrix = 1 - torch.matmul(gt_models[bi, :num_models[bi], 0:3], tp_models[0:3]).abs()

                        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix.detach().cpu().numpy())
                        loss = cost_matrix[row_ind, col_ind].sum().detach()
                        val_losses += [loss.cpu().numpy()]

                mean_entropy = torch.mean(all_entropies)
                entropies += [mean_entropy.cpu().detach().numpy()]

        print("Eval loss: ", np.mean(val_losses))
        tensorboard_writer.add_scalar('val/loss', np.mean(val_losses), iteration)
        tensorboard_writer.add_scalar('val/entropy', np.mean(entropies), iteration)

        model.train()

    # ============================================================================

    avg_losses_epoch = []
    avg_per_model_losses_epoch = [[] for _ in range(opt.instances)]

    for data, gt_models, num_data, num_models, masks in trainset_loader:

        data = data.to(device)
        gt_models = gt_models.to(device)
        masks = masks.to(device)

        data_and_state = torch.zeros((opt.outerhyps, opt.instances, opt.samplecount, data.size(0), data.size(1), data_dim), device=device)
        all_grads = torch.zeros((opt.outerhyps, opt.instances, opt.samplecount, data.size(0), data.size(1)), device=device)
        all_inliers = torch.zeros((opt.outerhyps, opt.hyps, opt.instances, opt.samplecount, data.size(0), data.size(1)), device=device)
        all_best_inliers = torch.zeros((opt.outerhyps, opt.instances, opt.samplecount, data.size(0), data.size(1)), device=device)
        all_models = torch.zeros((opt.outerhyps, opt.hyps, opt.instances, opt.samplecount, data.size(0), model_dim), device=device)
        best_models = torch.zeros((opt.outerhyps, opt.instances, opt.samplecount, data.size(0), model_dim), device=device)
        all_best_hypos = torch.zeros((opt.outerhyps, opt.instances, opt.samplecount, data.size(0)), device=device)
        all_log_probs = torch.zeros((opt.hyps, opt.instances, opt.samplecount, data.size(0), data.size(1)), device=device)
        all_entropies = torch.zeros((opt.outerhyps, opt.instances, opt.samplecount, data.size(0),), device=device)
        all_losses = torch.zeros((opt.samplecount, data.size(0)), device=device)
        all_losses_per_model = torch.zeros((opt.samplecount, data.size(0), opt.instances), device=device)
        all_max_probs = torch.ones((opt.outerhyps, opt.instances, opt.samplecount, data.size(0), data.size(1)), device=device)
        all_joint_inliers = torch.zeros((opt.outerhyps, opt.hyps, opt.instances, opt.samplecount, data.size(0), data.size(1)), device=device)

        model.eval()

        neg_inliers = torch.ones((opt.samplecount, opt.outerhyps, opt.instances + 1, data.size(0), data.size(1)),
                                 device=device)

        for mi in range(opt.instances):

            for oh in range(opt.outerhyps):
                for si in range(0, opt.samplecount):
                    data_and_state[oh, mi, si, :, :, 0:(data_dim - 1)] = data[:, :, 0:(data_dim - 1)]
                    
            data_and_state_batched = data_and_state[:, mi].contiguous().view((-1, data.size(1), data_dim))
            log_probs_batched = model(data_and_state_batched)
            log_probs = log_probs_batched.view((opt.outerhyps, opt.samplecount, data.size(0), data.size(1)))
            probs = torch.exp(log_probs)

            for oh in range(opt.outerhyps):
                for bi in range(0, data.size(0)):

                    all_max_probs[oh, mi, :, bi, :] = neg_inliers[:, oh, mi, bi, :]

                    for si in range(0, opt.samplecount):

                        cur_probs = probs[oh, si, bi, 0:num_data[bi]]

                        entropy = torch.distributions.categorical.Categorical(
                            probs=cur_probs).entropy()
                        all_entropies[oh, mi, si, bi] = entropy

                        models, _, choices, distances = \
                            sampling.sample_model_pool(data[bi], num_data[bi], opt.hyps, minimal_set_size,
                                                       inlier_fun, sampling.vp_from_lines,
                                                       sampling.vp_consistency_measure_angle, cur_probs,
                                                       device=device, min_prob=opt.min_prob)

                        all_grads[oh, mi, si, bi] = choices.sum(0)
                        
                        inliers = sampling.soft_inlier_fun(distances, 5. / opt.threshold, opt.threshold)

                        all_inliers[oh,:,mi,si, bi] = inliers
                        all_models[oh,:, mi, si, bi, :] = models

                        if mi > 0:
                            all_joint_inliers[oh, :, mi, si, bi] = torch.max(inliers, all_best_inliers[oh, mi - 1, si, bi].unsqueeze(0).expand(opt.hyps, -1))
                        else:
                            all_joint_inliers[oh, :, mi, si, bi] = inliers

                        inlier_counts = torch.sum(inliers, dim=-1)
                        best_hypo = torch.argmax(inlier_counts)
                        best_inliers = inliers[best_hypo]
                        all_best_hypos[oh, mi, si, bi] = best_hypo
                        all_best_inliers[oh, mi, si, bi] = best_inliers

                        best_joint_inliers = all_joint_inliers[oh, best_hypo, mi, si, bi]
                        neg_inliers[si, oh, mi + 1, bi, :] = 1 - best_joint_inliers

                        best_models[oh, mi, si, bi] = models[best_hypo]

                        if not opt.unconditional and mi+1 < opt.instances:
                            data_and_state[oh, mi + 1, si, bi, :, (data_dim - 1)] = torch.max(data_and_state[oh, mi, si, bi, :, (data_dim - 1)], best_inliers)
                                
        exclusive_inliers, _ = torch.max(all_best_inliers, dim=1)
        inlier_counts = torch.sum(exclusive_inliers, dim=-1)
        best_hypo = torch.argmax(inlier_counts, dim=0)

        sampled_models = torch.zeros((opt.samplecount, data.size(0), opt.instances, 3), device=device)

        for si in range(0, opt.samplecount):
            for bi in range(0, data.size(0)):
                sampled_models[si, bi] = best_models[best_hypo[si, bi], :, si, bi]

        if opt.selfsupervised:
            for bi in range(0, data.size(0)):
                for si in range(0, opt.samplecount):

                    inlier_count = 0
                    last_inlier_count = 0
                    for mi in range(opt.instances):
                        exclusive_inliers, _ = torch.max(all_best_inliers[best_hypo[si, bi], 0:mi + 1, si, bi], dim=0)
                        current_inlier_count = torch.sum(exclusive_inliers, dim=-1)
                        inlier_count += current_inlier_count
                        inlier_increase = current_inlier_count - last_inlier_count
                        all_losses_per_model[si, bi, mi] = -(inlier_increase * 1. / opt.max_num_data)
                        last_inlier_count = current_inlier_count

                    loss = -(inlier_count * 1. / opt.max_num_data * 1. / opt.instances)

                    all_losses[si, bi] = loss

        else:
            for bi in range(0, data.size(0)):
                for si in range(0, opt.samplecount):
                    
                    models = sampled_models[si, bi]

                    gt_tp_models = torch.transpose(gt_models[bi, :num_models[bi]], 0, 1)

                    tp_models_np = models.detach().cpu().numpy()
                    if False in np.isfinite(tp_models_np):
                        print(tp_models_np)

                    cost_matrix = 1 - torch.matmul(models[:, 0:3], gt_tp_models[0:3]).abs()

                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix.detach().cpu().numpy())
                    loss = cost_matrix[row_ind, col_ind].sum().detach()
                    all_losses[si,bi] = loss
                    for mi in range(num_models[bi]):
                        all_losses_per_model[si,bi,mi] = cost_matrix[mi, col_ind[mi]]

        baselines = all_losses.mean(dim=0)
        avg_per_model_losses = all_losses_per_model.mean(dim=0).mean(dim=0)
        
        for bi in range(0, data.size(0)):
            baseline = baselines[bi]
            for s in range(0, opt.samplecount):
                all_grads[:,:,s, bi, :] *= (all_losses[s, bi] - baseline)

        model.train()

        data_and_state_batched = data_and_state.view((-1, data.size(1), data_dim))
        log_probs_batched = model(data_and_state_batched)
        grads_batched = all_grads.view((-1, 1, data.size(1), 1))

        if opt.loss_clamp > 0:
            grads_batched = torch.clamp(grads_batched, max=opt.loss_clamp, min=-opt.loss_clamp)

        mean_entropy = torch.mean(all_entropies)

        if opt.max_prob_loss > 0:
            log_probs = log_probs_batched.view(opt.outerhyps, opt.instances, opt.samplecount, data.size(0), data.size(1))
            probs = torch.softmax(log_probs, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
            probs = probs / torch.clamp(max_probs, min=1e-8)
            max_prob_loss = torch.clamp(probs - all_max_probs, min=0)
            max_prob_grad = opt.max_prob_loss * torch.ones_like(max_prob_loss, device=device)
            if opt.max_prob_loss_only:
                torch.autograd.backward((max_prob_loss), (max_prob_grad))
            else:
                torch.autograd.backward((log_probs_batched, max_prob_loss), (grads_batched, max_prob_grad))

            avg_max_prob_loss = torch.sum(max_prob_loss)
            tensorboard_writer.add_scalar('train/max_prob_loss', avg_max_prob_loss.item(), iteration)
        else:
            torch.autograd.backward((log_probs_batched), (grads_batched))

        optimizer.step()
        optimizer.zero_grad()

        avg_loss = all_losses.mean()
        avg_losses_epoch += [avg_loss]

        tensorboard_writer.add_scalar('train/loss', avg_loss.item(), iteration)
        tensorboard_writer.add_scalar('train/entropy', mean_entropy.cpu().detach().numpy(), iteration)
        for mi in range(opt.instances):
            tensorboard_writer.add_scalar('train/model_loss_%d' % mi, avg_per_model_losses[mi].item(), iteration)
            avg_per_model_losses_epoch[mi] += [avg_per_model_losses[mi].item()]

        iteration += 1

    avg_loss_epoch = sum([l.item() for l in avg_losses_epoch]) / len(avg_losses_epoch)
    print("Avg epoch loss: ", avg_loss_epoch)
    tensorboard_writer.add_scalar('train/loss_avg', avg_loss_epoch, iteration)

    for mi in range(opt.instances):
        avg_per_model_loss_epoch = sum([l for l in avg_per_model_losses_epoch[mi]]) / len(avg_per_model_losses_epoch[mi])
        tensorboard_writer.add_scalar('train/model_loss_avg_%d' % mi, avg_per_model_loss_epoch, iteration)

    if opt.calr:
        scheduler.step()
        adjust_learning_rate(optimizer, scheduler.get_lr()[0])

