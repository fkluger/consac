from util import sampling
from util.misc import adjust_learning_rate
from util.tee import Tee
from util.cn_net import CNNet
from datasets.homographies.sfmdata import HomographyDataset
import numpy as np
import argparse
import os
import glob
import platform
import random
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(
    description='',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--ckpt_dir', default='./tmp/checkpoints/homography',
                    help='directory for storing NN weight checkpoints')
parser.add_argument('--data_path', default='./datasets/traindata',
                    help='path to training dataset')
parser.add_argument('--threshold', '-t', type=float, default=0.0001, help='tau - inlier threshold')
parser.add_argument('--hyps', type=int, default=2, help='S - inner hypotheses (single instance hypotheses) ')
parser.add_argument('--outerhyps', type=int, default=2, help='P - outer hypotheses (multi-hypotheses)')
parser.add_argument('--batch', '-bs', type=int, default=1, help='B - batch size')
parser.add_argument('--instances', type=int, default=6, help='M - max. number of instances')
parser.add_argument('--samplecount', '-ss', type=int, default=8, help='K - sample count')
parser.add_argument('--learningrate', '-lr', type=float, default=0.000002, help='learning rate')
parser.add_argument('--max_num_points', type=int, default=512, help='max. number of data points')
parser.add_argument('--resblocks', '-rb', type=int, default=6, help='CNN residual blocks')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--eval_freq', type=int, default=1, help='eval on validation set every n epochs')
parser.add_argument('--val_iter', type=int, default=1, help='number of eval runs on validation set')
parser.add_argument('--calr', dest='calr', action='store_true', help='use cosine annealing LR schedule')
parser.add_argument('--batch_norm', dest='batch_norm', action='store_true', help='use batch normalisation')
parser.add_argument('--fair_sampling', dest='fair_sampling', action='store_true', help='sample scenes uniformly')
parser.add_argument('--unconditional', dest='unconditional', action='store_true', help='sample minimal sets unconditionally')
parser.add_argument('--loss_clamp', type=float, default=0.3, help='clamp absolute value of losses')
parser.add_argument('--lowe_ratio', type=float, default=0.9, help='Lowe ratio filter threshold')
parser.add_argument('--min_prob', type=float, default=1e-8, help='min sampling weight to avoid degenerate distributions')
parser.add_argument('--max_prob_loss', type=float, default=0.01, help='kappa - inlier masking regularisation')
parser.add_argument('--max_prob_loss_only', dest='max_prob_loss_only', action='store_true', help='')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID to use')
parser.add_argument('--load', default=None, type=str, help='load pretrained NN weights from file')
parser.add_argument('--seed', default=1, type=int, help='random seed')

opt = parser.parse_args()
opt.max_inlier_state = True

data_dim = 5
model_dim = 9
minimal_set_size = 4

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

trainset = HomographyDataset(opt.data_path, opt.max_num_points, split='train', max_score_ratio=opt.lowe_ratio,
                             use_indoor=True, fair_sampling=opt.fair_sampling, augmentation=True)
valset = HomographyDataset(opt.data_path, opt.max_num_points, split='val', max_score_ratio=opt.lowe_ratio)

trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=1, batch_size=opt.batch)
valset_loader = torch.utils.data.DataLoader(valset, shuffle=False, num_workers=1, batch_size=opt.batch)

device = torch.device('cuda' if torch.cuda.is_available() and not (opt.gpu == 'no') else 'cpu', 0)

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

model = CNNet(opt.resblocks, data_dim, batch_norm=opt.batch_norm)
if opt.load is not None:
    model.load_state_dict(torch.load(opt.load))
model = model.to(device)

inlier_fun = sampling.soft_inlier_fun_gen(5. / opt.threshold, opt.threshold)

optimizer = optim.Adam(model.parameters(), lr=opt.learningrate, eps=1e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(opt.epochs), eta_min=opt.learningrate*0.01)

tensorboard_directory = ckpt_dir + "/tensorboard/"
if not os.path.exists(tensorboard_directory):
    os.makedirs(tensorboard_directory)
tensorboard_writer = SummaryWriter(tensorboard_directory)

M = opt.instances
P = opt.outerhyps
S = opt.hyps
K = opt.samplecount

iteration = 0
for epoch in range(0, opt.epochs):

    print("Epoch ", epoch)

    if epoch % opt.eval_freq == 0 or epoch == opt.epochs-1:
        torch.save(model.state_dict(), '%s/weights_%06d.net' % (ckpt_dir, epoch))

    if epoch % opt.eval_freq == 0 or epoch == opt.epochs-1:

        # ============================================================================

        print("Evaluation.")
        model.eval()

        val_losses = []

        entropies = []

        for vi in range(opt.val_iter):

            for data, num_data, masks in valset_loader:

                data = data.to(device)
                masks = masks.to(device)

                data_and_state = torch.zeros((opt.outerhyps, opt.instances, data.size(0), data.size(1), data_dim),
                                             device=device)
                all_grads = torch.zeros((opt.outerhyps, opt.hyps, opt.instances, data.size(0), data.size(1)),
                                        device=device)
                all_best_inliers = torch.zeros((opt.outerhyps, opt.instances, data.size(0), data.size(1)),
                                               device=device)
                all_joint_inliers = torch.zeros((opt.outerhyps, opt.hyps, opt.instances, data.size(0), data.size(1)),
                                                device=device)
                all_models = torch.zeros((opt.outerhyps, opt.hyps, opt.instances, data.size(0), model_dim),
                                         device=device)
                all_best_hypos = torch.zeros((opt.outerhyps, opt.instances,), device=device, dtype=torch.int)
                all_entropies = torch.zeros((opt.outerhyps,opt.instances, data.size(0)), device=device)

                for oh in range(opt.outerhyps):

                    neg_inliers = torch.ones((opt.instances+1, data.size(0), data.size(1)), device=device)

                    for mi in range(opt.instances):

                        data_and_state[oh, mi, :, :, 0:data_dim - 1] = data[:, :, 0:data_dim - 1]
                        log_probs = model(data_and_state[oh, mi])

                        for bi in range(0, data.size(0)):

                            log_prob_grads = []
                            losses = []
                            inlier_counts = []

                            cur_probs = torch.softmax(log_probs[bi, :, 0:num_data[bi]].squeeze(), dim=-1)

                            models, inliers, choices, distances = \
                                sampling.sample_model_pool(data[bi], num_data[bi], opt.hyps, minimal_set_size,
                                                           inlier_fun, sampling.homography_from_points,
                                                           sampling.homography_consistency_measure, cur_probs,
                                                           device=device, model_size=model_dim)

                            all_grads[oh, :,mi, bi] = choices

                            inliers = sampling.soft_inlier_fun(distances, 5. / opt.threshold, opt.threshold)

                            all_models[oh, :, mi, bi, :] = models

                            if mi > 0:
                                all_joint_inliers[oh, :, mi, bi] = torch.max(
                                    inliers, all_best_inliers[oh, mi - 1, bi].unsqueeze(0).expand(opt.hyps, -1))
                            else:
                                all_joint_inliers[oh, :, mi, bi] = inliers

                            inlier_counts = torch.sum(inliers, dim=-1)
                            cumulative_inlier_counts = torch.sum(all_joint_inliers[oh, :, mi, bi], dim=-1)
                            best_hypo = torch.argmax(cumulative_inlier_counts)
                            best_inliers = inliers[best_hypo]
                            best_joint_inliers = all_joint_inliers[oh, :, mi, bi][best_hypo]
                            all_best_hypos[oh,mi] = best_hypo
                            all_best_inliers[oh, mi, bi] = best_joint_inliers

                            neg_inliers[mi + 1, bi] = 1 - best_joint_inliers

                            entropy = torch.distributions.categorical.Categorical(
                                probs=cur_probs).entropy()
                            all_entropies[oh, mi, bi] = entropy

                            if not opt.unconditional and mi+1 < opt.instances:
                                data_and_state[oh, mi + 1, bi, :, data_dim - 1] = \
                                    torch.max(data_and_state[oh, mi, bi, :, data_dim - 1], best_inliers)

                exclusive_inliers, _ = torch.max(all_best_inliers, dim=1)
                inlier_counts = torch.sum(exclusive_inliers, dim=-1)
                best_hypo = torch.argmax(inlier_counts, dim=0)

                for bi in range(0, data.size(0)):

                    inlier_count = 0
                    for mi in range(opt.instances):
                        exclusive_inliers, _ = torch.max(all_best_inliers[best_hypo[bi],0:mi+1,bi], dim=0)
                        inlier_count += torch.sum(exclusive_inliers, dim=-1)

                    loss = -(inlier_count * 1./opt.max_num_points * 1./opt.instances)

                    val_losses += [loss.cpu().numpy()]

                mean_entropy = torch.mean(all_entropies)
                entropies += [mean_entropy.cpu().detach().numpy()]

        print("Eval loss: ", np.mean(val_losses))
        tensorboard_writer.add_scalar('val/loss', np.mean(val_losses), iteration)
        tensorboard_writer.add_scalar('val/entropy', np.mean(entropies), iteration)

        torch.cuda.empty_cache()

        model.train()

    # ============================================================================

    avg_losses_epoch = []
    avg_per_model_losses_epoch = [[] for _ in range(opt.instances)]

    for idx, (data, num_data, _) in enumerate(trainset_loader):

        print("batch %6d / %d" % (idx+1, len(trainset_loader)), end="\r")

        data = data.to(device)

        B = data.size(0)
        Y = data.size(1)

        data_and_state = torch.zeros((opt.outerhyps, opt.instances, opt.samplecount, data.size(0), data.size(1), data_dim),
                                     device=device)
        all_grads = torch.zeros((opt.outerhyps, opt.instances, opt.samplecount, data.size(0), data.size(1)),
                                device=device)
        all_max_probs = torch.ones((opt.outerhyps, opt.instances, opt.samplecount, data.size(0), data.size(1)),
                                   device=device)
        all_log_probs = torch.ones((opt.outerhyps, opt.instances, opt.samplecount, data.size(0), data.size(1)),
                                   device=device)
        all_best_inliers = torch.zeros((opt.outerhyps, opt.instances, opt.samplecount, data.size(0), data.size(1)),
                                       device=device)
        all_models = torch.zeros((opt.outerhyps, opt.hyps, opt.instances, opt.samplecount, data.size(0), model_dim),
                                 device=device)
        all_best_hypos = torch.zeros((opt.outerhyps, opt.instances, opt.samplecount, data.size(0)), device=device)
        all_entropies = torch.zeros((opt.outerhyps, opt.instances, opt.samplecount, data.size(0),), device=device)
        all_losses = torch.zeros((opt.samplecount, data.size(0)), device=device)
        all_losses_per_model = torch.zeros((opt.samplecount, data.size(0), opt.instances), device=device)

        # model.eval()
        model.train()

        neg_inliers = \
            torch.ones((opt.outerhyps, opt.instances + 1, opt.samplecount, data.size(0), data.size(1)), device=device)

        for mi in range(opt.instances):

            all_joint_inliers = torch.zeros(
                (opt.outerhyps, opt.hyps, opt.samplecount, data.size(0), data.size(1)),
                device=device)

            data_and_state[:, mi, :, :, :, 0:data_dim - 1] = \
                data[:, :, 0:data_dim - 1].repeat(opt.outerhyps, opt.samplecount, 1, 1, 1)

            segments_and_selection_batched = data_and_state[:, mi].contiguous().view((-1, data.size(1), data_dim))
            log_probs_batched = model(segments_and_selection_batched)
            log_probs = log_probs_batched.view((opt.outerhyps, opt.samplecount, data.size(0), data.size(1)))

            all_log_probs[:, mi] = log_probs

            all_max_probs[:, mi] = neg_inliers[:, mi]

            cur_probs_ = log_probs.clone()
            for bi in range(B):
                cur_probs_[:, :, bi, :num_data[bi]] = torch.softmax(log_probs[:, :, bi, 0:num_data[bi]], dim=-1)
            cur_probs = cur_probs_.unsqueeze(1).expand(P, S, K, B, -1)

            entropy = torch.distributions.categorical.Categorical(probs=cur_probs_).entropy()
            all_entropies[:, mi] = entropy

            models, inliers, choices, distances = \
                sampling.sample_model_pool_multiple_parallel_batched(
                    data, num_data, minimal_set_size, inlier_fun, sampling.homographies_from_points_parallel_batched,
                    sampling.homographies_consistency_measure_parallel_3dim, probs=cur_probs, device=device,
                    model_size=model_dim, sample_count=opt.samplecount, min_prob=opt.min_prob)

            all_grads[:, mi] = choices.sum(1)

            all_models[:, :, mi] = models

            if mi > 0:
                all_joint_inliers = \
                    torch.max(inliers,
                              all_best_inliers[:, mi - 1].unsqueeze(1).expand(opt.outerhyps, opt.hyps, -1, -1, -1))
            else:
                all_joint_inliers = inliers

            cumulative_inlier_counts = torch.sum(all_joint_inliers, dim=-1)
            inlier_counts = torch.sum(inliers, dim=-1)
            best_hypo = torch.argmax(cumulative_inlier_counts, dim=1)

            for bi in range(0, data.size(0)):
                for oh in range(opt.outerhyps):
                    for si in range(opt.samplecount):
                        best_inliers = inliers[oh, best_hypo[oh, si, bi], si, bi]
                        best_joint_inliers = all_joint_inliers[oh, :, :, bi][best_hypo[oh, si, bi], si]
                        neg_inliers[oh, mi+1, si, bi, :] = 1 - best_joint_inliers
                        all_best_hypos[oh, mi, si, bi] = best_hypo[oh, si, bi]
                        all_best_inliers[oh, mi, si, bi] = best_joint_inliers

                        if not opt.unconditional:
                            if mi + 1 < opt.instances:
                                data_and_state[oh, mi + 1, si, bi, :, data_dim - 1] = torch.max(
                                    data_and_state[oh, mi, si, bi, :, data_dim - 1], best_inliers)

        exclusive_inliers, _ = torch.max(all_best_inliers, dim=1)
        inlier_counts = torch.sum(exclusive_inliers, dim=-1)
        best_hypo = torch.argmax(inlier_counts, dim=0)

        for bi in range(0, data.size(0)):
            for si in range(0, opt.samplecount):

                inlier_count = 0
                last_inlier_count = 0
                for mi in range(opt.instances):
                    exclusive_inliers, _ = torch.max(all_best_inliers[best_hypo[si, bi], 0:mi + 1, si, bi], dim=0)
                    current_inlier_count = torch.sum(exclusive_inliers, dim=-1)
                    inlier_count += current_inlier_count
                    inlier_increase = current_inlier_count - last_inlier_count
                    all_losses_per_model[si, bi, mi] = -(inlier_increase * 1. / opt.max_num_points)
                    last_inlier_count = current_inlier_count

                loss = -(inlier_count * 1. / opt.max_num_points * 1. / opt.instances)

                all_losses[si,bi] = loss

        baselines_per_model = all_losses_per_model.mean(dim=0)
        baselines = all_losses.mean(dim=0)

        for mi in range(opt.instances):
            avg_per_model_losses_epoch[mi] += [baselines_per_model[:,mi].mean().cpu().numpy().squeeze()]

        for bi in range(0, data.size(0)):
            baseline = baselines[bi]
            for si in range(0, opt.samplecount):
                all_grads[:, :, si, bi, :] *= (all_losses[si, bi] - baseline)


        avg_loss = all_losses.mean()
        avg_losses_epoch += [avg_loss]

        del all_best_inliers
        torch.cuda.empty_cache()

        optimizer.zero_grad()

        for bi in range(B):
            segments_and_selection_batched = data_and_state[:, :, :, bi].view((-1, Y, data_dim))
            log_probs_batched = model(segments_and_selection_batched)
            grads_batched = all_grads[:, :, :, bi].view((-1, 1, data.size(1), 1))

            if opt.loss_clamp > 0:
                grads_batched = torch.clamp(grads_batched, max=opt.loss_clamp, min=-opt.loss_clamp)

            if opt.max_prob_loss > 0:
                log_probs = log_probs_batched.view(
                    opt.outerhyps, opt.instances, opt.samplecount, 1, data.size(1))
                probs = torch.softmax(log_probs, dim=-1)
                max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
                probs = probs / torch.clamp(max_probs, min=1e-8)
                max_prob_loss = torch.clamp(probs-all_max_probs[:, :, :, bi].unsqueeze(3), min=0)
                max_prob_grad = opt.max_prob_loss * torch.ones_like(max_prob_loss, device=device)
                if opt.max_prob_loss_only:
                    torch.autograd.backward(max_prob_loss, max_prob_grad)
                else:
                    torch.autograd.backward((log_probs_batched, max_prob_loss), (grads_batched, max_prob_grad))

            else:
                torch.autograd.backward(log_probs_batched, grads_batched)

        optimizer.step()

        mean_entropy = torch.mean(all_entropies)
        tensorboard_writer.add_scalar('train/entropy', mean_entropy.cpu().detach().numpy(), iteration)

        iteration += 1

        if iteration % 100 == 0:
            avg_loss_epoch = sum([l.item() for l in avg_losses_epoch]) / len(avg_losses_epoch)
            tensorboard_writer.add_scalar('train/loss_avg', avg_loss_epoch, iteration)

    avg_loss_epoch = sum([l.item() for l in avg_losses_epoch]) / len(avg_losses_epoch)
    print("Avg epoch loss: ", avg_loss_epoch)
    tensorboard_writer.add_scalar('train/loss_epoch_avg', avg_loss_epoch, iteration)

    for mi in range(opt.instances):
        avg_per_model_loss_epoch = sum([l for l in avg_per_model_losses_epoch[mi]]) / len(avg_per_model_losses_epoch[mi])
        tensorboard_writer.add_scalar('train/model_loss_avg_%d' % mi, avg_per_model_loss_epoch, iteration)

    if opt.calr:
        scheduler.step()
        adjust_learning_rate(optimizer, scheduler.get_lr()[0])

torch.save(model.state_dict(), '%s/weights_%06d.net' % (ckpt_dir, opt.epochs))
