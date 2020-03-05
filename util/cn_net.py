import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNet(nn.Module):
    '''
    Reimplementation of the network "Learning to find good correspondences"

    '''

    def __init__(self, blocks, input_dim, batch_norm=True):
        '''
        Constructor.
        '''
        super(CNNet, self).__init__()

        self.input_dim = input_dim

        self.p_in = nn.Conv2d(self.input_dim, 128, 1, 1, 0)

        self.res_blocks = []

        self.batch_norm = batch_norm

        for i in range(0, blocks):
            if batch_norm:
                self.res_blocks.append((
                    nn.Conv2d(128, 128, 1, 1, 0),
                    nn.BatchNorm2d(128),
                    nn.Conv2d(128, 128, 1, 1, 0),
                    nn.BatchNorm2d(128),
                ))
            else:
                self.res_blocks.append((
                    nn.Conv2d(128, 128, 1, 1, 0),
                    nn.Conv2d(128, 128, 1, 1, 0),
                ))

        for i, r in enumerate(self.res_blocks):
            super(CNNet, self).add_module(str(i) + 's0', r[0])
            super(CNNet, self).add_module(str(i) + 's1', r[1])
            if batch_norm:
                super(CNNet, self).add_module(str(i) + 's2', r[2])
                super(CNNet, self).add_module(str(i) + 's3', r[3])

        self.p_out = nn.Conv2d(128, 1, 1, 1, 0)

    def masked_instance_norm(self, data, mask):

        B = data.size(0)

        num_elements = mask.sum(-1)

        new_data_batch = []
        for bi in range(B):
            new_data_a = F.instance_norm(data[bi, :, :num_elements[bi]])
            if num_elements[bi] < data.size(2):
                new_data_b = data[bi, :, num_elements[bi]:]
                new_data = torch.cat([new_data_a, new_data_b], dim=1)
            else:
                new_data = new_data_a
            new_data_batch += [new_data]

        data = torch.stack(new_data_batch, dim=0)

        return data

    def forward(self, inputs, mask=None):
        '''
        Forward pass.

        inputs -- 4D data tensor (BxCxHxW)
        '''
        inputs = torch.transpose(inputs, 1, 2).unsqueeze(-1)

        batch_size = inputs.size(0)
        data_size = inputs.size(2)

        x = inputs[:, 0:self.input_dim]
        x = F.relu(self.p_in(x))

        for r in self.res_blocks:
            res = x
            if mask is None:
                if self.batch_norm:
                    x = F.relu(r[1](F.instance_norm(r[0](x))))
                    x = F.relu(r[3](F.instance_norm(r[2](x))))
                else:
                    x = F.relu(F.instance_norm(r[0](x)))
                    x = F.relu(F.instance_norm(r[1](x)))
            else:
                x = F.relu(r[1](self.masked_instance_norm(r[0](x), mask)))
                x = F.relu(r[3](self.masked_instance_norm(r[2](x), mask)))
            x = x + res

        log_probs = F.logsigmoid(self.p_out(x))

        # normalization
        log_probs = log_probs.view(batch_size, -1)
        normalizer = torch.logsumexp(log_probs, dim=1)
        normalizer = normalizer.unsqueeze(1).expand(-1, data_size)
        log_probs = log_probs - normalizer
        log_probs = log_probs.view(batch_size, 1, data_size, 1)

        return log_probs
