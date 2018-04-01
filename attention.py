import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import math


class Energy(nn.Module):
    def __init__(self, enc_dim=10, dec_dim=10, att_dim=10, init_r=-4):
        """
        Modified Bahdahnau attention from
        "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
        http://arxiv.org/abs/1704.00784
        """
        super().__init__()
        self.W = nn.Linear(enc_dim, att_dim, bias=False)
        self.V = nn.Linear(dec_dim, att_dim, bias=False)
        self.b = nn.Parameter(torch.Tensor(att_dim).normal_())

        self.v = nn.utils.weight_norm(nn.Linear(10, 1))
        self.v.weight_g.data = torch.Tensor([1 / math.sqrt(att_dim)])

        self.r = nn.Parameter(torch.Tensor([init_r]))
        self.tanh = nn.Tanh()

    def forward(self, encoder_outputs, decoder_h):
        """
        Args:
            encoder_outputs: [batch_size, sequence_length, enc_dim]
            decoder_h: [batch_size, dec_dim]
        Return:
            Energy [batch_size, sequence_length]
        """
        batch_size, sequence_length, enc_dim = encoder_outputs.size()
        encoder_outputs = encoder_outputs.view(-1, enc_dim)
        energy = self.tanh(self.W(encoder_outputs) +
                           self.V(decoder_h).repeat(sequence_length, 1) + self.b)
        energy = self.v(energy).squeeze(-1) + self.r

        return energy.view(batch_size, sequence_length)


class MonotonicAttention(nn.Module):
    def __init__(self):
        """
        [Monotonic Attention] from
        "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
        http://arxiv.org/abs/1704.00784
        """
        super().__init__()

        self.monotonic_energy = Energy()
        self.sigmoid = nn.Sigmoid()

    def gaussian_noise(self, *size):
        """Additive gaussian nosie to encourage discreteness"""
        if torch.cuda.is_available():
            return Variable(torch.cuda.FloatTensor(*size).normal_())
        else:
            return Variable(torch.Tensor(*size).normal_())

    def cumprod(self, x):
        """Numerically stable cumulative product by cumulative sum in log-space"""
        return torch.exp(torch.cumsum(torch.log(torch.clamp(x, min=1e-10, max=1)), dim=1))

    def forward_soft(self, encoder_outputs, decoder_h, previous_alpha=None):
        """Soft monotonic attention"""
        batch_size, sequence_length, enc_dim = encoder_outputs.size()

        monotonic_energy = self.monotonic_energy(encoder_outputs, decoder_h)
        p_select = self.sigmoid(monotonic_energy + self.gaussian_noise(monotonic_energy.size()))
        cumprod_1_minus_p = self.cumprod(1 - p_select)

        if not previous_alpha is None:
            alpha = p_select * cumprod_1_minus_p * \
                torch.cumsum(previous_alpha / cumprod_1_minus_p, dim=1)
        # First iteration => alpha = [1, 0, 0 ... 0]
        else:
            alpha = torch.zeros(batch_size, sequence_length)
            alpha[:, 0] = torch.ones(batch_size)
            if torch.cuda.is_available:
                alpha = alpha.cuda()
            alpha = Variable(alpha)

        return alpha

    def forward_hard(self, encoder_outputs, decoder_h):
        """Hard monotonic attention"""
        pass


class MoChA(MonotonicAttention):
    def __init__(self, chunk_size=3):
        """
        [Monotonic Chunkwise Attention] from
        "Monotonic Chunkwise Attention" (ICLR 2018)
        https://openreview.net/forum?id=Hko85plCW
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_energy = Energy()

    def moving_sum(self, x, back, forward):
        # Pad window before applying convolution
        # [batch_size,    back + sequence_length + forward]
        x_padded = F.pad(x, pad=[back, forward])

        # Fake channel dimension for conv1d
        # [batch_size, 1, back + sequence_length + forward]
        x_padded = x_padded.unsqueeze(1)

        # Apply conv1d with filter of all ones for moving sum
        filters = Variable(torch.ones(1, 1, back + forward + 1), requires_grad=False)
        if torch.cuda.is_available():
            filters = filters.cuda()
        x_sum = F.conv1d(x_padded, filters)

        # Remove channel dimension
        return x_sum.squeeze(1)

    def chunkwise_attention(self, alpha, u):
        """
        Args:
            alpha [batch_size, sequence_length]: emission probability in monotonic attention
            u [batch_size, sequence_length]: chunk energy
            chunk_size (int): window size of chunk
        Return
            beta [batch_size, sequence_length]: MoChA weights
        """

        # numerical stability (divide by same exponent => doesn't affect softmax)
        u -= torch.max(u, dim=1, keepdim=True)[0]
        exp_u = torch.exp(u)
        # numerical stability (limit range of logit)
        exp_u = torch.clamp(exp_u, min=1e-5)

        # [batch_size, sequence_length]
        denominators = self.moving_sum(exp_u, back=self.chunk_size - 1, forward=0)

        # compute beta
        beta = exp_u * self.moving_sum(alpha / denominators, back=0, forward=self.chunk_size - 1)
        return beta

    def forward_soft(self, encoder_outputs, decoder_h, previous_alpha=None):
        """Soft monotonic chunkwise attention"""
        alpha = super().forward_train(encoder_outputs, decoder_h, previous_alpha)
        chunk_energy = self.chunk_energy(encoder_outputs, decoder_h)
        beta = self.chunkwise_attention(alpha, chunk_energy)
        return alpha, beta

    def forward_hard(self):
        """Hard monotonic chunkwise attention"""
        pass
