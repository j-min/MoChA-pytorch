import torch
from torch.autograd import Variable
from torch import nn

from attention import MoChA


class Decoder(nn.Module):
    def __init__(self, enc_dim=10, dec_dim=10, embedding_dim=10, att_dim=10,
                 out_dim=10, vocab_size=100, chunk_size=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cell = nn.RNNCell(embedding_dim, dec_dim)
        self.attention = MoChA(chunk_size)

        # Effective Approaches to Attention-based Neural Machine Translation (EMNLP 2015)
        self.combine_c_h = nn.Linear(enc_dim + dec_dim, out_dim, bias=False)
        self.tanh = nn.Tanh()
        self.proj_vocab = nn.Linear(out_dim, vocab_size)

    def init_x(self, batch_size, sos_id=1):
        if torch.cuda.is_available:
            return Variable(torch.cuda.LongTensor([sos_id] * batch_size))
        else:
            return Variable(torch.LongTensor([sos_id] * batch_size))

    def init_h(self, batch_size, dec_dim=10):
        if torch.cuda.is_available:
            return Variable(torch.cuda.FloatTensor(batch_size, dec_dim).normal_())
        else:
            return Variable(torch.Tensor(batch_size, dec_dim).normal_())

    def forward_soft(self, encoder_outputs, decoder_inputs):
        """
        Args:
            encoder_outputs [batch_size, enc_sequence_length, enc_dim]
            decoder_inputs [batch_size, dec_sequence_length]
        Return:
            logits: [batch_size, dec_sequence_length, vocab_size]
        """
        batch_size, enc_sequence_length, enc_dim = encoder_outputs.size()
        batch_size, dec_sequence_length = decoder_inputs.size()

        x = self.init_x(batch_size)
        h = self.init_h(batch_size)
        alpha = None
        logit_list = []
        for i in range(dec_sequence_length):
            x = self.embedding(x)
            h = self.cell(x, h)
            alpha, beta = self.attention.forward_soft(encoder_outputs, h, alpha)

            # Weighted-sum
            context = torch.sum(beta.unsqueeze(1) * encoder_outputs, dim=1)

            # [batch_size, out_dim]
            attentional = self.tanh(self.combine_c_h(torch.cat([context, h], dim=1)))

            # [batch_size, vocab_size]
            logit = self.proj_vocab(attentional)
            logit_list.append(logit)

            x = decoder_inputs[:, i]

        return torch.stack(logit_list, dim=1)

    def forward_hard(self, encoder_outputs):
        all_finished = [False] * batch_size
        pass
#         while sum(all_finished) < batch_size:
