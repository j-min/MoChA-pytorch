import unittest

from decoder import Decoder

import torch
from torch.autograd import Variable


class MoChATest(unittest.TestCase):

    def test_soft(self):

        batch_size = 5
        sequence_length = 10
        chunk_size = 3
        dim = 10
        vocab_size = 100

        enc_outputs = Variable(torch.Tensor(batch_size, sequence_length, dim).normal_()).cuda()
        dec_inputs = Variable(torch.LongTensor(
            batch_size, sequence_length).clamp_(min=0, max=vocab_size - 1)).cuda()

        decoder = Decoder(vocab_size=vocab_size, chunk_size=chunk_size).cuda()
        decoder.forward_train(enc_outputs, dec_inputs)

    def test_hard(self):
        pass


if __name__ == '__main__':
    unittest.main()
