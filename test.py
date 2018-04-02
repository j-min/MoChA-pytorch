import unittest
import torch

from decoder import MoChADecoder


class MoChATest(unittest.TestCase):

    def setUp(self):
        self.batch_size = 5
        self.sequence_length = 40
        self.chunk_size = 3
        self.dim = 10
        self.vocab_size = 100

    def test_soft(self):
        """Soft Monotonic Chunkwise Attention"""

        enc_outputs = torch.Tensor(
            self.batch_size, self.sequence_length, self.dim).normal_()
        dec_inputs = torch.LongTensor(
            self.batch_size, self.sequence_length).clamp_(min=0, max=self.vocab_size - 1)
        decoder = MoChADecoder(vocab_size=self.vocab_size, chunk_size=self.chunk_size)

        if torch.cuda.is_available():
            enc_outputs = enc_outputs.cuda()
            dec_inputs = dec_inputs.cuda()
            decoder = decoder.cuda()

        decoder.forward_train(enc_outputs, dec_inputs)

    def test_hard(self):
        """Hard Monotonic Chunkwise Attention"""

        enc_outputs = torch.Tensor(
            self.batch_size, self.sequence_length, self.dim).normal_()
        decoder = MoChADecoder(
            vocab_size=self.vocab_size, chunk_size=self.chunk_size)

        if torch.cuda.is_available():
            enc_outputs = enc_outputs.cuda()
            decoder = decoder.cuda()

        decoder.forward_test(enc_outputs)


if __name__ == '__main__':
    unittest.main()
