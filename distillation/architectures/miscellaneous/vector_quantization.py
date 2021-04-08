from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import distillation.architectures.tools as tools

class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon=1e-5,
        temperature=None):

        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings,
            self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(
            torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

        self._temperature = temperature
        if self._temperature is not None:
            assert isinstance(self._temperature, (float, int))
            assert self._temperature > 0.0

    def forward(self, inputs, encodings_out=False):
        # convert inputs from BCHW -> BHWC
        # if self._normalizeF:
        #     inputs = F.normalize(inputs, p=2, dim=1)
        #
        # elif self._global_pooling:
        #     inputs = tools.global_pooling(inputs, pool_type='avg')

        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = distances.new_full(
            (encoding_indices.shape[0], self._num_embeddings), 0.0)
        assert encodings.device == distances.device
        assert encodings.type() == distances.type()

        encodings.scatter_(1, encoding_indices, 1)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = (
                self._ema_cluster_size * self._decay +
                (1 - self._decay) * torch.sum(encodings, 0))

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon) /
                (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        loss = self._commitment_cost * e_latent_loss


        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        encoding_indices = encoding_indices.view(
            input_shape[0], 1, input_shape[1], input_shape[2])

        quantized = None

        if self._temperature is not None:
            encodings = F.softmax(- self._temperature * distances, dim=1)
            mean_assign_score = F.softmax(- self._temperature * distances, dim=1).max(dim=1)[0].mean()
                # F.softmax(- self._temperature * distances, dim=1).max(dim=1)[0].mean() ==> ~ 0.99

        assert encodings.size(1) == self._num_embeddings
        encodings = encodings.view(
            input_shape[0], input_shape[1], input_shape[2], self._num_embeddings)
        if encodings_out:
            return quantized, loss, perplexity, encoding_indices, encodings, mean_assign_score
        else:
            return quantized, loss, perplexity, encoding_indices


def create_model(opt):
    num_embeddings = opt['num_embeddings']
    embedding_dim = opt['embedding_dim']
    commitment_cost = opt['commitment_cost']
    decay = opt.get('decay', 0.0)
    epsilon = opt.get('epsilon', 1e-5)
    temperature = opt.get('temperature', None)

    return VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay, epsilon, temperature)
