import math

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, fixed_weight=False):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if fixed_weight:
            self.weight = Variable(
                torch.Tensor(num_embeddings, embedding_dim),
                requires_grad=False)
        else:
            self.weight = nn.Parameter(
                torch.Tensor(num_embeddings, embedding_dim))
        self.sparse = sparse

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, input):
        padding_idx = self.padding_idx
        if padding_idx is None:
            padding_idx = -1
        return self._backend.Embedding.apply(
            input, self.weight,
            padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse
        )

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Embed(nn.ModuleList):

    def __init__(self, *args, dropout=0.0, padding_idx=None):
        embeds = []
        self.size = 0
        for i, _args in enumerate(args):
            if isinstance(_args, dict):
                vocab_size = _args.get('num_embeddings',
                                       _args.get('in_size', None))
                embed_size = _args.get('embedding_dim',
                                       _args.get('out_size', None))
                embeddings = _args.get('weight',
                                       _args.get('initialW', None))
                fixed_weight = _args.get('fixed_weight', False)
                if vocab_size is None or embed_size is None:
                    if embeddings is None:
                        raise ValueError('embeddings or in_size/out_size '
                                         'must be specified')
                    vocab_size, embed_size = embeddings.shape
            else:
                fixed_weight = False
                if isinstance(_args, np.ndarray):
                    vocab_size, embed_size = _args.shape
                    embeddings = _args
                elif isinstance(embeddings, tuple) and len(embeddings) == 2:
                    vocab_size, embed_size = embeddings
                    embeddings = None
                else:
                    raise ValueError('embeddings must be '
                                     'np.ndarray or tuple(len=2)')
            embed = Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_size,
                padding_idx=padding_idx,
                fixed_weight=fixed_weight,
            )
            if embeddings is not None:
                embed.weight.data.copy_(torch.from_numpy(embeddings))
            embeds.append(embed)
            self.size += embed_size
        super(Embed, self).__init__(embeds)

        assert dropout == 0 or type(dropout) == float
        self._dropout_ratio = dropout
        if dropout > 0:
            self._dropout = nn.Dropout(p=self._dropout_ratio)
        else:
            self._dropout = lambda x: x

    def forward(self, *xs):
        if next(self.parameters()).is_cuda:
            device_id = next(self.parameters()).get_device()
            hs = [_hs for _hs in self.embed_gpu(*xs, device_id=device_id)]
        else:
            hs = [_hs for _hs in self.embed_cpu(*xs)]
        return hs

    def embed_cpu(self, *xs):
        batch = len(xs[0])
        for i in range(batch):
            _hs = torch.cat(
                [self._dropout(
                    embed(Variable(
                        torch.from_numpy(_xs[i].astype(np.int64)))))
                 for _xs, embed in zip(xs, self)], dim=1)
            yield _hs

    def embed_gpu(self, *xs, device_id=None):
        batch = len(xs[0])
        for i in range(batch):
            _hs = torch.cat(
                [self._dropout(
                    embed(Variable(
                        torch.from_numpy(_xs[i].astype(np.int64))
                        .cuda(device_id))))
                 for _xs, embed in zip(xs, self)], dim=1)
            yield _hs


class MLP(nn.ModuleList):

    def __init__(self, layers):
        assert all(type(layer) == MLP.Layer for layer in layers)
        super(MLP, self).__init__(layers)

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

    class Layer(nn.Linear):

        def __init__(self, in_features, out_features,
                     activation=None, dropout=0.0, bias=True):
            super(MLP.Layer, self).__init__(in_features, out_features, bias)
            if activation is None:
                self._activate = lambda x: x
            else:
                if not callable(activation):
                    raise ValueError("activation must be callable: type={}"
                                     .format(type(activation)))
                self._activate = activation
            assert dropout == 0 or type(dropout) == float
            self._dropout_ratio = dropout
            if dropout > 0:
                self._dropout = nn.Dropout(p=self._dropout_ratio)
            else:
                self._dropout = lambda x: x

        def forward(self, x):
            size = x.size()
            if len(size) > 2:
                y = super(MLP.Layer, self).forward(
                    x.contiguous().view(-1, size[-1]))
                y = y.view(size[0:-1] + (-1,))
            else:
                y = super(MLP.Layer, self).forward(x)
            return self._dropout(self._activate(y))


class Biaffine(nn.Module):
    """
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/linalg.py#L116  # NOQA
    """

    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self._use_bias = bias

        shape = (in1_features + int(bias[0]),
                 in2_features + int(bias[1]),
                 out_features)
        self.weight = nn.Parameter(torch.Tensor(*shape))
        if bias[2]:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        is_cuda = next(self.parameters()).is_cuda
        device_id = next(self.parameters()).get_device() if is_cuda else None
        out_size = self.out_features
        batch_size, len1, dim1 = input1.size()
        if self._use_bias[0]:
            ones = torch.ones(batch_size, len1, 1)
            if is_cuda:
                ones = ones.cuda(device_id)
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            dim1 += 1
        len2, dim2 = input2.size()[1:]
        if self._use_bias[1]:
            ones = torch.ones(batch_size, len2, 1)
            if is_cuda:
                ones = ones.cuda(device_id)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1
        input1_reshaped = input1.contiguous().view(batch_size * len1, dim1)
        W_reshaped = torch.transpose(self.weight, 1, 2) \
            .contiguous().view(dim1, out_size * dim2)
        affine = torch.mm(input1_reshaped, W_reshaped) \
            .view(batch_size, len1 * out_size, dim2)
        biaffine = torch.transpose(
            torch.bmm(affine, torch.transpose(input2, 1, 2))
            .view(batch_size, len1, out_size, len2), 2, 3)
        if self._use_bias[2]:
            biaffine += self.bias.expand_as(biaffine)
        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'
