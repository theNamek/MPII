# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import torch
from torch.autograd import grad as grad_fn


def penalize_sent_grad(d_model, real, fake, x_mask, desired_styles, k=1.0, lamb=10, gp_norm_seq=False):
    """
    :param d_model: sent_D
    :param real: [N, x_len, vocab_size], one-hot format
    :param fake: [N, x_len, vocab_size], after softmax
    :param x_mask: [N, x_len], true or one for mast be masked.
    :param desired_styles: [N]
    :param k:
    :param lamb:
    :param gp_norm_seq:
        the default implementation of wgan-gp by caogang version norms across the dim=1,
        which means norm on x_len axis. see: https://github.com/caogang/wgan-gp/issues/25 , for details.

        However, in the official implementation of the original paper "Improved Training of Wasserstein GANs",
        they pat the input of discriminator to shape [-1, dim].

        In our implementation, we simply reshape the gradients to [-1, dim], and norm across axis 1 .
    :return:
    """
    batch_size = real.size(0)
    # [0, 1)
    alpha = torch.rand(batch_size, 1, 1).type_as(real).expand_as(real)
    interpolates = alpha * real + ((1 - alpha) * fake)
    interpolates = interpolates.detach()
    interpolates.requires_grad = True
    d_interpolates = d_model(interpolates, style_ids=desired_styles, vocab_type=True)
    ones = torch.ones_like(d_interpolates)
    gradients = grad_fn(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=ones, create_graph=True,
        retain_graph=True, only_inputs=True)[0]
    # grad_penalty = ((gradients.norm(2, dim=1) - k) ** 2).mean() * lamb
    if not gp_norm_seq:
        # [N, dim]
        grad_norm = gradients.norm(2, dim=1)
        grad_penalty = ((grad_norm - k) ** 2).mean() * lamb
    else:
        last_dim = gradients.size(-1)
        grad_norm = gradients.contiguous().view(-1, last_dim).norm(2, dim=1)
        x_mask = x_mask.contiguous().view(-1)
        part = (grad_norm - k) ** 2
        part.masked_fill_(x_mask, 0.0)
        n_valid = torch.sum(1.0 - x_mask.float())
        grad_penalty = lamb * torch.sum(part) / n_valid
    return grad_penalty


def penalize_gen_style_grad(d_model, real, fake, x_ids, x_mask, src_styles,
                            k=1.0, lamb=10, gp_norm_seq=False):
    """
    :param d_model: style_D
    :param real: [N, x_len, dim]
    :param fake: [N, x_len, dim]
    :param x_ids: [N, x_len]
    :param x_mask: [N, x_len], true or one for mast be masked.
    :param src_styles: [N]
    :param k:
    :param lamb:
    :param gp_norm_seq:
        the default implementation of wgan-gp by caogang version norms across the dim=1,
        which means norm on x_len axis. see: https://github.com/caogang/wgan-gp/issues/25 , for details.

        However, in the official implementation of the original paper "Improved Training of Wasserstein GANs",
        they pat the input of discriminator to shape [-1, dim].

        In our implementation, we simply reshape the gradients to [-1, dim], and norm across axis 1 .
    :return:
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1).type_as(real).expand_as(real)
    interpolates = alpha * real + ((1 - alpha) * fake)
    interpolates = interpolates.detach()
    interpolates.requires_grad = True

    d_interpolates = d_model(x_ids=x_ids, x_states=interpolates, x_mask=x_mask, style_ids=src_styles)

    ones = torch.ones_like(d_interpolates)
    gradients = grad_fn(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=ones, create_graph=True,
        retain_graph=True, only_inputs=True)[0]
    # grad_penalty = ((gradients.norm(2, dim=1) - k) ** 2).mean() * lamb
    if not gp_norm_seq:
        # [N, dim]
        grad_norm = gradients.norm(2, dim=1)
        grad_penalty = ((grad_norm - k) ** 2).mean() * lamb
    else:
        last_dim = gradients.size(-1)
        grad_norm = gradients.contiguous().view(-1, last_dim).norm(2, dim=1)
        x_mask = x_mask.contiguous().view(-1)
        part = (grad_norm - k) ** 2
        part.masked_fill_(x_mask, 0.0)
        n_valid = torch.sum(1.0 - x_mask.float())
        grad_penalty = lamb * torch.sum(part) / n_valid
    return grad_penalty


