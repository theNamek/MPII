"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from onmt.encoders.transformer import TransformerEncoder

from onmt.modules import Embeddings, VecEmbedding
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
import copy
from onmt.utils.misc import sequence_mask


def build_embeddings(opt, text_field, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    old_opt = opt
    opt = copy.deepcopy(opt)
    opt.src_word_vec_size = opt.critic_src_word_vec_size

    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    if opt.model_type == "vec" and for_encoder:
        return VecEmbedding(
            opt.feat_vec_size,
            emb_dim,
            position_encoding=opt.position_encoding,
            dropout=(opt.dropout[0] if type(opt.dropout) is list
                     else opt.dropout),
        )

    pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    fix_word_vecs = opt.fix_word_vecs_enc if for_encoder \
        else opt.fix_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        fix_word_vecs=fix_word_vecs
    )
    return emb


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    enc_type = opt.encoder_type if opt.model_type == "text" \
        or opt.model_type == "vec" else opt.model_type
    old_opt = opt
    opt = copy.deepcopy(old_opt)
    opt.enc_layers = opt.critic_enc_layers
    opt.enc_rnn_size = opt.critic_enc_rnn_size
    opt.heads = opt.critic_heads
    opt.transformer_ff = opt.critic_transformer_ff

    return TransformerEncoder.from_opt(opt, embeddings)


class CriticOld(nn.Module):
    def __init__(self, encoder, emb_size, d_model, n_labels=3):
        super(CriticOld, self).__init__()
        self.encoder = encoder
        self.cls_vec = nn.Parameter(torch.randn(1, 1, emb_size))
        self.out_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_labels)
        )

    def enc_fn(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`

        src: [seq_len, batch_size, 1]
        src_lengths: [batch_size]
        """
        self.encoder._check_args(src, lengths)
        emb = self.encoder.embeddings(src)
        batch_size = src.size(1)
        cls_tensor = self.cls_vec.expand(1, batch_size, self.cls_vec.size(-1))

        emb = torch.cat((cls_tensor, emb), 0)
        out = emb.transpose(0, 1).contiguous()

        lengths = lengths + 1
        mask = ~sequence_mask(lengths, max_len=src.size(0) + 1).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        for layer in self.encoder.transformer:
            out = layer(out, mask)
        out = self.encoder.layer_norm(out)
        # out: [batch_size, seq_len, emb_size]
        return out

    def forward(self, src, src_lengths):
        """
        :param src: [seq_len, N, 1] or [seq_len, N, vocab_size]
        :param src_lengths: [N]
        :return:
        """
        out = self.enc_fn(src, src_lengths)
        # [N, d_model]
        cls_vec = out[:, 0]
        cls_logits = self.out_layer(cls_vec)
        return cls_logits


class Critic(nn.Module):
    def __init__(self, encoder, emb_size, d_model, n_labels=3):
        super(Critic, self).__init__()
        self.encoder = encoder
        self.label_emb = nn.Embedding(n_labels, emb_size)
        self.out_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def x_emb_fn(self, x_src, x_lengths, premises_lens, types_ids):
        src, lengths = x_src, x_lengths

        self.encoder._check_args(src, lengths)
        emb = self.encoder.embeddings(src, premises_lens=premises_lens)
        types_emb = self.encoder.types_embeddings(types_ids)
        emb = emb + types_emb

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)  # [N, 1, seq_len]

        return out, mask

    def exp_emb_fn(self, exp_src, exp_lengths, labels):
        """
        src: [seq_len, batch_size, 1]
        src_lengths: [batch_size]
        """
        src, lengths = exp_src, exp_lengths

        self.encoder._check_args(src, lengths)
        emb = self.encoder.embeddings(src)
        if len(labels.size()) > 1:
            label_emb = torch.matmul(labels, self.label_emb.weight)
        else:
            label_emb = self.label_emb(labels)
        label_emb = label_emb.unsqueeze(0)

        emb = torch.cat((label_emb, emb), 0)
        out = emb.transpose(0, 1).contiguous()

        lengths = lengths + 1
        mask = ~sequence_mask(lengths, max_len=src.size(0) + 1).unsqueeze(1)

        return out, mask

    def forward(self, x_src, x_src_lengths, premises_lens, types_ids,
                exp_src, exp_src_lengths, labels):
        """
        :param src: [seq_len, N, 1] or [seq_len, N, vocab_size]
        :param src_lengths: [N]
        :return:
        """
        x_emb, x_mask = self.x_emb_fn(x_src, x_src_lengths, premises_lens, types_ids)
        e_emb, e_mask = self.exp_emb_fn(exp_src, exp_src_lengths, labels)

        out = torch.cat((x_emb, e_emb), 1)
        mask = torch.cat((x_mask, e_mask), -1)
        # Run the forward pass of every layer of the tranformer.
        for layer in self.encoder.transformer:
            out = layer(out, mask)
        # out: [batch_size, seq_len, emb_size]
        out = self.encoder.layer_norm(out)
        # [N, d_model]
        cls_vec = out[:, 0]
        cls_logits = self.out_layer(cls_vec)
        return cls_logits


def build_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """

    # for back compat when attention_dropout was not defined
    try:
        model_opt.attention_dropout
    except AttributeError:
        model_opt.attention_dropout = model_opt.dropout

    # Build embeddings.
    if model_opt.model_type == "text" or model_opt.model_type == "vec":
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None

    # Build encoder.
    encoder = build_encoder(model_opt, src_emb)

    critic = Critic(encoder, src_emb.word_vec_size, model_opt.critic_enc_rnn_size)

    # Build NMTModel(= encoder + decoder).
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")
    model = critic

    if model_opt.param_init != 0.0:
        for p in model.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    if model_opt.param_init_glorot:
        for p in model.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    if hasattr(model.encoder, 'embeddings'):
        model.encoder.embeddings.load_pretrained_vectors(
            model_opt.pre_word_vecs_enc)

    model.to(device)
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        model.half()
    return model


def build_model(model_opt, opt, fields, checkpoint):
    logger.info('Building model...')
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    if opt.show_model_struct:
        logger.info(model)
    return model
