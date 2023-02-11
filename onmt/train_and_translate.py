#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import torch
from torch.nn import functional as F
LABEL_ENTAIL = 'label_ent'
LABEL_NEUTRAL = 'label_neu'
LABEL_CONTRADICTION = 'label_con'
CLS_TOKEN = '<cls>'
SEP_TOKEN = '<sep>'
import logging


class TrainTranslator(object):
    def __init__(
            self,
            model,
            fields,
            max_dec_steps,
            ues_ori_dec=False,
            opt=None
    ):
        self.model = model
        self.fields = fields
        src_field = dict(self.fields)["src"].base_field
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._src_vocab = src_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_entail_idx = self._tgt_vocab.stoi[LABEL_ENTAIL]
        self._tgt_neutral_idx = self._tgt_vocab.stoi[LABEL_NEUTRAL]
        self._tgt_contradict_idx = self._tgt_vocab.stoi[LABEL_CONTRADICTION]
        self._cls_idx = self._tgt_vocab.stoi[CLS_TOKEN]
        self._sep_idx = self._tgt_vocab.stoi[SEP_TOKEN]
        # self._tgt_dot_idx = self._tgt_vocab.stoi['.']
        self._tgt_vocab_len = len(self._tgt_vocab)
        self.max_dec_steps = max_dec_steps
        self.ues_ori_dec = ues_ori_dec
        self.label_ids = [self._tgt_entail_idx, self._tgt_neutral_idx, self._tgt_contradict_idx]
        logging.info('*' * 100)
        logging.info('label_ids: {}'.format(self.label_ids))
        self._src_entail_idx = self._src_vocab.stoi[LABEL_ENTAIL]
        self._src_neutral_idx = self._src_vocab.stoi[LABEL_NEUTRAL]
        self._src_contradict_idx = self._src_vocab.stoi[LABEL_CONTRADICTION]
        self.src_label_ids = [self._src_entail_idx, self._src_neutral_idx, self._src_contradict_idx]
        logging.info('src_label_ids: {}'.format(self.src_label_ids))
        logging.info('*' * 100)

    def fetch_premise_lens(self, real_src):
        wids = real_src.squeeze(2).transpose(0, 1).tolist()
        premises_lens = [sent_ids.index(self._sep_idx) for sent_ids in wids]
        return premises_lens

    def fetch_types(self, premises_lens, max_len, device):
        types_ids = []
        for plen in premises_lens:
            types_ids.append([0] + [1] * plen + [2] + [3] * (max_len - 2 - plen))
        types_ids_tensor = torch.LongTensor(types_ids).to(device)
        return types_ids_tensor

    def handle_sep_types(self, batch):
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
            else (batch.src, None)

        with torch.no_grad():
            premises_lens = self.fetch_premise_lens(src)
            types_ids_tensor = self.fetch_types(premises_lens, src.size(0), src.device).transpose(0, 1)
        return premises_lens, types_ids_tensor

    def _fetch_label(self, labels):
        labels_ids = labels.tolist()
        projected_ids_list = [self.label_ids.index(x) for x in labels_ids]
        projected_ids = torch.LongTensor(projected_ids_list).type_as(labels)
        return projected_ids, projected_ids_list

    def repack_batch_for_train(self, batch):
        with torch.no_grad():
            labels_ground_truth = batch.src[0][0].squeeze(1)
            labels_ground_truth, labels_gt_list = self._fetch_label(labels_ground_truth)
            batch.src = (batch.src[0][1:], batch.src[1] - 1)
            batch.labels_ground_truth, batch.labels_gt_list = labels_ground_truth, labels_gt_list
            premises_lens, types_ids_tensor = self.handle_sep_types(batch)
            batch.premises_lens, batch.types_ids_tensor = premises_lens, types_ids_tensor
            # print('indices: {}'.format(batche.indices))  # 存放样本来自哪个, 在数据集上的下标

    def repack_batch_for_test(self, batch):
        with torch.no_grad():
            premises_lens, types_ids_tensor = self.handle_sep_types(batch)
            batch.premises_lens, batch.types_ids_tensor = premises_lens, types_ids_tensor

    # def _run_encoder(self, batch):
    #     src, src_lengths = batch.src if isinstance(batch.src, tuple) \
    #                        else (batch.src, None)
    #     with torch.no_grad():
    #         premises_lens, types_ids_tensor = self.handle_sep_types(batch)
    #
    #     enc_states, memory_bank, src_lengths = self.model.encoder(
    #         src, types_ids=types_ids_tensor, lengths=src_lengths, premises_lens=premises_lens)
    #     if src_lengths is None:
    #         assert not isinstance(memory_bank, tuple), \
    #             'Ensemble decoding only supported for text data'
    #         src_lengths = torch.Tensor(batch.batch_size) \
    #                            .type_as(memory_bank) \
    #                            .long() \
    #                            .fill_(memory_bank.size(0))
    #     return src, enc_states, memory_bank, src_lengths
    def _run_encoder(self, batch):
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                           else (batch.src, None)
        premises_lens, types_ids_tensor = batch.premises_lens, batch.types_ids_tensor

        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, types_ids=types_ids_tensor, lengths=src_lengths, premises_lens=premises_lens)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                               .type_as(memory_bank) \
                               .long() \
                               .fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths

    def do_purely_classify(self, batch):
        """
        batch: 需要剔除第一个 label token
        """
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        cls_vec = memory_bank[0]
        entail_logits = self.model.decoder.first_step_map(cls_vec)
        return entail_logits

    def _decode_and_generate(
            self,
            decoder_in,
            memory_bank,
            memory_lengths,
            step=None):
        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
        )
        # log_probs = self.model.generator(dec_out.squeeze(0))
        # return log_probs
        n_g_layers = len(self.model.generator)
        # [1, N, vocab_size]
        probs = dec_out
        for i in range(n_g_layers - 1):
            probs = self.model.generator[i](probs)
        probs = F.softmax(probs, -1)
        return probs

    def decode_batch_gen(self, src, enc_states, memory_bank, src_lengths, max_dec_steps=None):
        if max_dec_steps is None:
            max_dec_steps = self.max_dec_steps

        # (0) Prep the components of the search.
        batch_size = src_lengths.size(0)
        # (1) Run the encoder on the src.
        self.model.decoder.init_state(src, memory_bank, enc_states)
        # (3) Begin decoding step by step:
        # [1, N, 1]
        decoder_input = torch.LongTensor([self._tgt_bos_idx] * batch_size).type_as(src_lengths)
        decoder_input = decoder_input.unsqueeze(0).unsqueeze(2)
        dec_outs = []
        for step in range(max_dec_steps):
            # decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
            # probs: [1, N, vocab_size]
            probs = self._decode_and_generate(
                decoder_in=decoder_input,
                memory_bank=memory_bank,
                memory_lengths=src_lengths,
                step=step)
            dec_outs.append(probs[-1:])
            decoder_input = dec_outs[-1]
        results = torch.cat(dec_outs, 0)
        entail_logits = self.model.decoder.state['entail_logits']
        tgt_lens = self.get_gen_len(results)

        final_entail_logits = []
        for bidx in range(batch_size):
            final_entail_logits.append(entail_logits[tgt_lens[bidx]][bidx].unsqueeze(0))
        final_entail_logits = torch.cat(final_entail_logits, 0)

        return results, entail_logits, final_entail_logits, tgt_lens

    def fetch_tgt_lens(self, batch):
        tgt_outer = batch.tgt
        with torch.no_grad():
            tgt_lengths = tgt_outer.ne(self._tgt_pad_idx).squeeze(2)
            # [batch_size]
            tgt_lengths = torch.sum(tgt_lengths, 0) - 1
        return tgt_lengths.tolist()

    def decode_batch_with_tgt(self, src, enc_states, memory_bank, src_lengths, tgt_in, tgt_lens):
        max_dec_steps = tgt_in.size(0)

        # (0) Prep the components of the search.
        batch_size = src_lengths.size(0)
        # (1) Run the encoder on the src.
        self.model.decoder.init_state(src, memory_bank, enc_states)
        # (3) Begin decoding step by step:
        dec_outs = []
        for step in range(max_dec_steps):
            # [1, N, 1]
            decoder_input = tgt_in[step].unsqueeze(0)
            # probs: [1, N, vocab_size]
            probs = self._decode_and_generate(
                decoder_in=decoder_input,
                memory_bank=memory_bank,
                memory_lengths=src_lengths,
                step=step)
            dec_outs.append(probs)
        results = torch.cat(dec_outs, 0)
        entail_logits = self.model.decoder.state['entail_logits']

        final_entail_logits = []
        for bidx in range(batch_size):
            final_entail_logits.append(entail_logits[tgt_lens[bidx]][bidx].unsqueeze(0))
        final_entail_logits = torch.cat(final_entail_logits, 0)

        return results, entail_logits, final_entail_logits, tgt_lens

    def run_batch_gen(self, batch, max_dec_steps=None):
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        vocab_probs_tensor, entail_probs, new_entail_probs, tgt_lens = self.decode_batch_gen(
            src, enc_states, memory_bank, src_lengths,
            max_dec_steps=max_dec_steps
        )
        return vocab_probs_tensor, entail_probs, new_entail_probs, tgt_lens

    def run_batch_with_tgt(self, batch):
        tgt_lens = self.fetch_tgt_lens(batch)

        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        vocab_probs_tensor, entail_probs, new_entail_probs, tgt_lens = self.decode_batch_with_tgt(
            src, enc_states, memory_bank, src_lengths,
            tgt_in=batch.tgt[:-1], tgt_lens=tgt_lens
        )
        return vocab_probs_tensor, entail_probs, new_entail_probs, tgt_lens

    def get_gen_len(self, vocab_probs):
        """
        vocab_probs: [x_len, batch_size, vocab_size]
        """
        seq_len = vocab_probs.size(0)
        batch_size = vocab_probs.size(1)
        seq_lens = [seq_len] * batch_size
        pred_ids_tensor = torch.argmax(vocab_probs, 2)
        pred_ids = pred_ids_tensor.transpose(0, 1).tolist()
        for i in range(batch_size):
            cur_sent = pred_ids[i]
            cur_eos_idx = cur_sent.index(self._tgt_eos_idx) if self._tgt_eos_idx in cur_sent else seq_len - 1
            if cur_eos_idx <= 1:
                cur_eos_idx = seq_len - 1
            cur_sent_len = cur_eos_idx + 1
            seq_lens[i] = cur_sent_len
        return seq_lens

