#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import torch

import onmt.model_builder
import onmt.inputters as inputters
import onmt.decoders.ensemble
from onmt.translate.beam_search import BeamSearch
from onmt.utils.misc import set_random_seed

import json
from onmt.train_and_translate import TrainTranslator


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')
    print('RESTORING ...')
    print('opt.models: {}'.format(opt.models))
    load_test_model = onmt.decoders.ensemble.load_test_model \
        if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

    translator = Translator.from_opt(
        model,
        fields,
        opt,
        model_opt,
        global_scorer=scorer,
        out_file=out_file,
        report_align=opt.report_align,
        report_score=report_score,
        logger=logger
    )
    return translator


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        # max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    src_elements = count * max_src_in_batch
    return src_elements


class Translator(object):
    def __init__(
            self,
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=-1,
            n_best=1,
            min_length=0,
            max_length=100,
            ratio=0.,
            beam_size=30,
            random_sampling_topk=1,
            random_sampling_temp=1,
            stepwise_penalty=None,
            dump_beam=False,
            block_ngram_repeat=0,
            ignore_when_blocking=frozenset(),
            replace_unk=False,
            tgt_prefix=False,
            phrase_table="",
            data_type="text",
            verbose=False,
            report_time=False,
            copy_attn=False,
            global_scorer=None,
            out_file=None,
            report_align=False,
            report_score=True,
            logger=None,
            seed=-1, opt=None):
        self.model = model
        self.fields = fields
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_vocab_len = len(self._tgt_vocab)

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = torch.device("cuda", self._gpu) \
            if self._use_cuda else torch.device("cpu")

        self.n_best = n_best
        self.max_length = max_length

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk

        self.min_length = min_length
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {
            self._tgt_vocab.stoi[t] for t in self.ignore_when_blocking}
        self.src_reader = src_reader
        self.tgt_reader = tgt_reader
        self.replace_unk = replace_unk
        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError(
                "replace_unk requires an attentional decoder.")
        self.tgt_prefix = tgt_prefix
        self.phrase_table = phrase_table
        self.data_type = data_type
        self.verbose = verbose
        self.report_time = report_time

        self.copy_attn = copy_attn

        self.global_scorer = global_scorer
        if self.global_scorer.has_cov_pen and \
                not self.model.decoder.attentional:
            raise ValueError(
                "Coverage penalty requires an attentional decoder.")
        self.out_file = out_file
        self.report_align = report_align
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

        set_random_seed(seed, self._use_cuda)

        assert opt is not None
        self.train_translator = TrainTranslator(
            model=model,
            fields=fields, max_dec_steps=opt.max_dec_steps,
            opt=opt
        )

    @classmethod
    def from_opt(
            cls,
            model,
            fields,
            opt,
            model_opt,
            global_scorer=None,
            out_file=None,
            report_align=False,
            report_score=True,
            logger=None):
        """Alternate constructor.

        Args:
            model (onmt.modules.NMTModel): See :func:`__init__()`.
            fields (dict[str, torchtext.data.Field]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (onmt.translate.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_align (bool) : See :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """

        src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
        tgt_reader = inputters.str2reader["text"].from_opt(opt)
        return cls(
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=opt.gpu,
            n_best=opt.n_best,
            min_length=opt.min_length,
            max_length=opt.max_length,
            ratio=opt.ratio,
            beam_size=opt.beam_size,
            random_sampling_topk=opt.random_sampling_topk,
            random_sampling_temp=opt.random_sampling_temp,
            stepwise_penalty=opt.stepwise_penalty,
            dump_beam=opt.dump_beam,
            block_ngram_repeat=opt.block_ngram_repeat,
            ignore_when_blocking=set(opt.ignore_when_blocking),
            replace_unk=opt.replace_unk,
            tgt_prefix=opt.tgt_prefix,
            phrase_table=opt.phrase_table,
            data_type=opt.data_type,
            verbose=opt.verbose,
            report_time=opt.report_time,
            copy_attn=model_opt.copy_attn,
            global_scorer=global_scorer,
            out_file=out_file,
            report_align=report_align,
            report_score=report_score,
            logger=logger,
            seed=opt.seed, opt=opt)

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _gold_score(self, batch, memory_bank, src_lengths, src_vocabs,
                    use_src_map, enc_states, batch_size, src):
        if "tgt" in batch.__dict__:
            gs = self._score_target(
                batch, memory_bank, src_lengths, src_vocabs,
                batch.src_map if use_src_map else None)
            self.model.decoder.init_state(src, memory_bank, enc_states)
        else:
            gs = [0] * batch_size
        return gs

    def translate(
            self,
            src,
            tgt=None,
            src_dir=None,
            batch_size=None,
            batch_type="sents",
            attn_debug=False,
            align_debug=False,
            phrase_table=""):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            src: See :func:`self.src_reader.read()`.
            tgt: See :func:`self.tgt_reader.read()`.
            src_dir: See :func:`self.src_reader.read()` (only relevant
                for certain types of data).
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging
            align_debug (bool): enables the word alignment logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """

        if batch_size is None:
            raise ValueError("batch_size must be set")

        if self.tgt_prefix and tgt is None:
            raise ValueError('Prefix should be feed to tgt if -tgt_prefix.')

        src_data = {"reader": self.src_reader, "data": src, "dir": src_dir}
        tgt_data = {"reader": self.tgt_reader, "data": tgt, "dir": None}
        _readers, _data, _dir = inputters.Dataset.config(
            [('src', src_data), ('tgt', tgt_data)])

        # corpus_id field is useless here
        if self.fields.get("corpus_id", None) is not None:
            self.fields.pop('corpus_id')
        data = inputters.Dataset(
            self.fields, readers=_readers, data=_data, dirs=_dir,
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred
        )

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=self._dev,
            batch_size=batch_size,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,
            train=False,
            sort=False,
            sort_within_batch=False,
            shuffle=False
        )

        for batch in data_iter:
            self.train_translator.repack_batch_for_test(batch)

            vocab_probs, entail_logits, new_entail_logits, tgt_lens = self.train_translator.run_batch_gen(
                batch
            )

            pred_labels = torch.argmax(new_entail_logits, 1).tolist()
            pred_ids_tensor = torch.argmax(vocab_probs, 2)
            pred_ids = pred_ids_tensor.transpose(0, 1).tolist()
            gen_texts = []
            sidx = 0
            for sent_ids in pred_ids:
                gen_texts.append(
                    ' '.join([self.train_translator._tgt_vocab.itos[wid] for wid in sent_ids][:tgt_lens[sidx]])
                )
                sidx += 1
            # for pred_label, sent in zip(pred_labels, gen_texts):
            #     self.out_file.write('{}\n'.format(json.dumps({
            #         'pred_label': pred_label,
            #         'gen_exp': sent
            #     })))
            #     self.out_file.flush()

            src_ids = batch.src[0].squeeze(2).transpose(0, 1).tolist()
            src_sents = [' '.join([self.train_translator._tgt_vocab.itos[widx] for widx in sent_ids]) for sent_ids in src_ids]
            for pred_label, sent, src_sent in zip(pred_labels, gen_texts, src_sents):
                src_sent = src_sent.replace(' <blank>', '')
                self.out_file.write('{}\n'.format(json.dumps({
                    'src_sent': src_sent,
                    'pred_label': pred_label,
                    'gen_exp': sent
                })))
                self.out_file.flush()
        print('Finished.')
