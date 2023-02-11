"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

import torch
import traceback
import random

import onmt.utils
from onmt.utils.logging import logger
from onmt.train_and_translate import TrainTranslator
from onmt.utils.misc import sequence_mask
from torch.nn import NLLLoss, CrossEntropyLoss
from torch.nn import functional as F


def build_trainer(opt, device_id, model, fields, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    tgt_field = dict(fields)["tgt"].base_field
    train_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, tgt_field, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    # shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    shard_size = 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    source_noise = None
    if len(opt.src_noise) > 0:
        src_field = dict(fields)["src"].base_field
        corpus_id_field = dict(fields).get("corpus_id", None)
        if corpus_id_field is not None:
            ids_to_noise = corpus_id_field.numericalize(opt.data_to_noise)
        else:
            ids_to_noise = None
        source_noise = onmt.modules.source_noise.MultiNoise(
            opt.src_noise,
            opt.src_noise_prob,
            ids_to_noise=ids_to_noise,
            pad_idx=src_field.pad_token,
            end_of_sentence_mask=src_field.end_of_sentence_mask,
            word_start_mask=src_field.word_start_mask,
            device_id=device_id
        )

    report_manager = onmt.utils.build_report_manager(opt, gpu_rank)
    trainer = Trainer(model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           with_align=True if opt.lambda_align > 0 else False,
                           model_saver=model_saver if gpu_rank == 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           dropout=dropout,
                           dropout_steps=dropout_steps,
                           source_noise=source_noise,
                           opt=opt, fields=fields
                           )
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1, gpu_verbose_level=0,
                 report_manager=None, with_align=False, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0],
                 source_noise=None, opt=None, fields=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.with_align = with_align
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.source_noise = source_noise
        self.opt = opt
        max_dec_steps = opt.max_dec_steps
        self.train_translator = TrainTranslator(
            model=model,
            fields=fields, max_dec_steps=max_dec_steps,
            opt=opt
        )
        # self.label_loss_fn = NLLLoss()
        self.label_loss_fn = CrossEntropyLoss()

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()
        # self.optimizer_2 = torch.optim.Adam(self.model.parameters())
        self.optimizer_2 = torch.optim.Adam(
            [p for p in self.model.encoder.parameters()] + [p for p in self.model.decoder.first_step_map.parameters()],
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2)
        )

        self.total_label_loss = 0.0
        self.total_label_acc = 0.0
        self.total_label_step = 0.0
        self.total_train_step = 0.0
        self.report_every = opt.report_every
        self.dynamic_gen_prob = opt.dynamic_gen_prob
        self.penalty_each_step_cls = opt.penalty_each_step_cls

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d"
                            % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1)/(step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        best_acc = 0

        for i, (batches, normalization) in enumerate(
                self._accum_batches(train_iter)):
            # UPDATE DROPOUT
            self._maybe_update_dropout(self.total_train_step)

            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d \
                            n_minibatch %d"
                            % (self.gpu_rank, i + 1, len(batches)))

            if self.n_gpu > 1:
                normalization = sum(onmt.utils.distributed
                                    .all_gather_list
                                    (normalization))

            self._gradient_accumulation(
                batches, normalization, total_stats,
                report_stats)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(self.total_train_step)

            self._maybe_report_training(
                self.total_train_step, train_steps,
                self.optim.learning_rate(),
                report_stats)

            if valid_iter is not None and self.total_train_step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d'
                                % (self.gpu_rank, self.total_train_step))
                valid_acc = self.validate(
                    valid_iter, moving_average=self.moving_average)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    logger.info('#### New Best.')
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat \
                                step %d' % (self.gpu_rank, self.total_train_step))
            if (self.model_saver is not None
                and (save_checkpoint_steps != 0
                     and self.total_train_step % save_checkpoint_steps == 0)):
                self.model_saver.save(self.total_train_step, moving_average=self.moving_average)

            if train_steps > 0 and self.total_train_step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(self.total_train_step, moving_average=self.moving_average)
        return total_stats

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        total_label_loss = 0.0
        total_label_acc = 0.0
        total_steps = 0.0

        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" \
                    else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                with torch.no_grad():
                    # [batch_size]
                    labels_ground_truth = batch.src[0][0].squeeze(1)
                    labels_ground_truth = self._handle_label(labels_ground_truth)
                    batch.src = (batch.src[0][1:], batch.src[1] - 1)
                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    entail_log_probs = self.do_classify(batch)

                    label_loss = self.label_loss_fn(entail_log_probs, labels_ground_truth)
                    label_loss_val = label_loss.item()
                    acc_val = torch.mean(torch.argmax(entail_log_probs, -1).eq(labels_ground_truth).float())
                    total_label_loss += label_loss_val
                    total_label_acc += acc_val
                    total_steps += 1
        logger.info('Validation Classifier:::::::: ACC: {}, Loss: {}'.format(
            100 * total_label_acc / total_steps,
            total_label_loss / total_steps
        ))
        # Set model back to training mode.
        valid_model.train()

        return total_label_acc / total_steps

    def _handle_label(self, labels):
        labels_ids = labels.tolist()
        projected_ids = [self.train_translator.label_ids.index(x) for x in labels_ids]
        projected_ids = torch.LongTensor(projected_ids).type_as(labels)
        return projected_ids

    def handle_types(self, src):
        max_len = src.size(0)
        src_toks = src.squeeze(2).transpose(0, 1).tolist()
        types_ids = []
        sep_idx = self.train_translator._tgt_vocab.stoi['<sep>']
        for sent_ids in src_toks:
            idx = sent_ids.index(sep_idx)
            types_ids.append([0] + [1] * idx + [2] * (max_len - 1 - idx))
        types_ids_tensor = torch.LongTensor(types_ids).type_as(src)
        return types_ids_tensor

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()
        self.model.train()

        for k, batch in enumerate(true_batches):
            with torch.no_grad():
                # [batch_size]
                labels_ground_truth = batch.src[0][0].squeeze(1)
                labels_ground_truth = self._handle_label(labels_ground_truth)
                batch.src = (batch.src[0][1:], batch.src[1] - 1)
            for j in range(1):
                # # 1. Create truncated target.
                # tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                # if self.accum_count == 1:
                    # self.optim.zero_grad()
                self.optimizer_2.zero_grad()

                # with torch.cuda.amp.autocast(enabled=self.optim.amp):
                entail_log_probs = self.train_translator.do_purely_classify(batch)

                label_loss = self.label_loss_fn(entail_log_probs, labels_ground_truth)
                label_loss_val = label_loss.item()
                acc_val = torch.mean(torch.argmax(entail_log_probs, -1).eq(labels_ground_truth).float())
                self.total_label_loss += label_loss_val
                self.total_label_acc += acc_val
                self.total_label_step += 1
                self.total_train_step += 1
                loss = label_loss
                try:
                    if loss is not None:
                        # self.optim.backward(loss)
                        loss.backward()
                        self.optimizer_2.step()
                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d", self.optim.training_step, k)

                # # 4. Update the parameters and statistics.
                # if self.accum_count == 1:
                #     # Multi GPU gradient gather
                #     if self.n_gpu > 1:
                #         grads = [p.grad.data for p in self.model.parameters()
                #                  if p.requires_grad
                #                  and p.grad is not None]
                #         onmt.utils.distributed.all_reduce_and_rescale_tensors(
                #             grads, float(1))
                #     self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            if 0 == step % self.report_every:
                logger.info("CLASSIFIER :::::::: Step %s; _CLS_ACC_: %6.2f; loss: %7.5f"
                            % (
                                step,
                                100 * self.total_label_acc / self.total_label_step,
                                self.total_label_loss / self.total_label_step
                            )
                )
                self.total_label_loss = 0.0
                self.total_label_acc = 0.0
                self.total_label_step = 0.0

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def maybe_noise_source(self, batch):
        if self.source_noise is not None:
            return self.source_noise(batch)
        return batch
