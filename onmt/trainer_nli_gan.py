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
from torch.autograd import grad as grad_fn
from torch.optim import Adam


def build_trainer(opt, device_id, model, fields, optim, critic, model_saver=None):
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
    trainer = Trainer(model, train_loss, valid_loss, optim,
                      critic,
                      trunc_size,
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


class BatchItem:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt


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
                 critic,
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
        self.train_translator = TrainTranslator(
            model=model,
            fields=fields, max_dec_steps=opt.max_dec_steps,
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

        self.total_label_loss = 0.0
        self.total_label_acc = 0.0
        self.total_label_loss_first_step = 0.0
        self.total_label_acc_first_step = 0.0
        self.total_label_step = 0.0
        self.report_every = opt.report_every
        self.dynamic_gen_prob = opt.dynamic_gen_prob
        self.penalty_each_step_cls = opt.penalty_each_step_cls
        self.loss_with_first_cls = opt.loss_with_first_cls

        self.best_label_acc = 0.0

        # critic
        self.one_tensor = torch.ones([]).cuda()
        self.mone_tensor = -1 * torch.ones([]).cuda()
        self.critic = critic
        self.critic_optimizer = Adam(self.critic.parameters(), lr=opt.critic_lr)
        self.critic_real_gain = 0.0
        self.critic_fake_gain = 0.0
        self.critic_gp = 0.0
        self.critic_wdistance = 0.0
        self.critic_wcost = 0.0
        self.critic_train_step = 0.0
        self.ap_critic_gain = 0.0
        self.critic_apply_step = 0.0

        self.critic_steps = opt.critic_steps
        self.ap_critic_steps = opt.ap_critic_steps

        self.save_model_path = opt.save_model

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
        cur_critic_steps = 0
        cur_ap_critic_steps = 0

        for i, (batches, normalization) in enumerate(
                self._accum_batches(train_iter)):
            step = self.optim.training_step
            # UPDATE DROPOUT
            self._maybe_update_dropout(step)

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
            # batches_bk = [BatchItem(bc.src, bc.tgt) for bc in batches]
            for bc_idx in range(len(batches)):
                self.train_translator.repack_batch_for_train(batches[bc_idx])
            self._gradient_accumulation(
                batches, normalization, total_stats,
                report_stats)
            if cur_critic_steps < self.critic_steps:
                self.train_critic(batches)
                cur_critic_steps += 1
            elif cur_ap_critic_steps < self.ap_critic_steps:
                self.apply_critic(batches)
                cur_ap_critic_steps += 1
            else:
                cur_critic_steps = 0
                cur_ap_critic_steps = 0

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim.learning_rate(),
                report_stats)

            if valid_iter is not None and step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d'
                                % (self.gpu_rank, step))
                valid_stats = self.validate(
                    valid_iter, moving_average=self.moving_average)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat \
                                step %d' % (self.gpu_rank, step))
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d'
                                % (self.gpu_rank, step))
                self._report_step(self.optim.learning_rate(),
                                  step, valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

            if (self.model_saver is not None
                and (save_checkpoint_steps != 0
                     and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)
                torch.save(self.critic, self.save_model_path + '_critic_{}.pt'.format(step))

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
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
        total_label_loss_first_step = 0.0
        total_label_acc_first_step = 0.0

        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                self.train_translator.repack_batch_for_train(batch)
                labels_ground_truth = batch.labels_ground_truth

                tgt_outer = batch.tgt
                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    vocab_probs, entail_logits, new_entail_logits, tgt_lens = self.train_translator.run_batch_gen(
                        batch, max_dec_steps=tgt_outer.size(0) - 1
                    )
                    # Compute loss.
                    _, batch_stats = self.valid_loss(batch, torch.log(vocab_probs), attns=None)

                    label_loss = self.label_loss_fn(new_entail_logits, labels_ground_truth)
                    acc_val = torch.mean(torch.argmax(new_entail_logits, -1).eq(labels_ground_truth).float())
                    label_loss_val = label_loss.item()
                    total_label_loss += label_loss_val
                    total_label_acc += acc_val
                    total_steps += 1
                    if self.loss_with_first_cls:
                        first_cls_logits = entail_logits[0]
                        label_loss_first = self.label_loss_fn(first_cls_logits, labels_ground_truth)
                        label_loss_first_val = label_loss_first.item()
                        acc_first_val = torch.mean(torch.argmax(first_cls_logits, -1).eq(labels_ground_truth).float())
                        total_label_loss_first_step += label_loss_first_val
                        total_label_acc_first_step += acc_first_val
                # Update statistics.
                stats.update(batch_stats)
        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data
        if self.best_label_acc < total_label_acc / total_steps:
            logger.info('FOUND NEW BEST VALID ACC:: {} %'.format(100 * total_label_acc / total_steps))
            self.best_label_acc = total_label_acc / total_steps
        if self.loss_with_first_cls:
            logger.info('Validation Classifier:::::::: ACC: {}, Loss: {}, ACC0: {}, Loss0: {}'.format(
                100 * total_label_acc / total_steps,
                total_label_loss / total_steps,
                100 * total_label_acc_first_step / total_steps,
                total_label_loss_first_step / total_steps
            ))
        else:
            logger.info('Validation Classifier:::::::: ACC: {}, Loss: {}'.format(
                100 * total_label_acc / total_steps,
                total_label_loss / total_steps
            ))
        # Set model back to training mode.
        valid_model.train()

        return stats

    def _handle_label(self, labels):
        labels_ids = labels.tolist()
        projected_ids = [self.train_translator.label_ids.index(x) for x in labels_ids]
        projected_ids = torch.LongTensor(projected_ids).type_as(labels)
        return projected_ids

    def _handle_label2(self, labels):
        labels_ids = labels.tolist()
        projected_ids_list = [self.train_translator.label_ids.index(x) for x in labels_ids]
        projected_ids = torch.LongTensor(projected_ids_list).type_as(labels)
        return projected_ids, projected_ids_list

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            # with torch.no_grad():
            #     # [batch_size]
            #     labels_ground_truth = batch.src[0][0].squeeze(1)
            #     labels_ground_truth = self._handle_label(labels_ground_truth)
            #     batch.src = (batch.src[0][1:], batch.src[1] - 1)
            labels_ground_truth = batch.labels_ground_truth
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            trunc_size = target_size

            batch = self.maybe_noise_source(batch)

            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt
            for j in range(1):
                # # 1. Create truncated target.
                # tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    if self.dynamic_gen_prob > 0 and random.random() <= self.dynamic_gen_prob:
                        vocab_probs, entail_logits, new_entail_logits, tgt_lens = self.train_translator.run_batch_gen(
                            batch, max_dec_steps=tgt_outer.size(0) - 1
                        )
                    else:
                        vocab_probs, entail_logits, new_entail_logits, tgt_lens = self.train_translator.run_batch_with_tgt(
                            batch
                        )
                    label_loss = self.label_loss_fn(new_entail_logits, labels_ground_truth)
                    acc_val = torch.mean(torch.argmax(new_entail_logits, -1).eq(labels_ground_truth).float())
                    label_loss_val = label_loss.item()
                    self.total_label_loss += label_loss_val
                    self.total_label_acc += acc_val
                    self.total_label_step += 1
                    if self.loss_with_first_cls:
                        first_cls_logits = entail_logits[0]
                        label_loss_first = self.label_loss_fn(first_cls_logits, labels_ground_truth)
                        label_loss_first_val = label_loss_first.item()
                        acc_first_val = torch.mean(torch.argmax(first_cls_logits, -1).eq(labels_ground_truth).float())
                        self.total_label_loss_first_step += label_loss_first_val
                        self.total_label_acc_first_step += acc_first_val
                    # 3. Compute loss.
                    loss, batch_stats = self.train_loss(
                        batch,
                        torch.log(vocab_probs),
                        attns=None,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size)
                    loss = loss + label_loss
                    if self.loss_with_first_cls:
                        loss = loss + label_loss_first
                try:
                    if loss is not None:
                        self.optim.backward(loss)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d", self.optim.training_step, k)

                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    self.optim.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

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

    def apply_critic(self, true_batches):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            # with torch.no_grad():
            #     # [batch_size]
            #     labels_ground_truth = batch.src[0][0].squeeze(1)
            #     labels_ground_truth, labels_gt_list = self._handle_label2(labels_ground_truth)
            #     batch.src = (batch.src[0][1:], batch.src[1] - 1)
            labels_ground_truth, labels_gt_list = batch.labels_ground_truth, batch.labels_gt_list
            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            premises_lens, types_ids = batch.premises_lens, batch.types_ids_tensor

            tgt_outer = batch.tgt
            for j in range(1):
                # # 1. Create truncated target.
                # tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    vocab_probs, entail_logits, new_entail_logits, tgt_lens = self.train_translator.run_batch_gen(
                        batch, max_dec_steps=tgt_outer.size(0) - 1
                    )
                    fake_src = vocab_probs
                    fake_src_lengths = torch.LongTensor(tgt_lens).to(batch.tgt.device)
                    pred_label_probs = torch.softmax(new_entail_logits, -1)

                    # fake_logits = self.critic(fake_src, fake_src_lengths)
                    gain_t_fake = self.critic(
                        x_src=src, x_src_lengths=src_lengths, premises_lens=premises_lens, types_ids=types_ids,
                        exp_src=fake_src, exp_src_lengths=fake_src_lengths, labels=pred_label_probs
                    )
                    # gain_t_fake = torch.mean(cal_critic_loss(fake_logits, labels_gt_list))
                    gain_t_fake = torch.mean(gain_t_fake)
                    gain_t_fake_val = gain_t_fake.item()
                    gain_t_fake.backward(self.mone_tensor)
                    self.ap_critic_gain += gain_t_fake_val
                    self.critic_apply_step += 1
                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    self.optim.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

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
        pass

    def train_critic(self, true_batches):
        self.critic.train()
        self.model.eval()

        for k, batch in enumerate(true_batches):
            with torch.no_grad():
                # [batch_size]
                # labels_ground_truth = batch.src[0][0].squeeze(1)
                # labels_ground_truth, labels_gt_list = self._handle_label2(labels_ground_truth)
                # batch.src = (batch.src[0][1:], batch.src[1] - 1)
                labels_ground_truth, labels_gt_list = batch.labels_ground_truth, batch.labels_gt_list

                tgt_outer = batch.tgt
                tgt_lengths = tgt_outer.ne(self.train_translator._tgt_pad_idx).squeeze(2)
                # [batch_size]
                tgt_lengths = torch.sum(tgt_lengths, 0) - 1

                vocab_probs, entail_logits, new_entail_logits, tgt_lens = self.train_translator.run_batch_gen(
                    batch, max_dec_steps=tgt_outer.size(0) - 1
                )
                pred_label_probs = torch.softmax(new_entail_logits, -1)

                real_critic_src = tgt_outer[1:]
                real_critic_src_lengths = tgt_lengths.long()
                fake_src = vocab_probs
                fake_src_lengths = torch.LongTensor(tgt_lens).to(tgt_outer.device)
                # batch first
                real_critic_src_onehot_t = convert_sent_ids_to_onehot(
                    real_critic_src.transpose(0, 1).squeeze(2), fake_src.transpose(0, 1)
                )

            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            premises_lens, types_ids = batch.premises_lens, batch.types_ids_tensor
            self.critic_optimizer.zero_grad()
            # REAL
            gain_t_real = self.critic(
                x_src=src, x_src_lengths=src_lengths, premises_lens=premises_lens, types_ids=types_ids,
                exp_src=real_critic_src, exp_src_lengths=real_critic_src_lengths, labels=labels_ground_truth
            )
            # gain_t_real = torch.mean(cal_critic_loss(real_logits, labels_gt_list))
            gain_t_real = torch.mean(gain_t_real)
            gain_t_real_val = gain_t_real.item()
            gain_t_real.backward(self.mone_tensor)
            # FAKE
            gain_t_fake = self.critic(
                x_src=src, x_src_lengths=src_lengths, premises_lens=premises_lens, types_ids=types_ids,
                exp_src=fake_src, exp_src_lengths=fake_src_lengths, labels=pred_label_probs
            )
            # gain_t_fake = torch.mean(cal_critic_loss(fake_logits, labels_gt_list))
            gain_t_fake = torch.mean(gain_t_fake)
            gain_t_fake_val = gain_t_fake.item()
            gain_t_fake.backward(self.one_tensor)
            # t_gradient_penalty = penalize_grad(
            #     self.critic, real_critic_src_onehot_t, fake_src.transpose(0, 1),
            #     real_critic_src_lengths, labels_gt_list
            # )
            t_gradient_penalty = penalize_grad(
                self.critic, real_critic_src_onehot_t.transpose(0, 1), fake_src,
                src, src_lengths, premises_lens, types_ids,
                real_critic_src_lengths, labels_ground_truth
            )

            t_gradient_penalty_val = t_gradient_penalty.item()
            t_gradient_penalty.backward()
            D_cost_t = gain_t_fake_val - gain_t_real_val + t_gradient_penalty_val
            Wasserstein_D_t = gain_t_real_val - gain_t_fake_val
            self.critic_optimizer.step()

            self.critic_real_gain += gain_t_real_val
            self.critic_fake_gain += gain_t_fake_val
            self.critic_gp += t_gradient_penalty_val
            self.critic_wdistance += Wasserstein_D_t
            self.critic_wcost += D_cost_t
            self.critic_train_step += 1

        self.model.train()
        self.critic.eval()

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
                if self.loss_with_first_cls:
                    logger.info(
                        "CLASSIFIER :::::::: Step %s; _CLS_ACC_: %6.2f; loss: %7.5f; _CLS_ACC0_: %6.2f; loss0: %7.5f"
                        % (
                            step,
                            100 * self.total_label_acc / self.total_label_step,
                            self.total_label_loss / self.total_label_step,
                            100 * self.total_label_acc_first_step / self.total_label_step,
                            self.total_label_loss_first_step / self.total_label_step
                        )
                    )
                else:
                    logger.info(
                        "CLASSIFIER :::::::: Step %s; _CLS_ACC_: %6.2f; loss: %7.5f"
                        % (
                            step,
                            100 * self.total_label_acc / self.total_label_step,
                            self.total_label_loss / self.total_label_step
                        )
                    )
                self.total_label_loss = 0.0
                self.total_label_acc = 0.0
                self.total_label_step = 0.0
                self.total_label_loss_first_step = 0.0
                self.total_label_acc_first_step = 0.0

                train_cric_has_log = self.critic_train_step > 0
                logger.info(
                    'CRITIC :::::::: r_gain: {:.4f}, f_gain: {:.4f}, gp: {:.4f}, wdist: {:.4f}, wcost: {:.4f}'.format(
                        self.critic_real_gain / self.critic_train_step if train_cric_has_log else 0.0,
                        self.critic_fake_gain / self.critic_train_step if train_cric_has_log else 0.0,
                        self.critic_gp / self.critic_train_step if train_cric_has_log else 0.0,
                        self.critic_wdistance / self.critic_train_step if train_cric_has_log else 0.0,
                        self.critic_wcost / self.critic_train_step if train_cric_has_log else 0.0,
                    )
                )
                self.critic_real_gain = 0.0
                self.critic_fake_gain = 0.0
                self.critic_gp = 0.0
                self.critic_wdistance = 0.0
                self.critic_wcost = 0.0
                self.critic_train_step = 0.0

                logger.info('APPLY CRITIC #### gain: {:.4f}'.format(
                    self.ap_critic_gain / self.critic_apply_step if self.critic_apply_step > 0 else 0.0
                ))
                self.ap_critic_gain = 0.0
                self.critic_apply_step = 0.0
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

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


def cal_critic_loss(probs, labels):
    """
    probs: [N, n_styles]
    """
    bz = probs.size(0)
    gains = []
    for idx in range(bz):
        gains.append(probs[idx][labels[idx]].unsqueeze(-1))
    return torch.cat(gains, 0)


def convert_sent_ids_to_onehot(sent_ids, refer_logits):
    """
    :param sent_ids: [N, max_x_len]
    :param refer_logits: [N, max_x_len, vocab_size]
    :return:
    """
    vocab_size = refer_logits.size(2)
    batch_size, max_x_len = sent_ids.size()
    sent_ids = sent_ids.contiguous().view(-1, 1)
    one_hots = torch.zeros(batch_size * max_x_len, vocab_size).to(refer_logits.device)
    one_hots = one_hots.scatter_(1, sent_ids, 1)
    one_hots = one_hots.contiguous().view(batch_size, max_x_len, vocab_size)
    return one_hots


def penalize_grad(d_model, real, fake,
                  src, src_lengths, premises_lens, types_ids,
                  real_critic_src_lengths, labels_ground_truth,
                  k=1.0, lamb=10, gp_norm_seq=True):
    """
    :param d_model: style_D
    :param real: [x_len, N, dim]
    :param fake: [x_len, N, dim]
    :param real_src_lengths: [N]
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
    batch_size = real.size(1)
    alpha = torch.rand(1, batch_size, 1).type_as(real).expand_as(real)
    interpolates = alpha * real + ((1 - alpha) * fake)
    interpolates = interpolates.detach()
    interpolates.requires_grad = True

    d_interpolates = d_model(
        x_src=src, x_src_lengths=src_lengths, premises_lens=premises_lens, types_ids=types_ids,
        exp_src=interpolates, exp_src_lengths=real_critic_src_lengths, labels=labels_ground_truth
    )
    error_middle = d_interpolates

    ones = torch.ones_like(error_middle)
    gradients = grad_fn(
        outputs=error_middle, inputs=interpolates,
        grad_outputs=ones, create_graph=True,
        retain_graph=True, only_inputs=True)[0]
    gradients = gradients.transpose(0, 1)
    # grad_penalty = ((gradients.norm(2, dim=1) - k) ** 2).mean() * lamb
    if gp_norm_seq:
        last_dim = gradients.size(-1)
        gradients = gradients.contiguous().view(-1, last_dim)
        useful_mask = sequence_mask(real_critic_src_lengths)
        useful_mask = useful_mask.contiguous().view(-1, 1).expand_as(gradients)
        gradients = gradients[useful_mask].view(-1, last_dim)

    # [N, dim]
    grad_norm = gradients.norm(2, dim=1)
    grad_penalty = ((grad_norm - k) ** 2).mean() * lamb

    return grad_penalty



