"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn

import onmt
import onmt.inputters as inputters
from onmt.utils.logging import logger

def build_loss_compute(model, tgt_vocab, opt, train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength)
    else:
        compute = NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing if train else 0.0)
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[inputters.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _make_shard_state_valid(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError


    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def print_output(self, train, output, target, **kwargs):
        return NotImplementedError


    def monolithic_compute_loss(self, batch, output, attns, train):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        shard_state_valid = self._make_shard_state_valid(batch, output, range_, attns)
        self.print_output(train, **shard_state_valid)
        _, batch_stats = self._compute_loss(batch, train, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization, train=True):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = onmt.utils.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns)
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, train, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, scores, target, train=False):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        if not train:
            logger.info("num_correct " + str(num_correct) + " num_non_padding " + str(num_non_padding))
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        assert 0.0 <= label_smoothing <= 1.0
        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight, size_average=False)
        self.confidence = 1.0 - label_smoothing


    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _make_shard_state_valid(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "source": batch.src[0][range_[0] + 1: range_[1]],
            "answer": batch.ans[0][range_[0] + 1: range_[1]],
        }

    def print_output(self, train, output, target, source, answer):
        scores = self.generator(self._bottle(output))
        gtruth = target.view(-1)
        l, b_sz = target.size()
        l_src, b_sz_src = source.size()
        if not train:
            logger.info(str(len(scores)) + " " + str(len(gtruth)))
            logger.info(scores.size())
            logger.info(gtruth.size())

            pred = scores.data.max(1)[1]
            gtruth_data = target.view(-1).data
            src_data = source.view(-1).data
            pred = pred.view(l, b_sz)
            gtruth_data = gtruth_data.view(l, b_sz)
            src_data = src_data.view(l_src, b_sz_src)

            for i in range(b_sz):
                b_sent = []
                b_sent_pred = []
                b_sent_src = []
                for j in range(l):
                    b_sent.append(self.tgt_vocab.itos[gtruth_data[j][i]])
                    b_sent_pred.append(self.tgt_vocab.itos[pred[j][i]])

                for j in range(l_src):
                    b_sent_src.append(self.tgt_vocab.itos[src_data[j][i]])
                logger.info("Question: " + " ".join(b_sent_src))
                logger.info("groundtruth: " + " ".join(b_sent))
                logger.info("prediction: " + " ".join(b_sent_pred))
                logger.info("\n\n")
                print("question: " + " ".join(b_sent_src))
                print("groundtruth: " + " ".join(b_sent))
                print("prediction: " + " ".join(b_sent_pred))
                print("\n\n")

            #logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    def _compute_loss(self, batch, train, output, target):
        scores = self.generator(self._bottle(output))
        gtruth = target.view(-1)

        ########## Modified ###################
        '''
        logger.info("scores")
        logger.info(scores.size())
        logger.info("target")
        logger.info(target.size())
        '''

        '''
        l, b_sz = target.size()
        l_src, b_sz_src = source.size()
        logger.info("")
        if not train:
            logger.info(str(len(scores)) + " " + str(len(gtruth)))
            logger.info(scores.size())
            logger.info(gtruth.size())

            pred = scores.data.max(1)[1]
            gtruth_data = target.view(-1).data
            src_data = source.view(-1).data
            pred = pred.view(l, b_sz)
            gtruth_data = gtruth_data.view(l, b_sz)
            src_data = src_data.view(l_src, b_sz_src)

            for i in range(b_sz):
                b_sent = []
                b_sent_pred = []
                b_sent_src = []
                for j in range(l):
                    b_sent.append(self.tgt_vocab.itos[gtruth_data[j][i]])
                    b_sent_pred.append(self.tgt_vocab.itos[pred[j][i]])

                for j in range(l_src):
                    b_sent_src.append(self.tgt_vocab.itos[src_data[j][i]])
                logger.info("Question: " + b_sent_src)
                logger.info("groundtruth: " + b_sent)
                logger.info("prediction: " + b_sent_pred)

                print("question: " + b_sent_src)
                print("groundtruth: " + b_sent)
                print("prediction: " + b_sent_pred)
                print("\n\n")

            logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        '''

        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze(-1)
            # log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.size(0) > 0:
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_
        loss = self.criterion(scores, gtruth)
        if self.confidence < 1:
            # Default: report smoothed ppl.
            # loss_data = -log_likelihood.sum(0)
            loss_data = loss.data.clone()
        else:
            loss_data = loss.data.clone()

        stats = self._stats(loss_data, scores.data, target.view(-1).data, train)

        return loss, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)

