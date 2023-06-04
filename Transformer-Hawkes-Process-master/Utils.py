import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import get_non_pad_mask


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:] # this is to caculate time lag between events
    #--------------?????????????????????????-----------------------
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:] # why difference in lambda is additive?

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model,data_prev, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100
   
    if data_prev.shape != data.shape:
       data_prev = data
    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) *\
          torch.rand([*diff_time.size(), num_samples], device=data.device) 
    # \ this is the symbol for line continuation
    # *diff_time.size() the aestrisk in from of size just gets the value inside the set containg shape of temsor --works with temsors
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)
    temp_hid = model.linear(data)[:, 1:, :]
    temp_hid_prev = model.linear(data_prev)[:, 1:, :]
    temp_hid_prev_next = torch.sum(temp_hid*temp_hid_prev * type_mask[:, 1:, :], dim=2, keepdim=True)
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)
    # all_lambda = softplus(model.gamma_1*temp_hid1 + model.gamma_2*temp_hid2*temp_hid2 + model.alpha * temp_time, model.beta)
    all_lambda = softplus( model.gamma_1*temp_hid + model.alpha_1 * temp_time, model.beta) \
    +torch.pow(softplus( model.gamma_2*temp_hid_prev_next + model.alpha_2 * (temp_time**model.param_time), model.beta),2)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(model, data_prev, data, time, types):
    """ Log-likelihood of sequence. """
    if data_prev.shape != data.shape:
       data_prev = data
      
    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device) # num_types is no. of distinct events
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)
    # type_mask isd the make for event type for each entry in event stream a vector of size num_types is created with 0 
    # at event type of that event and rest 1

    all_hid = model.linear(data)
    all_hid_prev_next = model.linear(data*data_prev)
    # all_lambda = softplus( model.gamma_1*all_hid + model.gamma_2*all_hid*all_hid, model.beta)
    all_lambda = softplus( model.gamma_1*all_hid , model.beta) \
    + torch.pow(softplus(model.gamma_2*all_hid*all_hid_prev_next, model.beta), 2)
    # all_hid = model.linear(data)
    # all_lambda = softplus(all_hid, model.beta)
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, data_prev, data, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1 
    # truth's row will be actual event type for each stream of event
    prediction = prediction[:, :-1, :] # original prediction (1,5,75) ---> (1,3,75) 75:no of events, 5 = max steam length

    pred_type = torch.max(prediction, dim=-1)[1] # maximum out of last dimension 
    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth) # transpose changes (m,n,p) to (m,p,n)
        # essentially for each event type we collate the prediction value in each row-- row 1 prob of event type 0 for 
        # various events in the same event stream

    loss = torch.sum(loss)
    return loss, correct_num


def time_loss(prediction, event_time):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff)
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
