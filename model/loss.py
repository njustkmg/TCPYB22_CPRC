# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
loss.py
~~~~~~~

Loss functions used to compute loss.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from utils.utils import init_scorer, get_self_critical_reward

class RewardCriterion(nn.Layer):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.reshape((-1, ))
        reward = reward.reshape((-1, ))
        mask = paddle.to_tensor((seq > 0), dtype='float32')
        mask = paddle.concat(x=[paddle.full(shape=[mask.shape[0], 1], fill_value=1, dtype='float32'),
                                mask[:, :-1]], axis=1).reshape((-1, ))
        output = - input * reward * mask
        output = paddle.sum(output) / paddle.sum(mask)

        return output

class LabelSmoothing(nn.Layer):
    "Implement label smoothing."
    def __init__(self, vocab_size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.vocab_size = vocab_size + 1
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.shape[1]]
        mask = mask[:, :input.shape[1]]

        input = input.reshape((-1, input.shape[-1]))
        target = target.reshape((-1, ))
        mask = mask.reshape((-1, ))

        self.size = input.shape[1]
        target_one_hot = F.one_hot(target, num_classes=self.vocab_size)
        x = paddle.full(target_one_hot.shape, dtype=target_one_hot.dtype, fill_value=self.confidence)
        y = paddle.full(target_one_hot.shape, dtype=target_one_hot.dtype, fill_value=self.smoothing / (self.size - 1))
        true_dist = paddle.where(target_one_hot!=0, x, y)

        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()


class XECriterion(nn.Layer):
    def __init__(self):
        super(XECriterion, self).__init__()

    def forward(self, pred, target, mask):
        """Inputs:
         - pred: logits has shape of [batch_size, seq_len, vocab_size].
         - target: [batch_size, seq_len].
         - mask: [batch_size, seq_len].
        """
        # truncate to the same size
        target = target[:, :pred.shape[1]]
        mask = mask[:, :pred.shape[1]]

        loss_ = F.cross_entropy(pred, target, reduction='none')
        loss_ *= mask

        return paddle.sum(loss_) / paddle.sum(mask)


def fast_cdist(x1, x2):
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

    # Compute squared distance matrix using quadratic expansion
    # But be clever and do it with a single matmul call
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = paddle.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = paddle.ones_like(x2_norm)
    x1_ = paddle.concat([-2. * x1, x1_norm, x1_pad], dim=-1)
    x2_ = paddle.concat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    # Zero out negative values
    res.clamp_min_(1e-30).sqrt_()
    return res

class LossWrapper(nn.Layer):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = LabelSmoothing(vocab_size=opt.vocab_size, smoothing=opt.label_smoothing)
        else:
            self.crit = XECriterion()
        self.rl_crit = RewardCriterion()
        self.class_criterion = nn.BCELoss()
        self.KL = nn.KLDivLoss(reduction='mean')
        self.log_soft = nn.LogSoftmax()
        self.softmax = nn.Softmax()

    def forward(self, fc_feats, att_feats, att_feats_1, att_feats_2, att_feats_3, 
                labels, class_label, masks, att_masks, att_masks_1, att_masks_2, att_masks_3, gts, gt_indices, sc_flag):
        out = {}
        logit_pred, prob_pred, encoder_output, decoder_output = self.model(fc_feats, att_feats, labels, att_masks)
        logit_pred_1, prob_pred_1, encoder_output_1, decoder_output_1 = self.model(fc_feats, att_feats_1, labels, att_masks_1)
        logit_pred_2, prob_pred_2, encoder_output_2, decoder_output_2 = self.model(fc_feats, att_feats_2, labels, att_masks_2)
        logit_pred_3, prob_pred_3, encoder_output_3, decoder_output_3 = self.model(fc_feats, att_feats_3, labels, att_masks_3)

        cat_encoder_output = paddle.stack([encoder_output, encoder_output_1, 
                                          encoder_output_2, encoder_output_3]).permute(1, 0, 2)
        cat_decoder_output = paddle.stack([decoder_output.squeeze(), decoder_output_1.squeeze(),
                                          decoder_output_2.squeeze(), decoder_output_3.squeeze()]).permute(1, 0, 2)
        # 有监督数据的预测loss
        img_loss = self.class_criterion(encoder_output, class_label)
        cap_loss = self.class_criterion(decoder_output.squeeze(), class_label)
        # print('img_loss: ', img_loss.item())
        # print('cap_loss: ', cap_loss.item())

        # 无监督的预测一致性loss
        img_cap = self.KL(F.log_softmax(decoder_output.squeeze()), F.softmax(encoder_output, dim=-1))
        # img_cap = 0
        # print('img_cap_loss: ', img_cap.item())

        # 无监督的结构一致性loss and tao
        img_l2 = fast_cdist(cat_encoder_output, cat_encoder_output)
        img_l2 = paddle.triu(img_l2, diagonal=1)
        img_relation = img_l2[img_l2.nonzero(as_tuple=True)].reshape(img_l2.size(0), 6)
        img_relation = F.softmax(img_relation, dim=-1)

        cap_l2 = fast_cdist(cat_decoder_output, cat_decoder_output)
        cap_l2 = paddle.triu(cap_l2, diagonal=1)
        cap_relation = cap_l2[cap_l2.nonzero(as_tuple=True)].reshape(cap_l2.size(0), 6)
        cap_relation = F.softmax(cap_relation, dim=-1)

        img_cap_realtion = self.KL(cap_relation.log(), img_relation)
        # print('img_cap_relation_loss: ', img_cap_realtion.item())

        if not sc_flag:
            self.model.train()
            # xe
            
            if self.opt.label_smoothing > 0:
                g_loss = self.crit(prob_pred, labels[:, 1:], masks[:, 1:])
                # g_loss_1 = self.crit(prob_pred_1, labels[:, 1:], masks[:, 1:])
                # g_loss_2 = self.crit(prob_pred_2, labels[:, 1:], masks[:, 1:])
                # g_loss_3 = self.crit(prob_pred_3, labels[:, 1:], masks[:, 1:])
            else:
                g_loss = self.crit(logit_pred, labels[:, 1:], masks[:, 1:])
                # g_loss_1 = self.crit(logit_pred_1, labels[:, 1:], masks[:, 1:])
                # g_loss_2 = self.crit(logit_pred_2, labels[:, 1:], masks[:, 1:])
                # g_loss_3 = self.crit(logit_pred_3, labels[:, 1:], masks[:, 1:])

            loss = g_loss
            # print('g_loss_total: ', loss.item())
        else:
            self.model.eval()
            with paddle.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_method': 'sample'},
                                                     mode='sample')
            gts = [gts[_].numpy().astype('uint32') for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = paddle.to_tensor(reward, dtype='float32')
            loss = self.rl_crit(sample_logprobs, gen_result, reward)
            out['reward'] = reward[:, 0].mean()
        out['loss'] = loss + img_cap + img_loss + cap_loss + img_cap_realtion*0.01
        return out
