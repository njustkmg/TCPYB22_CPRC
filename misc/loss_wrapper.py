import torch
import torch.nn as nn
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
import torch.nn.functional as F



def fast_cdist(x1, x2):
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

    # Compute squared distance matrix using quadratic expansion
    # But be clever and do it with a single matmul call
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    # Zero out negative values
    res.clamp_min_(1e-30).sqrt_()
    return res


class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()
        self.class_criterion = nn.BCELoss()
        self.KL = nn.KLDivLoss(reduction='mean')
        self.log_soft = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        

    def forward(self, epoch, fc_feats, fc_feats_1, fc_feats_2, fc_feats_3,  
                      att_feats, att_feats_1, att_feats_2, att_feats_3, labels, class_label, masks, 
                      att_masks, att_masks_1, att_masks_2, att_masks_3, gts, gt_indices,
                sc_flag):
        out = {}

        

        if epoch > 25:
            self.model.predict[0].weight.requires_grad = False
            self.model.predict[3].weight.requires_grad = False

            model_out, encoder_output, decoder_output = self.model(fc_feats, att_feats, labels, att_masks)
            model_out_1, encoder_output_1, decoder_output_1 = self.model(fc_feats_1, att_feats_1, labels, att_masks_1)
            model_out_2, encoder_output_2, decoder_output_2 = self.model(fc_feats_2, att_feats_2, labels, att_masks_2)
            model_out_3, encoder_output_3, decoder_output_3 = self.model(fc_feats_3, att_feats_3, labels, att_masks_3)
        
            cat_encoder_output = torch.stack([encoder_output, encoder_output_1, 
                                            encoder_output_2, encoder_output_3]).permute(1, 0, 2)
            cat_decoder_output = torch.stack([decoder_output.squeeze(), decoder_output_1.squeeze(),
                                          decoder_output_2.squeeze(), decoder_output_3.squeeze()]).permute(1, 0, 2)

            # 无监督的预测一致性loss
            img_cap = self.KL(F.log_softmax(decoder_output.squeeze()), F.softmax(encoder_output, dim=-1))
            # img_cap = 0
            print('img_cap_loss: ', img_cap.item())

            # 无监督的结构一致性loss and tao
            # img_l2 = torch.cdist(cat_encoder_output, cat_encoder_output, p=2)
            img_l2 = fast_cdist(cat_encoder_output, cat_encoder_output)
            img_l2 = torch.triu(img_l2, diagonal=1)
            img_relation = img_l2[img_l2.nonzero(as_tuple=True)].reshape(img_l2.size(0), 6)
            img_relation = F.softmax(img_relation, dim=-1)

            # cap_l2 = torch.cdist(cat_decoder_output, cat_decoder_output, p=2)
            cap_l2 = fast_cdist(cat_decoder_output, cat_decoder_output)
            cap_l2 = torch.triu(cap_l2, diagonal=1)
            cap_relation = cap_l2[cap_l2.nonzero(as_tuple=True)].reshape(cap_l2.size(0), 6)
            cap_relation = F.softmax(cap_relation, dim=-1)

            img_cap_realtion = self.KL(cap_relation.log(), img_relation)*0.01
            # img_cap_realtion = 0
            print('img_cap_relation_loss: ', img_cap_realtion.item())

            img_loss = 0
            cap_loss = 0
        else:
            model_out, encoder_output, decoder_output = self.model(fc_feats, att_feats, labels, att_masks)
            # 有监督数据的预测loss
            img_loss = self.class_criterion(encoder_output, class_label)
            cap_loss = self.class_criterion(decoder_output.squeeze(), class_label)
            print('img_loss: ', img_loss.item())
            print('cap_loss: ', cap_loss.item())

            img_cap = 0
            img_cap_realtion = 0

        if not sc_flag:
            # 有监督的生成loss
            g_loss = self.crit(model_out, labels[:,1:], masks[:,1:])
            # g_loss_1 = self.crit(model_out_1, labels[:,1:], masks[:,1:])
            # g_loss_2 = self.crit(model_out_2, labels[:,1:], masks[:,1:])
            # g_loss_3 = self.crit(model_out_3, labels[:,1:], masks[:,1:])
            
            loss = g_loss
            print('g_loss_total: ', loss.item())
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_method':'sample'}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss + img_cap*0.1 + img_loss + cap_loss + img_cap_realtion*0.01
        # out['loss'] = loss + img_cap*0.1 + img_cap_realtion
        return out
