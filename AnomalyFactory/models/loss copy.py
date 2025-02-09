import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import pdb
from torchvision.utils import save_image

##depthAnything2 loss: not work, all nan
class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask=None):
        if valid_mask is not None:
            valid_mask = valid_mask.detach()
            diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        else:
            diff_log = torch.log(target) - torch.log(pred)
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))
        # pdb.set_trace()
        return loss


def get_dice_target(target, ncls=4):
    k_target = target.repeat(1,ncls,1,1)
    for k in range(ncls):
        k_target[:,k][k_target[:,k]!=k]=0
        k_target[:,k][k_target[:,k]==k]=255        
    return k_target/255

def dice_loss(pred, target, smooth=1.):
    """Dice loss
    """
    pred = pred.contiguous()
    target = target.contiguous()
    if target.shape[1]==1:
        target = get_dice_target(target)
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def dice_loss2(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

def compute_unsupervised_loss(predict, target, percent, pred_teacher):
    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():
        # drop pixels with high entropy
        prob = torch.softmax(pred_teacher, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
       
        thresh = np.percentile(entropy[target != 255].detach().cpu().numpy().flatten(), percent) 
        thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool() #ge: greater and equal than

        target[thresh_mask] = 255  #drop high entropy
        weight = batch_size * h * w / torch.sum(target != 255)
    
    loss = weight * F.cross_entropy(predict, target.long(), ignore_index=255)  # [10, 321, 321]
    
    return loss


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]
        # print(logit.shape, num_class)   #8, 2, 256, 256
        # # save_image(target[0,0], 'anomaly_mask.png')
        # pdb.set_trace()

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.contiguous().view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

# def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None, use_dhard=False):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        l = max_val - min_val
    else:
        l = val_range

    padd = window_size//2
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    c1 = (0.01 * l) ** 2
    c2 = (0.03 * l) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

    if size_average:
        if use_dhard:
            d_hard = np.quantile(ssim_map.detach().cpu().numpy(), q=0.5)
            ret = torch.mean(ssim_map[ssim_map < d_hard]) #org
             
        else:
            ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret, ssim_map


# class SSIM(torch.nn.Module):
#     def __init__(self, window_size=11, size_average=True, val_range=None, use_dhard=False):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.val_range = val_range

#         # Assume 1 channel for SSIM
#         self.channel = 1
#         self.window = create_window(window_size).cuda()

#         # use hard example mining
#         self.use_dhard = use_dhard

#     # def _ohem_forward(self, score, target, **kwargs):
#     #     ph, pw = score.size(2), score.size(3)
#     #     h, w = target.size(1), target.size(2)
#     #     if ph != h or pw != w:
#     #         score = F.interpolate(input=score, size=(
#     #             h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
#     #     pred = F.softmax(score, dim=1)
#     #     pixel_losses = self.criterion(score, target).contiguous().view(-1)
#     #     mask = target.contiguous().view(-1) != self.ignore_label

#     #     tmp_target = target.clone()
#     #     tmp_target[tmp_target == self.ignore_label] = 0
#     #     pred = pred.gather(1, tmp_target.unsqueeze(1))
#     #     pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
#     #     min_value = pred[min(self.min_kept, pred.numel() - 1)]
#     #     threshold = max(min_value, self.thresh)

#     #     pixel_losses = pixel_losses[mask][ind]
#     #     pixel_losses = pixel_losses[pred < threshold]
#     #     return pixel_losses.mean()

#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()

#         if channel == self.channel and self.window.dtype == img1.dtype:
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
#             self.window = window
#             self.channel = channel

#         s_score, ssim_map = ssim(img1, img2, window=window, window_size=self.window_size, 
#                     size_average=self.size_average, use_dhard=self.use_dhard)
#         return 1.0 - s_score


# class CharbonnierLoss(nn.Module):
#     """Charbonnier Loss (L1)"""

#     def __init__(self, eps=1e-3):
#         super(CharbonnierLoss, self).__init__()
#         self.eps = eps

#     def forward(self, x, y):
#         diff = x - y
#         # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
#         loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
#         return loss

# class EdgeLoss(nn.Module):
#     def __init__(self, gpu_ids):
#         super(EdgeLoss, self).__init__()
#         k = torch.Tensor([[.05, .25, .4, .25, .05]])
#         self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
#         self.n_channels, _, self.kw, self.kh = self.kernel.shape
#         if torch.cuda.is_available():
#             self.kernel = self.kernel.cuda()
#         # self.kernel = torch.nn.DataParallel(self.kernel, device_ids=gpu_ids)
#         self.loss = CharbonnierLoss()

#     def conv_gauss(self, img):
#         # n_channels, _, kw, kh = self.kernel.shape
#         img = F.pad(img, (self.kw//2, self.kh//2, self.kw//2, self.kh//2), mode='replicate')
#         return F.conv2d(img, self.kernel, groups=self.n_channels)

#     def laplacian_kernel(self, current):
#         filtered    = self.conv_gauss(current)    # filter
#         down        = filtered[:,:,::2,::2]               # downsample
#         new_filter  = torch.zeros_like(filtered)
#         new_filter[:,:,::2,::2] = down*4                  # upsample
#         filtered    = self.conv_gauss(new_filter) # filter
#         diff = current - filtered
#         return diff

#     def forward(self, x, y):
#         loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
#         return loss


# ######################################
# #     edge specific functions        #
# ######################################


# def cross_entropy_loss_RCF(prediction, labelf, beta=1.1):
#     label = labelf.long()
#     mask = labelf.clone()
#     num_positive = torch.sum(label==1).float()
#     num_negative = torch.sum(label==0).float()

#     mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
#     mask[label == 0] = beta * num_positive / (num_positive + num_negative)
#     mask[label == 2] = 0
#     cost = F.binary_cross_entropy(
#             prediction, labelf, weight=mask, reduction='sum')

#     return cost