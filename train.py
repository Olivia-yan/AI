import collections
import math
import statistics
from functools import reduce
from apex.parallel import DistributedDataParallel

import torch
import torch.nn as nn
from apex import amp
from torch import distributed
from torch.nn import functional as F

from utils import get_regularizer
from utils.loss import (NCA, BCESigmoid, BCEWithLogitsLossWithIgnoreIndex,
                        ExcludedKnowledgeDistillationLoss, FocalLoss,
                        FocalLossNew, IcarlLoss, KnowledgeDistillationLoss,
                        UnbiasedCrossEntropy,
                        UnbiasedKnowledgeDistillationLoss, UnbiasedNCA,
                        soft_crossentropy)

from model.attention.CoTAttention import CoTAttention
from model.attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention,SequentialPolarizedSelfAttention


def features_distillation_channel(
        list_attentions_a,
        list_attentions_b,
        index_new_class=16,
        nb_current_classes=16,
        nb_new_classes=1,
        opts=None
):
    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    list_attentions_a = list_attentions_a[:-1]
    list_attentions_b = list_attentions_b[:-1]
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        n, c, h, w = a.shape
        layer_loss = torch.tensor(0.).to(a.device)

        if a.shape[1] != b.shape[1]:
            _b = torch.zeros_like(a).to(a.dtype).to(a.device)
            _b[:, 0] = b[:, 0] + b[:, index_new_class:].sum(dim=1)
            _b[:, 1:] = b[:, 1:index_new_class]
            b = _b

        a = a ** 2
        b = b ** 2

        a_p = F.avg_pool2d(a.permute(0, 2, 1, 3), (3, 1), stride=1, padding=(1, 0))
        b_p = F.avg_pool2d(b.permute(0, 2, 1, 3), (3, 1), stride=1, padding=(1, 0))

        # psa = ParallelPolarizedSelfAttention(channel=a_p.shape[1],device=a.device)
        # a_p = psa(a_p)
        # psb = ParallelPolarizedSelfAttention(channel=b_p.shape[1],device=b.device)
        # b_p = psb(b_p)

        layer_loss = torch.frobenius_norm((a_p - b_p).view(a.shape[0], -1), dim=-1).mean()

        if i == len(list_attentions_a) - 1:
            if opts.dataset == "ade":
                pckd_factor = 5e-7
            elif opts.dataset == "voc" or opts.dataset == 'cityscape':
                pckd_factor = 0.0005
            elif opts.dataset == 'cityscapes_domain':
                pckd_factor = 0.
        else:
            if opts.dataset == "ade":
                pckd_factor = 5e-6
            elif opts.dataset == "voc" or opts.dataset == "cityscape":
                pckd_factor = 0.01
            elif opts.dataset == 'cityscapes_domain':
                pckd_factor = 0.0001

        loss = loss + layer_loss.mean() * 1.0 * math.sqrt(nb_current_classes / nb_new_classes) * pckd_factor


    return loss / len(list_attentions_a)


# 主要目的是计算两组注意力图（list_attentions_a 和 list_attentions_b）之间的某种“蒸馏”损失（通常用于知识蒸馏的场景）
def features_distillation_spatial(
        list_attentions_a,
        list_attentions_b,
        index_new_class=16,
        nb_current_classes=16,
        nb_new_classes=1,
        opts=None
):
    loss = torch.tensor(0.).to(list_attentions_a[0].device)  # 使用与 list_attentions_a 中第一个注意力图相同的设备初始化一个零张量 loss
    # list_attentions_a = list_attentions_a[:-1]
    # list_attentions_b = list_attentions_b[:-1]
    # print('list_attentions_a',list_attentions_a)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        n, c, h, w = a.shape
        # print('a.shape',a.shape)#[6, 256, 128, 128]，[6, 512, 64, 64]，[6, 1024, 32, 32]，[6, 2048, 32, 32]，[6, 256, 32, 32]，[6, 12, 32, 32]
        layer_loss = torch.tensor(0.).to(a.device)

        if a.shape[1] != b.shape[1]:
            # print('a.shape[1]', a.shape[1])#12
            # print('b.shape[1]', b.shape[1])#13
            _b = torch.zeros_like(a).to(a.dtype).to(a.device)
            _b[:, 0] = b[:, 0] + b[:, index_new_class:].sum(dim=1)
            _b[:, 1:] = b[:, 1:index_new_class]
            b = _b
            # print('b.shape', b.shape)#[6, 12, 32, 32]

        a = a ** 2#[6, 256, 128, 128],[6, 512, 64, 64],[6, 1024, 32, 32],[6, 2048, 32, 32],[6, 256, 32, 32],[6, 11, 32, 32]





        b = b ** 2#[6, 256, 128, 128],[6, 512, 64, 64],[6, 1024, 32, 32],[6, 2048, 32, 32],[6, 256, 32, 32],[6, 11, 32, 32]

        # if a.shape[1]%4 == 0 or b.shape[1] %4 == 0:
        #     cot = CoTAttention(list_attentions_a[0].device, dim=a.shape[1], kernel_size=3)
        #     a = cot(a)
        #     # print('aaaaaaaaa',a.shape)
        #     cot2 = CoTAttention(list_attentions_a[0].device, dim=b.shape[1], kernel_size=3)
        #     b = cot2(b)
            # print('bbbbbbbbb', b.shape)
        # psa = ParallelPolarizedSelfAttention(channel=a.shape[1], device=a.device)
        # a = psa(a)
        # psb = ParallelPolarizedSelfAttention(channel=b.shape[1], device=b.device)
        # b = psb(b)
        a_affinity_4 = F.avg_pool2d(a, (4, 4), stride=1, padding=2)
        b_affinity_4 = F.avg_pool2d(b, (4, 4), stride=1, padding=2)
        a_affinity_8 = F.avg_pool2d(a, (8, 8), stride=1, padding=4)
        b_affinity_8 = F.avg_pool2d(b, (8, 8), stride=1, padding=4)
        a_affinity_12 = F.avg_pool2d(a, (12, 12), stride=1, padding=6)
        b_affinity_12 = F.avg_pool2d(b, (12, 12), stride=1, padding=6)
        a_affinity_16 = F.avg_pool2d(a, (16, 16), stride=1, padding=8)
        b_affinity_16 = F.avg_pool2d(b, (16, 16), stride=1, padding=8)
        a_affinity_20 = F.avg_pool2d(a, (20, 20), stride=1, padding=10)
        b_affinity_20 = F.avg_pool2d(b, (20, 20), stride=1, padding=10)
        a_affinity_24 = F.avg_pool2d(a, (24, 24), stride=1, padding=12)
        b_affinity_24 = F.avg_pool2d(b, (24, 24), stride=1, padding=12)

        layer_loss = torch.frobenius_norm((a_affinity_4 - b_affinity_4).view(a.shape[0], -1), dim=-1).mean() + \
                     torch.frobenius_norm((a_affinity_8 - b_affinity_8).view(a.shape[0], -1), dim=-1).mean() + \
                     torch.frobenius_norm((a_affinity_16 - b_affinity_16).view(a.shape[0], -1), dim=-1).mean() + \
                     torch.frobenius_norm((a_affinity_20 - b_affinity_20).view(a.shape[0], -1), dim=-1).mean() + \
                     torch.frobenius_norm((a_affinity_24 - b_affinity_24).view(a.shape[0], -1), dim=-1).mean() + \
                     torch.frobenius_norm((a_affinity_12 - b_affinity_12).view(a.shape[0], -1), dim=-1).mean()
        layer_loss = layer_loss / 6.


        if i == len(list_attentions_a) - 1:
            if opts.dataset == "ade":
                pckd_factor = 5e-7
            elif opts.dataset == "voc" or opts.dataset == 'cityscape':
                pckd_factor = 0.0005
            elif opts.dataset == "cityscapes_domain":
                pckd_factor = 1e-4
        else:
            if opts.dataset == "ade":
                pckd_factor = 5e-6
            elif opts.dataset == "voc" or opts.dataset == 'cityscape':
                pckd_factor = 0.01
            elif opts.dataset == "cityscapes_domain":
                pckd_factor = 5e-4

        loss = loss + layer_loss.mean() * 1.0 * math.sqrt(nb_current_classes / nb_new_classes) * pckd_factor

    return loss / len(list_attentions_a)


class Trainer:

    def __init__(self, model, model_old, device, opts, trainer_state=None, classes=None, step=0):

        self.model_old = model_old
        self.model = model
        self.device = device
        self.step = step
        self.opts = opts

        if opts.dataset == "cityscapes_domain":
            if self.step > 0:
                self.old_classes = opts.num_classes
                self.nb_classes = opts.num_classes
            else:
                self.old_classes = 0
                self.nb_classes = None
            self.nb_current_classes = opts.num_classes
            self.nb_new_classes = opts.num_classes
        elif classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = tot_classes - new_classes
            self.nb_classes = opts.num_classes
            self.nb_current_classes = tot_classes
            self.nb_new_classes = new_classes
        else:
            self.old_classes = 0
            self.nb_classes = None

        # Select the Loss Type
        reduction = 'none'

        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(
                old_cl=self.old_classes, ignore_index=255, reduction=reduction
            )
        elif opts.nca and self.old_classes != 0:
            self.criterion = UnbiasedNCA(
                old_cl=self.old_classes,
                ignore_index=255,
                reduction=reduction,
                scale=model.module.scalar,
                margin=opts.nca_margin
            )
        elif opts.nca:
            self.criterion = NCA(
                scale=model.module.scalar,
                margin=opts.nca_margin,
                ignore_index=255,
                reduction=reduction
            )
        elif opts.focal_loss:
            self.criterion = FocalLoss(ignore_index=255, reduction=reduction, alpha=opts.alpha,
                                       gamma=opts.focal_loss_gamma)
        elif opts.focal_loss_new:
            self.criterion = FocalLossNew(ignore_index=255, reduction=reduction, index=self.old_classes,
                                          alpha=opts.alpha, gamma=opts.focal_loss_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)  # 在标签中有一个特殊的类别（255），在计算损失时应该被忽略

        # ILTSS
        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and model_old is not None
        self.lde_loss = nn.MSELoss()

        self.lkd = opts.loss_kd
        self.lkd_mask = opts.kd_mask
        self.kd_mask_adaptative_factor = opts.kd_mask_adaptative_factor
        self.lkd_flag = self.lkd > 0. and model_old is not None
        self.kd_need_labels = False
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(reduction="none", alpha=opts.alpha)
        elif opts.kd_bce_sig:
            self.lkd_loss = BCESigmoid(reduction="none", alpha=opts.alpha, shape=opts.kd_bce_sig_shape)
        elif opts.exkd_gt and self.old_classes > 0 and self.step > 0:
            self.lkd_loss = ExcludedKnowledgeDistillationLoss(
                reduction='none', index_new=self.old_classes, new_reduction="gt",
                initial_nb_classes=opts.inital_nb_classes,
                temperature_semiold=opts.temperature_semiold
            )
            self.kd_need_labels = True
        elif opts.exkd_sum and self.old_classes > 0 and self.step > 0:
            self.lkd_loss = ExcludedKnowledgeDistillationLoss(
                reduction='none', index_new=self.old_classes, new_reduction="sum",
                initial_nb_classes=opts.inital_nb_classes,
                temperature_semiold=opts.temperature_semiold
            )
            self.kd_need_labels = True
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

        # Regularization
        regularizer_state = trainer_state['regularizer'] if trainer_state is not None else None
        self.regularizer = get_regularizer(model, model_old, device, opts, regularizer_state)
        self.regularizer_flag = self.regularizer is not None
        self.reg_importance = opts.reg_importance

        self.ret_intermediate = self.lde or (opts.pod is not None)

        self.pseudo_labeling = opts.pseudo
        self.threshold = opts.threshold
        self.step_threshold = opts.step_threshold
        self.ce_on_pseudo = opts.ce_on_pseudo
        self.pseudo_nb_bins = opts.pseudo_nb_bins
        self.pseudo_soft = opts.pseudo_soft
        self.pseudo_soft_factor = opts.pseudo_soft_factor
        self.pseudo_ablation = opts.pseudo_ablation
        self.classif_adaptive_factor = opts.classif_adaptive_factor
        self.classif_adaptive_min_factor = opts.classif_adaptive_min_factor

        self.kd_new = opts.kd_new
        self.pod = opts.pod
        self.pod_options = opts.pod_options if opts.pod_options is not None else {}
        self.pod_factor = opts.pod_factor
        self.pod_prepro = opts.pod_prepro
        self.use_pod_schedule = not opts.no_pod_schedule
        self.pod_deeplab_mask = opts.pod_deeplab_mask
        self.pod_deeplab_mask_factor = opts.pod_deeplab_mask_factor
        self.pod_apply = opts.pod_apply
        self.pod_interpolate_last = opts.pod_interpolate_last
        self.deeplab_mask_downscale = opts.deeplab_mask_downscale
        self.spp_scales = opts.spp_scales
        self.pod_logits = opts.pod_logits
        self.pod_large_logits = opts.pod_large_logits

        self.align_weight = opts.align_weight
        self.align_weight_frequency = opts.align_weight_frequency

        self.dataset = opts.dataset

        self.entropy_min = opts.entropy_min

        self.kd_scheduling = opts.kd_scheduling

        self.sample_weights_new = opts.sample_weights_new

        self.temperature_apply = opts.temperature_apply
        self.temperature = opts.temperature

        # CIL
        self.ce_on_new = opts.ce_on_new

    def before(self, train_loader, logger):
        if self.pseudo_labeling is None:
            return
        if self.pseudo_labeling.split("_")[0] == "median" and self.step > 0:
            logger.info("Find median score")
            self.thresholds, _ = self.find_median(train_loader, self.device, logger)
        elif self.pseudo_labeling.split("_")[0] == "entropy" and self.step > 0:
            logger.info("Find median score")
            self.thresholds, self.max_entropy = self.find_median(
                train_loader, self.device, logger, mode="entropy"
            )

    def train(self, cur_epoch, optim, train_loader, scheduler=None, print_int=10, logger=None):
        """Train and return epoch loss"""
        logger.info(
            f"Pseudo labeling is: {self.pseudo_labeling}")  # 这行代码用于记录伪标签的信息。它将在训练日志中输出 "Pseudo labeling is: " 后跟着 self.pseudo_labeling 的值。这是为了让用户知道当前是否在使用伪标签。
        logger.info("Epoch %d, lr = %f" % (
        cur_epoch, optim.param_groups[0]['lr']))  # 这一行代码记录当前的训练周期 (cur_epoch) 以及使用的学习率。这对于在训练过程中监视学习率的变化非常有用。

        device = self.device
        model = self.model
        criterion = self.criterion  # criterion 是损失函数

        model.module.in_eval = False
        if self.model_old is not None:
            #运行这里
            self.model_old.in_eval = False

        epoch_loss = 0.0  # epoch_loss: 用于追踪整个epoch的总体损失。
        reg_loss = 0.0  # reg_loss: 用于追踪正则化项的总和。
        interval_loss = 0.0  # interval_loss: 用于追踪每个小批次（batch）的损失，以便在每个print_int步骤后输出平均损失。
        # 不同类型的损失项，它们被初始化为零，用于累积每个batch的相应损失。
        lkd = torch.tensor(0.)  # 知识蒸馏损失
        lde = torch.tensor(0.)  # 表示差异性
        l_icarl = torch.tensor(0.)  # iCaRL损失
        l_reg = torch.tensor(0.)  # 正则化项
        pod_loss = torch.tensor(0.)  # 注意力distillation
        loss_entmin = torch.tensor(0.)  # 熵最小化

        sample_weights = None

        train_loader.sampler.set_epoch(cur_epoch)

        #### xjw fixed
        # model.train()
        # 整个循环的目的是遍历训练数据集中的所有批次,并对每个批次执行训练步骤。在循环内的代码将图像和标签加载到设备（通常是GPU）上，然后通过神经网络进行前向传播（计算模型的输出），接着计算损失，并最终通过反向传播更新模型的参数。这是深度学习训练过程中的基本步骤。
        # enumerate(train_loader): 这个部分是使用enumerate函数来遍历train_loader中的数据批次。enumerate同时返回循环的索引（cur_step）和train_loader中的数据批次。
        # (images, labels): 这是解包操作，将从train_loader中加载的数据批次分配给images和labels两个变量。通常，images包含输入图像的张量，而labels包含对应的标签。
        for cur_step, (images, labels) in enumerate(train_loader):
            images = images.to(device,
                               dtype=torch.float32)  # 将输入图像张量移动到指定的设备（通常是 GPU），并将数据类型设置为 torch.float32。这是为了确保输入数据与模型的数据类型匹配。
            labels = labels.to(device, dtype=torch.long)  # 将标签张量移动到指定的设备，并将数据类型设置为 torch.long。通常，标签是整数类型。

            original_labels = labels.clone()  # 创建标签的深拷贝。这是因为在后续的处理中，可能会修改 labels，但我们想要保留原始标签以供后续使用。

            # 它们检查一些训练中使用的标志（self.lde_flag，self.lkd_flag等）以及旧模型是否存在(self.model_old is not None)。
            # 如果这些条件为真，它们执行一些计算，包括获取旧模型的输出( outputs_old)和特征(features_old)
            if (
                    self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.pod is not None or
                    self.pseudo_labeling is not None
            ) and self.model_old is not None:
                # 运行这里
                with torch.no_grad():  # 使用torch.no_grad()上下文管理器，表示在这个块中的计算不会影响梯度计算，因为在这里我们只是获取旧模型的输出以进行一些处理，而不是通过这些计算来训练模型。
                    outputs_old, features_old = self.model_old(
                        images, ret_intermediate=self.ret_intermediate
                    )

            classif_adaptive_factor = 1.0  # 初始化分类自适应因子，该因子在后续计算中用于动态调整损失
            if self.step > 0:  # 这个条件检查是否是训练的第一个步骤之后的步骤。如果是第一个步骤，可能不需要执行以下的伪标签逻辑
                # 运行这里
                mask_background = labels < self.old_classes  # 创建一个布尔掩码，标识标签中属于旧类别（background）的部分。

                # 下面一系列逻辑的目的是使用旧模型的输出来生成一些伪标签，以帮助训练模型。这在迁移学习或者类别增量学习中经常被使用，以利用旧模型对新类别的适应性。
                if self.pseudo_labeling == "naive":
                    labels[mask_background] = outputs_old.argmax(dim=1)[
                        mask_background]  # "naive"方法，使用旧模型的输出的最大预测值作为伪标签
                elif self.pseudo_labeling is not None and self.pseudo_labeling.startswith(
                        "threshold_"
                ):
                    threshold = float(self.pseudo_labeling.split("_")[
                                          1])  # 如果采用 "threshold_" 方法，根据设定的阈值，将输出概率高于阈值的样本的伪标签设置为对应类别，其他设置为 255。
                    probs = torch.softmax(outputs_old, dim=1)
                    pseudo_labels = probs.argmax(dim=1)
                    pseudo_labels[probs.max(dim=1)[0] < threshold] = 255
                    labels[mask_background] = pseudo_labels[mask_background]
                elif self.pseudo_labeling == "confidence":  # 如果采用 "confidence" 方法，使用旧模型的输出概率，将伪标签设置为预测概率最高的类别
                    probs_old = torch.softmax(outputs_old, dim=1)
                    labels[mask_background] = probs_old.argmax(dim=1)[mask_background]
                    sample_weights = torch.ones_like(labels).to(device, dtype=torch.float32)
                    sample_weights[mask_background] = probs_old.max(dim=1)[0][mask_background]
                elif self.pseudo_labeling == "median":  # 如果采用 "median" 方法，将输出概率最大值低于设定阈值的样本的伪标签设置为 255
                    probs = torch.softmax(outputs_old, dim=1)
                    max_probs, pseudo_labels = probs.max(dim=1)
                    pseudo_labels[max_probs < self.thresholds[pseudo_labels]] = 255
                    labels[mask_background] = pseudo_labels[mask_background]
                elif self.pseudo_labeling == "entropy":  # 如果采用 "entropy" 方法，计算样本的熵，根据设定的阈值，将伪标签设置为熵低于阈值的样本的预测类别，其他设置为 255。
                    probs = torch.softmax(outputs_old, dim=1)
                    max_probs, pseudo_labels = probs.max(dim=1)

                    mask_valid_pseudo = (entropy(probs) /
                                         self.max_entropy) < self.thresholds[pseudo_labels]

                    # 这一部分是在前面生成的伪标签基础上进一步处理和修改真实标签
                    if self.pseudo_soft is None:
                        # All old labels that are NOT confident enough to be used as pseudo labels:
                        labels[~mask_valid_pseudo & mask_background] = 255  # 对于那些不够自信（即熵大于阈值）的旧标签，将其设为 255（即无效标签）

                        if self.pseudo_ablation is None:
                            # All old labels that are confident enough to be used as pseudo labels:对于自信的旧标签，将其替换为相应的伪标签
                            labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                                        mask_background]
                        elif self.pseudo_ablation == "corrected_errors":
                            pass  # If used jointly with data_masking=current+old, the labels already
                            # contrain the GT, thus all potentials errors were corrected.如果与 data_masking=current+old 一起使用，标签已经包含 GT（Ground Truth），因此所有潜在的错误都已纠正。
                        elif self.pseudo_ablation == "removed_errors":  # 保留伪标签中没有错误的标签，删除有错误的标签
                            pseudo_error_mask = labels != pseudo_labels
                            kept_pseudo_labels = mask_valid_pseudo & mask_background & ~pseudo_error_mask
                            removed_pseudo_labels = mask_valid_pseudo & mask_background & pseudo_error_mask

                            labels[kept_pseudo_labels] = pseudo_labels[kept_pseudo_labels]
                            labels[removed_pseudo_labels] = 255
                        else:
                            raise ValueError(f"Unknown type of pseudo_ablation={self.pseudo_ablation}")
                    elif self.pseudo_soft == "soft_uncertain":  # 对于自信的旧标签，将其替换为相应的伪标签
                        labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                                    mask_background]

                    # 这一部分的代码是关于分类自适应因子的计算和处理
                    if self.classif_adaptive_factor:  # 控制是否启用分类自适应因子
                        # Number of old/bg pixels that are certain对于每个样本，首先计算在伪标签中被认为是确定的旧/背景像素的数量 num 和总的旧/背景像素的数量 den
                        num = (mask_valid_pseudo & mask_background).float().sum(dim=(1, 2))
                        # Number of old/bg pixels
                        den = mask_background.float().sum(dim=(1, 2))
                        # If all old/bg pixels are certain the factor is 1 (loss not changed)如果所有旧/背景像素都被认为是确定的，那么 classif_adaptive_factor 保持为1（损失不变）
                        # Else the factor is < 1, i.e. the loss is reduced to avoid否则，classif_adaptive_factor 被设置为小于1的值，以减小对新像素的重要性，以避免引入太多的不确定性。
                        # giving too much importance to new pixels
                        classif_adaptive_factor = num / (den + 1e-6)
                        classif_adaptive_factor = classif_adaptive_factor[:, None, None]

                        if self.classif_adaptive_min_factor:  # 如果设置了 self.classif_adaptive_min_factor，则将 classif_adaptive_factor 截断到指定的最小值。
                            classif_adaptive_factor = classif_adaptive_factor.clamp(
                                min=self.classif_adaptive_min_factor)

            optim.zero_grad()  # 这部分代码是进行了优化器梯度的清零,用于清零之前参数的梯度，以避免梯度在多次反向传播中累积
            outputs, features = model(images,
                                      ret_intermediate=self.ret_intermediate)  # 然后通过模型 model 计算了输出 outputs 和可能的中间层特征 features

            # xxx BCE / Cross Entropy Loss
            # 这部分代码包含了计算损失函数 loss 的逻辑，这通常是训练深度学习模型时的一个关键步骤。损失函数衡量模型输出与实际标签之间的差异，而优化器使用这个差异来更新模型的权重，从而减小这个差异
            if self.pseudo_soft is not None:  # 如果 self.pseudo_soft 不为空，说明使用了一种软标签（soft label）的损失计算方法 soft_crossentropy
                loss = soft_crossentropy(
                    outputs,
                    labels,
                    outputs_old,
                    mask_valid_pseudo,
                    mask_background,
                    self.pseudo_soft,
                    pseudo_soft_factor=self.pseudo_soft_factor
                )
            ##根据一些条件选择使用交叉熵损失（Cross Entropy Loss）或是其他自定义的损失函数，比如 self.licarl
            elif not self.icarl_only_dist:
                # 运行这里
                if self.ce_on_pseudo and self.step > 0:  # 如果 self.ce_on_pseudo 为真，并且 self.step 大于 0，那么应用了一种混合的损失，其中对于未选为伪标签（pseudo label）的背景类使用交叉熵损失，对于被选为伪标签的旧类使用交叉熵损失。
                    assert self.pseudo_labeling is not None
                    assert self.pseudo_labeling == "entropy"
                    # Apply UNCE on:
                    #   - all new classes (foreground)
                    #   - old classes (background) that were not selected for pseudo
                    loss_not_pseudo = criterion(
                        outputs,
                        original_labels,
                        mask=mask_background & mask_valid_pseudo  # what to ignore
                    )

                    # Apply CE on:
                    # - old classes that were selected for pseudo
                    _labels = original_labels.clone()
                    _labels[~(mask_background & mask_valid_pseudo)] = 255
                    _labels[mask_background & mask_valid_pseudo] = pseudo_labels[mask_background &
                                                                                 mask_valid_pseudo]
                    loss_pseudo = F.cross_entropy(
                        outputs, _labels, ignore_index=255, reduction="none"
                    )
                    # Each loss complete the others as they are pixel-exclusive
                    loss = loss_pseudo + loss_not_pseudo
                elif self.ce_on_new:  # 如果 self.ce_on_new 为真，那么对新类应用交叉熵损失
                    _labels = labels.clone()
                    _labels[_labels == 0] = 255
                    loss = criterion(outputs, _labels)  # B x H x W
                else:
                    # 运行这里
                    loss = criterion(outputs, labels)  # B x H x W
            else:
                loss = self.licarl(outputs, labels,
                                   torch.sigmoid(outputs_old))  # 使用 self.licarl 函数计算损失，该函数可能是一种特定的损失函数，与交叉熵不同。

            if self.sample_weights_new is not None:  # 这段代码检查是否存在 self.sample_weights_new。如果存在，说明你想要为每个样本指定不同的权重。这可以用于调整损失函数中每个样本的重要性
                sample_weights = torch.ones_like(original_labels).to(device,
                                                                     dtype=torch.float32)  # 创建了一个和 original_labels 相同形状的张量 sample_weights，初始值都是 1。
                sample_weights[
                    original_labels >= 0] = self.sample_weights_new  # 对于那些标签大于等于 0 的位置，将对应的权重设置为 self.sample_weights_new。这样，你可以为不同的类别或样本赋予不同的权重。

            if sample_weights is not None:  # 如果 sample_weights 不为 None，就将损失乘以权重。这样，计算的损失将按照权重进行加权。
                loss = loss * sample_weights
            loss = classif_adaptive_factor * loss  # 这个因子被设计成根据数据集中新旧类别的比例进行调整。这是为了确保新类别和旧类别对损失的贡献相对平衡。
            loss = loss.mean()  # scalar最后，将所有样本的损失取平均，得到一个标量值。这是为了确保最终的损失是一个可比较的标量，方便优化算法使用。

            if self.icarl_combined:  # 如果 self.icarl_combined 为真，那么这段代码将计算 l_icarl，它是 iCaRL 算法的一部分。具
                # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                n_cl_old = outputs_old.shape[1]
                # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                # 体地说，它使用 n_cl_old 表示旧类别的数量，然后使用 self.licarl 函数计算 iCaRL 损失。
                l_icarl = self.icarl * n_cl_old * self.licarl(
                    outputs.narrow(1, 0, n_cl_old), torch.sigmoid(outputs_old)
                    # outputs.narrow(1, 0, n_cl_old) 是为了选择输出中与旧类别相关的部分，然后使用 torch.sigmoid(outputs_old) 对旧输出进行 sigmoid 处理
                )

            # xxx ILTSS (distillation on features or logits)
            # self.lde_flag是一个标志，指示是否应该使用Knowledge Distillation
            # self.lde是一个超参数，表示Knowledge Distillation损失的权重或缩放系数
            # self.lde_loss是一个函数，用于计算Knowledge Distillation损失。这个函数接受两个参数，features['body']是当前模型的某个层（可能是特征提取的一部分），而features_old['body']是旧模型相应层的特征。
            if self.lde_flag:  # 如果 self.lde_flag 为真，那么这段代码将计算 lde，这是用于 Knowledge Distillation 的损失。
                lde = self.lde * self.lde_loss(features['body'], features_old['body'])

            # 这一部分的代码主要用于实现知识蒸馏，通过利用旧模型的输出（outputs_old）和当前模型的输出（outputs）之间的差异来帮助模型学习。
            if self.lkd_flag:  # 是一个标志，指示是否应该使用知识蒸馏（Knowledge Distillation）
                # 运行这里
                # resize new output to remove new logits and keep only the old ones
                # 是一个可选的标志，用于指定在计算知识蒸馏损失时要考虑的样本的掩码。在这里，它通过比较 labels 和 self.old_classes 来创建一个二进制掩码 kd_mask
                if self.lkd_mask is not None and self.lkd_mask == "oldbackground":
                    kd_mask = labels < self.old_classes
                elif self.lkd_mask is not None and self.lkd_mask == "new":
                    kd_mask = labels >= self.old_classes
                else:
                    # 运行这里
                    kd_mask = None

                if self.temperature_apply is not None:  # 是一个可选的标志，用于控制是否应该应用温度缩放
                    temp_mask = torch.ones_like(labels).to(outputs.device).to(outputs.dtype)

                    if self.temperature_apply == "all":
                        temp_mask = temp_mask / self.temperature
                    elif self.temperature_apply == "old":
                        mask_bg = labels < self.old_classes
                        temp_mask[mask_bg] = temp_mask[mask_bg] / self.temperature
                    elif self.temperature_apply == "new":
                        mask_fg = labels >= self.old_classes
                        temp_mask[mask_fg] = temp_mask[mask_fg] / self.temperature
                    temp_mask = temp_mask[:, None]
                else:
                    # 运行这里
                    temp_mask = 1.0

                if self.kd_need_labels:  # 是一个可选的标志，指示知识蒸馏损失是否需要使用真实标签（labels）
                    lkd = self.lkd * self.lkd_loss(  # self.lkd是一个超参数，表示知识蒸馏损失的权重或缩放系数
                        outputs * temp_mask, outputs_old * temp_mask, labels, mask=kd_mask
                    )
                else:
                    # 运行这里
                    lkd = self.lkd * self.lkd_loss(
                        outputs * temp_mask, outputs_old * temp_mask, mask=kd_mask
                    )

                if self.kd_new:  # WTF?
                    mask_bg = labels == 0
                    lkd = lkd[mask_bg]

                if kd_mask is not None and self.kd_mask_adaptative_factor:
                    lkd = lkd.mean(dim=(1, 2)) * kd_mask.float().mean(dim=(1, 2))
                lkd = torch.mean(lkd)

            if self.pod is not None and self.step > 0:
                # 运行这里
                attentions_old = features_old["attentions"]
                attentions_new = features["attentions"]

                if self.pod_logits:
                    # 运行这里
                    attentions_old.append(features_old["sem_logits_small"])
                    attentions_new.append(features["sem_logits_small"])
                elif self.pod_large_logits:
                    attentions_old.append(outputs_old)
                    attentions_new.append(outputs)

                # print('self.old_classes', self.old_classes)#12
                # print('self.nb_current_classes', self.nb_current_classes)#15
                # print('self.nb_new_classes', self.nb_new_classes)#1
                # print('self.opts', self.opts)#Namespace(align_weight=None, align_weight_frequency='never', alpha=1.0, backbone='resnet101', base_weights=False, batch_size=6, bce=False, ce_on_new=False, ce_on_pseudo=False, checkpoint='checkpoints/step/', ckpt=None, ckpt_interval=1, classif_adaptive_factor=True, classif_adaptive_min_factor=0.0, code_directory='.', cosine=False, crop_size=512, crop_val=True, cross_val=False, data_masking='current', data_root='data/PascalVOC12', dataset='voc', date='2024-05-16', debug=False, deeplab_mask_downscale=False, disable_background=False, dont_predict_bg=False, entropy_min=0.0, entropy_min_mean_pixels=False, epochs=1, exkd_gt=False, exkd_sum=False, fix_bn=False, focal_loss=False, focal_loss_gamma=2, focal_loss_new=False, freeze=False, fusion_mode='mean', icarl=False, icarl_bkg=False, icarl_disjoint=False, icarl_importance=1.0, ignore_test_bg=False, init_balanced=True, init_multimodal=None, inital_nb_classes=11, kd_bce_sig=False, kd_bce_sig_shape='trim', kd_mask=None, kd_mask_adaptative_factor=False, kd_new=False, kd_scheduling=False, local_rank=1, logdir='./logs', loss_de=0.0, loss_kd=100, lr=0.001, lr_decay_factor=0.1, lr_decay_step=5000, lr_old=None, lr_policy='poly', lr_power=0.9, method='RCIL', momentum=0.9, multimodal_fusion='sum', name='RCIL_overlap', nb_background_modes=1, nca=False, nca_margin=0.0, no_cross_val=True, no_mask=False, no_overlap=False, no_pod_schedule=False, no_pretrained=False, norm_act='iabn_sync', num_classes=21, num_workers=4, opt_level='O1', output_stride=16, overlap=True, pod='local', pod_apply='all', pod_deeplab_mask=False, pod_deeplab_mask_factor=None, pod_factor=0.01, pod_interpolate_last=False, pod_large_logits=False, pod_logits=True, pod_options={'switch': {'after': {'extra_channels': 'sum', 'factor': 0.0005, 'type': 'local'}}}, pod_prepro='pow', pooling=32, print_interval=10, pseudo=None, pseudo_ablation=None, pseudo_nb_bins=None, pseudo_soft=None, pseudo_soft_factor=1.0, random_seed=42, reg_alpha=0.9, reg_importance=1.0, reg_iterations=10, reg_no_normalize=False, regularizer=None, sample_num=0, sample_weights_new=None, spp_scales=[1, 2, 4], step=4, step_ckpt=None, step_threshold=None, strict_weights=True, task='10-1', temperature=1.0, temperature_apply=None, temperature_semiold=1.0, test=False, test_on_val=False, threshold=0.9, unce=True, unkd=True, val_interval=15, val_on_trainset=False, visualize=True, weight_decay=0.0001)
                pod_loss = features_distillation_spatial(
                    attentions_old,
                    attentions_new,
                    index_new_class=self.old_classes,
                    nb_current_classes=self.nb_current_classes,
                    nb_new_classes=self.nb_new_classes,
                    opts=self.opts
                ) + features_distillation_channel(
                    attentions_old,
                    attentions_new,
                    index_new_class=self.old_classes,
                    nb_current_classes=self.nb_current_classes,
                    nb_new_classes=self.nb_new_classes,
                    opts=self.opts
                )
                # psa = SequentialPolarizedSelfAttention(channel=pod_loss.shape[1], device=pod_loss.device)
                # pod_loss = psa(pod_loss)
                # print('pod_loss', pod_loss)
                # tensor(26.7944, device='cuda:2', grad_fn=<AddBackward0>)
                # tensor(27.4486, device='cuda:3', grad_fn= < AddBackward0 >)
                # tensor(29.1848, device='cuda:1', grad_fn=<AddBackward0>)
                # tensor(26.5714, device='cuda:0', grad_fn=<AddBackward0>)
            if self.entropy_min > 0. and self.step > 0:
                mask_new = labels > 0
                entropies = entropy(torch.softmax(outputs, dim=1))
                entropies[mask_new] = 0.
                pixel_amount = (~mask_new).float().sum(dim=(1, 2))
                loss_entmin = (entropies.sum(dim=(1, 2)) / pixel_amount).mean()

            if self.kd_scheduling:
                lkd = lkd * math.sqrt(self.nb_current_classes / self.nb_new_classes)

            # xxx first backprop of previous loss (compute the gradients for regularization methods)

            loss_tot = loss + lkd + lde + l_icarl + pod_loss + loss_entmin

            with amp.scale_loss(loss_tot, optim) as scaled_loss:
                scaled_loss.backward()

            # xxx Regularizer (EWC, RW, PI)
            if self.regularizer_flag:
                if distributed.get_rank() == 0:
                    self.regularizer.update()
                l_reg = self.reg_importance * self.regularizer.penalty()
                if l_reg != 0.:
                    with amp.scale_loss(l_reg, optim) as scaled_loss:
                        scaled_loss.backward()

            optim.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item() + pod_loss.item(
            ) + loss_entmin.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(
                    f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                    f" Loss={interval_loss}"
                )
                logger.info(
                    f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}, POD {pod_loss} EntMin {loss_entmin}"
                )
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss', interval_loss, x)
                interval_loss = 0.0

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
            reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")

        return (epoch_loss, reg_loss)

    def find_median(self, train_loader, device, logger, mode="probability"):
        """Find the median prediction score per class with the old model.

        Computing the median naively uses a lot of memory, to allievate it, instead
        we put the prediction scores into a histogram bins and approximate the median.

        https://math.stackexchange.com/questions/2591946/how-to-find-median-from-a-histogram
        """
        if mode == "entropy":
            max_value = torch.log(torch.tensor(self.nb_current_classes).float().to(device))
            nb_bins = 100
        else:
            max_value = 1.0
            nb_bins = 20  # Bins of 0.05 on a range [0, 1]
        if self.pseudo_nb_bins is not None:
            nb_bins = self.pseudo_nb_bins

        histograms = torch.zeros(self.nb_current_classes, nb_bins).float().to(self.device)

        for cur_step, (images, labels) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs_old, features_old = self.model_old(images, ret_intermediate=False)

            mask_bg = labels == 0
            probas = torch.softmax(outputs_old, dim=1)
            max_probas, pseudo_labels = probas.max(dim=1)

            if mode == "entropy":
                values_to_bins = entropy(probas)[mask_bg].view(-1) / max_value
            else:
                values_to_bins = max_probas[mask_bg].view(-1)

            x_coords = pseudo_labels[mask_bg].view(-1)
            y_coords = torch.clamp((values_to_bins * nb_bins).long(), max=nb_bins - 1)

            histograms.index_put_(
                (x_coords, y_coords),
                torch.LongTensor([1]).expand_as(x_coords).to(histograms.device).float(),
                accumulate=True
            )

            if cur_step % 10 == 0:
                logger.info(f"Median computing {cur_step}/{len(train_loader)}.")

        thresholds = torch.zeros(self.nb_current_classes, dtype=torch.float32).to(
            self.device
        )  # zeros or ones? If old_model never predict a class it may be important

        logger.info("Approximating median")
        for c in range(self.nb_current_classes):
            total = histograms[c].sum()
            if total <= 0.:
                continue

            half = total / 2
            running_sum = 0.
            for lower_border in range(nb_bins):
                lower_border = lower_border / nb_bins
                bin_index = int(lower_border * nb_bins)
                if half >= running_sum and half <= (running_sum + histograms[c, bin_index]):
                    break
                running_sum += lower_border * nb_bins

            median = lower_border + ((half - running_sum) /
                                     histograms[c, bin_index].sum()) * (1 / nb_bins)

            thresholds[c] = median

        base_threshold = self.threshold
        if "_" in mode:
            mode, base_threshold = mode.split("_")
            base_threshold = float(base_threshold)
        if self.step_threshold is not None:
            self.threshold += self.step * self.step_threshold

        if mode == "entropy":
            for c in range(len(thresholds)):
                thresholds[c] = max(thresholds[c], base_threshold)
        else:
            for c in range(len(thresholds)):
                thresholds[c] = min(thresholds[c], base_threshold)
        logger.info(f"Finished computing median {thresholds}")
        return thresholds.to(device), max_value

    def validate(self, loader, metrics, ret_samples_ids=None, logger=None, end_task=False):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        model.eval()

        model.module.in_eval = True
        if self.model_old is not None:
            self.model_old.in_eval = True

        class_loss = 0.0
        reg_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        if self.step > 0 and self.align_weight_frequency == "epoch":
            model.module.align_weight(self.align_weight)
        elif self.step > 0 and self.align_weight_frequency == "task" and end_task:
            model.module.align_weight(self.align_weight)

        ret_samples = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                if (
                        self.lde_flag or self.lkd_flag or self.icarl_dist_flag
                ) and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old, features_old = self.model_old(images, ret_intermediate=True)

                outputs, features = model(images, ret_intermediate=True)

                # xxx BCE / Cross Entropy Loss
                if not self.icarl_only_dist:
                    loss = criterion(outputs, labels)  # B x H x W
                else:
                    loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                loss = loss.mean()  # scalar

                if self.icarl_combined:
                    # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                    n_cl_old = outputs_old.shape[1]
                    # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                    l_icarl = self.icarl * n_cl_old * self.licarl(
                        outputs.narrow(1, 0, n_cl_old), torch.sigmoid(outputs_old)
                    )

                # xxx ILTSS (distillation on features or logits)
                if self.lde_flag:
                    lde = self.lde_loss(features['body'], features_old['body'])

                if self.lkd_flag and not self.kd_need_labels:
                    lkd = self.lkd_loss(outputs, outputs_old).mean()

                # xxx Regularizer (EWC, RW, PI)
                if self.regularizer_flag:
                    l_reg = self.regularizer.penalty()

                class_loss += loss.item()
                reg_loss += l_reg.item() if l_reg != 0. else 0.
                reg_loss += lkd.item() + lde.item() + l_icarl.item()

                _, prediction = outputs.max(dim=1)

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((images[0].detach().cpu().numpy(), labels[0], prediction[0]))

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

            class_loss = torch.tensor(class_loss).to(self.device)
            reg_loss = torch.tensor(reg_loss).to(self.device)

            torch.distributed.reduce(class_loss, dst=0)
            torch.distributed.reduce(reg_loss, dst=0)

            if distributed.get_rank() == 0:
                class_loss = class_loss / distributed.get_world_size() / len(loader)
                reg_loss = reg_loss / distributed.get_world_size() / len(loader)

            if logger is not None:
                logger.info(
                    f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)"
                )

        return (class_loss, reg_loss), score, ret_samples

    def test(self, loader, metrics, logger=None):
        """Do test and return all output"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        model.eval()

        class_loss = 0.0
        reg_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        ret_samples = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                if self.model_old is not None:
                    with torch.no_grad():
                        outputs_old, features_old = self.model_old(images, ret_intermediate=True)
                    outputs, features = model(images, x_b_old=features_old['body'],
                                              x_pl_old=features_old['pre_logits'],
                                              ret_intermediate=self.ret_intermediate)
                    # print("calculate old feature")
                else:
                    outputs, features = model(images, ret_intermediate=True)
                # xxx BCE / Cross Entropy Loss
                if not self.icarl_only_dist:
                    loss = criterion(outputs, labels)  # B x H x W
                else:
                    loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                loss = loss.mean()  # scalar

                if self.icarl_combined:
                    # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                    n_cl_old = outputs_old.shape[1]
                    # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                    l_icarl = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                                  torch.sigmoid(outputs_old))

                # xxx ILTSS (distillation on features or logits)
                if self.lde_flag:
                    lde = self.lde_loss(features['body'], features_old['body'])

                if self.lkd_flag:
                    lkd = self.lkd_loss(outputs, outputs_old)

                # xxx Regularizer (EWC, RW, PI)
                if self.regularizer_flag:
                    l_reg = self.regularizer.penalty()

                class_loss += loss.item()
                reg_loss += l_reg.item() if l_reg != 0. else 0.
                reg_loss += lkd.item() + lde.item() + l_icarl.item()

                _, prediction = outputs.max(dim=1)

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                #### attention
                out_size = images.shape[-2:]

                a = torch.sum(features['body'] ** 2, dim=1)
                for l in range(a.shape[0]):
                    a[l] = a[l] / torch.norm(a[l])
                a = torch.unsqueeze(a, 1)
                a = F.interpolate(a, size=out_size, mode="bilinear", align_corners=False).squeeze()
                metrics.update(labels, prediction)
                # ## save confusion matrx
                # fig.savefig(os.path.join(colormap_dir, str(i)+'_tsne.png'))

                ###
                ### normal
                for j in range(images.shape[0]):
                    ret_samples.append((images[j].detach().cpu().numpy(),
                                        labels[j],
                                        prediction[j],
                                        a[j].cpu().numpy()))

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

            class_loss = torch.tensor(class_loss).to(self.device)
            reg_loss = torch.tensor(reg_loss).to(self.device)

            torch.distributed.reduce(class_loss, dst=0)
            torch.distributed.reduce(reg_loss, dst=0)

            if distributed.get_rank() == 0:
                class_loss = class_loss / distributed.get_world_size() / len(loader)
                reg_loss = reg_loss / distributed.get_world_size() / len(loader)

            if logger is not None:
                logger.info(f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)")

        return (class_loss, reg_loss), score, ret_samples

    def state_dict(self):
        state = {"regularizer": self.regularizer.state_dict() if self.regularizer_flag else None}

        return state

    def load_state_dict(self, state):
        if state["regularizer"] is not None and self.regularizer is not None:
            self.regularizer.load_state_dict(state["regularizer"])


def entropy(probabilities):
    """Computes the entropy per pixel.

    # References:
        * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
          Saporta et al.
          CVPR Workshop 2020

    :param probabilities: Tensor of shape (b, c, w, h).
    :return: One entropy per pixel, shape (b, w, h)
    """
    factor = 1 / math.log(probabilities.shape[1] + 1e-8)
    return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)


def features_distillation(
        list_attentions_a,
        list_attentions_b,
        collapse_channels="spatial",
        normalize=True,
        labels=None,
        index_new_class=None,
        pod_apply="all",
        pod_deeplab_mask=False,
        pod_deeplab_mask_factor=None,
        interpolate_last=False,
        pod_factor=1.,
        prepro="pow",
        deeplabmask_upscale=True,
        spp_scales=[1, 2, 4],
        pod_options=None,
        outputs_old=None,
        use_pod_schedule=True,
        nb_current_classes=-1,
        nb_new_classes=-1
):
    """A mega-function comprising several features-based distillation.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """
    device = list_attentions_a[0].device

    assert len(list_attentions_a) == len(list_attentions_b)

    if pod_deeplab_mask_factor is None:
        pod_deeplab_mask_factor = pod_factor

    # if collapse_channels in ("spatial_tuple", "spp", "spp_noNorm", "spatial_noNorm"):
    normalize = False

    apply_mask = "background"
    upscale_mask_topk = 1
    mask_position = "last"  # Others choices "all" "backbone"
    use_adaptative_factor = False
    mix_new_old = None

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        adaptative_pod_factor = 1.0
        difference_function = "frobenius"
        pool = True
        use_adaptative_factor = False
        handle_extra_channels = "sum"
        normalize_per_scale = False

        if pod_options and pod_options.get("switch"):
            if i < len(list_attentions_a) - 1:
                if "before" in pod_options["switch"]:
                    collapse_channels = pod_options["switch"]["before"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["before"].get("factor", pod_factor)
                    normalize = pod_options["switch"]["before"].get("norm", False)
                    prepro = pod_options["switch"]["before"].get("prepro", prepro)
                    use_adaptative_factor = pod_options["switch"]["before"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
            else:
                if "after" in pod_options["switch"]:
                    collapse_channels = pod_options["switch"]["after"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["after"].get("factor", pod_factor)
                    normalize = pod_options["switch"]["after"].get("norm", False)
                    prepro = pod_options["switch"]["after"].get("prepro", prepro)

                    apply_mask = pod_options["switch"]["after"].get("apply_mask", apply_mask)
                    upscale_mask_topk = pod_options["switch"]["after"].get(
                        "upscale_mask_topk", upscale_mask_topk
                    )
                    use_adaptative_factor = pod_options["switch"]["after"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
                    mix_new_old = pod_options["switch"]["after"].get("mix_new_old", mix_new_old)

                    handle_extra_channels = pod_options["switch"]["after"].get(
                        "extra_channels", handle_extra_channels
                    )
                    spp_scales = pod_options["switch"]["after"].get(
                        "spp_scales", spp_scales
                    )
                    use_pod_schedule = pod_options["switch"]["after"].get(
                        "use_pod_schedule", use_pod_schedule
                    )

            mask_position = pod_options["switch"].get("mask_position", mask_position)
            normalize_per_scale = pod_options["switch"].get(
                "normalize_per_scale", normalize_per_scale
            )
            pool = pod_options.get("pool", pool)

        if a.shape[1] != b.shape[1]:
            assert i == len(list_attentions_a) - 1
            assert a.shape[0] == b.shape[0]
            assert a.shape[2] == b.shape[2]
            assert a.shape[3] == b.shape[3]

            assert handle_extra_channels in ("trim", "sum"), handle_extra_channels

            if handle_extra_channels == "sum":
                _b = torch.zeros_like(a).to(a.dtype).to(a.device)
                _b[:, 0] = b[:, 0] + b[:, index_new_class:].sum(dim=1)
                _b[:, 1:] = b[:, 1:index_new_class]
                b = _b
            elif handle_extra_channels == "trim":
                b = b[:, :index_new_class]

        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if not pod_deeplab_mask and use_adaptative_factor:
            adaptative_pod_factor = (labels == 0).float().mean()

        if prepro == "pow":
            a = torch.pow(a, 2)
            b = torch.pow(b, 2)
        elif prepro == "none":
            pass
        elif prepro == "abs":
            a = torch.abs(a, 2)
            b = torch.abs(b, 2)
        elif prepro == "relu":
            a = torch.clamp(a, min=0.)
            b = torch.clamp(b, min=0.)

        if collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        elif collapse_channels == "global":
            a = _global_pod(a, spp_scales, normalize=False)
            b = _global_pod(b, spp_scales, normalize=False)
        elif collapse_channels == "local":
            if pod_deeplab_mask and (
                    (i == len(list_attentions_a) - 1 and mask_position == "last") or
                    mask_position == "all"
            ):
                if pod_deeplab_mask_factor == 0.:
                    continue

                pod_factor = pod_deeplab_mask_factor

                if apply_mask == "background":
                    mask = labels < index_new_class
                elif apply_mask == "old":
                    pseudo_labels = labels.clone()
                    mask_background = labels == 0
                    pseudo_labels[mask_background] = outputs_old.argmax(dim=1)[mask_background]

                    mask = (labels < index_new_class) & (0 < pseudo_labels)
                else:
                    raise NotImplementedError(f"Unknown apply_mask={apply_mask}.")

                if deeplabmask_upscale:
                    a = F.interpolate(
                        torch.topk(a, k=upscale_mask_topk, dim=1)[0],
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    b = F.interpolate(
                        torch.topk(b, k=upscale_mask_topk, dim=1)[0],
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                else:
                    mask = F.interpolate(mask[:, None].float(), size=a.shape[-2:]).bool()[:, 0]

                if use_adaptative_factor:
                    adaptative_pod_factor = mask.float().mean(dim=(1, 2))

                a = _local_pod_masked(
                    a, mask, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
                b = _local_pod_masked(
                    b, mask, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
            else:
                a = _local_pod(
                    a, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
                b = _local_pod(
                    b, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if i == len(list_attentions_a) - 1 and pod_options is not None:
            if "difference_function" in pod_options:
                difference_function = pod_options["difference_function"]
        elif pod_options is not None:
            if "difference_function_all" in pod_options:
                difference_function = pod_options["difference_function_all"]

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        if difference_function == "frobenius":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.frobenius_norm(aa - bb, dim=-1) for aa, bb in zip(a, b)]
                ).to(device)
            else:
                layer_loss = torch.frobenius_norm(a - b, dim=-1)
        elif difference_function == "frobenius_mix":
            layer_loss_old = torch.frobenius_norm(a[0] - b[0], dim=-1)
            layer_loss_new = torch.frobenius_norm(a[1] - b[1], dim=-1)

            layer_loss = mix_new_old * layer_loss_old + (1 - mix_new_old) * layer_loss_new
        elif difference_function == "l1":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.norm(aa - bb, p=1, dim=-1) for aa, bb in zip(a, b)]
                ).to(device)
            else:
                layer_loss = torch.norm(a - b, p=1, dim=-1)
        elif difference_function == "kl":
            d1, d2, d3 = a.shape
            a = (a.view(d1 * d2, d3) + 1e-8).log()
            b = b.view(d1 * d2, d3) + 1e-8

            layer_loss = F.kl_div(a, b, reduction="none").view(d1, d2, d3).sum(dim=(1, 2))
        elif difference_function == "bce":
            d1, d2, d3 = a.shape
            layer_loss = bce(a.view(d1 * d2, d3), b.view(d1 * d2, d3)).view(d1, d2,
                                                                            d3).mean(dim=(1, 2))
        else:
            raise NotImplementedError(f"Unknown difference_function={difference_function}")

        assert torch.isfinite(layer_loss).all(), layer_loss
        assert (layer_loss >= 0.).all(), layer_loss

        layer_loss = torch.mean(adaptative_pod_factor * layer_loss)
        if pod_factor <= 0.:
            continue

        layer_loss = pod_factor * layer_loss
        if use_pod_schedule:
            layer_loss = layer_loss * math.sqrt(nb_current_classes / nb_new_classes)
        loss += layer_loss

    return loss / len(list_attentions_a)


def bce(x, y):
    return -(y * torch.log(x + 1e-6) + (1 - y) * torch.log((1 - x) + 1e-6))


def _local_pod(x, spp_scales=[1, 2, 4], normalize=False, normalize_per_scale=False):
    b = x.shape[0]
    w = x.shape[-1]
    emb = []

    for scale_index, scale in enumerate(spp_scales):
        k = w // scale

        nb_regions = scale ** 2

        for i in range(scale):
            for j in range(scale):
                tensor = x[..., i * k:(i + 1) * k, j * k:(j + 1) * k]

                horizontal_pool = tensor.mean(dim=3).view(b, -1)
                vertical_pool = tensor.mean(dim=2).view(b, -1)

                if normalize_per_scale is True:
                    horizontal_pool = horizontal_pool / nb_regions
                    vertical_pool = vertical_pool / nb_regions
                elif normalize_per_scale == "spm":
                    if scale_index == 0:
                        factor = 2 ** (len(spp_scales) - 1)
                    else:
                        factor = 2 ** (len(spp_scales) - scale_index)
                    horizontal_pool = horizontal_pool / factor
                    vertical_pool = vertical_pool / factor

                if normalize:
                    horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                    vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                emb.append(horizontal_pool)
                emb.append(vertical_pool)

    return torch.cat(emb, dim=1)


def _local_pod_masked(
        x, mask, spp_scales=[1, 2, 4], normalize=False, normalize_per_scale=False
):
    b = x.shape[0]
    c = x.shape[1]
    w = x.shape[-1]
    emb = []

    mask = mask[:, None].repeat(1, c, 1, 1)
    x[mask] = 0.

    for scale in spp_scales:
        k = w // scale

        nb_regions = scale ** 2

        for i in range(scale):
            for j in range(scale):
                tensor = x[..., i * k:(i + 1) * k, j * k:(j + 1) * k]

                horizontal_pool = tensor.mean(dim=3).view(b, -1)
                vertical_pool = tensor.mean(dim=2).view(b, -1)

                if normalize_per_scale is True:
                    horizontal_pool = horizontal_pool / nb_regions
                    vertical_pool = vertical_pool / nb_regions
                elif normalize_per_scale == "spm":
                    if scale_index == 0:
                        factor = 2 ** (len(spp_scales) - 1)
                    else:
                        factor = 2 ** (len(spp_scales) - scale_index)
                if normalize:
                    horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                    vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                emb.append(horizontal_pool)
                emb.append(vertical_pool)

    return torch.cat(emb, dim=1)


def _global_pod(x, spp_scales=[2, 4, 8], normalize=False):
    b = x.shape[0]
    w = x.shape[-1]

    emb = []
    for scale in spp_scales:
        tensor = F.avg_pool2d(x, kernel_size=w // scale)
        horizontal_pool = tensor.sum(dim=2).view(b, -1)
        vertical_pool = tensor.sum(dim=3).view(b, -1)

        if normalize:
            horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
            vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

        tensor_pool = torch.cat([horizontal_pool, vertical_pool], dim=-1)

        emb.append(tensor_pool)

    return torch.cat(emb, dim=1)





