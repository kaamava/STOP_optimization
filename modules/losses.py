# coding=utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossEn(nn.Module):
    def __init__(self,):
        """cross entropy loss"""
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class MILNCELoss(nn.Module):
    def __init__(self, batch_size=1, n_pair=1,):
        super(MILNCELoss, self).__init__()
        self.batch_size = batch_size
        self.n_pair = n_pair
        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix):
        mm_mask = np.eye(self.batch_size)
        mm_mask = np.kron(mm_mask, np.ones((self.n_pair, self.n_pair)))
        mm_mask = torch.tensor(mm_mask).float().to(sim_matrix.device)

        from_text_matrix = sim_matrix + mm_mask * -1e12
        from_video_matrix = sim_matrix.transpose(1, 0)

        new_sim_matrix = torch.cat([from_video_matrix, from_text_matrix], dim=-1)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)

        mm_mask_logpt = torch.cat([mm_mask, torch.zeros_like(mm_mask)], dim=-1)
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12

        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)

        logpt_choice = torch.zeros_like(new_logpt)
        mark_ind = torch.arange(self.batch_size).to(sim_matrix.device) * self.n_pair + (self.n_pair//2)
        logpt_choice[mark_ind] = 1
        sim_loss = new_logpt.masked_select(logpt_choice.to(dtype=self.bool_dtype)).mean()
        return sim_loss


class MaxMarginRankingLoss(nn.Module):
    def __init__(self,
                 margin=1.0,
                 negative_weighting=False,
                 batch_size=1,
                 n_pair=1,
                 hard_negative_rate=0.5,
        ):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1 and batch_size > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = torch.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float()

    def forward(self, x):
        d = torch.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + \
                     F.relu(self.margin + x - d.view(1, -1))
        if self.negative_weighting and self.n_pair > 1 and self.batch_size > 1:
            max_margin = max_margin * self.mm_mask.to(max_margin.device)
        return max_margin.mean()







class RelationJSLoss(nn.Module):
    """
    用样本间关系分布（余弦相似 -> softmax 概率）对齐两个表示空间的 JS 散度损失。
    - 输入: x, y 形状均为 [N, D]
    - 输出: 标量 loss
    - tau: 温度（越小分布越尖锐）
    - detach_target: 将对向的 KL 目标分布从计算图中分离，常用于蒸馏场景提高稳定性（可选）
    """
    def __init__(self, tau: float = 0.1, detach_target: bool = False):
        super().__init__()
        self.tau = tau
        self.detach_target = detach_target

    @staticmethod
    def _l2_normalize(z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z, p=2, dim=1)

    def _pairwise_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算批内两两余弦相似度并做温度 softmax，得到每行的关系分布（概率）。
        返回形状 [N, N]。
        """
        z = self._l2_normalize(z)             # [N, D]
        sim = z @ z.t()                        # [N, N] 余弦相似（已归一化因此点乘≈cos）
        prob = F.softmax(sim / self.tau, dim=1)
        return prob

    def _kl(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        计算 KL(p || q)，其中 p、q 都是概率分布（每行和为 1）。
        使用 batchmean 规约，返回标量。
        """
        p_log = torch.log(p.clamp_min(1e-12))  # 数值稳定
        # PyTorch: kl_div(input=log_probs, target=probs)
        return F.kl_div(p_log, q, reduction="batchmean")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算对称的 JS 散度:
            JS(P||Q) = 0.5 * KL(P||Q) + 0.5 * KL(Q||P)
        其中 P、Q 为两空间的关系分布矩阵（按行 softmax）。
        """
        assert x.dim() == 2 and y.dim() == 2, "x, y 必须是 [N, D] 的 2D 张量"
        assert x.size(0) == y.size(0), "x 与 y 的 batch 大小（N）必须一致"

        p = self._pairwise_prob(x)             # 来自 x 空间的关系分布
        q = self._pairwise_prob(y)             # 来自 y 空间的关系分布

        if self.detach_target:
            # 可选：将对向 KL 的目标分布从计算图中分离，常用于蒸馏稳定训练
            kl_pq = self._kl(p, q.detach())
            kl_qp = self._kl(q, p.detach())
        else:
            kl_pq = self._kl(p, q)
            kl_qp = self._kl(q, p)

        js = 0.5 * (kl_pq + kl_qp)
        return js