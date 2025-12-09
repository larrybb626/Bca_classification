import torch
import torch.nn.functional as F
import torch.nn as nn

Tensor = torch.Tensor


def equalized_focal_loss(logits: Tensor,
                         targets: Tensor,
                         gamma_b=2,
                         scale_factor=8,
                         reduction="mean"):
    """ EFL loss"""
    ce_loss = F.binary_cross_entropy(logits, targets, reduction="none", )
    outputs = F.binary_cross_entropy(logits, targets)  # 求导使用，不能带 reduction 参数
    log_pt = -ce_loss
    pt = torch.exp(log_pt)  # softmax 函数打分

    # targets = targets.view(-1, 1)  # 多加一个维度，为使用 gather 函数做准备
    grad_i = torch.autograd.grad(outputs=-outputs, inputs=logits)[0]  # 求导
    # grad_i = grad_i.gather(1, targets)  # 每个类对应的梯度
    pos_grad=[]
    neg_grad=[]
    for i in range(len(grad_i)):
        if targets[i]==1:
            pos_grad_i = F.relu(grad_i[i])
            pos_grad.append(pos_grad_i)
        if targets[i]==0:
            neg_grad_i = F.relu(grad_i[i])
            neg_grad.append(neg_grad_i)
    # pos_grad_i = F.relu(grad_i).sum()
    # neg_grad_i = F.relu(-grad_i).sum()
    pos_grad_sum = sum(pos_grad)
    neg_grad_sum = sum(neg_grad)
    neg_grad_sum += 1e-9  # 防止除数为0
    grad_i = pos_grad_sum / neg_grad_sum
    grad_i = torch.clamp(grad_i, min=0, max=1)  # 裁剪梯度

    dy_gamma = gamma_b + scale_factor * (1 - grad_i)
    dy_gamma = dy_gamma.view(-1)  # 去掉多的一个维度
    # weighting factor
    wf = dy_gamma / gamma_b
    weights = wf * (1 - pt) ** dy_gamma

    efl = weights * ce_loss

    if reduction == "sum":
        efl = efl.sum()
    elif reduction == "mean":
        efl = efl.mean()
    else:
        raise ValueError(f"reduction '{reduction}' is not valid")
    return efl


def balanced_equalized_focal_loss(logits: Tensor,
                                  targets: Tensor,
                                  alpha_t=0.22,
                                  gamma_b=2,
                                  scale_factor=8,
                                  reduction="mean"):
    """balanced EFL loss"""
    return alpha_t * equalized_focal_loss(logits, targets, gamma_b,
                                          scale_factor, reduction)


if __name__ == "__main__":
    logits = torch.tensor([[0.3, 0.6, 0.9, 1],
                           [0.6, 0.4, 0.9, 0.5]],
                          requires_grad=True)
    labels = torch.tensor([1, 0])
    print(equalized_focal_loss(logits, labels))
    print(equalized_focal_loss(logits, labels, reduction="sum"))
    print(balanced_equalized_focal_loss(logits, labels))
    print(balanced_equalized_focal_loss(logits, labels, reduction="sum"))