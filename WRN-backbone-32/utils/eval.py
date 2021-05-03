from __future__ import print_function, absolute_import
import pdb

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # torch.Size([5, 64])

    res = []
    # pdb.set_trace()
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].float().sum(0).bool().sum(0).float()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res