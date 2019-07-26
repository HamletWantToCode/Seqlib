import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=2)
        target = target.unsqueeze(dim=-1)
        assert pred.size() == target.size()
        correct = 0
        for i in range(output.size(0)):
            sentence_len = torch.sum(target[i]!=0)
            correct += torch.sum(pred[i]==target[i])*1.0 / sentence_len
    return correct.item() / len(target)
