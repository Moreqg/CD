import torch
import torchmetrics
class DiceMetric(object):
    def __init__(self, 
                force_binary = False, 
                per_image = False, 
                drop_background = False):
        self.force_binary = force_binary
        self.per_image = per_image
        self.drop_background = drop_background

    def preprocess(self, output):
        # y, yhat = output
        return output


    def __call__(self, outputs):
        y = torch.cat([_[0] for _ in outputs]).detach().cpu()
        yhat = torch.cat([_[1] for _ in outputs]).detach().cpu().float()
        # y, yhat = outputs
        # y = y[1].detach().cpu().float()
        # yhat = yhat[1].detach().cpu().float()
        f1 = torchmetrics.functional.f1_score(yhat, y)  # yhat is prediction
        # if len(y.shape) != len(yhat.shape) and yhat.shape[1] != 1:
        #     num_classes = yhat.shape[1]
        #     yhat = yhat.argmax(1)
        #     inte_and_card = []
        #     for c in range(num_classes):
        #         inte = ((y == c).long() * (yhat == c).long()).sum((1, 2))
        #         card = ((y == c).long() + (yhat == c).long()).sum((1, 2))
        #         inte_and_card.append(torch.stack([inte, card], 1))
        #     return torch.stack(inte_and_card, 2)
        # else:
        #     if yhat.shape[1] == 1:
        #         yhat = yhat.squeeze(1)
        #     yhat = yhat.sigmoid().round()  # 512 512
        #     inte = ((y == 1).long() * (yhat == 1).long()).sum((1, 2))
        #     card = ((y == 1).long() + (yhat == 1).long()).sum((1, 2))
        #     return torch.stack([inte, card], 1)
        #
        #
        # dice = torch.cat(outputs).numpy()
        #
        # if not self.per_image:
        #     dice = dice.sum(0)
        #     dice = 2 * dice[0] / (dice[1] + 1e-6)
        # else:
        #     dice = 2 * dice[:,0] / (dice[:,1] + 1e-6)
        #     dice = dice.mean(0)

        return {"val_dice": f1}

