import torch
from torchmetrics import Accuracy, F1
from torch.nn import BCEWithLogitsLoss
from ..base import Base


class BaseClassification(Base):
    def __init__(self, *args,
                        lr=1e-3,
                        optim_type='adam',
                        debug=False,
                        **kwargs):
        super().__init__(*args, **kwargs)
        self.crit = BCEWithLogitsLoss()
        self.acc = Accuracy(num_classes=1)
        self.f1 = F1(num_classes=1)
        self.lr = lr
        self.debug = debug

        if optim_type in ['adam', 'sgd']:
            self.optim_type = optim_type
        else:
            raise ValueError(f'Unsupported optimizer type: {optim_type}')

    def training_step(self, batch, batch_idx):
        self.train_iter += 1
        self.log(f'train/iter', self.train_iter)
        x, targs = batch

        logits = self(x)
        loss = self.crit(logits, targs.float())
        preds = torch.sigmoid(logits)

        self.log(f'train/loss', loss.item())
        self.log(f'train/acc', self.acc(preds, targs))
        self.log(f'train/f1' , self.f1(preds, targs))

        return loss

    def validation_step(self, batch, batch_idx):
        self.valid_iter += 1
        self.log(f'valid/iter', self.valid_iter)
        x, targs = batch
        logits = self(x)
        loss = self.crit(logits, targs.float())
        preds = torch.sigmoid(logits)

        self.log(f'valid/loss', loss.item())
        self.log(f'valid/acc', self.acc(preds, targs))
        self.log(f'valid/f1' , self.f1(preds, targs))
    
    def test_step(self, batch, batch_idx):
        x, targs = batch
        logits = self(x)
        preds = torch.sigmoid(logits)
        self.log(f'test/acc', self.acc(preds, targs))
        self.log(f'test/f1' , self.f1(preds, targs))
