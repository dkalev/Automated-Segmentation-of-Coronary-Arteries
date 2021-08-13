import torch
from torchmetrics import Accuracy, F1, Precision, Recall
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
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()
        self.train_f1 = F1()
        self.valid_f1 = F1()
        self.test_f1 = F1()
        self.train_prec = Precision()
        self.valid_prec = Precision()
        self.test_prec = Precision()
        self.train_recall = Recall()
        self.valid_recall = Recall()
        self.test_recall = Recall()
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
        self.log(f'train/acc', self.train_acc(preds, targs), prog_bar=True)
        self.log(f'train/f1', self.train_f1(preds, targs), prog_bar=True)
        self.log(f'train/prec', self.train_prec(preds, targs))
        self.log(f'train/rec', self.train_recall(preds, targs))

        return loss

    def validation_step(self, batch, batch_idx):
        self.valid_iter += 1
        self.log(f'valid/iter', self.valid_iter)
        x, targs = batch
        logits = self(x)
        loss = self.crit(logits, targs.float())
        preds = torch.sigmoid(logits)

        self.log(f'valid/loss', loss.item())
        self.log(f'valid/acc', self.valid_acc(preds, targs))
        self.log(f'valid/f1', self.valid_f1(preds, targs))
        self.log(f'valid/prec', self.valid_prec(preds, targs))
        self.log(f'valid/rec', self.valid_recall(preds, targs))
    
    def test_step(self, batch, batch_idx):
        x, targs = batch
        logits = self(x)
        preds = torch.sigmoid(logits)
        self.log(f'test/acc', self.test_acc(preds, targs))
        self.log(f'test/f1' , self.test_f1(preds, targs))
        self.log(f'test/prec', self.test_prec(preds, targs))
        self.log(f'test/rec', self.test_recall(preds, targs))
