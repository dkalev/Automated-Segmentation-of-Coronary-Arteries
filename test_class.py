from typing import Dict
from torch.utils.data import DataLoader
import wandb
import torch
import shutil
import sys
sys.path.append('..')
from train import get_class
from data_utils.datamodule import AsocaClassificationDataModule
import pytorch_lightning as plt
import argparse
import e3cnn.nn as enn
import e3cnn.gspaces as gspaces
import numpy as np
import traceback
from tqdm import tqdm



def test_model(model, x, samples, G):

    errors = []
    trivial_type = enn.FieldType(G, [G.trivial_repr])

    for el in tqdm(samples):

        out1 = model(x).detach().cpu().numpy()
        x_tr = enn.GeometricTensor(x, trivial_type).transform(el).tensor
        out2 = model(x_tr).detach().cpu().numpy()

        errs = ((out1 - out2)**2).sum(axis=1)
        errs = np.sqrt(errs).reshape(-1)
        maxnorm = np.sqrt((out1**2).sum(axis=1)).reshape(-1)
        rel_err = errs / maxnorm
        errors.append(rel_err.mean())

    return errors

def test_equivariance(model:plt.LightningModule, G: gspaces.GSpace3D, gpu:int=0):

    # build 300 random inputs to check
    x = torch.randn(20, 1, 68, 68, 68)
    x = x.to(f'cuda:{gpu}')
    model = model.to(f'cuda:{gpu}')

    # check 20 random rotations
    samples = G.fibergroup.grid('cube')
    # if you want to test equivariance to only the cube symmetries, use this
    # samples = G.fibergroup.grid('cube')

    errors = test_model(model, x, samples, G)
    
    return errors

def get_test_metrics(model:plt.LightningModule, dm:plt.LightningDataModule, gpu, bs:int=16, n_total:int=20000) -> Dict:
    preds = torch.empty(n_total)
    targs = torch.empty(n_total)
    model.to(f'cuda:{gpu}')
    for i, (x, targs_b) in enumerate(tqdm(dm.test_dataloader(batch_size=bs))):
        x = x.to(f'cuda:{gpu}')
        preds_b = model(x).cpu().detach()
        preds_b = torch.sigmoid(preds_b)
        preds[i*bs:i*bs+bs] = preds_b.flatten()
        targs[i*bs:i*bs+bs] = targs_b.flatten()
    
    preds = preds.round().numpy()
    targs = targs.numpy()

    acc = np.mean(preds == targs)

    tp = np.sum((preds == targs) & (preds == 1.))
    tn = np.sum((preds == targs) & (preds == 0.))
    fp = np.sum((preds != targs) & (preds == 1.))
    fn = np.sum((preds != targs) & (preds == 0.))

    assert sum([tp, tn, fp, fn]) == n_total

    precision = tp / (tp+fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        'test/acc': acc,
        'test/prec': precision,
        'test/rec': recall,
        'test/f1': f1
    }


def test(model_dir, data_dir='dataset/classification', gpu=0):
    runs = wandb.Api().runs(path='ASOCA_final', filters={
        'config.seed': {'$in': [0,11,42]},
        'config.model/model': {"$regex": "models.classification.*"}, 
        'createdAt': {'$gte': '2021-08-15' },
        'state': { '$ne': 'Running' },
    })
    print(len(runs))

    # trainer = plt.Trainer(gpus=4, accelerator='ddp', replace_sampler_ddp=False)
    trainer = plt.Trainer(gpus=[gpu])
    dm = AsocaClassificationDataModule(data_dir=data_dir)


    for i, run in enumerate(runs):
        # run.summary.update({
        #     'predicting': None,
        #     'eq_max_error': None,
        #     'eq_mean_error': None,
        #     'eq_std_error': None,
        #     'test/acc': None,
        #     'test/prec': None,
        #     'test/rec': None,
        #     'test/f1': None,
        # })
        # continue
        if run.name == 'hopeful-cherry-773': continue
        if run.name == 'lunar-surf-693': continue
        if run.name == 'stilted-valley-657': continue
        if run.summary.get('eq_max_error') is not None and run.summary.get('test/acc') is not None: continue
        # if run.summary.get('predicting') is not None: continue
        # run.summary.update({'predicting': gpu})
        print(f'Predicting on {run.name}, {run.id} ({i+1}/{len(runs)})')

        model_params = { k.split('/')[-1]:v for k,v in run.config.items() if 'model' in k }
        class_name = model_params['model']
        del model_params['model']
        if 'initialize' in model_params: model_params['initialize'] = False
            
        G = gspaces.rot3dOnR3() 

        try:
            with run.files()[0].download(model_dir, replace=True)as model_f:
                model: plt.LightningModule = get_class(class_name)(**model_params)
                with open(model_f.name, 'rb') as f:
                    ckpt = torch.load(f)
                state_dict = {k:v for k,v in ckpt['state_dict'].items() if 'in_indices_' not in k}
                model.load_state_dict(state_dict, strict=False)

                # if run.summary.get('eq_max_error') is None:
                #     print('checking equivariance')
                #     equiv_errors = test_equivariance(model, G, gpu)
                #     run.summary.update({
                #         'eq_max_error': np.max(equiv_errors),
                #         'eq_mean_error': np.mean(equiv_errors),
                #         'eq_std_error': np.std(equiv_errors),
                #     })
                if run.summary.get('test/acc') is None:
                # test_res = trainer.test(model=model, datamodule=dm)[0]
                    test_res = get_test_metrics(model, dm, gpu)
                    run.summary.update(test_res)

        except RuntimeError as e:
            print(f'Failed for {run.name} {run.id} with error {e}')
            traceback.print_exc()
        finally:
            shutil.rmtree(f'{model_dir}/{run.project}/{run.id}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build and preprocess ASOCA dataset')
    parser.add_argument('--gpu', type=int, default=0)
    hparams = vars(parser.parse_args())
    model_dir = '/var/scratch/ebekkers/damyan/models/'
    test(model_dir, gpu=hparams['gpu'])
