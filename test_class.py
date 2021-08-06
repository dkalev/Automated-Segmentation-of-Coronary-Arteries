import wandb
import torch
import shutil
import sys
sys.path.append('..')
from train import get_class
from data_utils.datamodule import AsocaClassificationDataModule
import pytorch_lightning as plt


def test(model_dir, data_dir='dataset/classification'):
    runs = wandb.Api().runs(path='ASOCA_final', filters={
        'config.seed': {'$in': [0,11,42]},
        'config.model/model': {"$regex": "models.classification.*"}, 
        })

    # trainer = plt.Trainer(gpus=4, accelerator='ddp', replace_sampler_ddp=False)
    trainer = plt.Trainer(gpus=1)
    dm = AsocaClassificationDataModule(data_dir=data_dir)

    for i, run in enumerate(runs):
        if run.summary.get('test/acc') is not None: continue
        print(f'Predicting on {run.name}, {run.id} ({i+1}/{len(runs)})')

        model_params = { k.split('/')[-1]:v for k,v in run.config.items() if 'model' in k }
        class_name = model_params['model']
        del model_params['model']
        if 'initialize' in model_params: model_params['initialize'] = False
            
        try:
            with run.files()[0].download(model_dir, replace=True)as model_f:
                model: plt.LightningModule = get_class(class_name)(**model_params)
                with open(model_f.name, 'rb') as f:
                    ckpt = torch.load(f)
                state_dict = {k:v for k,v in ckpt['state_dict'].items() if 'in_indices_' not in k}
                model.load_state_dict(state_dict, strict=False)
                test_res = trainer.test(model=model, datamodule=dm)[0]
                run.summary.update(test_res)
            shutil.rmtree(f'{model_dir}/{run.project}/{run.id}')
        except RuntimeError as e:
            print(f'Failed for {run.name} {run.id} with error {e}')
            continue



if __name__ == '__main__':
    model_dir = '/var/scratch/ebekkers/damyan/models/'
    test(model_dir)
