from resnet1d import ResNet1d
from tqdm import tqdm
import h5py
import torch
import os
import json
import numpy as np
import argparse
from warnings import warn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dataloader import Code15Dataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(add_help=True,
                                 description='Train model to predict rage from the raw ecg tracing.')
parser.add_argument('--root_dir', type=str,
                    default='../data/code15')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--model_path', type=str, default='out/resnet1d_bs512')
parser.add_argument('--save_csv', type=str, default='predicted_age.csv')
parser.add_argument('--load_csv', type=str,
                    # default='')
                    default='out/resnet1d_bs512/predicted_age.csv')
args, unk = parser.parse_known_args()

model_path = args.model_path
batch_size = args.batch_size
num_workers = args.num_workers
save_csv = os.path.join(model_path, args.save_csv)
root_dir = args.root_dir
# csv_file = '../../data/code15/exams.csv'


def cal_eval(csv_file):
    # from sklearn.metrics import mean_absolute_error, r2_score
    from torchmetrics import MeanAbsoluteError, R2Score, MatthewsCorrCoef
    from torchmetrics.functional import pearson_corrcoef

    df = pd.read_csv(csv_file)
    age = df['age'].to_numpy()
    predicted_age = df['predicted_age'].to_numpy()

    age = torch.tensor(age).float()
    predicted_age = torch.tensor(predicted_age).float()

    mae = MeanAbsoluteError()(predicted_age, age)
    corrcoef = pearson_corrcoef(predicted_age, age)
    r2_square = R2Score()(predicted_age, age)

    print('MAE: {:.2f}'.format(float(mae)))
    print('Correlation coefficient: {:.2f}'.format(float(corrcoef)))
    print('R square: {:.2f}'.format(float(r2_square)))

if args.load_csv != '':
    cal_eval(args.load_csv)

    import sys
    sys.exit(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(os.path.join(model_path, 'model.pth'), map_location=lambda storage, loc: storage)
config = os.path.join(model_path, 'config.json')
with open(config, 'r') as f:
    config_dict = json.load(f)

# Get model
N_LEADS = 12
model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
    blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
    n_classes=1,
    kernel_size=config_dict['kernel_size'],
    dropout_rate=config_dict['dropout_rate'])
model.load_state_dict(ckpt["model"])
model = model.to(device)

dataset = Code15Dataset(root_dir, mode='test',
                        cut_by_rpeaks=config_dict['cut'],
                        seq_length=config_dict['seq_length'],
                        filtered=config_dict['filtered'],
                        oned=config_dict['oned'])
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False)

ids = []
predicted_age = []
age = []
is_male = []
death = []
# timey = []
normal_ecg = []

print('Start evaluate')

for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
    tracing, _, _, data_info = data

    with torch.no_grad():
        tracing = tracing.float().to(device)
        y_pred = model(tracing)

    exam_id = data_info['exam_id']

    ids += list(exam_id.flatten().numpy())
    predicted_age += list(y_pred.flatten().cpu().numpy())

    age += list(data_info['age'].numpy())
    is_male += list(data_info['is_male'].numpy())
    death += list(data_info['death'].numpy())
    # timey += list(data_info['timey'].numpy())
    normal_ecg += list(data_info['normal_ecg'].numpy())

df = pd.DataFrame({'exam_id': ids, 'predicted_age': predicted_age,
                   'age': age,
                   # 'is_male': is_male,
                   # 'death': death,
                   # 'timey': timey,
                   # 'normal_ecg': normal_ecg
                   })
df = df.set_index('exam_id')
df.to_csv(save_csv)

cal_eval(save_csv)

