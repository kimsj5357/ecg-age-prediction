import json
import torch
import os
from tqdm import tqdm
from resnet1d import ResNet1d
import torch.optim as optim
import numpy as np
import h5py
import pandas as pd
import argparse
import time
from warnings import warn
import warnings
warnings.filterwarnings("ignore")

from dataloader import Code15Dataset
from torch.utils.data import DataLoader
from tools import ETA

def parse_args():
    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('--root_dir', default='../../data/code15')

    parser.add_argument('--epochs', type=int, default=70,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--sample_freq', type=int, default=400,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    parser.add_argument('--seq_length', type=int, default=4096,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                             'to fit into the given size. (default: 4096)')
    parser.add_argument('--scale_multiplier', type=int, default=10,
                        help='multiplicative factor used to rescale inputs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    parser.add_argument('--dropout_rate', type=float, default=0.8,
                        help='dropout rate (default: 0.8).')
    parser.add_argument('--kernel_size', type=int, default=17,
                        help='kernel size in convolutional layers (default: 17).')
    parser.add_argument('--folder', default='my_model/',
                        help='output folder (default: ./my_model)')
    parser.add_argument('--traces_dset', default='tracings',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('--ids_dset', default='',
                        help='by default consider the ids are just the order')
    parser.add_argument('--age_col', default='age',
                        help='column with the age in csv file.')
    parser.add_argument('--ids_col', default=None,
                        help='column with the ids in csv file.')
    parser.add_argument('--cuda', default=True,
                        help='use cuda for computations. (default: False)')
    parser.add_argument('--n_valid', type=int, default=100,
                        help='the first `n_valid` exams in the hdf will be for validation.'
                             'The rest is for training')
    # parser.add_argument('path_to_traces',
    #                     help='path to file containing ECG traces')
    # parser.add_argument('path_to_csv',
    #                     help='path to csv file containing attributes.')

    parser.add_argument('--num_workers', '-nw', default=4)

    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    return args

def compute_loss(ages, pred_ages, weights):
    diff = ages.flatten() - pred_ages.flatten()
    loss = torch.sum(weights.flatten() * diff * diff)
    return loss


def train(epoch, dataload):
    model.train()
    total_loss = 0
    n_entries = 0
    train_desc = "Epoch [{:2d}/{}]: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(epoch, 0, 0), position=0)
    for traces, ages, weights, data_info in dataload:
        # traces = traces.transpose(1, 2)
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
        # Reinitialize grad
        model.zero_grad()
        # Send to device
        # Forward pass
        pred_ages = model(traces)
        loss = compute_loss(ages, pred_ages, weights)
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        bs = len(traces)
        total_loss += loss.detach().cpu().numpy()
        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, args.epochs, total_loss / n_entries)
        train_bar.update(1)
    train_bar.close()
    return total_loss / n_entries


def eval(epoch, dataload):
    model.eval()
    total_loss = 0
    n_entries = 0
    eval_desc = "Epoch [{:2d}/{}]: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(epoch, 0, 0), position=0)
    for traces, ages, weights, data_info in dataload:
        traces = traces.transpose(1, 2)
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
        with torch.no_grad():
            # Forward pass
            pred_ages = model(traces)
            loss = compute_loss(ages, pred_ages, weights)
            # Update outputs
            bs = len(traces)
            # Update ids
            total_loss += loss.detach().cpu().numpy()
            n_entries += bs
            # Print result
            eval_bar.desc = eval_desc.format(ep, args.epochs, total_loss / n_entries)
            eval_bar.update(1)
    eval_bar.close()
    return total_loss / n_entries



if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    print(args)
    # Set device
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    folder = args.folder

    # Generate output folder if needed
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    # Save config file
    config = vars(args)
    # config.update({'data': {'train': train_dataset.file_list, 'val': val_dataset.file_list}})
    with open(os.path.join(args.folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent='\t')

    tqdm.write("Building data loaders...")
    train_dataset = Code15Dataset(args.root_dir, mode='train')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True)
    val_dataset = Code15Dataset(args.root_dir, mode='test')
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=False)
    tqdm.write("Done!")

    tqdm.write("Define model...")
    N_LEADS = 12  # the 12 leads
    N_CLASSES = 1  # just the age
    model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                     blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                     n_classes=N_CLASSES,
                     kernel_size=args.kernel_size,
                     dropout_rate=args.dropout_rate)
    model.to(device=device)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    optimizer = optim.Adam(model.parameters(), args.lr)
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience,
                                                     min_lr=args.lr_factor * args.min_lr,
                                                     factor=args.lr_factor)
    tqdm.write("Done!")

    tqdm.write("Training...")
    start_epoch = 0
    best_loss = np.Inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'weighted_rmse', 'weighted_mae', 'rmse', 'mse'])
    eta = ETA(args.epochs)
    st = time.time()

    for ep in range(start_epoch, args.epochs):
        train_loss = train(ep, train_dataloader)
        valid_loss = eval(ep, val_dataloader)
        # Save best model
        if valid_loss < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'valid_loss': valid_loss,
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_loss = valid_loss
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < args.min_lr:
            break

        time_epoch = time.time() - st
        time_left = eta(time_epoch)
        time_log = time.strftime('%dD %X', time.gmtime(time_left))
        st = time.time()

        # Print message
        tqdm.write('Epoch [{:2d}/{}]: \tTrain Loss {:.6f} ' \
                   '\tValid Loss {:.6f} \tLearning Rate {:.7f} \tETA {}'
                   .format(ep, args.epochs, train_loss, valid_loss, learning_rate, time_log))
        # Save history
        history = history.append({"epoch": int(ep), "train_loss": train_loss,
                                  "valid_loss": valid_loss, "lr": learning_rate}, ignore_index=True)
        history.to_csv(os.path.join(folder, 'history1.csv'), index=False)
        # Update learning rate
        scheduler.step(valid_loss)
    tqdm.write("Done!")
