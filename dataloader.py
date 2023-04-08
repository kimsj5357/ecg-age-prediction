import os
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import Dataset


class Code15Dataset(Dataset):
    def __init__(self, root_dir, mode='train',
                 cut_by_rpeaks=False, seq_length=4096,
                 filtered=False,
                 oned=True, image_size=64):
        self.root_dir = root_dir
        self.mode = mode

        self.cut_by_rpeaks = cut_by_rpeaks
        self.seq_length = seq_length
        self.filtered = filtered
        self.oned = oned

        csv_file = os.path.join(root_dir, 'exams.csv')
        self.csv_data = pd.read_csv(csv_file)
        # self.csv_data = self.csv_data.set_index('exam_id')

        self.exam_ids = list(self.csv_data['exam_id'])

        if filtered or cut_by_rpeaks:
            rpeaks_file = os.path.join(root_dir, 'rpeaks.json')
            self.rpeaks = {}
            with open(rpeaks_file, 'r') as f:
                self.rpeaks = json.load(f)
            self.filtered_path = os.path.join(root_dir, 'filtered')

            self.exam_ids = list(map(int, self.rpeaks))

        train_num = int(len(self.exam_ids) * 0.8)
        val_num = int(len(self.exam_ids) * 0.1)
        # print(len(self.exam_ids), train_num, val_num, len(self.exam_ids) - train_num - val_num)

        if mode == 'train':
            self.exam_ids = self.exam_ids[:train_num]
            # self.weights = self.compute_weights()
        elif mode == 'val':
            self.exam_ids = self.exam_ids[train_num:train_num + val_num]
        elif mode == 'test':
            self.exam_ids = self.exam_ids[train_num + val_num:]
        # elif mode == 'all':

        self.ecg_order = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        if not self.oned:
            # import pyts
            from pyts.image import GramianAngularField, MarkovTransitionField

            self.gaf = GramianAngularField(image_size=image_size, method='summation')
            self.mtf = MarkovTransitionField(image_size=image_size)


    def __len__(self):
        return len(self.exam_ids)

    def __getitem__(self, idx):
        exam_id = self.exam_ids[idx]
        data_info = self.csv_data[self.csv_data['exam_id'] == exam_id].iloc[0].to_dict()

        trace_file = os.path.join(self.root_dir, data_info['trace_file'])
        with h5py.File(trace_file, "r") as f:
            exam_ids = f['exam_id'][:-1]
            id_index = np.where(exam_ids == exam_id)[0][0]
            tracing = f['tracings'][id_index]
        tracing = tracing.T
        raw_tracing = tracing

        if self.filtered:
            filtered_file = os.path.join(self.filtered_path, self.rpeaks[str(exam_id)]['filtered_file'])
            with h5py.File(filtered_file, 'r') as f:
                exam_ids = f['exam_id'][:]
                id_index = np.where(exam_ids == exam_id)[0][0]
                tracing = f['filtered'][id_index]

        if self.cut_by_rpeaks:
            rpeaks = self.rpeaks[str(exam_id)]['rpeaks']
            new_tracing = np.zeros((len(rpeaks), self.seq_length))

            for i in range(len(rpeaks)):
                # start_point = 0
                # if len(rpeaks[i]) > 0:
                #     start_point = rpeaks[i][0]
                # elif len(np.where(tracing[i] != 0)[0]) > 0:
                #     start_point = np.where(tracing[i] != 0)[0][0]
                # elif len(np.where(raw_tracing[i] != 0)[0]) > 0:
                #     start_point = np.where(raw_tracing[i] != 0)[0][0]
                #
                # if start_point + self.seq_length > tracing.shape[-1]:
                #     start_point = tracing.shape[-1] - self.seq_length
                #
                # new_tracing[i] = tracing[i, start_point:start_point + self.seq_length]

                if len(rpeaks[i]) > 1:
                    cut_tracing = tracing[i, rpeaks[i][0]: rpeaks[i][-1]]
                    cut_tracing_repeat = np.tile(cut_tracing, (self.seq_length // len(cut_tracing) + 1))
                    cut_tracing_repeat = cut_tracing_repeat[:self.seq_length]
                    new_tracing[i] = cut_tracing_repeat
                else:
                    new_tracing[i] = tracing[i]


            tracing = new_tracing


        if not self.oned:
            tracing = self.timeseries2image(tracing)

        age = data_info['age']# if self.mode == 'train' else None
        # weight = self.weights[idx] if self.mode == 'train' else 1.

        weight = 1 - abs(60 - age) * 0.01

        return tracing, age, weight, data_info

    def compute_weights(self, max_weight=np.inf):
        ages = self.csv_data['age'].values

        _, inverse, counts = np.unique(ages, return_inverse=True, return_counts=True)
        weights = 1 / counts[inverse]
        normalized_weights = weights / sum(weights)
        w = len(ages) * normalized_weights
        # Truncate weights to a maximum
        if max_weight < np.inf:
            w = np.minimum(w, max_weight)
            w = len(ages) * w / sum(w)
        return w

    def timeseries2image(self, tracing):
        if tracing.ndim != 2:
            raise Exception('Tracing must be 2D (12, 4096) but current tracing shape: ' + tracing.shape)

        im_gaf = self.gaf.fit_transform(tracing)

        return im_gaf



if __name__ == "__main__":
    root_dir = '../../data/code15'
    csv_file = os.path.join(root_dir, 'exams.csv')

    train_dataset = Code15Dataset(root_dir, mode='train')
    val_dataset = Code15Dataset(root_dir, mode='val')
    test_dataset = Code15Dataset(root_dir, mode='test')
    print('train dataset num:', len(train_dataset))
    print('val dataset num:', len(val_dataset))
    print('test dataset num:', len(test_dataset))
    print('total dataset num:', len(train_dataset) + len(val_dataset) + len(test_dataset))

    # from tqdm import tqdm
    # dataset = Code15Dataset(root_dir, mode='all')
    # from biosppy.signals import ecg
    # rpeaks = {}
    # min_rpeaks = 1000
    # zero_rpeaks = []
    #
    # filtered = []
    # exam_id = []
    # save_file = os.path.join(root_dir, 'filtered')
    # os.makedirs(save_file, exist_ok=True)
    # import h5py
    #
    # for i, data in tqdm(enumerate(dataset), total=len(dataset)):
    #     tracing, _, _, data_info = data
    #
    #     id = data_info['exam_id']
    #     rpeaks[id] = {}
    #     rpeaks[id]['rpeaks'] = []
    #
    #     filtered.append([])
    #
    #     for t in tracing:
    #         try:
    #             ecg_data = ecg.ecg(t, sampling_rate=1000, show=False)
    #             rpeaks_i = ecg_data['rpeaks']
    #             filtered_i = ecg_data['filtered']
    #         except ValueError:
    #             zero_rpeaks.append(id)
    #             rpeaks_i = np.array([])
    #             filtered_i = np.zeros(4096)
    #
    #         rpeaks_i = list(rpeaks_i)
    #         rpeaks_i = list(map(int, rpeaks_i))
    #         rpeaks[id]['rpeaks'] += [rpeaks_i]
    #
    #         filtered[-1].append(filtered_i)
    #
    #         if min_rpeaks > len(rpeaks_i) and len(rpeaks_i) != 0: min_rpeaks = len(rpeaks_i)
    #
    #     save_name = 'part' + str(i // 10000) + '.hdf5'
    #     rpeaks[id]['filtered_file'] = save_name
    #     exam_id.append(id)
    #
    #     if (i + 1) % 10000 == 0:
    #         save_part_file = os.path.join(save_file, save_name)
    #         exam_id = np.array(exam_id)
    #         filtered = np.array(filtered)
    #
    #         with h5py.File(save_part_file, 'w') as f:
    #             f.create_dataset('filtered', data=filtered)
    #             f.create_dataset('exam_id', data=exam_id)
    #
    #         exam_id = []
    #         filtered = []
    #
    #         # break
    #
    #
    # import json
    # save_json = os.path.join(root_dir, 'rpeaks.json')
    # with open(save_json, 'w') as f:
    #     json.dump(rpeaks, f, indent=2)
    # print(min_rpeaks)
    # zero_rpeaks = list(set(zero_rpeaks))
    # print('zero repaks:', len(zero_rpeaks))