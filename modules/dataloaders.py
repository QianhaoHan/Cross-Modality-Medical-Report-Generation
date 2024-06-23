import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),   # 由于我有两张图片所以如果之间使用随机裁剪的话，会导致两张图片的尺寸裁剪的地方不一样，从而导致一定的误差
                # transforms.Resize((224,224)),  
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),  # 这个地方存在一个变形的问题，你的训练集都没有变形，这个直接resize两边会导致变形
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch ,retrival_report_ids_batch , retrival_report_masks_batch, seq_retrival_length = zip(*data) # 增加一个检索到的报告
        image_batch = torch.stack(image_batch, 0)
        max_seq_length = max(seq_lengths_batch)

        target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)   #建立一个全为0的矩阵，长度为seq_length
        target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int) #建立一个全为1的矩阵

        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks
        # 把batch内的长度统一为max_seq_length的长度


        max_len_retrival = max(seq_retrival_length)
        retrival_batch = np.zeros((len(retrival_report_ids_batch), max_len_retrival), dtype=int)   #建立一个全为0的矩阵，长度为seq_length
        retrival_masks_batch = np.zeros((len(retrival_report_masks_batch), max_len_retrival), dtype=int) #建立一个全为1的矩阵

        for i, ids in enumerate(retrival_report_ids_batch):
            retrival_batch[i, :len(ids)] = ids

        for i, masks in enumerate(retrival_report_masks_batch):
            retrival_masks_batch[i, :len(masks)] = masks




        return image_id_batch, image_batch, torch.LongTensor(target_batch), torch.FloatTensor(target_masks_batch) , torch.LongTensor(retrival_batch) , torch.LongTensor(retrival_masks_batch)
