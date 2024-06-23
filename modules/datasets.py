import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.use_rebuild_data = args.use_rebuild_data
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            self.examples[i]['retrival_ids'] = tokenizer(self.examples[i]['retrival_reports'][0])[:self.max_seq_length]
            self.examples[i]['retrival_mask'] = [1] * len(self.examples[i]['retrival_ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        
        # 检索的部分
        retrival_report_ids = example['retrival_ids']
        retrival_report_masks = example['retrival_mask']
        seq_retrival_length = len(retrival_report_ids)
        
        sample = (image_id, image, report_ids, report_masks, seq_length ,retrival_report_ids , retrival_report_masks , seq_retrival_length)

        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_id = os.path.join(self.image_dir, image_path[0])
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        #if self.use_diff_picture():

        if self.use_rebuild_data:
            rebuld_data_dir = '/root/paddlejob/workspace/liujunyi_new_work_space/liujunyi05/mimic-abn-rebulid/rebulid_data/rebulid_data'
            dir = os.path.join(rebuld_data_dir, image_path[0].replace('jpg','png'))
            if os.path.exists(dir):
                rebuild_image = Image.open(dir).convert('RGB')
                if self.transform is not None:
                    rebuild_image = self.transform(rebuild_image)
                image = torch.stack((image, rebuild_image), 0)
            else:
                image = torch.stack((image, image), 0)
            sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
