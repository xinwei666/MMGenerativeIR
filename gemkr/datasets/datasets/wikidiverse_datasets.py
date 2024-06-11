import pickle
import random

from torch.utils.data import DataLoader, Dataset
import json
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import base64

import torch
import os.path as osp
import os
from typing import Union
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import re

def load_json(file_path):
    with open(file_path,'r',encoding='utf-8') as fr:
        data = json.load(fr)

    return data

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: Union[None, str] = None,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                     input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

class ELDataset_train(Dataset):

    def __init__(self, args, tokenizer, preprocess, gen_max_length=256):
        super(ELDataset_train, self).__init__()
        path = args.data_path
        print(path)
        self.args = args
        self.image_size = 224

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = 0  # unk
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2
        self.transform = preprocess["train"].transform

        self.prompter = Prompter("alpaca")

        raw_data = load_json(path)
        data = []
        for item in raw_data:
            data.append({
                'input': item['input'],# + ' ' + item['caption'],
                'output': item['output'],
                'image': item['image']
            })

        self.data = data #random.shuffle(data)

        self.cutoff_len = gen_max_length

        self.imgs_root = args.img_root
        self.region_feat_root = args.region_feat_path
        region_feat_files = os.listdir(self.region_feat_root)
        region_file_dict = {}
        for file in region_feat_files:
            name = file.replace('.pkl', '')
            img_name = '_'.join(name.split('_')[:-1])
            if img_name in region_file_dict:
                region_file_dict[img_name].append(file)
            else:
                region_file_dict[img_name] = [file]

        self.region_file_dict = region_file_dict


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_prompt = self.prompter.generate_prompt(
            item["instruction"] if 'instruction' in item else None,
            item["input"] if "input" in item else None,
        )

        image_id = item['image']

        try:
            image_path = os.path.join(self.imgs_root, image_id + '.jpg')
            image = Image.open(image_path).convert('RGB')
            image_feat = self.transform(image)
        except:
            print(f'Exception happened when load image id: {idx} name: {image_id}')
            image_feat = torch.zeros((3, self.image_size, self.image_size))

        image_name = image_id
        if image_name in self.region_file_dict:
            region_feats = []
            region_feat_files = self.region_file_dict[image_name]
            for file in region_feat_files:
                with open(os.path.join(self.region_feat_root, file), 'rb') as ff:
                    region_feats.append(pickle.load(ff))
        else:
            region_feats = []

        if len(region_feats) != 0:
            region_feats = torch.stack(region_feats)
            padd_feats = torch.zeros(10 - region_feats.shape[0], 1408)
            region_feats = torch.cat([region_feats, padd_feats], dim=0)
        else:
            region_feats = torch.zeros(10, 1408)
        return {
            'id': idx,
            'image_feat': image_feat,
            'full_prompt': full_prompt,
            'label': item["output"],
            'region_feat': region_feats
        }

    def collate_fn(self, batch):
        data_id = [data['id'] for data in batch]
        image_feat = [data['image_feat'] for data in batch]
        full_prompts = [data['full_prompt'] for data in batch]
        labels = [data['label'] for data in batch]

        image_feat = torch.stack(image_feat)

        region_feat = [data['region_feat'] for data in batch]
        region_feat = torch.stack(region_feat)

        return {
            # 'input_ids': input_ids,
            # 'attention_mask': attention_mask,
            # 'labels': labels,
            'full_prompts': full_prompts,
            'labels': labels,
            'images': image_feat,
            'id': data_id,
            'region_feat': region_feat
        }

class ELDataset_eval(Dataset):

    def __init__(self, args, tokenizer, preprocess, path='./dataset/okvqa/test.json', gen_max_length=256):
        super(ELDataset_eval, self).__init__()
        print(path)
        self.args = args
        self.image_size = 224
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = 0  # unk
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2
        # self.transform = preprocess['train'].transform

        self.prompter = Prompter("alpaca")

        raw_data = load_json(path)
        data = []
        for item in raw_data:
            data.append({
                'input': item['input'],
                'output': item['output'],
                'image': item['image']
            })

        self.data = data

        self.cutoff_len = gen_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # tokenized_full_prompt = self.generate_and_tokenize_prompt(item)
        full_prompt = self.prompter.generate_prompt(
            input=item["input"],
            # item["output"],
        )

        return {
            'id': idx,
            'image_feat': None, #image_feat,
            'full_prompt': full_prompt,
            'label': item["output"],
            'region_feat': None
        }

    def collate_fn(self, batch):
        data_id = [data['id'] for data in batch]
        # image_feat = [data['image_feat'] for data in batch]
        full_prompts = [data['full_prompt'] for data in batch]
        labels = [data['label'] for data in batch]

        return {
            'full_prompts': full_prompts,
            'labels': labels,
            # 'images': image_feat,
            'id': data_id,
            # 'region_feat': region_feat
        }

class ELDataset_test(Dataset):

    def __init__(self, args, tokenizer, preprocess, gen_max_length=256):
        super(ELDataset_test, self).__init__()

        path = args.data_path
        self.args = args
        self.image_size = 224
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = 0  # unk
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2
        self.transform = preprocess.transform

        self.prompter = Prompter("alpaca")

        raw_data = load_json(path)
        data = []
        for item in raw_data:
            data.append({
                'input': item['input'],
                'output': item['output'],
                'image': item['image']
            })
        # random.shuffle(data)
        self.data = data

        self.cutoff_len = gen_max_length

        self.imgs_root = args.img_root
        self.region_feat_root = args.region_feat_path
        region_feat_files = os.listdir(self.region_feat_root)
        region_file_dict = {}
        for file in region_feat_files:
            name = file.replace('.pkl', '')
            img_name = name.split('_')[0]
            if img_name in region_file_dict:
                region_file_dict[img_name].append(file)
            else:
                region_file_dict[img_name] = [file]

        self.region_file_dict = region_file_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_prompt = self.prompter.generate_prompt(
            input=item["input"],
        )

        image_id = item['image']

        try:
            image_path = os.path.join(self.imgs_root, image_id + '.jpg')
            image = Image.open(image_path).convert('RGB')
            image_feat = self.transform(image)
        except:
            print(f'Exception happened when load image id: {idx} name: {image_id}')
            image_feat = torch.zeros((3, self.image_size, self.image_size))

        image_name = image_id
        if image_name in self.region_file_dict:
            region_feats = []
            region_feat_files = self.region_file_dict[image_name]
            for file in region_feat_files:
                with open(os.path.join(self.region_feat_root, file), 'rb') as ff:
                    region_feats.append(pickle.load(ff))
        else:
            region_feats = []

        if len(region_feats) != 0:
            region_feats = torch.stack(region_feats)
            padd_feats = torch.zeros(10 - region_feats.shape[0], 1408)
            region_feats = torch.cat([region_feats, padd_feats], dim=0)
        else:
            region_feats = torch.zeros(10, 1408)

        return {
            'id': idx,
            'image_feat': image_feat,
            'full_prompt': full_prompt,
            'label': item["output"],
            'region_feat': region_feats
        }

    def collate_fn(self, batch):
        data_id = [data['id'] for data in batch]
        image_feat = [data['image_feat'] for data in batch]
        full_prompts = [data['full_prompt'] for data in batch]
        labels = [data['label'] for data in batch]

        image_feat = torch.stack(image_feat)
        region_feat = [data['region_feat'] for data in batch]
        region_feat = torch.stack(region_feat)

        return {
            'full_prompts': full_prompts,
            'labels': labels,
            'images': image_feat,
            'id': data_id,
            'region_feat': region_feat
        }