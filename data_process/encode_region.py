import os
import argparse
from gemkr.common.registry import registry
from gemkr.models.eva_vit import create_eva_vit_g
from gemkr.common.config import Config
from torch.utils.data import DataLoader, Dataset
import re
from PIL import Image
import torch
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=False, help="path to configuration file.", default='../eval_configs/gemkr_eval.yaml')
    parser.add_argument("--region_img_root", required=False, help="path to configuration file.",
                        default='/root/data1/yolov7-main/runs/detect/exp29/cropped')
    parser.add_argument("--output_feat_root", required=False, help="path to configuration file.",
                        default='/root/data1/yolov7-main/runs/detect/exp29/feat')

    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

class ELDataset_test(Dataset):

    def __init__(self, args, preprocess, image_list, region_root='/root/data1/yolov7-main/runs/detect/exp29/cropped'):
        super(ELDataset_test, self).__init__()
        self.args = args
        self.image_size = 224
        self.image_list = image_list
        self.transform = preprocess.transform
        self.region_root = region_root

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        suffix = re.sub(r'(\S+(?=\.(jpg|JPG|png|PNG|svg|SVG|gif|GIF)))|(\S+(?=\.(jpeg|JPEG)))', '', image_path)
        image_name = image_path.replace(suffix, '')
        image_name = image_name.replace('.', '')
        # tokenized_full_prompt = self.generate_and_tokenize_prompt(item)
        image_path = os.path.join(self.region_root, image_path)

        try:
            print(image_name)
            image = Image.open(image_path).convert('RGB')
            image_feat = self.transform(image)
        except:
            print(f'Exception happened when load image id: {idx} name: {image_path}')
            image_feat = torch.zeros((3, self.image_size, self.image_size))

        return {
            'image_name': image_name,
            'image_feat': image_feat,
        }


    def collate_fn(self, batch):
        image_names = [data['image_name'] for data in batch]
        image_feats = [data['image_feat'] for data in batch]
        image_feats = torch.stack(image_feats)

        return {
            'names': image_names,
            'images': image_feats,
        }

if __name__ == '__main__':
    clip_model = create_eva_vit_g(adapter_layer=0).cuda()
    clip_model = clip_model.half()
    args = parse_args()
    region_img_root = args.region_img_root
    output_region_feat_root = args.output_feat_root
    region_img_list = os.listdir(region_img_root)
    cfg = Config(args)

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    dataset = ELDataset_test(args, vis_processor, region_img_list, region_root=args.region_img_root)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False
    )
    with torch.no_grad():
        for id, data in enumerate(dataloader):
            names = data['image_name']
            images = data['image_feat'].cuda()
            feats = clip_model.forward_features(images.half())[0].cpu()
            feats = feats[:, 0, :]
            for name, feat in zip(names, feats):
                with open(os.path.join(output_region_feat_root, name + '.pkl'), 'wb') as ff:
                    pickle.dump(feat, ff)

