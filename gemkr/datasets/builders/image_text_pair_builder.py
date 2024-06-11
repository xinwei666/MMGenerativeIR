import os
import logging
import warnings
from transformers import LlamaTokenizer
from gemkr.common.registry import registry
from gemkr.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from gemkr.datasets.datasets.laion_dataset import LaionDataset
from gemkr.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from torch.utils.data import Dataset, ConcatDataset
from gemkr.datasets.datasets.wikidiverse_datasets import ELDataset_train, ELDataset_eval
@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets

#
# class EL_dataset(Dataset):
#
#     def __init__(self, data_file, ):

@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        datasets = dict()
        llamatokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        train_dataset = ELDataset_train(self.config, llamatokenizer, self.vis_processors)
        datasets['train'] = train_dataset
        eval_dataset = ELDataset_eval(self.config, llamatokenizer, self.vis_processors)
        datasets['eval'] = eval_dataset

        # build_info = self.config.build_info
        # storage_path = build_info.storage
        #
        # datasets = dict()
        #
        # if not os.path.exists(storage_path):
        #     warnings.warn("storage path {} does not exist.".format(storage_path))
        #
        # # create datasets
        # dataset_cls = self.train_dataset_cls
        # datasets['train'] = dataset_cls(
        #     vis_processor=self.vis_processors["train"],
        #     text_processor=self.text_processors["train"],
        #     ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
        #     vis_root=os.path.join(storage_path, 'image'),
        # )

        return datasets
