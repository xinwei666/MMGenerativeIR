import argparse
import random
from transformers import GenerationConfig
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from gemkr.common.config import Config
from gemkr.common.registry import registry
import json
from tqdm import tqdm
from gemkr.datasets.datasets.wikidiverse_datasets import ELDataset_test
from torch.utils.data import DataLoader
from seal import FMIndex

def _prefix_allowed_token_fn(batch_id, sent):
    sent_list = sent.tolist()

    if len(sent_list) == 1:
        next_seq = index.occurring_distinct
    else:
        next_seq= index.get_continuations(sent_list[1:])
    if len(sent_list) >= 10:
        return [2]
    return next_seq

class Evaluator:
    def __init__(self, model, tokenizer, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model.to(self.device)

        self.vis_processor = vis_processor
        self.llama_tokenizer = tokenizer


    def evaluate(self, test_dataloader, ckp_path):
        self.model.eval()
        test_bar = tqdm(test_dataloader)

        predictions = []

        p5 = 0
        r5 = 0
        r10 = 0

        for id, batch in enumerate(test_bar):
            answer = batch['labels'][0]
            outputs = self.answer(batch)
            predictions.append(outputs)

            predicts = []
            for pred in outputs:
                if pred not in predicts:
                    predicts.append(pred)

            predicts = [id2text[id] for id in predicts]
            result = []
            for pred in predicts:
                kk = 0
                for ans in answer:
                    if ans in pred:
                        kk = 1
                        break
                result.append(kk)

            p5 += sum(result[0: 5])
            r5 += 1 if sum(result[0: 5]) >= 1 else 0
            r10 += 1 if sum(result[0: 10]) >= 1 else 0

            test_bar.desc = f"test iter [{id}] P@5= {p5 / len(predictions) / 5:.3f}, R@5= {r5 / len(predictions) :.3f}, R@10={r10 / len(predictions):.3f}"

        with open(f"./results/results_okvqa_{ckp_path}_p5: {p5 / len(predictions) / 5:.3f}.json", "w", encoding='utf-8') as f:
            json.dump(predictions, f)

    def answer(self, batch, num_beams=10,top_p=0.75, top_k=40, temperature=1.0):

        full_prompts = batch['full_prompts']
        # print("Input: %s" % full_prompts)
        images = batch['images'].to(self.device)
        region_feat = batch["region_feat"].to(self.device).half() if "region_feat" in batch else None
        p_before = []
        p_after = ''
        for prompt in full_prompts:
            aaa = prompt.split('<ImageHere>')
            p_before.append(aaa[0])
            p_after = aaa[1]

        self.llama_tokenizer.padding_side = "left"
        p_before_tokens = self.llama_tokenizer(p_before,
                                                return_tensors="pt",
                                                padding="longest",
                                                truncation=True,
                                                max_length=256,
                                                add_special_tokens=True).to(self.device)
        batch_size = len(p_before)
        p_after_tokens = self.llama_tokenizer(p_after,
                                               return_tensors="pt",
                                               add_special_tokens=False).to(self.device)
        p_before_embeds = self.model.llama_model.base_model.model.model.embed_tokens(p_before_tokens.input_ids)  # .expand(batch_size, -1, -1)
        p_after_embeds = self.model.llama_model.base_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

        with torch.no_grad():
            img_embeds, _ = self.model.encode_img(images, region_feat=region_feat)

        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            num_beam_groups=10,
            diversity_penalty=0.5,
        )

        # Without streaming
        with torch.no_grad():

            generation_output = self.model.llama_model.generate(
                inputs_embeds=wrapped_img_embeds,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256,
                # stopping_criteria=self.stopping_criteria,
                length_penalty=1.5,
                num_beams=num_beams,
                eos_token_id=2,
                pad_token_id=0,
                bos_token_id=1,
                prefix_allowed_tokens_fn=_prefix_allowed_token_fn,
                num_return_sequences = num_beams,
            )

        sentences = generation_output.sequences
        aa = [self.llama_tokenizer.decode(o) for o in sentences]
        outputs = []
        for sent in sentences:
            idx13 = [loc for loc, val in enumerate(sent) if val == 2]
            idx13 = max(idx13)
            if idx13 != len(sent) - 1:
                outputs.append(sent[1:idx13 + 1])
            else:
                outputs.append(sent[1:idx13])

        labels = []
        for output in outputs:
            label_idx = [index.labels[dd] for dd in index.get_doc_indices(output.tolist())]
            labels.append(label_idx[0])

        return labels


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=False, help="path to configuration file.", default='./eval_configs/gemkr_eval.yaml')
    parser.add_argument("--data_path", required=False, help="path to configuration file.",
                        default="./dataset/okvqa/test.json")
    parser.add_argument("--img_root", required=False, help="path to configuration file.",
                        default="/root/datasets/MSCOCO/coco2014/validation/data")
    parser.add_argument("--region_feat_path", required=False, help="path to configuration file.",
                        default="/root/data1/yolov7-main/runs/detect/exp29/feat")
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


def setup_seeds(config):
    seed = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


if __name__ == '__main__':

    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    ckpt = model_config.ckpt
    ckpt = ckpt.split('/')[-2]
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    tokenizer = model.llama_tokenizer
    tokenizer.padding_side = "left"
    model.llama_model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.llama_model.config.bos_token_id = 1
    model.llama_model.config.eos_token_id = 2
    tokenizer.bos_token_id = model.llama_model.config.bos_token_id
    tokenizer.eos_token_id = model.llama_model.config.eos_token_id

    global index
    index = FMIndex.load('./index/okvqa_corpus12k.fm_index')
    label = index.labels

    with open("./index/id2text.json", 'r', encoding='utf-8') as f:
        global id2text
        id2text = json.load(f)

    test_dataset = ELDataset_test(args, tokenizer, vis_processor)
    loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        drop_last=False,
    )

    ckp_path = ckpt

    evaluator = Evaluator(model, tokenizer, vis_processor, device='cuda:{}'.format(args.gpu_id))
    evaluator.evaluate(test_dataloader=loader, ckp_path=ckp_path)

