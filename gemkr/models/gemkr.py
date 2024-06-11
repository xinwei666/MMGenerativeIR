import logging
import random
from peft import PeftModel
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from gemkr.models.eva_vit import PadPrompter
from gemkr.common.registry import registry
from gemkr.models.blip2 import Blip2Base, disabled_train
from gemkr.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from transformers.activations import get_activation

class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)

class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, input_dim, out_dim, non_linearity="gelu_new", reduction_factor=2):
        super().__init__()
        # self.config = config
        self.input_dim = input_dim
        reduction_factor = reduction_factor
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = Activations(non_linearity.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)
        self.up_sampler = nn.Linear(self.down_sample_size, out_dim)

        # self.track_z = config.track_z

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        # if self.track_z:
        #     self.z = z
        output = self.up_sampler(z)
        return output

@registry.register_model("gemkr")
class GeMKR(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/gemkr.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        llama_model="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        adapter_layer=20
    ):
        super().__init__()

        # self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        # self.visual_prompt = PadPrompter(image_size=img_size, prompt_size=30)

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, adapter_layer=adapter_layer
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")

        for name, param in self.visual_encoder.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
                param.data = param.data.float()

        for name, param in self.visual_encoder.blocks[-1 * adapter_layer:].named_parameters():
            if 'gate' in name or 'adapter' in name:
                param.data = param.data.float()
                param.requires_grad = True

        print('Loading VIT Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "left"
        self.llama_tokenizer.pad_token_id = 0  # unk
        self.llama_tokenizer.bos_token_id = 1
        self.llama_tokenizer.eos_token_id = 2


        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "v_proj",
                ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        #if True:
        self.llama_model = get_peft_model(self.llama_model, lora_config)
        # else:
        #     lora_path = "/root/data1/minigpt4_okvqa_mm/minigpt4/output/minigpt4_stage2_finetune/20240521213"
        #     # lora_path = '/root/data1/minigpt4_remuq_mm/minigpt4/output/minigpt4_stage2_finetune/20230728011'
        #     # lora_path = '/root/data1/lora_llama_remuq/outputs/lora-alpaca_okvqa_lora8_0805'
        #     self.llama_model = PeftModel.from_pretrained(
        #         self.llama_model,
        #         lora_path,
        #         device_map={"": "cuda:0"},
        #         # lora_config=lora_config
        #     # torch_dtype=torch.float16,
        #     )
        #     print("load lora weight from %s" % lora_path)

        # for name, param in self.llama_model.named_parameters():
        #     param.requires_grad = False

        # for name, param in self.llama_model.named_parameters():
        #     param.requires_grad = False
        print('Loading LLAMA Done')

        # self.llama_proj = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(1408, 768), #
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(768, self.llama_model.config.hidden_size),
        # )
        self.llama_proj = nn.Linear(
            1408, self.llama_model.config.hidden_size
        )

        # self.llama_proj = Adapter(1408, 4096)

        # self.llama_proj = nn.Linear(
        #     self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        # )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        self.llama_model.config.pad_token_id = self.llama_tokenizer.pad_token_id = 0  # unk
        self.llama_model.config.bos_token_id = 1
        self.llama_model.config.eos_token_id = 2

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image, region_feat=None):
        device = image.device
        if False:# self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        # image = self.visual_prompt(image)

        with self.maybe_autocast():
            # image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            (image_embeds, region_embeds) = self.visual_encoder(image, region_feat=region_feat)  #.unsqueeze(1).to(device)
            image_embeds = image_embeds[:, 0, :].unsqueeze(1)
            if region_embeds is not None:
                region_embeds = region_embeds[:, :2, :]
                image_embeds = torch.cat([image_embeds, region_embeds], dim=1)
                image_embeds = self.ln_vision(image_embeds)

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            # query_output = self.Qformer.bert(
            #     query_embeds=query_tokens,
            #     encoder_hidden_states=image_embeds,
            #     encoder_attention_mask=image_atts,
            #     return_dict=True,
            # )
            # query_output = query_output.last_hidden_state #[:, :8, :]
            inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, full_prompt):
        # if prompt:
        batch_size = img_embeds.shape[0]
        # p_before, p_after = full_prompt.split('<ImageHere>')
        p_before = []
        p_after = ''
        for prompt in full_prompt:
            aaa = prompt.split('<ImageHere>')
            p_before.append(aaa[0])
            p_after= aaa[1]

        self.llama_tokenizer.padding_side = "left"
        p_before_tokens = self.llama_tokenizer(p_before,
                                               return_tensors="pt",
                                               padding="longest",
                                               truncation=True,
                                               max_length=256,
                                               add_special_tokens=True).to(img_embeds.device)

        p_after_tokens = self.llama_tokenizer(p_after,
                                              return_tensors="pt",
                                              add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.llama_model.base_model.model.model.embed_tokens(p_before_tokens.input_ids) # .expand(batch_size, -1, -1)
        p_after_embeds = self.llama_model.base_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        # p_before_embeds = self.llama_model.base_model.embed_tokens(
        #     p_before_tokens.input_ids)  # .expand(batch_size, -1, -1)
        # p_after_embeds = self.llama_model.base_model.embed_tokens(p_after_tokens.input_ids).expand(
        #     batch_size, -1, -1)

        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        # wrapped_img_embeds = torch.cat([p_before_embeds, p_after_embeds], dim=1)
        # wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        wrapped_atts_img = torch.cat([p_before_tokens['attention_mask'], atts_img, p_after_tokens['attention_mask'].expand(batch_size, -1)], dim=1)
        # wrapped_atts_img = torch.cat(
        #     [p_before_tokens['attention_mask'], p_after_tokens['attention_mask'].expand(batch_size, -1)],
        #     dim=1)
        return wrapped_img_embeds, wrapped_atts_img
        # else:
        #     return img_embeds, atts_img

    def forward(self, samples):
        image = samples["images"]
        region_feat = samples["region_feat"].to(self.device).half() if "region_feat" in samples else None
        img_embeds, atts_img = self.encode_img(image, region_feat=region_feat)
        full_prompt = samples['full_prompts']
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, full_prompt)

        text = [t for t in samples["labels"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            # return_tensors="pt",
            # padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
            # add_eos_token=True,
            # add_bos_token=False
        )#.to(image.device)

        trt_input_ids = to_regress_tokens['input_ids']
        trt_attention_mask = to_regress_tokens['attention_mask']
        trt_attention_mask = [[1] + data for data in trt_attention_mask]
        trt_input_ids = [data + [2] for data in trt_input_ids]
        trt_lens = [len(data) for data in trt_input_ids]
        trt_max_lens = max(trt_lens)
        to_regress_token_input_ids = [trt_input_ids[i] + [0] * (trt_max_lens - trt_lens[i]) for i in range(len(trt_input_ids))]
        trt_attention_mask = [trt_attention_mask[i] + [0] * (trt_max_lens - trt_lens[i]) for i in range(len(trt_attention_mask))]

        to_regress_token_input_ids = torch.LongTensor(to_regress_token_input_ids).to(image.device)
        trt_attention_mask = torch.FloatTensor(trt_attention_mask).to(image.device)

        targets = to_regress_token_input_ids.masked_fill(
            to_regress_token_input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]],
                       dtype=torch.long).to(image.device).fill_(-100)  ### plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]

        to_regress_embeds = self.llama_model.base_model.model.model.embed_tokens(to_regress_token_input_ids)
        # to_regress_embeds = self.llama_model.base_model.embed_tokens(to_regress_token_input_ids)
        # inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        inputs_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_img, trt_attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        llama_model = cfg.get("llama_model")
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of Checkpoint for evaluation
        if ckpt_path:
            print("Load GeMKR Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
