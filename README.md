# GeMKR 
Official code of ''Generative Multi-Modal Knowledge Retrieval with Large Language Models''

<p align="left">
<a href="https://arxiv.org/abs/2401.08206" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2405.05615-b31b1b.svg?style=flat" /></a>
</p>

## Environment
```shell script
conda create -n gemkr python==3.9
conda activate gemkr
pip install -r requirements.txt
```

## TODO 
- [x] Code of experiments on LLaMA-1.
- [ ] Code of extensive version on the advanced MLLMs.
- [ ] Code of extensive experiments on downstream tasks, such as knowledge-based VQA.


## Preparation
### Raw data and model weights
For the raw OKVQA-GS112K dataset, which includes the corpus, train, and dev-test files, please refer to the [official repo](https://arxiv.org/abs/2109.04014) of VRR. Or you can directly use the data that we have already saved in the './dataset' directory.

For the weights of LLaMA, please refer to the [official form](https://forms.gle/jk851eBVbX1m5TAv5) or unofficial HuggingFace repo [LLaMA-7B](https://huggingface.co/nyanko7/LLaMA-7B/tree/main).

### Region features
To obtain the RoIs of each image, we used [YOLOv7](https://arxiv.org/abs/2207.02696) and have shared our cropped ROIs files at this [Link](https://arxiv.org/abs/2207.02696). Next, please download the files and use the following command to encode the features of each ROI using the CLIP model (it should take no more than 30 minutes).
```
python ./data_process/encode_region.py --region_img_root your_roi_file_path --output_feat_root your_save_path
```

### FM-Index Initialization
You can run the following command to generate the corresponding FM-Index, which includes three files (*.fmi, *.oth, and id2text.json). Before running the command, please specify the locations of the corpus and Llama tokenizer in the './data_process/my_build_fm_index.py' file. 
```
python ./data_process/my_build_fm_index.py
```
**Alternatively, you can directly obtain them from this [Link](https://arxiv.org/abs/2207.02696). Please save these three files in the './index' directory.**

If you need the FM-index files for Wiki-21M, please contact us via email (longxw22@mails.tsinghua.edu.cn). Due to Google Drive storage limitations, we will provide it to you in a different way.

## Fine-tuning and Evaluation

### Fine-tuning
```
python train.py
```

### Evaluation
```
python eval.py
```
If you want to directly evaluate our model using the checkpoint we provided, please download the checkpoint from this [Link](https://arxiv.org/abs/2207.02696).
**If you want to directly obtain our retrieval results without running our model**, we can provide you with the final JSON-formatted result file (No more than 10 documents retrieved for each question). Please contact us via email.


## Acknowledgements

- [SEAL](https://arxiv.org/abs/2204.10628)
- [Minigpt-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [VRR](https://arxiv.org/abs/2109.04014)
- [YOLOv7](https://arxiv.org/abs/2207.02696)


## Citation
If you are using GeMKR in your research or applications, please cite using this BibTex:

```bibtex
@article{Long_Zeng_Meng_Ma_Zhang_Zhou_Zhou_2024, title={Generative Multi-Modal Knowledge Retrieval with Large Language Models}, volume={38}, url={https://ojs.aaai.org/index.php/AAAI/article/view/29837}, DOI={10.1609/aaai.v38i17.29837}, number={17}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Long, Xinwei and Zeng, Jiali and Meng, Fandong and Ma, Zhiyuan and Zhang, Kaiyan and Zhou, Bowen and Zhou, Jie}, year={2024}, month={Mar.}, pages={18733-18741} }
```
