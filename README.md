<div align="center">
    <a href="https://github.com/Weixiang-Sun/Bora/stargazers"><img src="https://img.shields.io/github/stars/Weixiang-Sun/Bora?style=social"></a>
    <a href="https://weixiang-sun.github.io/Bora/"><img src="https://img.shields.io/badge/Gallery-View-orange?logo=&amp"></a>
    <a href="https://huggingface.co/Sweson/Bora"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>
</div>

## Bora: Biomedical Generalist Video Generation Model
**Abstract:** Generative models hold promise for revolutionizing medical education, robot-assisted surgery, and data augmentation for medical AI development. Diffusion models can now generate realistic images from text prompts, while recent advancements have demonstrated their ability to create diverse, high-quality videos. However, these models often struggle with generating accurate representations of medical procedures and detailed anatomical structures. This paper introduces Bora, the first spatio-temporal diffusion probabilistic model designed for text-guided biomedical video generation. Bora leverages Transformer architecture and is pre-trained on general-purpose video generation tasks. It is fine-tuned through model alignment and instruction tuning using a newly established medical video corpus, which includes paired text-video data from various biomedical fields. To the best of our knowledge, this is the first attempt to establish such a comprehensive annotated biomedical video dataset. Bora is capable of generating high-quality video data across four distinct biomedical domains, adhering to medical expert standards and demonstrating consistency and diversity. This generalist video generative model holds significant potential for enhancing medical consultation and decision-making, particularly in resource-limited settings. Additionally, Bora could pave the way for immersive medical training and procedure planning. Extensive experiments on distinct medical modalities such as endoscopy, ultrasound, MRI, and cell tracking validate the effectiveness of our model in understanding biomedical instructions and its superior performance across subjects compared to state-of-the-art generation models.

## 📰 News
- **[2024.6.19]** We release **Bora**, a video generation model specificaly for biomedical domain.

## 🎥 Some Demo
| Endoscopy | Ultrasound | RT-MRI | Cell |
| ------ | ------ | ------ | ------ |
| <img src="examples/endo/sample_0.gif" width=""> | <img src="examples/uls/sample_1.gif" width=""> | <img src="examples/mri/sample_1.gif" width=""> | <img src="examples/cell/sample_0.gif" width=""> |
| <img src="examples/endo/sample_4.gif" width=""> | <img src="examples/uls/sample_6.gif" width=""> | <img src="examples/mri/sample_2.gif" width=""> | <img src="examples/cell/sample_4.gif" width=""> |
| <img src="examples/endo/sample_6.gif" width=""> | <img src="examples/uls/sample_8.gif" width=""> | <img src="examples/mri/sample_3.gif" width=""> | <img src="examples/cell/sample_7.gif" width=""> |

## Contents
- [Installation](#installation)
- [Prepare](#prepare)
- [Inference](#inference)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contribution](#contribution)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Installation
```bash
# create a virtual env
conda create -n bora python=3.10
# activate virtual environment
conda activate bora
# install torch
# We recommend torch==2.2.2 under CUDA12.1
pip install torch torchvision

# install flash attention
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex
# We recommend install from source
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# install xformers
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121

# install opensora
pip install -v .
```

## Prepare
Before running, besides Bora's weights, you also need to download the weights for the VAE and Text Encoder. We have provided all the links in the table below:
|Bora|Video Encoder|Text Encoder|
|----|----|----|
|[Bora](https://huggingface.co/Sweson/Bora)|[VAE](https://huggingface.co/stabilityai/sd-vae-ft-ema)|(T5)[https://huggingface.co/DeepFloyd/t5-v1_1-xxl]|

## Inference


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Weixiang-Sun/Bora&type=Date)](https://star-history.com/#Weixiang-Sun/Bora&Date)