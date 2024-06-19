## Bora: Biomedical Generalist Video Generation Model
**Abstract:** Generative models hold promise for revolutionizing medical education, robot-assisted surgery, and data augmentation for medical AI development. Diffusion models can now generate realistic images from text prompts, while recent advancements have demonstrated their ability to create diverse, high-quality videos. However, these models often struggle with generating accurate representations of medical procedures and detailed anatomical structures. This paper introduces Bora, the first spatio-temporal diffusion probabilistic model designed for text-guided biomedical video generation. Bora leverages Transformer architecture and is pre-trained on general-purpose video generation tasks. It is fine-tuned through model alignment and instruction tuning using a newly established medical video corpus, which includes paired text-video data from various biomedical fields. To the best of our knowledge, this is the first attempt to establish such a comprehensive annotated biomedical video dataset. Bora is capable of generating high-quality video data across four distinct biomedical domains, adhering to medical expert standards and demonstrating consistency and diversity. This generalist video generative model holds significant potential for enhancing medical consultation and decision-making, particularly in resource-limited settings. Additionally, Bora could pave the way for immersive medical training and procedure planning. Extensive experiments on distinct medical modalities such as endoscopy, ultrasound, MRI, and cell tracking validate the effectiveness of our model in understanding biomedical instructions and its superior performance across subjects compared to state-of-the-art generation models.

## 📰 News
- **[2024.6.19]** We release **Bora**, a video generation model specificaly for biomedical domain.

## 🎥 Some Demo
| Endoscopy | Ultrasound | RT-MRI | Cell |
| ------ | ------ | ------ | ------ |
| <img src="examples/endo/sample_0.gif" width=""> | <img src="example/uls/sample_1.gif" width=""> | <img src="example/mri/sample_1.gif" width=""> | <img src="example/cell/sample_0.gif" width=""> |
| <img src="examples/endo/sample_4.gif" width=""> | <img src="example/uls/sample_6.gif" width=""> | <img src="example/mri/sample_2.gif" width=""> | <img src="example/cell/sample_4.gif" width=""> |
| <img src="examples/endo/sample_6.gif" width=""> | <img src="example/uls/sample_8.gif" width=""> | <img src="example/mri/sample_3.gif" width=""> | <img src="example/cell/sample_7.gif" width=""> |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Weixiang-Sun/Bora&type=Date)](https://star-history.com/#Weixiang-Sun/Bora&Date)