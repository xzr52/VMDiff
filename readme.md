
<div align="center">

<h1> VMDiff: Visual Mixing Diffusion for Limitless Cross-Object Synthesis </h1>

Zeren Xiong<sup>1</sup> · Yue Yu<sup>1</sup> · Zedong Zhang<sup>1</sup> · Shuo Chen<sup>2</sup> · Jian Yang<sup>1,2</sup> · Jun Li<sup>1,2*</sup>  

<sup>1</sup>Nanjing University of Science and Technology · <sup>2</sup>Nanjing University 

<a href="https://arxiv.org/abs/2509.23605"><img src="https://img.shields.io/badge/arXiv-2509.23605-b31b1b.svg" height=20.5></a>
<a href="https://xzr52.github.io/VMDiff_index/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
</div>


<div>
  Creating novel images by fusing visual cues from multiple sources is a fundamental yet underexplored problem in image-to-image generation, with broad applications in artistic creation, virtual reality, and visual media. Existing methods often face two key challenges: <em>coexistent generation</em>, where multiple objects are simply juxtaposed without true integration, and <em>bias generation</em>, where one object dominates the output due to semantic imbalance. To address these issues, we propose <strong>Visual Mixing Diffusion (VMDiff)</strong>, a simple yet effective diffusion-based framework that synthesizes a single, coherent object by integrating two input images at both noise and latent levels. Our approach comprises (1) a <strong>hybrid sampling process</strong> that combines guided denoising, inversion, and spherical interpolation with adjustable parameters to achieve structure-aware fusion, mitigating coexistent generation; and (2) an <strong>efficient adaptive adjustment</strong> module that introduces a novel similarity-based score to automatically and adaptively search for optimal parameters, countering semantic bias. Experiments on a curated benchmark of <strong>780 concept pairs</strong> demonstrate that our method outperforms strong baselines in visual quality, semantic consistency, and human-rated creativity.
</div>



<!-- ## 🚀 **News**
- **2024.12.20**: 🎉 Our code is released! Explore the possibilities of novel object synthesis with our framework. -->



## 🛠️ **1. Set Environment**

To set up the environment for running the code, follow these steps:

1. Clone the repository:
    ```bash
    git clone git@github.com:xzr52/VMDiff.git
    cd VMDiff
    ```

2. Create a conda environment and install dependencies:
    ```bash
    conda create -n VMDiff python=3.11
    conda activate VMDiff
    pip install -r requirements.txt
    ```

3. Set CUDA paths:
    ```bash
    export CUDA_HOME=/usr/local/cuda
    ```

4. Install the required submodule:
    ```bash
    cd GroundingDino
    pip install -e .
    ```
<!-- git clone https://github.com/IDEA-Research/GroundingDINO.git -->
5. Download the segmentation model weights [seg_ckpts](https://drive.google.com/file/d/12TP528_6FfDRSeczkHuZbMDMG8flRgb6/view?usp=drive_link), unzip them, and place them in the ckpts/ folder.


## 🚀 **2. Quick Start (Inference on an Image Folder)**

VMDiff can run in **two modes** depending on your GPU memory:

### ✅ Mode A: Single GPU (Full pipeline on one GPU)
To run the **complete inference pipeline on a single GPU**, you typically need **~48GB VRAM** (recommended).  
This mode is simplest, but may **OOM** on 24GB/32GB cards.
```python
python run.py --input_dir examples --output_dir Fuse_results  --device_mode single 
```
- `--image_path`: /path/to/images.
- `--output_dir`: /path/to/save.
### ✅ Mode B: Two GPUs (Recommended for 2×24GB)
If you have two 24GB GPUs, we strongly recommend using split mode.
In this mode, the pipeline is distributed across two GPUs to reduce peak VRAM on each card, making inference stable on 24GB devices.
```python
python run.py --input_dir examples --output_dir Fuse_results  --device_mode split 
```
- `--image_path`: /path/to/images.
- `--output_dir`: /path/to/save.
## 🎨 **3. Inference on a Single Image Pair**
To perform inference on image floder, use the following command:
```python
python run_single.py --img1 examples/astronaut2figurine.png --img2 examples/charizard2figurine.png --output_dir Fuse_results -device_mode single
```
- `--image_path`: /path/to/images.
- `--output_dir`: /path/to/save.





</p>
</table>
</details>



# 🙌 Acknowledgment

This work was supported by National Natural Science Fund of China (Nos. U24A20330, 62361166670, 62506155), Provincial Natural Science Fund of Jiangsu (Nos. BK20251985), and Suzhou Municipal Leading Talents Fund (Nos. ZXL2025320). We also thank the developers of the following projects, which our implementation builds upon:

- [Diffusers](https://github.com/huggingface/diffusers)  
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)  
- [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)  
- [FLUX](https://github.com/black-forest-labs/flux)
- [FLUX-Krea](https://github.com/krea-ai/flux-krea)

We deeply appreciate their contributions , which have been instrumental to our work.



# 📖BibTeX
```
@article{xiong2025vmdiff,
  title={VMDiff: Visual Mixing Diffusion for Limitless Cross-Object Synthesis},
  author={Xiong, Zeren and Yu, Yue and Zhang, Zedong and Chen, Shuo and Yang, Jian and Li, Jun},
  journal={arXiv preprint arXiv:2509.23605},
  year={2025}
}
```