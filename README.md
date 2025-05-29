<p align="center">
<h1 align="center"><strong> OmniMap: A Comprehensive Mapping Framework Integrating Optics, Geometry, and Semantics</strong></h1>
</p>



<p align="center">
  <a href="https://omnimap123.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-👔-green?">
  </a>
</p>


 ## 🏠  Abstract
Robotic systems demand accurate and comprehensive 3D environment perception, requiring simultaneous capture of a comprehensive representation of photo-realistic appearance (optical), precise layout shape (geometric), and open-vocabulary scene understanding (semantic). Existing methods typically achieve only partial fulfillment of these requirements while exhibiting optical blurring, geometric irregularities, and semantic ambiguities. To address these challenges, we propose OmniMap. Overall, OmniMap represents the first online mapping framework that simultaneously captures optical, geometric, and semantic scene attributes while maintaining real-time performance and model compactness. At the architectural level, OmniMap employs a tightly coupled 3DGS–Voxel hybrid representation that combines fine-grained modeling with structural stability. At the implementation level, OmniMap identifies key challenges across different modalities and introduces several innovations: adaptive camera modeling for motion blur and exposure compensation, hybrid incremental representation with normal constraints, and probabilistic fusion for robust instance-level understanding. Extensive experiments show OmniMap’s superior performance in rendering fidelity, geometric accuracy, and zero-shot semantic segmentation compared to state-of-the-art methods across diverse scenes. The framework’s versatility is further evidenced through a variety of downstream applications including multi-domain scene Q&A, interactive edition, perception-guided manipulation, and map-assisted navigation.

<img src="https://omnimap123.github.io/static/images/poster.png">


## TODO List

- [x] Anonymous open-source training code
- [ ] Anonymous interactive visualization code
- [ ] Anonymous open-source downstream task code


## 🛠  Install

### Clone this repo

```bash
git clone https://github.com/omnimap123/anonymous_code.git
cd anonymous_code
```

### Install the required libraries
Use conda to install the required environment. To avoid problems, it is recommended to follow the instructions below to set up the environment.


```bash
conda env create -f environment.yml
conda activate omnimap
```

###  Install YOLO-World Model
Follow the [instructions](https://github.com/AILab-CVC/YOLO-World#1-installation) to install the YOLO-World model and download the pretrained weights [YOLO-Worldv2-L (CLIP-Large)](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth).

###  Install TAP Model
Follow the [instructions](https://github.com/baaivision/tokenize-anything?tab=readme-ov-file#installation) to install the TAP model and download the pretrained weights [here](https://github.com/baaivision/tokenize-anything?tab=readme-ov-file#models).


###  Install SBERT Model
```bash
pip install -U sentence-transformers
```
Download pretrained weights
```bash
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

### Modify the model path

Change the address of the above model in the configuration file in ```omnimap/config/```.

## 📊 Prepare dataset
OmniMap has completed validation on Replica (as same with [vMap](https://github.com/kxhit/vMAP)) and Scannet. 
Please download the following datasets.

* [Replica Demo](https://huggingface.co/datasets/kxic/vMAP/resolve/main/demo_replica_room_0.zip) - Replica Room 0 only for faster experimentation.
* [Replica](https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip) - All Pre-generated Replica sequences.
* [ScanNet](https://github.com/ScanNet/ScanNet) - Official ScanNet sequences.



## 🏃 Run


### Main Code
Run the following command to start the formal execution of the incremental mapping.

```bash
# for replica
python demo.py  --dataset replica --scene {scene} --vis_gui
# for scannet
python main.py  --dataset scannet --scene {scene} --vis_gui
```

You can use ```--start {start_id}``` and ```--length {length}``` to specify the starting frame ID and the mapping duration, respectively. The ```--vis_gui``` flag controls online visualization; disabling it may improve processing speed.

After building the map, the results will be saved in folder ```outputs/{scene}```, which contains the rendered outputs and evaluation metrics.

### Gen 3D Mesh
We use the rendered depth and color images to generate the color mesh. You can run the following code to perform this operation.

```bash
# for replica
python tsdf_integrate.py --dataset replica --scene {scene}
# for scannet
python tsdf_integrate.py --dataset scannet --scene {scene}
```
