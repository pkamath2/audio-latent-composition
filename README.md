# Audio Latent Composition

This project is part of a paper titled "An Example-Based Framework for Perceptually Guided Audio Texture Generation" under review.
    
[Paper]() | [Demo Webpage](https://guided-control-by-prototypes.s3.ap-southeast-1.amazonaws.com/audio-guided-generation/index.html) | [Citation](#citation)
    
<img src='resources/feature-diag.png' style="background-color: #cccccc">
    
In this paper, we employ an exemplar based approach in conjunction with a pre-trained StyleGAN2 and GAN inversion techniques to find user-defined directions for semantic controllability. 
     
We generate synthetic examples based on William Gaver's "Everyday Listening" approach and find their matching real-world samples by inverting the synthetic samples from the latent space of a pre-trained StyleGAN2. These samples and their respective latent space embeddings are used to derive directional vectors to provide semantic guidance over audio texture generation. Such vectors are able to provide "synthesizer-like" continuous control while generation sounds from the latent space of the GAN.
    
This repo is adapted and modified for use with audio from Chai et al., "Using latent space regression to analyze and leverage compositionality in GAN".  [Paper](http://arxiv.org/abs/2103.10426) and [Code](https://github.com/chail/latent-composition).

### Table of Contents

* [Setup](#setup) 
* [Notebooks](#notebooks) 
* [Training](#training) 
* [Streamlit Interfaces](#interfaces) 
* [Citing this work](#citation)
     
### Setup
* Clone this repo
* Install dependencies by creating a new conda environment called ```audio-latent-composition```
```
conda env create -f environment.yml
```
Add the newly created environment to Jupyter Notebooks
```
python -m ipykernel install --user --name audio-latent-composition
```
    
### Notebooks
Notebooks outline how to generate synthetic Gaver sounds (see paper for algorithms) and invert them to real-world audio. Directional vectors generated in the notebooks can be used to edit any randomly generated audio sample.

* [Notebook to demonstrate perceptual guidance for Greatest Hits](perceptually_guided_generation/greatesthits-guidance.ipynb)
* [Notebook to demonstrate perceptual guidance for Water Filling](perceptually_guided_generation/water-guidance.ipynb)
* [Notebook to demonstrate Selective Attribute Transfer for Greatest Hits](perceptually_guided_generation/greatesthits-timbrepicker.ipynb)

     

### Training
We use pre-trained StyleGAN2 on audio textures of the [Greatest Hits Dataset](https://andrewowens.com/vis/) and [Water Filling a Container](https://animatedsound.com/ismir2022/metrics/). All StyleGAN2 checkpoints are downloaded when you run the notebooks in the section above.

Kickstart training of encoder. See [config.json](config/config.json) for various parameter settings.
```
python -m training.train_sgan_encoder
```
     
### Interfaces
We demonstrate the ease of using the directional vectors developed using this method to edit randomly generated samples by actualizing the vectors as sliders on a web-interface.   

The interfaces are developed using [Streamlit](https://streamlit.io/)  

To run the interface to generate Gaver sounds and perform analysis-by-synthesis in the latent space of the StyleGAN2 (as shown in the demo video) - 
```
cd interface
streamlit run gaver-sounds-interface.py
```

To perceptually edit any randomly generated samples for Greatest Hits dataset - 
```
streamlit run dim-control-interface-greatesthits.py
```
   
To perceptually edit any randomly generated samples for Water dataset -    
```
streamlit run dim-control-interface-water.py
```
    
### Citation
If you use this code for your research please cite as:
```
@ARTICLE{kamath2024example,
  author={Kamath, Purnima and Gupta, Chitralekha and Wyse, Lonce and Nanayakkara, Suranga},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Example-Based Framework for Perceptually Guided Audio Texture Generation}, 
  year={2024},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TASLP.2024.3393741}
}
```

