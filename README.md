# Spoken-Video-Grounding
This is a Pytorch implementation for MM 2022 paper "Video-Guided Curriculum Learning for Spoken Video Grounding".

## Introduction
We introduce a new task, spoken video grounding (SVG), which aims to localize the desired video fragments from spoken language descriptions. Compared with using text, employing audio requires the model to directly exploit the useful phonemes and syllables related to the video from raw speech. Moreover, we randomly add environmental noises to this speech audio, further increasing the difficulty of this task and better simulating real applications. To rectify the discriminative phonemes and extract video-related information from noisy audio, we develop a novel video-guided curriculum learning (VGCL) during the audio pre-training process, which can make use of the vital visual perceptions to help understand the spoken language and suppress the external noise. Considering during inference the model can not obtain ground truth video segments, we design a curriculum strategy that gradually shifts the input video from the ground truth to the entire video content during pre-training. Finally, the model can learn how to extract critical visual information from the entire video clip to help understand the spoken language. In addition, we collect the first large-scale spoken video grounding dataset based on ActivityNet, which is named as ActivityNet Speech dataset.


![SVG](figs/task_demo.png)


## Prerequisites

This package has the following requirements:

* `Python 3.7.6`
* `Pytorch 1.9.0`
* `CUDA 11.4`
* `h5py 2.10.0` 
* `numpy 1.21.5`

## Data preparation
The visual features of ActivityNet Caption can be downloaded from [here](https://cs.stanford.edu/people/ranjaykrishna/densevid/).

Download the pretrained glove models:

```bash
wget http://nlp.stanford.edu/data/glove.42B.300d.zip
unzip glove.42B.300d.zip
```

The spoken audio dataset is now available from [here](https://drive.google.com/file/d/11f6sC94Swov_opNfpleTlVGyLJDFS5IW/view?usp=sharing). 


## Training
```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```


