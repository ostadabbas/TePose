#!/usr/bin/env bash

mkdir -p data
cd data
gdown "https://drive.google.com/file/d/16z2zLaEuuQ33YIfYwg_3P4KYaYLWmPob/view?usp=sharing"
unzip base_data.zip
rm base_data.zip
cd ..
mv data/base_data/merged_courtyard_basketball_01.mp4 .
mkdir -p $HOME/.torch/models/
mv data/base_data/yolov3.weights $HOME/.torch/models/
