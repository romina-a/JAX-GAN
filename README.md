# Top-k Training of GANs

The goal of the project is to reproduce the results of 
*Top-k Training of GANs: Improving GAN Performance by Throwing Away Bad
Samples* https://arxiv.org/abs/2002.06224

# ------------***Vanilla GAN***-----------

# Training the GANs on Toy Tasks

To compare, train with ```--top_k 0``` and ```--top_k 1```

See ```python StaxToyGAN.py -h``` for options

The saved model will have "< #modes >-< variance >-< (no)topk >.pkl" naming convention.

A model with "-intermediate" to its name will also be saved which is the model in the middle of training. This is used for gradient update analysis.
````shell
StaxToyGAN.py --num_components 5 --top_k 0 --save_adr_plots_folder "./output_ims/experiment23/" --save_adr_model_folder "./Models/experiment23/" --seed 20 --batch_size 256 --batch_size_min 192 --num_iter 100000 --dataset 'circle'
````

# Testing trained GANs on Toy Tasks:

Trained models are in ./Trained_Models/GaussianMixtures-seed100-TopBottomRandom

See ```ToyGAN_eval_vis.py -h``` for options

````shell
ToyGAN_eval_vis.py --path_to_gan "./Models/experiment22-finished/No order k/gaussian_mixture-64-0.0025-randomk.pkl" --num_components 64 --variance 0.0025 --data 'gaussian_mixture'
````

# ------------***DC GAN***-----------

Checkout DC_GAN.ipynb

# Training DC-GAN

See ```python StaxDCGAN.py -h``` for options

````shell
python StaxDCGAN.py --save_adr_model "./" --save_adr_plots_folder "./" --dataset 'cifar10'
````

# Generating samples with trained DC-GAN

See ```python DC_GAN_GenerateAndSaveImages.py -h``` for options

Trained models are in ./Trained_Models/CIFAR10-40000 

````shell
python DC_GAN_GenerateAndSaveImages.py --path_to_images "./image_folder/" --path_to_gan "./Trained_Models/CIFAR10-40000/GAN-cifar10-0.pkl" --iter 1 --batch 10
````
