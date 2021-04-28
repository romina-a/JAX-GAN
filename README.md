# Top-k Training of GANs

The goal of the project is to reproduce the results of 
*Top-k Training of GANs: Improving GAN Performance by Throwing Away Bad
Samples* https://arxiv.org/abs/2002.06224

# Training the GANs on Gaussian Mixtures:

To compare, train with ```--top_k 0``` and ```--top_k 1```

See ```python StaxToyGAN.py -h``` for options

The saved model will have "< #modes >-< variance >-< (no)topk >.pkl" naming convention.

A model with "-intermediate" to its name will also be saved which is the model in the middle of training. This is used for gradient update analysis.
````shell
python StaxToyGAN.py --num_components 64 --top_k 0 --save_adr_plots_folder "<folder path to save training procedure plots>" --save_adr_model_folder "<path to save the final models>" --seed 1000
````

# Testing trained GANs on Gaussian Mixtures:

````shell
python3 ToyGAN_eval_vis.py --path_to_gan "<path to the saved model>" --num_components 25 --variance 0.0025
````