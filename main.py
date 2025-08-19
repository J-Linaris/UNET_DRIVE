"""
This file is an example of the main.py file used in
the experiments conducted in order to obtain the metrics:

Accuracy, Recall, Precision, F1-Score, Recall on Thin Vessels,
Precision on Thin Vessels, F1-Score on Thin Vessels

for the loss functions:

Dice, Dice + BCE with unbalanced classes weight mask, 
Dice + BCE with W1 weight mask, Dice + BCE with W0 weight 
mask, BCE with unbalanced classes weight mask, BCE with
W1 weight mask and BCE with W0 weight mask

Experimenting, also, different variations of the UNet and
the improvement caused by using the augmented dataset 
instead of the original one.
"""



from import_drive import DRIVE
from UNet import UNet_d5_simple, UNet_d5_double, UNet_d5_double_mean, UNet_d4_double, UNet_d4_double_mean, UNet_d4_simple
from train_model import train_unet
from test_model import calculate_metrics, record_results
import torch
import random
import numpy as np


def main():

    #~~~~~~~~~~~~~~~~~Improvement of reproducibility~~~~~~~~~~~~~~~~~
    
    torch.use_deterministic_algorithms(True)

    # Forces the usage of a deterministic algorithm in
    # CUDA convolutions with the disadvantage of greater
    # running time
    torch.backends.cudnn.benchmark = False 

    #~~~~~~~~~~~~~~~~~~~Creation of the datasets~~~~~~~~~~~~~~~~~~~~~

    # Data importation
    aug_root_path = "/path/to/augmented/DRIVE/"
    stdr_root_path = "/path/to/DRIVE/"

    # Creation of the datasets
    train_data_stdr = DRIVE(stdr_root_path, test_data=False, val_data=False, augmented_dataset=False) 
    val_data_stdr = DRIVE(stdr_root_path, test_data=False, val_data=True, augmented_dataset=False)
    train_data_aug = DRIVE(aug_root_path, test_data=False, val_data=False, augmented_dataset=True) 
    val_data_aug = DRIVE(aug_root_path, test_data=False, val_data=True, augmented_dataset=True)

    print("EXPERIMENTS FOR BEST LOSS DEFINITION (WITH AUG DATA)")

    print(f"\n\nDATA RELATION:\n(Training: {len(train_data_stdr)} images)\n(Validation: {len(val_data_stdr)} images)\n")

    # Random seed definition (to increase reproducibility)
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    print("\n\n\n\n\n\n\n\n\n\n\n\n\n~~~~~~~~~~~~~DICE+BCE W1 WEIGHT MASKS~~~~~~~~~~~~~~")
    model = train_unet(train_data_aug, val_data_aug, 0.001, 200, loss_function= "DiceBCE",
                             architecture=UNet_d5_simple, weights_function_index=1, num_mod=6)
    calculate_metrics(model, val_data_stdr, thin_vessels_max_width=1.0)
    record_results(model, val_data_stdr, "/path/to/results/")
    torch.save(model, '/path/to/model.pth')


    return 0

if __name__ == "__main__":
    main()