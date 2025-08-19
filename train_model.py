import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
import time
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc
from import_drive import DRIVE
from UNet import UNet_d5_simple
from loss import DiceBCELoss, DiceLoss, BinaryCrossEntropyLoss
from aux import save_plot, save_conj_plot, get_metrics
from weights import compute_weigths
from sklearn.utils.class_weight import compute_class_weight
from early_stopper import EarlyStopper

def train_unet(train_data, val_data, learning_rate, epochs, model=None, loss_function="DiceBCE", architecture=UNet_d5_simple, weights_function_index=-1, generate_plots=False, plots_dir=None):
    """
    Returns the trained model. As input, it is expected:
        -> train_data: training dataset
        -> val_data: validation dataset
        -> learining rate: initial learning rate
        -> epochs: number of epochs
        -> model: PyTorch model's object
        -> loss_function: Name of the loss function 
                          to be used: 3 values accepted:
                            Dice, BCE, DiceBCE
        -> architecture: class containing the wanted U-Net architecture
        -> weights_function_index: identifier of the weight function to be used:

            -1: no weight function will be used (standard value)
             0: => W_i = 2/R^2 (R: vessel radius)
             1: => W_i = (D_i + 1)/R^2 (D_i: distance of pixel i to background)
             2: => W_i = (2*D_i + 1)/R^2 (D_i: distance of pixel i to background)
             3: => W_i = (3*D_i + 1)/R^2 (D_i: distance of pixel i to background)

        -> generate_plots: Boolean indicating if plots must be generated or not
        -> plots_dir: Path to directory which will store the plots (only necessary
                      if generate_plots == True)

    A model of the specified architecture is trained using the number of epochs,
    loss function, initial learning rate (updated by the ReduceLROnPlateu scheduler),
    and weight function (if wanted), registering the accuracy and best value achieved
    of f1-score during the training process.
    """

    ti1 = time.time()

    # Inputs verification
    if plots_dir != None and generate_plots:
        print(f"Plots were requested ({generate_plots}), but no directory was provided ({plots_dir}).")
        exit(1)

    #------------------------Sets the device to be used--------------------------------
    
    torch.cuda.empty_cache()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print(f"GPU está disponível? {torch.cuda.is_available()}")
    print(f"Device que será usado: {device}")

   #----------------------------Data Loaders creation---------------------------------

    # Criação de DataLoaders para cada dataset

    val_loader = DataLoader(val_data, batch_size=2, shuffle=False, persistent_workers=False)
    train_loader = DataLoader(train_data, batch_size=2, shuffle= True, persistent_workers=False)


    #--------------------------Creation of the model object----------------------------
    if model == None:
        model = architecture(in_channels=3, num_classes=1)
    model.to(device)

    #--------------------Loss, optimizer and scheduler definition----------------------
    
    if loss_function == "DiceBCE":
        criterion = DiceBCELoss()
    elif loss_function == "Dice":
        criterion = DiceLoss()
    elif loss_function == "BCE":
        criterion = BinaryCrossEntropyLoss()
    else:
        print("Unrecognized formula. Choose between:\n'Dice' ; 'BCE' and 'DiceBCE'")
        exit(1)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    scheduler = ReduceLROnPlateau(optimizer)

    stopper = EarlyStopper(patience=10, min_delta=0.05)

    #--------------------------------Training loop-------------------------------------
    avg_loss_values_per_epoch_training = []
    validation_accuracy = []
    training_accuracy = []
    validation_f1_score = []
    validation_loss = []

    f1_max = 0

    for epoch in range(epochs):
        
        model.train()

        epoch_loss = 0.0
        accuracy = 0.0

        for imgs, segs, files_tuple in train_loader:
            if weights_function_index != -1:
                
                # Computes the weight mask
                W = torch.tensor([])

                for i in range(len(segs)):
                    
                    # Uses the wanted formula to compute the weights
                    seg_weights = compute_weigths(segs[i], weights_function=weights_function_index)
                    
                    # Sets the minimum value of pixels weight
                    # min_weight = seg_weights.min()
                    if "Dice" not in loss_function:
                        min_weight = 0.5
                        seg_weights = seg_weights + min_weight

                    # [512,512] --> [1,1,512,512]
                    seg_weights = np.expand_dims(np.expand_dims(seg_weights, axis=0), axis=0)
                    seg_weights = torch.tensor(seg_weights)
                    
                    # Concatenate the computed weights to the list of weights W
                    W = torch.cat((W, seg_weights), dim=0)

                W = W.to(device)
            else:
                
                if not (loss_function == "Dice" ):
                    # Weights definition for classes balancing
                    seg_np = segs.numpy()
                    seg_np = np.where(seg_np>0, 1, 0).astype(np.uint8)
                    valores_pixels = seg_np.flatten()

                    # Computes the class weights (returns: [pixel_0_weights, pixel_1_weights])
                    W = compute_class_weight(class_weight="balanced", classes=np.unique(valores_pixels), y=valores_pixels)
                    
                    # Sets, to each pixel, the corresponding weight based on the values returned above
                    cp_W = W.copy()
                    W = np.zeros(seg_np.shape)
                    W = np.where(seg_np > 0, cp_W[1], cp_W[0])

                    # Prepares the weights for training
                    W = torch.tensor(W, dtype=torch.float32)
                    W = W.to(device)
                
            # Input preparation
            images = imgs.to(device) 
            segmenteds = segs.to(device)

            # Computes model's prediction
            outputs = model(images)

            # Computes the loss
            if loss_function == "Dice":
                loss = criterion(outputs, segmenteds)
            else:
                loss = criterion(outputs, segmenteds, weight=W)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Updates the epoch loss
            epoch_loss += loss.item()

            # Gets the epoch accuracy
            accuracy += get_metrics(outputs, segmenteds)[0]
            

        #------------------Updates the vectors with the data from training-----------------

        total = len(train_loader)
        
        training_accuracy.append(accuracy/total)
        
        avg_loss_values_per_epoch_training.append(epoch_loss/total)

        #--------------------------------Validation Loop-----------------------------------

        model.eval()
        y_pred = torch.tensor([])
        y_pred = y_pred.to(device)
        y_true = torch.tensor([])
        y_true = y_true.to(device)
        validation_loss = []

        with torch.no_grad():
            accuracy = 0.0
            batch_val_loss = 0.0
            for imgs, segs, _ in val_loader:
                
                images, segmenteds = imgs.to(device), segs.to(device)

                # Computes model's prediction
                outputs = model(images)
                
                # Computes the loss (for the scheduler)
                loss = criterion(outputs, segmenteds)
                batch_val_loss += loss.item()
                accuracy += get_metrics(outputs, segmenteds)[0]

                # Updates the y_pred and y_true vectors
                y_pred = torch.cat((outputs, y_pred), dim=0)
                y_true = torch.cat((segmenteds, y_true), dim=0)

        val_loss /= len(val_loader)
        validation_loss.append(batch_val_loss)
        accuracy /= len(val_loader)

        #----------------------------Verifies Early Stopping-------------------------------
        
        # Verifies if the validation loss stopped decreasing for a 'patience' amount
        # of epochs and a 'min_delta' of tolerance (if so, stops training)
        if stopper.early_stop(val_loss):
            break

        #------------------------------Scheduler Update------------------------------------
        
        scheduler.step(val_loss)  

        #-------------------------------Metrics Update-------------------------------------

        print(f"Epoch: {epoch+1} | Validation Accuracy: {100 * accuracy:.2f}%")
        
        f1 = get_metrics(y_true, y_pred)[1]
        
        # Updates best f1-score and the epoch of best f1-score
        if f1 > f1_max:
            f1_max = f1
            best_f1_epoch = epoch
        
        # Updates the vecotr of validation accuracy and f1-score
        validation_accuracy.append(accuracy)
        validation_f1_score.append(f1)        
        
        # Deals with memory
        del outputs, loss
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nTraining time: {time.time() - ti1} sec, {(time.time() - ti1)/60} min\n")
    print(f"\nBest f1: {f1_max} (epoch: {best_f1_epoch+1})")
    
    #-------------------------------Plots the Metrics-------------------------------------
    
    if generate_plots:

        # Changes the directory temporarily to save the plots
        working_dir = os.getcwd()
        if plots_dir != None:
            os.chdir(plots_dir)

        # Plot with the validation accuracies

        save_plot(validation_accuracy, "Validation Accuracy Curve", "Accuracy","Validation Accuracy")

        # Plot with the training accuracies

        save_plot(training_accuracy, "Training Accuracy Curve", "Accuracy","Training Accuracy")
        
        # Plot with the training loss

        save_plot(avg_loss_values_per_epoch_training, "Loss Curve", "Loss","Training Loss")

        # Plot with the validation f1-score

        save_plot(validation_f1_score, "Validation F1 Score Curve", "F1 Score","F1 Score")

        # Plots training loss against validation loss
        
        save_conj_plot(avg_loss_values_per_epoch_training, validation_loss,
                    curve1_label="Training Loss", curve2_label="Validation Loss",
                    title="Losses Curve Comparison", y_axis_label="Loss")
        
        os.chdir(working_dir)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    torch.cuda.empty_cache() # We empty cache to avoid memory problems

    return model


def main():

    
    
    # Datasets creation

    # Data importation
    root_path = "/path/to/DRIVE/"
    
    train_data = DRIVE(root_path, test_data=False) 
    test_data = DRIVE(root_path, test_data=True)

    # Random division of datasets for validation and training
    total_test = len(test_data)
    test_size = int( 0.6 * total_test )
    val_size = total_test  - test_size
    test_data, val_data = random_split(test_data, [test_size, val_size])

    # Trains the model
    
    model = train_unet(train_data, test_data, val_data, 0.001, 1, 
                            loss_function= "BCE", architecture=UNet_d5_simple, weights_function_index=-1)

    
    return 0

if __name__=="__main__":
    main()