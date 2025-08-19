import numpy as np
import torch
import cv2
from PIL import Image
from import_drive import DRIVE
from aux import get_metrics, calculate_recall_straight, calculate_precision_straight

def calculate_metrics(model, test_data, thin_vessels_max_width=1.0):
    """
    Prints the accuracy score and the precision, recall and f1 scores
    all three computed both overall and using thin vessels only (radius=1).
    -> model: PyTorch model's object
    -> test_data: test dataset (can be any PyTorch's Dataset object)
    -> thin_vessels_max_width: maximum vessel's width for a vessel to be 
                               considered thin (impact on what's included 
                               in the final filtered segmentation mask, 
                               important for metrics calculation).
                               
    """
    #----------------------------------Testing Loop--------------------------------------
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    recall_straight_veins = 0
    precision_straight_veins = 0
    epoch_rec_str_vess = 0
    epoch_pre_str_vess = 0
    f1_straight_veins = 0
    general_metrics = []

    with torch.no_grad():
        device = torch.device('cpu')
        model.eval()
        for img, seg, files_tuple in test_data:            
            # Input preparation for testing
            img, seg = img.to(device), seg.to(device)
            img, seg = img.unsqueeze(0), seg.unsqueeze(0) # [1,H,W] ---> [1,1,H,W]
            
            # Model preparation for testing
            model.to(device)
            
            # Computes the logits (raw output)
            out = model(img)

            # [1,1,H,W] ---> [1,H,W]
            out, seg = out[0], seg[0]
            
            # Computes the metrics
            general_metrics = get_metrics(out, seg)
            accuracy += general_metrics[0]
            precision += general_metrics[3]
            recall += general_metrics[2]
            f1 += general_metrics[1]

            epoch_rec_str_vess = calculate_recall_straight(out, seg, ceil=thin_vessels_max_width)
            recall_straight_veins += epoch_rec_str_vess
            
            epoch_pre_str_vess = calculate_precision_straight(out, seg, ceil=thin_vessels_max_width)
            precision_straight_veins += epoch_pre_str_vess

            f1_straight_veins += 2*epoch_pre_str_vess*epoch_rec_str_vess/(epoch_pre_str_vess+epoch_rec_str_vess)
    #-------------------------------Metrics print----------------------------------------
    
    test_size = len(test_data)
    
    accuracy = accuracy / test_size
    precision = precision / test_size
    recall = recall / test_size
    f1 = f1/test_size

    recall_straight_veins = recall_straight_veins / test_size
    precision_straight_veins = precision_straight_veins / test_size
    f1_straight_veins = f1_straight_veins / test_size


    print("--------------TEST METRICS-----------------")
    print(f"Number of images for test: {test_size}")
    print(f"Accuracy: {100 * accuracy:.2f}%")
    print(f"Precision: {100 * precision:.2f}%")
    print(f"Recall: {100 * recall:.2f}%")
    print(f"F1-Score: {100 * f1:.2f}%")
    print(f"Recall on Thin Vessels: {100 * recall_straight_veins:.2f}%")
    print(f"Precision on Thin Vessels: {100 * precision_straight_veins:.2f}%")
    print(f"F1-Score on Thin Vessels: {100 * f1_straight_veins:.2f}%")
    print("-------------------------------------------")

    return 0

def record_results(model, test_data, path_to_results):
    """
    Records the results in a single image, in which
    results are disposed in the following order:

    Original RGB image | Seg Mask | Prediction | Reescaled Prediction 

    It is important to notice the diference:

    Prediciton ---> Application of a binary classification of pixels on the output of the
                    model based on a threshold of 0.5

    In order to record the results in a single image,
    it is required two main transformations:

    1 - Segmentation masks have shape:
        [1,576,576]
        However, we need them to have shape:
        [3,576,576]
        (we use torch.cat to solve this problem)

    2 - For achieving compatibility between the shape of tensors 
        and the shape of numpy arrays, it is important to know
        the following conventions:
        TORCH convention: [[B], C, H, W] ([batch_size], channel, height, width) ([]: optional)
        NUMPY convention: [[B], H, W, C] ([batch_size], height, width, channel) ([]: optional)
    
    """
    #-------------Saves: Original RGB images, Prediction Mask and Segmentation Mask in vectors---------
    original_images = []
    predictions = []
    predictions_reescaled = []
    original_segmented_imgs = []
    
    with torch.no_grad():
        device = torch.device('cpu')
        model.eval()
        for img, seg, files_tuple in test_data:            
            image, seg_mask = img.to(device), seg.to(device)
            model.to(device)


            # Increases Image's dimension (Necessary due to the expected shape of the model's input)
            image = image[None,:] #[1,576,576] --> [1,1,576,576]
            
            # Computes the model's prediction
            output = model(image)

            #----------------------------Saves the prediction----------------------------

            # Dimension reduction: [1,1,576,576] --> [1,576,576]
            output = output[0]

            # tensor([1,576,576]) ---> numpy([3,576,576])
            output = torch.cat([output, output, output], dim=0).numpy()
            
            # numpy([3,576,576]) ---> numpy([576,576,3])
            output = np.transpose(output, (1,2,0))

            # Stores the reescaled prediction
            predictions_reescaled.append((output*255).astype(np.uint8))

            # Classifies the pixels according to the sigmoid probability and the threshold (0.5)
            output = ((output > 0.5)*255).astype(np.uint8)

            # Stores the threshold-based prediction
            predictions.append(output)

            #----------------------------Saves the seg mask--------------------------------

            # tensor([1,576,576]) ---> numpy([3,576,576])
            seg_mask = torch.cat([seg_mask, seg_mask, seg_mask], dim=0)

            # Reescales the mask
            seg_mask = (seg_mask*255).to(dtype=torch.uint8)
            seg_mask = seg_mask.numpy()
            
            # numpy([3,576,576]) ---> numpy([576,576,3])
            seg_mask = np.transpose(seg_mask, (1,2,0))

            # img_show = Image.fromarray(seg_mask)
            # img_show.show()
            # exit(0)

            # Stores the reescaled mask
            original_segmented_imgs.append(seg_mask)

            #--------------------------Saves the original image----------------------------
            
            # Opens the image
            img_file_name = files_tuple[0]
            original_img = cv2.imread(img_file_name, cv2.IMREAD_COLOR) #[H,W,3]
            
            # Reshapes to 576x576 for compatibility
            original_img = cv2.resize(original_img, (576, 576), interpolation=cv2.INTER_LINEAR) #[576,576,3]
            
            # Stores the reshaped image
            original_images.append(original_img)

    # Convertion to a numpy array
    predictions = np.array(predictions)
    predictions_reescaled = np.array(predictions_reescaled)
    original_segmented_imgs = np.array(original_segmented_imgs)
    original_images = np.array(original_images)

    #---------------Creates the comparison image and saves into path_to_results--------------------


    # cria um vetor 576x10x3 (3 canais) com valores iguais a 128
    # Creates a divisory line vector
    divisory_line = np.ones((576, 10, 3)) * 128     

    # Stacks horizontally each set of images in the dataset into a comparison image and saves it

    for i in range(len(test_data)):

        concat_images = np.concatenate(
            [original_images[i], divisory_line, original_segmented_imgs[i], divisory_line, predictions[i], divisory_line, predictions_reescaled[i]],
            axis=1
        )

        cv2.imwrite(f"{path_to_results}result{i}.jpg", concat_images) 

    return 0

def vizualize_difference(y_true, y_pred, out_file_name="Difference_predicition_ground_truth.png"):
    """
    Shows the difference between prediction and segmentaition mask (even if you consider the
    maximum vessel radius, it will be slightly different due to the applicaiton of area 
    closing in new_seg_true). Inputs:
        -> y_true: Segmentation mask tensor of shape [1, H, W]
        -> y_pred: Segmentation mask prediction tensor of shape [1, H, W]
        -> out_file_name: Desired output file's name.   
    
    """

    seg_pred_copy = y_pred.clone().to(torch.device("cpu"))
    seg_true_copy = y_true.clone().to(torch.device("cpu"))

    seg_pred_copy = seg_pred_copy.detach().numpy()
    seg_pred_copy = seg_pred_copy > 0.5
    seg_pred_copy = seg_pred_copy.astype(np.uint8)
    
    seg_true_copy = seg_true_copy.detach().numpy()
    seg_true_copy = seg_true_copy > 0
    seg_true_copy = seg_true_copy.astype(np.uint8)

    seg_pred_copy = np.transpose(seg_pred_copy, (1,2,0)) #[1,H,W] --> [H,W,1] 
    seg_true_copy = np.transpose(seg_true_copy, (1,2,0)) #[1,H,W] --> [H,W,1]

    img_seg_pred = np.concatenate([seg_pred_copy, seg_pred_copy, seg_pred_copy], axis=-1) #[H,W,1] --> [H,W,3]
    img_seg_true = np.concatenate([seg_true_copy, seg_true_copy, seg_true_copy], axis=-1) #[H,W,1] --> [H,W,3]
    
    # Compares pixel to pixel if all channels are greater than 0
    true_positives  = (img_seg_true >  0).all(axis=-1) & (img_seg_pred >  0).all(axis=-1)
    false_positives = (img_seg_true == 0).all(axis=-1) & (img_seg_pred >  0).all(axis=-1)
    false_negatives = (img_seg_true >  0).all(axis=-1) & (img_seg_pred == 0).all(axis=-1)

    # Initializes the difference image
    difference_img = np.full_like(img_seg_true, fill_value=0, dtype=np.uint8)


    # # Applies colors to the interest points
    # difference_img[false_positives] = [255, 60, 40]      # Bright Red
    # difference_img[true_positives] = [6, 168, 0]       # Bright Green
    # difference_img[false_negatives] = [80, 180, 255]     # Bright Blue
    
    # Applies colors to the interest points
    difference_img[false_positives] = [255, 60, 40]      # Bright Red
    difference_img[true_positives] = [6, 168, 0]       # Bright Green
    difference_img[false_negatives] = [250, 225, 0]     # Bright yellow

    
    # # Initializes the difference image with a white background
    # difference_img = np.full_like(img_seg_true, fill_value=255, dtype=np.uint8)
    
    # # Applies colors that contrast well with white
    # difference_img[false_positives]  = [255, 85, 0]     # Deep Orange
    # difference_img[true_positives]   = [0, 200, 140]    # Teal Green
    # difference_img[false_negatives]  = [160, 0, 255]    # Purple

    
    img = Image.fromarray(difference_img.astype(np.uint8))
    img.save(out_file_name)


def main():

    # We empty the cache to avoid memory problems
    torch.cuda.empty_cache()

    # Data importation
    root_path = "/path/to/DRIVE/"
    path_to_results="/path/to/results/dir"
    test_data = DRIVE(root_path, test_data=True, augmentation=False, full_drive=True) 
    train_data = DRIVE(root_path, test_data=False, augmentation=False, full_drive=True)

    # Model importation
    model = torch.load("/path/to/model.pth", weights_only=False)
    model.eval()

    # Calculates the performance metrics
    calculate_metrics(model, test_data=test_data, thin_vessels_max_width=1.0)

    # Gets the prediction
    objects = train_data[2] # Gets the third element of the training dataset
    seg_mask = objects[1] # [1,H,W]
    
    input = objects[0].unsqueeze(0) # [3,H,W] ---> [1,3,H,W]
    out = model(input)
    out = out[0] # [1,1,H,W] --> [1,H,W]

    # Vizualizes the difference between both
    vizualize_difference(seg_mask, out)

    record_results(model, test_data, path_to_results)
    exit(0)
    return 0

if __name__ == "__main__":
    main()