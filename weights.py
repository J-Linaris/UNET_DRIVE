import numpy as np
import cv2
import imageio.v3 as iio
from skimage.morphology import medial_axis, area_closing
from scipy.ndimage import distance_transform_edt
from aux import get_shift_tuples
from import_drive import DRIVE
from PIL import Image
import os

# Prunning function importation
import sys
sys.path.append("./DSE-skeleton-pruning")
from dsepruning import skel_pruning_DSE


def compute_weigths(img, weights_function=0):
    """
    Returns a weight mask with the same shape as the image,
    using one of the following formulas, using the
    weigths_function parameter as the index to the desired
    formula:
        
        (0)    W_i = 2 / (D_esq ** 2)         (standard)
        
        (1)    W_i = (D_i + 1) / (D_esq ** 2)
        
        (2)    W_i = (2*D_i + 1) / (D_esq**2) 

        (3)    W_i = (3*D_i + 1) / (D_esq**2)

    -> img: must be a 1x512x512 tesnsor (as returned by
            import_drive's DRIVE class).
    """
    img_copy = img[0].numpy() #[1,512,512] --> [512,512]

    # Applies closing
    img_copy = area_closing(img_copy, 32)

    # Gets the skeleton and distances mask
    skel = medial_axis(img_copy, return_distance=False)
    distances = distance_transform_edt(img_copy)

    # Skeleton prunning
    skel = skel_pruning_DSE(skel, distances, np.ceil(distances.max()))

    # Computes the skeleton with the values of the distances
    dist_skel = np.where(skel>0, distances, 0) # Esqueleto com as distâncias 

    # Gets unique values of dist_skel excluding 0 (values of the radius of vessels)
    values_dist_skel = np.unique(dist_skel)[1:]

    # Puts in decreasing order (we choose to allow overwritting of pixels weights using greater values)
    values_dist_skel = -np.sort(-values_dist_skel)

    W = np.zeros(dist_skel.shape, dtype = np.float32)

    for value in values_dist_skel:

        # Gets the shifts for the considered radius
        shifts = get_shift_tuples(value)

        # Sets the weights for the surrounding pixels based on the shifts 
        linhas = len(W)
        colunas = len(W[0])
        for i in range(linhas):
            for j in range(colunas):
                if dist_skel[i][j] == value:
                    for dx, dy in shifts:
                        if 0 <= i+dx < linhas and 0 <= j+dy < colunas:
                            
                            match weights_function:
                                case 0:
                                    W[i+dx][j+dy] = 2/(value**2)
                                case 1:
                                    W[i+dx][j+dy] = (distances[i+dx][j+dy]+1) / (value**2)
                                case 2:
                                    W[i+dx][j+dy] = (2*distances[i+dx][j+dy]+1) / (value**2) 
                                case 3:
                                    W[i+dx][j+dy] = (3*distances[i+dx][j+dy]+1) / (value**2)
    

    return W

def main():

    root_path = "/path/to/DRIVE/"
    test_data = DRIVE(root_path, test_data=True, val_data=False, augmented_dataset=False, full_drive=False)
    i = 1
    for img, seg, file in test_data:
        W = compute_weigths(seg, 0)
        print_W = ((W-W.min())/(W.max()-W.min()))*255
        print_W = print_W.astype(np.uint8)
        print_W = Image.fromarray(print_W)
        print_W.save(f"weights0_img{i}.png")

        W = compute_weigths(seg, 1)
        print_W = ((W-W.min())/(W.max()-W.min()))*255
        print_W = print_W.astype(np.uint8)
        print_W = Image.fromarray(print_W)
        print_W.save(f"weights1_img{i}.png")
        i+=1 
        break
    
    return 0

if __name__=="__main__":
    main()
