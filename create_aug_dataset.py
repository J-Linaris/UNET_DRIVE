import os
from PIL import Image
import torch
from torchvision.transforms import v2
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode 

def augment_dataset(orig_dir_path, new_dir_path, img_suffix="training", segmentation_mask=False):
    """
    Applies horizontal and vertical transformations
    individually and combined on the images in 
    'orig_dir_path', then saves the augmented images
    in 'new_dir_path' using 'img_suffix' as the
    suffix.
    
    -> orig_dir_path: path to the directory with the 
                   original images

    -> new_dir_path:  path to the directory in which 
                   the images will be stored

    -> img_suffix: suffix of the generated augmented
                images

    -> segmentation_mask: Boolean indicating wether 
                       the images are segmentation 
                       masks or not

    """
    # Sets the transformations to be done on the image
    if segmentation_mask:
        transform = v2.Compose([
                v2.Resize(size = (576, 576), antialias=True, interpolation=InterpolationMode.NEAREST_EXACT), #O tamanho deve ser múltiplo de 32 (572 é o mais próximo da resolução original das imagens
                v2.ToImage(),
                v2.ToDtype(torch.float32)
        ])
    else:
        transform = v2.Compose([
                v2.Resize(size = (576, 576), antialias=True, interpolation=InterpolationMode.NEAREST_EXACT), #O tamanho deve ser múltiplo de 32 (572 é o mais próximo da resolução original das imagens
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
        ])
    
    to_pil = ToPILImage()

    for index, path in enumerate(orig_dir_path):
        
        # Opens the image and applies the necessary transformations
        
        if segmentation_mask:

            orig_img = Image.open(path)
            orig_img = transform(orig_img)
            orig_img = (orig_img - orig_img.min())/(orig_img.max() - orig_img.min())
            orig_img = torch.where(orig_img>0.5, torch.tensor(1),torch.tensor(0))


        else:
            orig_img = Image.open(path).convert("RGB")
            orig_img = transform(orig_img)

        # Applies the augmentation transformations saving a new image to each one of them

        # Saves the original img
        saving_img = to_pil((orig_img*255).to(dtype=torch.uint8))
        saving_img.save(new_dir_path+f"orig_img_{index}_{img_suffix}.png")

        # Generates and saves the horizontally flept image
        
        h_img = TF.hflip(orig_img)
        saving_img = to_pil((h_img*255).to(dtype=torch.uint8))
        saving_img.save(new_dir_path+f"h_flept_img_{index}_{img_suffix}.png")

        # Generates and saves the vertically flept image
        
        v_img = TF.vflip(orig_img)
        saving_img = to_pil((v_img*255).to(dtype=torch.uint8))
        saving_img.save(new_dir_path+f"v_flept_img_{index}_{img_suffix}.png")

        # Generates and saves the horizontally and vertically flept image
        
        hv_img = TF.hflip(orig_img)
        hv_img = TF.vflip(hv_img)
        saving_img = to_pil((hv_img*255).to(dtype=torch.uint8))
        saving_img.save(new_dir_path+f"hv_flept_img_{index}_{img_suffix}.png")

def create_dataset_drive(root_path, new_dir_path):
    """
    Populates the augmented images dataset, 
    indicated by 'new_dir_path' using the images in
    'root_path'.

    -> root_path: path to original dataset

    -> new_dir_path: path to the augmented dataset 
                  (must exist before calling the 
                  function)
    
    """

    # Gets the paths to each directory of the original dataset
    training_imgs = sorted([root_path+"training/input/"+ i for i in os.listdir(root_path + "training/input/")])
    training_segs = sorted([root_path+"training/target/"+ i for i in os.listdir(root_path + "training/target/")])
    testing_imgs = sorted([root_path+"test/input/"+ i for i in os.listdir(root_path + "test/input/")])
    testing_segs = sorted([root_path+"test/target/"+ i for i in os.listdir(root_path + "test/target/")])

    # Populates the augmented images dataset
    augment_dataset(training_imgs, new_dir_path+"training/input/", "training", segmentation_img=False)
    augment_dataset(training_segs, new_dir_path+"training/target/", "manual1", segmentation_img=True)
    augment_dataset(testing_imgs, new_dir_path+"test/input/", "test", segmentation_img=False)
    augment_dataset(testing_segs, new_dir_path+"test/target/", "manual1", segmentation_img=True)


    return 0

def main():

    root_path = "/home/joaolinaris/USP/IC/Projeto_retina/Datasets/DRIVE/"
    new_dir_path = "/home/joaolinaris/USP/IC/Projeto_retina/Datasets/AUGMENTED_DRIVE/"
    create_dataset_drive(root_path, new_dir_path)

    return 0

if __name__ == '__main__':
    main()