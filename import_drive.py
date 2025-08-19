"""
Logical structure used for the data in DRIVE/:
  Training set is split in 2:
      if val_data == True (validation set): 
          DRIVE returns a partition of 
          the training set of size len(val_identifiers)
      else (actual training set):
          DRIVE returns the complementar of the partition returned
          when val_data == True
  Tests set is unique and, if test_data == True, the argument val_data 
  has no effect. Also, to facilitate comparison of models, validation images
  were manually selected (DRIVE's: 21_training.png, 22_training.png, 
                                   23_training.png, 24_training.png,
                                   25_training.png)
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms import ToPILImage
from torchvision.transforms import InterpolationMode 

class DRIVE(Dataset):
    """
    Returns a class for using the images in DRIVE dataset. 
    As input, it requires the following parameters:
    
        -> root_path: path to DRIVE directory
        
        -> test_data: boolean variable indicating if it is
                      wanted the test dataset (20 imgs from 
                      DRIVE/test) or not (other 20 imgs 
                      from DRIVE/training)
        
        -> val_data: boolean variable indicating if it is
                     wanted the validation data partition
                     from the training set or the actual
                     training dataset (The current division
                     is 5 for validation and 15 for training)
        
        -> augmentation: boolean variable indicating if it
                     is wanted data augmentation or not
        
        -> full_drive: boolean variable indicating if it is
                     being used the 40 images version of DRIVE
                     (so called FULL_DRIVE) or the 20 images 
                     one
    There are two main functions:
        __len__: returns the length of the selected partition of
                 the dataset.

        __getitem__: returns a tuple with three informations:
                (image (rgb), seg mask, both files names tuple)
    """
    def __init__(self, root_path, test_data=False, val_data = False, augmented_dataset = False):
        
        # Basic settings
        if not test_data:
            self.imgs = sorted([root_path+"training/input/"+ i for i in os.listdir(root_path + "training/input/")])
                
            self.segs = sorted([root_path+"training/target/"+ i for i in os.listdir(root_path + "training/target/")])
        
            if not augmented_dataset:
                
                val_imgs = ["21_training.png", "22_training.png",
                            "23_training.png", "24_training.png", "25_training.png"]
                
                val_segs = ["21_manual1.png", "22_manual1.png",
                            "23_manual1.png", "24_manual1.png", "25_manual1.png"]
                    
            else:
                all_val_imgs = [i for i in self.imgs if "_0_" in i or"_1_" in i or"_2_" in i or "_3_" in i or "_4_" in i]                 
                all_val_segs = [i for i in self.segs if "_0_" in i or"_1_" in i or"_2_" in i or "_3_" in i or "_4_" in i]
                
                val_imgs = [path for path in all_val_imgs if "orig" in path]
                val_segs = [path for path in all_val_segs if "orig" in path]
            # val_imgs = [root_path + "training/input/" + suffix for suffix in val_imgs]
            # val_segs = [root_path + "training/target/" + suffix for suffix in val_segs]

            # Filters the image set based on if its val_data or not
            if val_data:
                
                
                self.imgs = val_imgs
                self.segs = val_segs
            
            else:
                if not augmented_dataset:
                    self.imgs = [path for path in self.imgs if path not in val_imgs]
                    self.segs = [path for path in self.segs if path not in val_segs]
                if augmented_dataset:
                    self.imgs = [path for path in self.imgs if path not in all_val_imgs]
                    self.segs = [path for path in self.segs if path not in all_val_segs]
        
        else:
            self.imgs = sorted([root_path+"test/input/"+ i for i in os.listdir(root_path + "test/input/")])

            self.segs = sorted([root_path+"test/target/"+ i for i in os.listdir(root_path + "test/target/")])
            
                
        # Transformations definitions (we don't apply pytorch's scale in segmentation images)
        self.transform_img = v2.Compose([
                v2.Resize(size = (576, 576), antialias=True, interpolation=InterpolationMode.NEAREST_EXACT), #O tamanho deve ser múltiplo de 32 (572 é o mais próximo da resolução original das imagens
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
        ])
        
        self.transform_seg = v2.Compose([
                v2.Resize(size = (576, 576), antialias=True, interpolation=InterpolationMode.NEAREST_EXACT), #O tamanho deve ser múltiplo de 32 (572 é o mais próximo da resolução original das imagens
                v2.ToImage(),
                v2.ToDtype(torch.float32)
        ])

        self.test = test_data

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
    
        image = Image.open(self.imgs[idx]).convert("RGB")
        seg = Image.open(self.segs[idx])
        files_name = (self.imgs[idx], self.segs[idx])
        
        # Applies the transformations according to the 'type' of the image (segmentation mask or RGB)
        image = self.transform_img(image)
        # image = image/255.0

        seg = self.transform_seg(seg)
        seg = (seg - seg.min())/(seg.max() - seg.min())

        # Makes sure the background is black (in images of DRIVE's test dataset, this is not always true)
        seg = torch.where(seg>0.5, torch.tensor(1.0),torch.tensor(0.0))

        return image, seg, files_name
    


def main():
    root_path_aug = "/path/to/augmented/DRIVE/"
    root_path_full = "/path/to/DRIVE/"

    train_data_aug = DRIVE(root_path_aug, test_data=False, val_data=False, augmented_dataset=True)
    train_data = DRIVE(root_path_full, test_data=False, val_data=False, augmented_dataset=False)

    # test_data = DRIVE(root_path, test_data=True, augmentation=True, full_drive=True)

    # Simple vizualization of a image retrieved by the dataset object
    sample_id = torch.randint(len(train_data), size=(1,)).item()
    to_pil = ToPILImage()

    objects_returned = train_data[sample_id]
    img = to_pil((objects_returned[1]*255).to(dtype=torch.uint8))
    img.save("seg_experiment.png")
    img = to_pil((objects_returned[0]*255).to(dtype=torch.uint8))
    img.save("img_experiment.png")

    # Gets a corresponding transformed image in the augmented dataset
    # (considering it was used the program create_aug_dataset.py to 
    # create the augmented dataset)
    objects_returned = train_data_aug[len(train_data)+sample_id]

    img = to_pil((objects_returned[1]*255).to(dtype=torch.uint8))
    # img.show()
    img.save("aug_seg_experiment.png")
    img = to_pil((objects_returned[0]*255).to(dtype=torch.uint8))
    # img.show()
    img.save("aug_img_experiment.png")

    
    return 0

if __name__ == '__main__':
    main()