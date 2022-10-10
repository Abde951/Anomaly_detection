#################
## DATA AUGMENTATION AND TRANSFORMATION


import torchvision
import torchvision.transforms as transforms

## for DATA AUGEMNTATION EXPERIMENTS
#transforms.RandomHorizontalFlip(),
#transforms.RandomVerticalFlip(),
#transforms.RandomRotation([90,180,270])

# Objects
fliph_classes = ['carpet','tile','leather','grid','wood','transistor','touthbrush','screw','hazelnut','zipper']

# Textiles
flipv_classes = ['carpet','tile','leather','grid','wood','screw','hazelnut']




def transform_data(classe,augemntation=False):
    """ Images data preprocessing 
        if augementation == True : Adding Basic Data Augmentation
    """
    list_transforms = []
    list_transforms.append(transforms.Resize((256,256)))

    if augemntation:
        choice_transforms = []
        if classe in fliph_classes:
            choice_transforms.append(transforms.RandomHorizontalFlip())
        if classe in flipv_classes:
            choice_transforms.append(transforms.RandomVerticalFlip())
        

        selected_transforms = transforms.RandomOrder(choice_transforms)
        list_transforms.append(selected_transforms)

    list_transforms.append(transforms.ToTensor())

    data_transforms = transforms.Compose(list_transforms)
    return data_transforms