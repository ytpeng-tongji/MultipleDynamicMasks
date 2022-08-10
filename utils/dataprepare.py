import torchvision.transforms as transforms
import pandas as pd
import numpy as np

label_path = './NIH_label/Data_Entry_2017.csv'
train_batch_size = 4
test_batch_size = 4
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
source_path = './NIH_datasets_crop/'
train_path = 'train'
test_path = 'test'
train_push_path = 'train_push'
label_dictionary = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltration': 3, 'Mass': 4, 'Nodule': 5, 'Pneumonia': 6, 'Pneumothorax': 7, 'Consolidation': 8, 'Edema': 9, 'Emphysema': 10, 'Fibrosis': 11, 'Pleural_Thickening': 12, 'Hernia': 13}

original_data = pd.read_csv(label_path) # 原数据
data = original_data.values # array数据
one_hot_label = {}
for i in data:
    file_name = i[0]
    file_label = i[1]
    one_hot = np.zeros([14, 1]).astype('uint8')

    if file_label == 'No Finding':
        one_hot_label[file_name] = one_hot
    else:
        result = file_label.split('|')
        for name in result:
            one_hot[label_dictionary[name]] = 1
        one_hot_label[file_name] = one_hot

normalize = transforms.Normalize(mean=mean,
                                 std=std)

img_size = 512
t=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])