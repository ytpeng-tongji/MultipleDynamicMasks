datasets_name_txt = 'Bird'
training_set_size = 5000
push_set_size = 5000
test_set_size = 1000
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
base_architecture = 'resnet34'
img_size = 224
prototype_shape = (2000, 128, 1, 1)
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'
experiment_run = '003'
data_path = './datasets/cub200_cropped/'
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5
warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 100
num_warm_epochs = 5
push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

original_img_path_ = './bird_datasets'
bb_img_path = './Active_test/test_nih_bbox/bbox_valid/'
original_img_path1_ = './Active_test/test_nih_bbox1/'
img_mask_root_path1_ = './img_mask_root_path_nb_single_plan_d1'
result_path1_ = './Result_nb_single_plan_d1'
pathology_path1_ = './pathology_detection_assemble/1'

img_size_ = 224
dynamic_img_batch_ = 2

dynamic_mask_lrs = {'mse': 1e-2,
                    'l1': 1}

base_mask_size_ = 10
base_mask_height_width_ =[6, 6]
best_loss_ = 1e8
best_epoch_ = -1
best_mask_position_ = -1
iteration_epoch_ = 20000
iteration_epoch_min_ = 2000
patient_ = 200
mask_optimizer_lr_ = 3e-3
check_epoch_ = 400
dynamic_img_batch_ = 2
dynamic_mask_batch_ = 80

IsPositivi_ = ['num', 'img_name', 'pathology', 'positive', 'predicted_value']
PredictedMax_ = ['num', 'img_name', 'pathology', 'predicted_pathology', 'predicted_value']

base_mask_size_list_ = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 30, 32, 56, 112]
base_mask_height_width_list_ = [[6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21], [22, 22], [23, 32], [24, 24], [25, 25], [26, 26], [27, 27], [28, 28], [29, 29], [30, 30], [31, 31], [32, 32]]
