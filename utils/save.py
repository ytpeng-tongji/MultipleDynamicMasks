import os
import torch
import csv

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))

def var_name(var,all_var=locals()):
    return [var_name for var_name in all_var if all_var[var_name] is var][0]

def TxtCreate(Path_list, Info_list):
    path = ''
    for i in Path_list:
        path = os.path.join(path, i)

    with open(file=path + str('.txt'), mode='w') as f:
        for i in Info_list:
            f.write(i)
            f.write("\n")

def CsvCreate(Path_list, Info_list):
    path = ''
    for i in Path_list:
        path = os.path.join(path, i)

    with open(file=path + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(Info_list)

def CsvSave(Path_list, Info_list):
    path = ''
    for i in Path_list:
        path = os.path.join(path, i)

    with open(file=path + '.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(Info_list)
