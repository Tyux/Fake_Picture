import os
import numpy as np
import shutil

np.random.seed(2019)

root_train = '/Users/tong/Workstation/Python/fake_data/t1'
root_val = '/Users/tong/Workstation/Python/fake_data/t2'

root_total = '/Users/tong/Workstation/Python/fake_data/val_split'

NewsCategory = ['true_pic', 'rumor_pic']

nbr_train_samples = 0
nbr_val_samples = 0

# Training proportion
split_proportion = 0.5

for news in NewsCategory:

    total_images = os.listdir(os.path.join(root_total, news))

    nbr_train = int(len(total_images) * split_proportion)

    np.random.shuffle(total_images)

    train_images = total_images[:nbr_train]

    val_images = total_images[nbr_train:]

    if news not in os.listdir(root_train):
        os.mkdir(os.path.join(root_train, news))

    for img in train_images:
        source = os.path.join(root_total, news, img)
        target = os.path.join(root_train, news, img)
        shutil.copy(source, target)
        nbr_train_samples += 1

    if news not in os.listdir(root_val):
        os.mkdir(os.path.join(root_val, news))

    for img in val_images:
        source = os.path.join(root_total, news, img)
        target = os.path.join(root_val, news, img)
        shutil.copy(source, target)
        nbr_val_samples += 1

print('Finish splitting train and val images!')
print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))
