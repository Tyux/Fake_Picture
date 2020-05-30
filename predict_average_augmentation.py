from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_width = 299
img_height = 299
batch_size = 32
nbr_test_samples = int(3354 / batch_size)  # 3409
nbr_augmentation = 5

weights_path = '/Users/tong/Workstation/Python/Fake_Picture_Detection/model-h5/weights-1577344991-res-val0.8969.h5'
test_data_dir = '/Users/tong/Workstation/Python/fake_data/test_split'

NewsCategory = ['rumor_pic', 'true_pic']

# test data generator for prediction
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)

for idx in range(nbr_augmentation):
    print('{}th augmentation for testing ...'.format(idx))
    random_seed = np.random.random_integers(0, 100000)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False,  # Important !!!
        seed=random_seed,
        classes=NewsCategory,
        class_mode='categorical')

    labels = test_generator.labels[0:nbr_test_samples * batch_size]
    test_image_list = test_generator.filenames
    # print('image_list: {}'.format(test_image_list[:10]))
    print('Begin to predict for testing data ...')
    if idx == 0:
        predictions = InceptionV3_model.predict_generator(test_generator, nbr_test_samples)
    else:
        predictions += InceptionV3_model.predict_generator(test_generator, nbr_test_samples)

predictions /= nbr_augmentation
print(predictions)
ans = predictions.argmax(axis=1)
print(ans)
print(sum(ans == labels) / len(labels))

