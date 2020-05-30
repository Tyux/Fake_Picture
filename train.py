from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
import os
import time
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

learning_rate = 0.0001
img_width = 299
img_height = 299
nbr_train_samples = 27276  # 3019
nbr_validation_samples = 6820  # 758
nbr_epochs = 25
batch_size = 32

train_data_dir = '/input/train_split'
val_data_dir = '/input/val_split'

NewsCategory = ['rumor_pic', 'true_pic']

print('Loading InceptionV3 Weights ...')
# base_model = InceptionV3(include_top=False, weights='imagenet',
#                                 input_tensor=None, input_shape=(299, 299, 3))

base_model = ResNet50(include_top=False, weights='imagenet',
                                input_tensor=None, input_shape=(299, 299, 3))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

print('Adding Average Pooling Layer and Softmax Output Layer ...')
output = base_model.get_layer(index=-1).output  # Shape: (8, 8, 2048)
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(2, activation='softmax', name='predictions')(output)

InceptionV3_model = Model(base_model.input, output)
# InceptionV3_model.summary()

optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# autosave best Model
# model.load_weights("") # 加载checkpoint权重模型，继续训练。
best_model_file = "./weights-{}.h5".format(int(time.time()))
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only=True)

# tensorboard
model_name = "fake_picture_detection-cnn-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='/output/{}'.format(model_name))

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=10.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True,
    # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
    # save_prefix = 'aug',
    classes=NewsCategory,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True,
    # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visulization',
    # save_prefix = 'aug',
    classes=NewsCategory,
    class_mode='categorical')

InceptionV3_model.fit_generator(
    train_generator,
    samples_per_epoch=nbr_train_samples,
    nb_epoch=nbr_epochs,
    validation_data=validation_generator,
    nb_val_samples=nbr_validation_samples,
    callbacks=[best_model, tensorboard])

os.system('/root/shutdown.sh')

# rsync -ah --progress /data/data.zip /input/ && unzip /input/data.zip -d /input/

# setsid python train2.py > /tmp/log3 2>&1
# tail -f /tmp/log3
