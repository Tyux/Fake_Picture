from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

img_width = 299
img_height = 299
batch_size = 32
nbr_test_samples = int(6695 / batch_size)

# root_path = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM'

weights_path = '/Users/tong/Workstation/Python/Fake Picture Detection/model-h5/weights-acc9205-val8841.h5'

test_data_dir = '/Users/tong/Workstation/Python/fake_data/test_split'

NewsCategory = ['rumor_pic', 'true_pic']

# test data generator for prediction
test_datagen = ImageDataGenerator(rescale=1. / 255)

# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     shuffle=False,  # Important !!!
#     classes=None,
#     class_mode=None)


test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=False,
    classes=NewsCategory,
    class_mode='categorical')

test_image_list = test_generator.filenames
# print(test_image_list)
labels = test_generator.labels[0:nbr_test_samples * batch_size]
print(labels)

print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)

print('Begin to predict for testing data ...')
predictions = InceptionV3_model.predict_generator(test_generator, nbr_test_samples)  # nbr_test_samples
ans = predictions.argmax(axis=1)
print(ans)
# evaluate = InceptionV3_model.evaluate_generator(test_generator, steps=2)
# # evaluate = InceptionV3_model.evaluate(imglist, labels)
# print(evaluate)
# # print("样本准确率%s: %.2f%%" % (InceptionV3_model.metrics_names[1], evaluate[1] * 100))
print(sum(ans == labels) / len(labels))







# np.savetxt('predictions.txt', predictions)
#
# print('Begin to write submission file ..')
# f_submit = open('submit.csv', 'w')
# f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
# for i, image_name in enumerate(test_image_list):
#     pred = ['%.6f' % p for p in predictions[i, :]]
#     if i % 100 == 0:
#         print('{} / {}'.format(i, nbr_test_samples))
#     f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
#
# f_submit.close()
#
# print('Submission file successfully generated!')
