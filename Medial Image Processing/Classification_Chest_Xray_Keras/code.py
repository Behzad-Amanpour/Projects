# Kaggle source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Data: https://data.mendeley.com/datasets/rscbjbr9sj/2

# Mounting Drive on Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Importing Libraries & Defining Functions
import numpy as np
from pandas import DataFrame

import matplotlib.pyplot as plt
def plot_figures(index, img_array1, img_array2):
  plt.subplot(2,3,1)
  plt.imshow(img_array1[index[0],:,:,:],cmap='gray')
  plt.subplot(2,3,2)
  plt.imshow(img_array1[index[1],:,:,:],cmap='gray')
  plt.subplot(2,3,3)
  plt.imshow(img_array1[index[2],:,:,:],cmap='gray')

  plt.subplot(2,3,4)
  plt.imshow(img_array2[index[0],:,:,:],cmap='gray')
  plt.subplot(2,3,5)
  plt.imshow(img_array2[index[1],:,:,:],cmap='gray')
  plt.subplot(2,3,6)
  plt.imshow(img_array2[index[2],:,:,:],cmap='gray')

def plot_hist(hist):
  plt.figure(figsize=(12, 6))
  plt.subplot(1,2,1)
  plt.plot(hist.history["loss"])
  plt.plot(hist.history["val_loss"])
  plt.title("model loss")
  plt.ylabel("loss")
  plt.xlabel("epoch")
  plt.legend(["train", "validation"], loc="upper left")
  plt.ylim(0, 10)

  plt.subplot(1,2,2)
  plt.plot(hist.history["accuracy"])
  plt.plot(hist.history["val_accuracy"])
  plt.title("model accuracy")
  plt.ylabel("accuracy")
  plt.xlabel("epoch")
  plt.legend(["train", "validation"], loc="upper left")
  plt.show()

# DATA LOADING ================================================================================
from keras.utils import load_img, img_to_array
from tensorflow import one_hot
from os import listdir
from os.path import join

def Load_Data(path, IMG_SIZE): # IMG_SIZE = 224  # based on our networks examples, original images have different sizes
  Num_Classes = 2  # for one-hot
  folders = listdir(path)
  images = []
  labels= []
  ix_PNEUMONIA = 0; ix_NORMAL=0

  for folder in folders:  # folder = 'NORMAL'
    path2 = join( path, folder )
    files = listdir( path2 )

    for file in files: # file = 'Copy of IM-0115-0001.jpeg'
      path3 = join(path2, file )
      img = load_img(
          path3,
          color_mode = "rgb",
          target_size = (IMG_SIZE, IMG_SIZE), # None
          interpolation= "bilinear",
          keep_aspect_ratio=False,  # True: The image is cropped in the center with target aspect ratio before resizing.
          )
      img = img_to_array( img ) # Converts a PIL Image instance to a NumPy array
      # img = np.uint8(img) # if you have a huge dataset, or your network input should be uint
      images.append( img )

      if folder == 'NORMAL':
        labels.append( 0 )
        ix_NORMAL+=1
      elif folder == 'PNEUMONIA':
        labels.append( 1 )
        ix_PNEUMONIA+=1
      else:
        raise ValueError('Check the folder name')
  images = np.array( images )
  labels = np.array( labels )
  # print( DataFrame(labels, columns=['Label']).to_string() )
  labels = one_hot(labels, Num_Classes)
  # print(DataFrame(labels).to_string())
  plt.imshow( np.uint8(img) )
  print('Normal Images: ', ix_NORMAL)
  print('PNEUMONIA Images: ', ix_PNEUMONIA)

  return images, labels

## Train Data
train_images, train_labels = Load_Data(
                                      path = '/content/drive/MyDrive/Projects/Classification_Chest_Xray_Keras/Train',
                                      IMG_SIZE = 224)  #224 is based on our network input size and its examples on Keras
## Validation Data (if you have a separate validation data)
train_images, train_labels = Load_Data(
                                      path = /content/drive/MyDrive/Projects/Classification_Chest_Xray_Keras/Validation,
                                      IMG_SIZE = 224)
## Test Data
test_images, test_labels = Load_Data(
                                      path = /content/drive/MyDrive/Projects/Classification_Chest_Xray_Keras/Validation,
                                      IMG_SIZE = 224)


# MODEL, TRAINING, TEST =======================================================================
##Callbacks (https://keras.io/api/callbacks/)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
my_callback = [EarlyStopping(monitor='val_loss', patience = 10),
               ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)]

##Data Augmentation
"""
(https://keras.io/2.15/api/layers/preprocessing_layers/image_augmentation/)
Note that you can apply augmentation layers as preprocessing layers of the network. By default, augmentation layers are only applied during training.
"""
from keras import layers
Rot_layer = layers.RandomRotation(factor=0.05, fill_mode="constant")
                                  # factor: a float represented as fraction of 2 Pi,    0.05 ≃ 20 degrees
                                  # fill_mode="constant": filling all values beyond the edge with k = 0.
Flip_layer = layers.RandomFlip(mode="horizontal")
Contrast_layer = layers.RandomContrast(factor=0.2)
                                  # The contrast_factor will be randomly picked between [1.0 - factor, 1.0 + factor]
                                  # For any pixel x in the channel, the output will be ((x - mean) * contrast_factor + mean) where mean is the mean value of the channel.
train_images_aug = np.concatenate((train_images, Rot_layer(train_images), Contrast_layer(train_images)), axis=0)
train_labels_aug = np.concatenate((train_labels, train_labels, train_labels), axis=0)





