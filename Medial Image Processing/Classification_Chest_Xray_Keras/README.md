# **Classification_Chest_Xray_Keras**
![photo](https://i.imgur.com/jZqpV51.png)

**This is a project for diagnosis of pneumonia using chest X-ray images.**


- [Kaggle Source] (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [Data] (https://data.mendeley.com/datasets/rscbjbr9sj/2)


### For Downloding the Dataset From Kaggle

From the `Datasetes`, download `Chest X-Ray Images (Pneumonia)` dataset.

### Content of Images

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
**Note**: Delete the "**.DS_Store**" after unzipping.


 ## Mounting Drive on your Google Colab
 ```
from google.colab import drive
drive.mount('/content/drive')
```
## Import Required Libraries
```
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
```
## Defining Functions
```
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
```

```
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
```
## DATA LOADING
**We changed this code of Keras for loadind JPEG format of data.**

### Import Required Libraries
from keras.utils import load_img, img_to_array
from tensorflow import one_hot
from os import listdir
from os.path import join

### Define Function
```
def Load_Data(path, IMG_SIZE):  # IMG_SIZE = 224  # Based on our networks examples, original images have different sizes.
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
      img = img_to_array( img )     # Converts a PIL Image instance to a NumPy array
      # img = np.uint8(img)         # if you have a huge dataset, or your network input should be uint
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
```
#### Train Data
```
train_images, train_labels = Load_Data(path, IMG_SIZE = 224)  #224 is based on our network input size and its examples on Keras.
```                                      
> The output will be a sample of your train dataset:

![Output:](https://storage.googleapis.com/kagglesdsdata/datasets/17810/23812/chest_xray/train/PNEUMONIA/person1004_bacteria_2935.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250317%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250317T230442Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=51bb4f1de2719f4fb174575c510d7aecbf84c0d64bd53aeeaa691cd16563a8f9b4661882b9208f28af04ceee1159a557a6ffec198e484613691199fb3a08f4215e5710c6421a83ba6c6f257613db45c4b4f5d09c2a83ba38640dbfaed3937ab2ad34582782a08be699380f2604a92efe26e9b07c6277bc451d79db789d8419af6e040ea556510e3c514e9ba0fcc6c320b9da31aedfdd94e7f62bf1fab145991b92f03076e50645eadb846c7868af8744fc0fb856148ec53f25871d0f2537237fecb9de0a969610501220f1d675e0fd80f759a4afca548dde6dbeb1393eb185f784e9e4202fa94edd24b4973ccf50d97bb9787d50fc45df090572bc427769cf85)

#### Validation Data
*If you have a separate validation data use this. But if not, you can use **`train_test_split`** for making validation dataset.*
```
train_images, train_labels = Load_Data(path, IMG_SIZE = 224)
```                                     
#### Test Data
```
test_images, test_labels = Load_Data(path, IMG_SIZE = 224)
```                                 

## Data Augmentation
**Note**: You can apply augmentation layers as preprocessing layers of the network. By default, augmentation layers are only applied during training.
In Keras 3 the number of Augmentation Layers has been expanded. 
For this project, we used three Augmentation layers 
- Keras 2 API (https://keras.io/2.15/api/layers/preprocessing_layers/image_augmentation)
- Keras 3 API (https://keras.io/api/layers/preprocessing_layers/image_augmentation)
```
from keras import layers
```
```
Rot_layer = layers.RandomRotation(factor=0.05, fill_mode="constant")
                                  # factor: a float represented as fraction of 2 Pi,    0.05 ≃ 20 degrees
                                  # fill_mode="constant": filling all values beyond the edge with k = 0.
Flip_layer = layers.RandomFlip(mode="horizontal")
Contrast_layer = layers.RandomContrast(factor=0.2)
                                  # The contrast_factor will be randomly picked between [1.0 - factor, 1.0 + factor]
                                  # For any pixel x in the channel, the output will be ((x - mean) * contrast_factor + mean) where mean is the mean value of the channel.
```
### Concatenate the created augmentation layers with train_images and their labels
```
train_images_aug = np.concatenate((train_images, Rot_layer(train_images), Contrast_layer(train_images)), axis=0)
train_labels_aug = np.concatenate((train_labels, train_labels, train_labels), axis=0)
```
## Callbacks 
- Available callbacks in Keras (https://keras.io/api/callbacks)

A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training.

