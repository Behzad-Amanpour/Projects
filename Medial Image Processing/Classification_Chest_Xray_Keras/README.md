# **Classification_Chest_Xray_Keras**
![photo](https://i.imgur.com/jZqpV51.png)

**This is a project for diagnosis of pneumonia using chest X-ray images.**


- [Kaggle Source] (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [Data] (https://data.mendeley.com/datasets/rscbjbr9sj/2)


### For Downloding the Dataset From Kaggle

From the `Datasetes`, download `Chest X-Ray Images (Pneumonia)` dataset.

### Content of Images

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
Note: Delete the "**.DS_Store**" after unzipping.


 ## Mounting Drive on your Google Colab
 ```
from google.colab import drive
drive.mount('/content/drive')
```
## Required Libraries:
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



