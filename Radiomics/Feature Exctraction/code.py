"""
Kaggle source: https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor
Sample Images and Mask: https://drive.google.com/drive/folders/1SiCvAmPyK3SjBZJ8C80N6HMS2OXsw7V2?usp=sharing
"""

# Mounting Drive on Google Colab
from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt
def plot_figures(img1, img2, img3, img4):
  plt.figure(figsize=(12, 6))
  plt.subplot(1,4,1)
  plt.imshow(img1, cmap='gray')
  plt.subplot(1,4,2)
  plt.imshow(img2, cmap='gray')
  plt.subplot(1,4,3)
  plt.imshow(img3, cmap='gray')
  plt.subplot(1,4,4)
  plt.imshow(img4, cmap='gray')

# Showing Some Images & Masks ===============================================================
from cv2 import cvtColor, imread, COLOR_BGR2GRAY

path = '/content/drive/MyDrive/Projects/Radiomics/Data/Images/Image1.jpg'
img1 = cvtColor( imread( path ), COLOR_BGR2GRAY)
path = '/content/drive/MyDrive/Projects/Radiomics/Data/Masks/Image1_mask.jpg'
mask1 = cvtColor( imread( path ), COLOR_BGR2GRAY)
path = '/content/drive/MyDrive/Projects/Radiomics/Data/Images/Image3.jpg'
img2 = cvtColor( imread( path ), COLOR_BGR2GRAY)
path = '/content/drive/MyDrive/Projects/Radiomics/Data/Masks/Image3_mask.jpg'
mask2 = cvtColor( imread( path ), COLOR_BGR2GRAY)

plot_figures( img1, mask1, img2, mask2 )

from numpy import unique
print( unique(mask1) )
print( unique(mask2) )  # the number of mask unique values should be equal to the number of classes, ideally "0" and "1" for a two classes task


# Feature Extraction ===========================================================================
!pip install pyradiomics
!pip install SimpleITK
from radiomics.featureextractor import RadiomicsFeatureExtractor
from SimpleITK import GetImageFromArray
from os import listdir
from os.path import join
from cv2 import cvtColor, imread, COLOR_BGR2GRAY
from numpy import array  # zeros, vstack
from pandas import DataFrame

path_image = '/content/drive/MyDrive/Projects/Radiomics/Data/Images'
path_mask = '/content/drive/MyDrive/Projects/Radiomics/Data/Masks'

extractor = RadiomicsFeatureExtractor()
extractor.enableFeatureClassByName('shape2D')

X = []  # X = zeros((1, 102))
files = listdir( path_image )

for file in files:    # file = files[0]
    path2 = join(path_image, file)
    image = cvtColor( imread( path2 ), COLOR_BGR2GRAY)
    path2 = join( path_mask, file[:-4]+'_mask.jpg')
    mask = cvtColor( imread( path2 ), COLOR_BGR2GRAY)
    mask[mask<=100] = 0; mask[mask>100] = 1   # unique values of the mask is more than the number of classes
    f = extractor.execute( GetImageFromArray(image), GetImageFromArray(mask))
    f = DataFrame(f.items()).drop(DataFrame(f.items()).index[0:22])
    X.append(f.values[:,1])  # X = vstack((X, f.values[:,1]))

X = array(X)     # X = X[ 1:, : ]
Data = DataFrame(data = X, index = files, columns = f.values[:,0])
name = '/content/drive/MyDrive/Projects/Radiomics/Brain Tumor2.xlsx'
Data.to_excel( name )













