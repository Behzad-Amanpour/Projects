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
from cv2 import cvtColor, imread, COLOR_BGR2GRAY
from pandas import DataFrame
from numpy import unique














