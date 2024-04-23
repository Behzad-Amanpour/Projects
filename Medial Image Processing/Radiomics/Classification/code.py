"""
Kaggle source: https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor
"""

# Mounting Drive on Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Loading Features Provided By Kaggle ===================================================
from pandas import read_csv
df = read_csv('/content/drive/MyDrive/Projects/Radiomics/Brain Tumor.csv')

from numpy import array
Feature_Names = df.columns[2:]

Data = df.values
X = Data[:, 2:].astype('float')   # feature values
y = Data[:, 1].astype('int')

print('Data: ', X[0:5, :])
print('Labels: ', y[0:5])
