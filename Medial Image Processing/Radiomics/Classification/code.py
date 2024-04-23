"""
Kaggle source: https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor
"""

# Mounting Drive on Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Loading Features Provided By Kaggle ===================================================
from pandas import read_csv
from numpy import array
df = read_csv('/content/drive/MyDrive/Projects/Radiomics/Brain Tumor.csv')
Feature_Names = df.columns[2:]

Data = df.values
X = Data[:, 2:].astype('float')   # feature values
y = Data[:, 1].astype('int')

print('Data: ', X[0:5, :])
print('Labels: ', y[0:5])

# Balance of classes
print('Tumor samples: ', sum(y==1), '  percentage: ', sum(y==1) / len(y) * 100 )
print('Normal samples: ', sum(y==0), '  percentage: ', sum(y==0) / len(y) * 100 )

# Variance of Features
from numpy import var
from numpy import set_printoptions
set_printoptions(precision = 4, suppress = True) #precision: Number of digits for floating point output,   suppress: scientific notation is suppressed
print (var(X, axis=0))

X = Data[:, 2:-1].astype('float') # Removing last columns that have zero variance

# Feature Selection & Classification ==============================================================
from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from numpy import mean, where
from scipy.stats import zscore

model = SVC()
X2 = zscore(X)
performance = 0
Result = []

for n in range(2,12): # n = 2
        fs_model=SequentialFeatureSelector(
                model,
                n_features_to_select = n,
                direction='forward'
                )
        fs_model.fit(X2,y)
        selected = fs_model.support_
        scores=cross_val_score(model, X2[:,selected], y, scoring = 'balanced_accuracy')
        # print('balanced_accuracy: ', scores.mean())
        if scores.mean() > performance:
          performance = scores.mean()
          Result.append([model, n, ix, scores.mean()])

print(Feature_Names[ix])
print(Result)





