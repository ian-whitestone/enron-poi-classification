# UD120 - Classifing Enron POIs

## Introduction

Text....

## Data Exploration

Text....

```python
%matplotlib inline
import numpy as np
import pandas as pd
import pickle
from ggplot import *
from feature_format import featureFormat, targetFeatureSplit
```

```python
## initiate seaborn plotting
import matplotlib.pyplot as plt
import seaborn as sns
p = sns.color_palette()
```

```python
## import dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    my_dataset = pickle.load(data_file)
```

```python
df = pd.DataFrame.from_dict(my_dataset,orient='index')
```

### Outlier Detection

```python
df['name'] = df.index
df = df.reset_index()
df['index'] = df.index
```

```python
df.head()
```

```python
df['bonus'] = df['bonus'].replace('NaN',0)
df['salary'] = df['salary'].replace('NaN',0)
```

```python
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

sns.lmplot('index', 'bonus', 
           data=df,
           fit_reg=False, 
           scatter_kws={"marker": "D", 
                        "s": 100},
          size=8);
plt.title('Enron Dataset - Employee Bonuses');
plt.xlabel('Index');
plt.ylabel('Bonus');

sns.lmplot('index', 'salary', 
           data=df,
           fit_reg=False, 
           scatter_kws={"marker": "D", 
                        "s": 100},
          size=8);
plt.title('Enron Dataset - Employee Salaries');
plt.xlabel('Index');
plt.ylabel('Salary');
```

```python
my_dataset.pop('TOTAL');
```

```python
df = pd.DataFrame.from_dict(my_dataset,orient='index')
```

### Data Overview

```python
df['poi'] = df['poi'].apply(lambda x: 1 if x else 0)
```

```python
sum(df['poi'])
```

```python
df.shape
```

```python
df.describe(include = 'all')
```

```python
poi = df.poi.value_counts()

plt.figure(figsize=(12,4));
sns.barplot(poi.index, poi.values, alpha=0.8, color=p[2]);
plt.xlabel('Person of Interest', fontsize=12);
plt.ylabel('Occurence count', fontsize=12);
```

```python
df['bonus'] = df['bonus'].replace('NaN',0)
df['salary'] = df['salary'].replace('NaN',0)

plt.figure(figsize=(12, 4))
plt.hist(df.bonus, bins=100, log=True)
plt.xlabel('Log Bonus (USD)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


plt.figure(figsize=(12, 4))
plt.hist(df.salary, bins=100, log=True)
plt.xlabel('Log Salary (USD)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()
```

```python
my_dataset['WHALLEY LAWRENCE G'].keys()
```

```python

```

```python

```

# Modelling

```python
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import tester
```

```python
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi'] 
```

```python
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
```

```python

```

```python

```

```python
### SVM Model
pca_svm = Pipeline([('pca',PCA(n_components=2)),('scaler',StandardScaler()),('svm',svm.SVC())])
param_grid = ([{'svm__C': [1000,10000],
                'svm__gamma': [0.01,0.0001],
                'svm__degree':[2,3],
                'svm__kernel': ['linear','rbf','poly']}])

svm_clf = GridSearchCV(pca_svm,param_grid,scoring='recall').fit(features,labels).best_estimator_


# ### KNB Model
# pca_knb = Pipeline([('pca',PCA(n_components=2)),('scaler',StandardScaler()),('knb',KNeighborsClassifier())])
# param_grid = ([{'knb__n_neighbors': [4,5,6]}])
# knb_clf = GridSearchCV(pca_knb,param_grid,scoring='recall').fit(features,labels).best_estimator_


# ### RFST Model
# pca_rfst = Pipeline([('pca',PCA(n_components=2)),('scaler',StandardScaler()),
#                  ('rfst',RandomForestClassifier())])
# param_grid = ([{'rfst__n_estimators': [4,5,6]}])
# rfst_clf = GridSearchCV(pca_rfst,param_grid,scoring='recall').fit(features,labels).best_estimator_
```

```python
print (svm_clf)
tester.test_classifier(svm_clf,my_dataset,features_list)

# print (knb_clf)
# tester.test_classifier(knb_clf,my_dataset,features_list)

# print (rfst_clf)
# tester.test_classifier(rfst_clf,my_dataset,features_list)
```

```python

```
