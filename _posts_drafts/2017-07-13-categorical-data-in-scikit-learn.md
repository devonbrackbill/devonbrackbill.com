---
layout: post
title:  "Using Categorical Variables in Scikit-learn Pipeline"
date:   2017-07-13
---


```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
```

<!--break-->

# Data


```python
X_train = pd.DataFrame({'var1': [0,1,2,2,6],
                       'var2' : ['a','b','c','d','e']})
y_train = pd.Series([0,1,1,0,1])
```


```python
X_test = pd.DataFrame({'var1': [0,10,3],
                      'var2' : ['z', 'a', 'b']})
y_test = pd.Series([0,1,1])
```

# pd.get_dummies() doesn't work


```python
pd.get_dummies(X_train,columns=['var2'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var1</th>
      <th>var2_a</th>
      <th>var2_b</th>
      <th>var2_c</th>
      <th>var2_d</th>
      <th>var2_e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.get_dummies(X_test,columns=['var2'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var1</th>
      <th>var2_a</th>
      <th>var2_b</th>
      <th>var2_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# pd.get_dummies in pipeline


```python
from sklearn.pipeline import Pipeline
```


```python
pipe = Pipeline([ ('dummies', pd.get_dummies() ),
                ('clf', RandomForestClassifier() )])
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-7-eaaf55f5d3d5> in <module>()
    ----> 1 pipe = Pipeline([ ('dummies', pd.get_dummies() ),
          2                 ('clf', RandomForestClassifier() )])


    TypeError: get_dummies() takes at least 1 argument (0 given)


# DictVectorizer


```python
X_train.to_dict('records')
```




    [{'var1': 0, 'var2': 'a'},
     {'var1': 1, 'var2': 'b'},
     {'var1': 2, 'var2': 'c'},
     {'var1': 2, 'var2': 'd'},
     {'var1': 6, 'var2': 'e'}]




```python
X_test.to_dict('records')
```




    [{'var1': 0, 'var2': 'z'}, {'var1': 10, 'var2': 'a'}, {'var1': 3, 'var2': 'b'}]



# DictVectorizer in Pipeline


```python
from sklearn.feature_extraction import DictVectorizer
```


```python
pipe = Pipeline([ ('dummies', DictVectorizer() ),
                ('clf', RandomForestClassifier() )])
```


```python
pipe.fit(X_train.to_dict('records'), y_train)
```




    Pipeline(steps=[('dummies', DictVectorizer(dtype=<type 'numpy.float64'>, separator='=', sort=True,
            sparse=True)), ('clf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e...imators=10, n_jobs=1, oob_score=False, random_state=None,
                verbose=0, warm_start=False))])




```python
pipe.predict(X_train.to_dict('records'))
```




    array([0, 1, 1, 0, 1])




```python
pipe.predict(X_test.to_dict('records'))
```




    array([0, 1, 1])




```python
pipe.steps[1][1].feature_importances_
```




    array([ 0.34444444,  0.08333333,  0.09236111,  0.00625   ,  0.31111111,
            0.0625    ])




```python
pipe.steps[0][1].get_feature_names()
```




    ['var1', 'var2=a', 'var2=b', 'var2=c', 'var2=d', 'var2=e']



# Custom Transformer


```python
from itertools import chain
from sklearn.pipeline import make_pipeline, TransformerMixin

class CategoricalTransformer(TransformerMixin):
    
    def __init__(self,cat_cols=None):
        self.cat_columns_ = cat_cols

    def fit(self, X, y=None, *args, **kwargs):
        self.columns_ = X.columns
        # convert X to categorical dtype
        for col in self.cat_columns_:
            X[col] = X[col].astype(str).astype('category')
        #self.cat_columns_ = X.select_dtypes(include=['category']).columns
        self.non_cat_columns_ = X.columns.drop(self.cat_columns_)

        self.cat_map_ = {col: X[col].cat.categories
                         for col in self.cat_columns_}
        self.ordered_ = {col: X[col].cat.ordered
                         for col in self.cat_columns_}

        self.dummy_columns_ = {col: ["_".join([col, v])
                                     for v in self.cat_map_[col]]
                               for col in self.cat_columns_}
        self.transformed_columns_ = pd.Index(
            self.non_cat_columns_.tolist() +
            list(chain.from_iterable(self.dummy_columns_[k]
                                     for k in self.cat_columns_))
        )

    def transform(self, X, y=None, *args, **kwargs):
        # fix the dtypes
        df = pd.get_dummies(X)
        missing_cols = [col for col in self.transformed_columns_ if col not in df.columns]
        df = df.reindex(columns=self.transformed_columns_).\
                fillna(np.int8(0))
        # convert missing_cols into np.uint8, which is the same as the original pd.dummies()
        if len(missing_cols) != 0:
            for col in missing_cols:
                df[col] = df[col].astype(np.uint8)
        
        return (df)
    
    def fit_transform(self, X, y=None,  *args, **kwargs):
        ct = self.fit(X)
        ct = self.transform(X)
        print(ct.head())
        return(ct)

    def inverse_transform(self, X):
        X = np.asarray(X)
        series = []
        non_cat_cols = (self.transformed_columns_
                            .get_indexer(self.non_cat_columns_))
        non_cat = pd.DataFrame(X[:, non_cat_cols],
                               columns=self.non_cat_columns_)
        for col, cat_cols in self.dummy_columns_.items():
            locs = self.transformed_columns_.get_indexer(cat_cols)
            codes = X[:, locs].argmax(1)
            cats = pd.Categorical.from_codes(codes, self.cat_map_[col],
                                             ordered=self.ordered_[col])
            series.append(pd.Series(cats, name=col))
        # concats sorts, we want the original order
        df = (pd.concat([non_cat] + series, axis=1)
                .reindex(columns=self.columns_))
        return df
```


```python
ct = CategoricalTransformer(cat_cols = ['var2'])
```


```python
ct.fit(X_train)
```


```python
ct.transform(X_train)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var1</th>
      <th>var2_a</th>
      <th>var2_b</th>
      <th>var2_c</th>
      <th>var2_d</th>
      <th>var2_e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
ct.transform(X_test)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var1</th>
      <th>var2_a</th>
      <th>var2_b</th>
      <th>var2_c</th>
      <th>var2_d</th>
      <th>var2_e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pipe = make_pipeline(CategoricalTransformer(cat_cols = ['var2']), 
                    RandomForestClassifier())
```


```python
pipe.fit(X_train, y_train)
```

       var1  var2_a  var2_b  var2_c  var2_d  var2_e
    0     0       1       0       0       0       0
    1     1       0       1       0       0       0
    2     2       0       0       1       0       0
    3     2       0       0       0       1       0
    4     6       0       0       0       0       1





    Pipeline(steps=[('categoricaltransformer', <__main__.CategoricalTransformer object at 0x7fd017cc6490>), ('randomforestclassifier', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1...imators=10, n_jobs=1, oob_score=False, random_state=None,
                verbose=0, warm_start=False))])




```python
pipe.predict(X_test)
```




    array([1, 1, 1])




```python
pipe.predict(X_train)
```




    array([0, 1, 1, 0, 1])



## Feature Importances


```python
pipe.steps[1][1].feature_importances_
```




    array([ 0.2       ,  0.03333333,  0.        ,  0.        ,  0.28333333,
            0.18333333])




```python
pipe.steps[0][1].transformed_columns_
```




    Index([u'var1', u'var2_a', u'var2_b', u'var2_c', u'var2_d', u'var2_e'], dtype='object')



## Cross Validation


```python
from sklearn.model_selection import cross_val_score
```


```python
cross_val_score(pipe, X_train, y_train)
```

    /home/r/anaconda2/envs/py27/lib/python2.7/site-packages/sklearn/model_selection/_split.py:581: Warning: The least populated class in y has only 2 members, which is too few. The minimum number of groups for any class cannot be less than n_splits=3.
      % (min_groups, self.n_splits)), Warning)
    /home/r/anaconda2/envs/py27/lib/python2.7/site-packages/ipykernel/__main__.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy


       var1  var2_c  var2_d  var2_e
    2     2       1       0       0
    3     2       0       1       0
    4     6       0       0       1
       var1  var2_a  var2_b  var2_e
    0     0       1       0       0
    1     1       0       1       0
    4     6       0       0       1
       var1  var2_a  var2_b  var2_c  var2_d
    0     0       1       0       0       0
    1     1       0       1       0       0
    2     2       0       0       1       0
    3     2       0       0       0       1





    array([ 0.5,  0.5,  0. ])




```python
# this is actually OK because it means the training set doesn't have some of the unique categories.
# It will ignore the categories it misses.
```
