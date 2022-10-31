# Machine_Learning_Project1

# 주어진 DataSet으로 이진 분류 구현
소득 $50000 초과 여부 분류   

Binary Classification for over or below $50000 income


```python
import os
import pandas as pd
import numpy as np
```

## I. 작업 경로 설정


```python
# 변경 전 확인
print(os.getcwd())

# 작업 경로 변경
os.chdir('/Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Machine-Learning/Project_data')

#Binary_Classification

# 변경 후 확인
print(os.getcwd())
```

    /Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Machine-Learning
    /Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Machine-Learning/Project_data


## II. 데이터 불러오기


```python
adult_data = pd.read_csv('./Binary_Classification/adult_data.csv', sep=',')
adult_test = pd.read_csv('./Binary_Classification/adult_test.csv', sep=',')
```

## III. 데이터 전처리 

### adult_data 전처리


```python
adult_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>eduaction-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>Listing of attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32556</th>
      <td>27</td>
      <td>Private</td>
      <td>257302</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32557</th>
      <td>40</td>
      <td>Private</td>
      <td>154374</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32558</th>
      <td>58</td>
      <td>Private</td>
      <td>151910</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32559</th>
      <td>22</td>
      <td>Private</td>
      <td>201490</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32560</th>
      <td>52</td>
      <td>Self-emp-inc</td>
      <td>287927</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>15024</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
<p>32561 rows × 15 columns</p>
</div>




```python
adult_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32561 entries, 0 to 32560
    Data columns (total 15 columns):
     #   Column                 Non-Null Count  Dtype 
    ---  ------                 --------------  ----- 
     0   age                    32561 non-null  int64 
     1   workclass              32561 non-null  object
     2   fnlwgt                 32561 non-null  int64 
     3   education              32561 non-null  object
     4   eduaction-num          32561 non-null  int64 
     5   marital-status         32561 non-null  object
     6   occupation             32561 non-null  object
     7   relationship           32561 non-null  object
     8   race                   32561 non-null  object
     9   sex                    32561 non-null  object
     10  capital-gain           32561 non-null  int64 
     11  capital-loss           32561 non-null  int64 
     12  hours-per-week         32561 non-null  int64 
     13  native-country         32561 non-null  object
     14  Listing of attributes  32561 non-null  object
    dtypes: int64(6), object(9)
    memory usage: 3.7+ MB



```python
adult_data.isnull().sum()
```




    age                      0
    workclass                0
    fnlwgt                   0
    education                0
    eduaction-num            0
    marital-status           0
    occupation               0
    relationship             0
    race                     0
    sex                      0
    capital-gain             0
    capital-loss             0
    hours-per-week           0
    native-country           0
    Listing of attributes    0
    dtype: int64




```python
print('age 값 분포',adult_data['age'].value_counts())
```

    age 값 분포 36    898
    31    888
    34    886
    23    877
    35    876
         ... 
    83      6
    88      3
    85      3
    86      1
    87      1
    Name: age, Length: 73, dtype: int64



```python
print('\n workclass 값 분포',adult_data['workclass'].value_counts())
```

    
     workclass 값 분포  Private             22696
     Self-emp-not-inc     2541
     Local-gov            2093
     ?                    1836
     State-gov            1298
     Self-emp-inc         1116
     Federal-gov           960
     Without-pay            14
     Never-worked            7
    Name: workclass, dtype: int64


### workclass 변수중 ' ?' 제거


```python
index = adult_data[adult_data['workclass']==' ?'].index
```


```python
adult_data.drop(index, inplace = True)
```


```python
print('\n fnlwgt 값 분포',adult_data['fnlwgt'].value_counts())
```

    
     fnlwgt 값 분포 203488    13
    164190    13
    123011    12
    148995    12
    121124    12
              ..
    33669      1
    177457     1
    312767     1
    43354      1
    201490     1
    Name: fnlwgt, Length: 20498, dtype: int64



```python
print('\n education 값 분포',adult_data['education'].value_counts())
```

    
     education 값 분포  HS-grad         9969
     Some-college    6777
     Bachelors       5182
     Masters         1675
     Assoc-voc       1321
     11th            1057
     Assoc-acdm      1020
     10th             833
     7th-8th          574
     Prof-school      558
     9th              463
     Doctorate        398
     12th             393
     5th-6th          303
     1st-4th          156
     Preschool         46
    Name: education, dtype: int64



```python
print('\n eduaction-num 값 분포',adult_data['eduaction-num'].value_counts())
```

    
     eduaction-num 값 분포 9     9969
    10    6777
    13    5182
    14    1675
    11    1321
    7     1057
    12    1020
    6      833
    4      574
    15     558
    5      463
    16     398
    8      393
    3      303
    2      156
    1       46
    Name: eduaction-num, dtype: int64



```python
print('\n marital-status 값 분포',adult_data['marital-status'].value_counts())
```

    
     marital-status 값 분포  Married-civ-spouse       14340
     Never-married             9917
     Divorced                  4259
     Separated                  959
     Widowed                    840
     Married-spouse-absent      389
     Married-AF-spouse           21
    Name: marital-status, dtype: int64



```python
print('\n occupation 값 분포',adult_data['occupation'].value_counts())
```

    
     occupation 값 분포  Prof-specialty       4140
     Craft-repair         4099
     Exec-managerial      4066
     Adm-clerical         3770
     Sales                3650
     Other-service        3295
     Machine-op-inspct    2002
     Transport-moving     1597
     Handlers-cleaners    1370
     Farming-fishing       994
     Tech-support          928
     Protective-serv       649
     Priv-house-serv       149
     Armed-Forces            9
     ?                       7
    Name: occupation, dtype: int64


### occupation 변수중 ' ?' 처리


```python
index = adult_data[adult_data['occupation']==' ?'].index
```


```python
adult_data.drop(index, inplace = True)
```


```python
print('\n occupation 값 분포',adult_data['occupation'].value_counts())
```

    
     occupation 값 분포  Prof-specialty       4140
     Craft-repair         4099
     Exec-managerial      4066
     Adm-clerical         3770
     Sales                3650
     Other-service        3295
     Machine-op-inspct    2002
     Transport-moving     1597
     Handlers-cleaners    1370
     Farming-fishing       994
     Tech-support          928
     Protective-serv       649
     Priv-house-serv       149
     Armed-Forces            9
    Name: occupation, dtype: int64



```python
print('\n relationship 값 분포',adult_data['relationship'].value_counts())
```

    
     relationship 값 분포  Husband           12704
     Not-in-family      7865
     Own-child          4525
     Unmarried          3271
     Wife               1435
     Other-relative      918
    Name: relationship, dtype: int64



```python
print('\n race 값 분포',adult_data['race'].value_counts())
```

    
     race 값 분포  White                 26301
     Black                  2909
     Asian-Pac-Islander      974
     Amer-Indian-Eskimo      286
     Other                   248
    Name: race, dtype: int64



```python
print('\n sex 값 분포',adult_data['sex'].value_counts())
```

    
     sex 값 분포  Male      20788
     Female     9930
    Name: sex, dtype: int64



```python
print('\n capital-gain 값 분포',adult_data['capital-gain'].value_counts())
```

    
     capital-gain 값 분포 0        28129
    15024      343
    7688       278
    7298       244
    99999      155
             ...  
    6097         1
    2538         1
    401          1
    1455         1
    1086         1
    Name: capital-gain, Length: 118, dtype: int64



```python
print('\n capital-loss 값 분포',adult_data['capital-loss'].value_counts())
```

    
     capital-loss 값 분포 0       29257
    1902      199
    1977      167
    1887      157
    1848       50
            ...  
    2457        1
    4356        1
    1539        1
    1844        1
    1411        1
    Name: capital-loss, Length: 90, dtype: int64



```python
print('\n hours-per-week 값 분포',adult_data['hours-per-week'].value_counts())
```

    
     hours-per-week 값 분포 40    14525
    50     2763
    45     1791
    60     1441
    35     1203
          ...  
    82        1
    94        1
    92        1
    87        1
    74        1
    Name: hours-per-week, Length: 94, dtype: int64



```python
print('\n capital-loss 값 분포',adult_data['capital-loss'].value_counts())
```

    
     capital-loss 값 분포 0       29257
    1902      199
    1977      167
    1887      157
    1848       50
            ...  
    2457        1
    4356        1
    1539        1
    1844        1
    1411        1
    Name: capital-loss, Length: 90, dtype: int64



```python
print('\n native-country 값 분포',adult_data['native-country'].value_counts())
```

    
     native-country 값 분포  United-States                 27504
     Mexico                          610
     ?                               556
     Philippines                     188
     Germany                         128
     Puerto-Rico                     109
     Canada                          107
     India                           100
     El-Salvador                     100
     Cuba                             92
     England                          86
     Jamaica                          80
     South                            71
     China                            68
     Italy                            68
     Dominican-Republic               67
     Vietnam                          64
     Guatemala                        63
     Japan                            59
     Poland                           56
     Columbia                         56
     Iran                             42
     Taiwan                           42
     Haiti                            42
     Portugal                         34
     Nicaragua                        33
     Peru                             30
     Greece                           29
     France                           27
     Ecuador                          27
     Ireland                          24
     Hong                             19
     Cambodia                         18
     Trinadad&Tobago                  18
     Thailand                         17
     Laos                             17
     Yugoslavia                       16
     Outlying-US(Guam-USVI-etc)       14
     Hungary                          13
     Honduras                         12
     Scotland                         11
     Holand-Netherlands                1
    Name: native-country, dtype: int64



```python
index = adult_data[adult_data['native-country']==' ?'].index
```


```python
adult_data.drop(index, inplace=True)
```


```python
adult_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>eduaction-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>Listing of attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32556</th>
      <td>27</td>
      <td>Private</td>
      <td>257302</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32557</th>
      <td>40</td>
      <td>Private</td>
      <td>154374</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32558</th>
      <td>58</td>
      <td>Private</td>
      <td>151910</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32559</th>
      <td>22</td>
      <td>Private</td>
      <td>201490</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32560</th>
      <td>52</td>
      <td>Self-emp-inc</td>
      <td>287927</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>15024</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
<p>30162 rows × 15 columns</p>
</div>




```python
adult_data.isnull().sum()
```




    age                      0
    workclass                0
    fnlwgt                   0
    education                0
    eduaction-num            0
    marital-status           0
    occupation               0
    relationship             0
    race                     0
    sex                      0
    capital-gain             0
    capital-loss             0
    hours-per-week           0
    native-country           0
    Listing of attributes    0
    dtype: int64



## adult_data_test


```python
adult_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 16281 entries, 0 to 16280
    Data columns (total 15 columns):
     #   Column                 Non-Null Count  Dtype 
    ---  ------                 --------------  ----- 
     0   age                    16281 non-null  int64 
     1   workclass              16281 non-null  object
     2   fnlwgt                 16281 non-null  int64 
     3   education              16281 non-null  object
     4   eduaction-num          16281 non-null  int64 
     5   marital-status         16281 non-null  object
     6   occupation             16281 non-null  object
     7   relationship           16281 non-null  object
     8   race                   16281 non-null  object
     9   sex                    16281 non-null  object
     10  capital-gain           16281 non-null  int64 
     11  capital-loss           16281 non-null  int64 
     12  hours-per-week         16281 non-null  int64 
     13  native-country         16281 non-null  object
     14  Listing of attributes  16281 non-null  object
    dtypes: int64(6), object(9)
    memory usage: 1.9+ MB



```python
adult_test.isnull().sum()
```




    age                      0
    workclass                0
    fnlwgt                   0
    education                0
    eduaction-num            0
    marital-status           0
    occupation               0
    relationship             0
    race                     0
    sex                      0
    capital-gain             0
    capital-loss             0
    hours-per-week           0
    native-country           0
    Listing of attributes    0
    dtype: int64




```python
print('age 값 분포',adult_test['age'].value_counts())
```

    age 값 분포 35    461
    33    460
    23    452
    36    450
    31    437
         ... 
    84      3
    88      3
    85      2
    89      2
    87      2
    Name: age, Length: 73, dtype: int64



```python
print('\n workclass 값 분포',adult_test['workclass'].value_counts())
```

    
     workclass 값 분포  Private             11210
     Self-emp-not-inc     1321
     Local-gov            1043
     ?                     963
     State-gov             683
     Self-emp-inc          579
     Federal-gov           472
     Without-pay             7
     Never-worked            3
    Name: workclass, dtype: int64



```python
index = adult_test[adult_test['workclass']==' ?'].index
```


```python
adult_test.drop(index, inplace = True)
```


```python
print('\n fnlwgt 값 분포',adult_test['fnlwgt'].value_counts())
```

    
     fnlwgt 값 분포 136986    9
    203488    8
    127651    8
    120277    8
    125892    8
             ..
    326005    1
    64102     1
    385793    1
    390537    1
    83891     1
    Name: fnlwgt, Length: 12081, dtype: int64



```python
print('\n education 값 분포',adult_test['education'].value_counts())
```

    
     education 값 분포  HS-grad         5005
     Some-college    3261
     Bachelors       2590
     Masters          915
     Assoc-voc        657
     11th             577
     Assoc-acdm       509
     10th             408
     7th-8th          271
     Prof-school      252
     9th              224
     12th             206
     Doctorate        178
     5th-6th          165
     1st-4th           73
     Preschool         27
    Name: education, dtype: int64



```python
print('\n eduaction-num 값 분포',adult_test['eduaction-num'].value_counts())
```

    
     eduaction-num 값 분포 9     5005
    10    3261
    13    2590
    14     915
    11     657
    7      577
    12     509
    6      408
    4      271
    15     252
    5      224
    8      206
    16     178
    3      165
    2       73
    1       27
    Name: eduaction-num, dtype: int64



```python
print('\n marital-status 값 분포',adult_test['marital-status'].value_counts())
```

    
     marital-status 값 분포  Married-civ-spouse       7112
     Never-married            4965
     Divorced                 2105
     Separated                 474
     Widowed                   456
     Married-spouse-absent     195
     Married-AF-spouse          11
    Name: marital-status, dtype: int64



```python
print('\n occupation 값 분포',adult_test['occupation'].value_counts())
```

    
     occupation 값 분포  Prof-specialty       2032
     Exec-managerial      2020
     Craft-repair         2013
     Sales                1854
     Adm-clerical         1841
     Other-service        1628
     Machine-op-inspct    1020
     Transport-moving      758
     Handlers-cleaners     702
     Tech-support          518
     Farming-fishing       496
     Protective-serv       334
     Priv-house-serv        93
     Armed-Forces            6
     ?                       3
    Name: occupation, dtype: int64



```python
index = adult_test[adult_test['occupation']==' ?'].index
```


```python
adult_test.drop(index, inplace = True)
```


```python
print('\n occupation 값 분포',adult_test['occupation'].value_counts())
```

    
     occupation 값 분포  Prof-specialty       2032
     Exec-managerial      2020
     Craft-repair         2013
     Sales                1854
     Adm-clerical         1841
     Other-service        1628
     Machine-op-inspct    1020
     Transport-moving      758
     Handlers-cleaners     702
     Tech-support          518
     Farming-fishing       496
     Protective-serv       334
     Priv-house-serv        93
     Armed-Forces            6
    Name: occupation, dtype: int64



```python
print('\n relationship 값 분포',adult_test['relationship'].value_counts())
```

    
     relationship 값 분포  Husband           6301
     Not-in-family     4051
     Own-child         2181
     Unmarried         1596
     Wife               704
     Other-relative     482
    Name: relationship, dtype: int64



```python
print('\n race 값 분포',adult_test['race'].value_counts())
```

    
     race 값 분포  White                 13143
     Black                  1447
     Asian-Pac-Islander      449
     Amer-Indian-Eskimo      149
     Other                   127
    Name: race, dtype: int64



```python
print('\n sex 값 분포',adult_test['sex'].value_counts())
```

    
     sex 값 분포  Male      10326
     Female     4989
    Name: sex, dtype: int64



```python
print('\n capital-gain 값 분포',adult_test['capital-gain'].value_counts())
```

    
     capital-gain 값 분포 0        14043
    15024      165
    7688       124
    7298       114
    99999       84
             ...  
    2329         1
    3273         1
    2346         1
    34095        1
    2036         1
    Name: capital-gain, Length: 111, dtype: int64



```python
print('capital-gain 값 분포',adult_test['capital-gain'].value_counts())
```

    capital-gain 값 분포 0        14043
    15024      165
    7688       124
    7298       114
    99999       84
             ...  
    2329         1
    3273         1
    2346         1
    34095        1
    2036         1
    Name: capital-gain, Length: 111, dtype: int64



```python
print('hours-per-week 값 분포',adult_test['hours-per-week'].value_counts())
```

    hours-per-week 값 분포 40    7242
    50    1402
    45     861
    60     688
    35     596
          ... 
    74       2
    89       1
    76       1
    79       1
    69       1
    Name: hours-per-week, Length: 89, dtype: int64



```python
print('\n capital-loss 값 분포',adult_test['capital-loss'].value_counts())
```

    
     capital-loss 값 분포 0       14595
    1902      102
    1977       84
    1887       73
    2415       23
            ...  
    1870        1
    2282        1
    1735        1
    1825        1
    1651        1
    Name: capital-loss, Length: 80, dtype: int64



```python
print('\n native-country 값 분포',adult_test['native-country'].value_counts())
```

    
     native-country 값 분포  United-States                 13788
     Mexico                          293
     ?                               255
     Philippines                      95
     Puerto-Rico                      66
     Germany                          65
     Canada                           56
     El-Salvador                      47
     India                            47
     China                            45
     Cuba                             41
     England                          33
     Italy                            32
     South                            30
     Japan                            30
     Dominican-Republic               30
     Portugal                         28
     Haiti                            27
     Columbia                         26
     Poland                           25
     Guatemala                        23
     Jamaica                          23
     Greece                           20
     Vietnam                          19
     Ecuador                          16
     Peru                             15
     Nicaragua                        15
     Iran                             14
     Taiwan                           13
     Ireland                          12
     Thailand                         12
     Hong                              9
     France                            9
     Scotland                          9
     Cambodia                          8
     Trinadad&Tobago                   8
     Outlying-US(Guam-USVI-etc)        8
     Yugoslavia                        7
     Honduras                          7
     Hungary                           5
     Laos                              4
    Name: native-country, dtype: int64



```python
index = adult_test[adult_test['native-country']==' ?'].index
```


```python
adult_test.drop(index, inplace=True)
```


```python
adult_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>eduaction-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>Listing of attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>Private</td>
      <td>226802</td>
      <td>11th</td>
      <td>7</td>
      <td>Never-married</td>
      <td>Machine-op-inspct</td>
      <td>Own-child</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>Private</td>
      <td>89814</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Farming-fishing</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>Local-gov</td>
      <td>336951</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Protective-serv</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44</td>
      <td>Private</td>
      <td>160323</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>7688</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>34</td>
      <td>Private</td>
      <td>198693</td>
      <td>10th</td>
      <td>6</td>
      <td>Never-married</td>
      <td>Other-service</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16275</th>
      <td>33</td>
      <td>Private</td>
      <td>245211</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>16276</th>
      <td>39</td>
      <td>Private</td>
      <td>215419</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Divorced</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>16278</th>
      <td>38</td>
      <td>Private</td>
      <td>374983</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>16279</th>
      <td>44</td>
      <td>Private</td>
      <td>83891</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Divorced</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>5455</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>16280</th>
      <td>35</td>
      <td>Self-emp-inc</td>
      <td>182148</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>United-States</td>
      <td>&gt;50K.</td>
    </tr>
  </tbody>
</table>
<p>15060 rows × 15 columns</p>
</div>



## IV. 인코딩


```python
adult_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 30162 entries, 0 to 32560
    Data columns (total 15 columns):
     #   Column                 Non-Null Count  Dtype 
    ---  ------                 --------------  ----- 
     0   age                    30162 non-null  int64 
     1   workclass              30162 non-null  object
     2   fnlwgt                 30162 non-null  int64 
     3   education              30162 non-null  object
     4   eduaction-num          30162 non-null  int64 
     5   marital-status         30162 non-null  object
     6   occupation             30162 non-null  object
     7   relationship           30162 non-null  object
     8   race                   30162 non-null  object
     9   sex                    30162 non-null  object
     10  capital-gain           30162 non-null  int64 
     11  capital-loss           30162 non-null  int64 
     12  hours-per-week         30162 non-null  int64 
     13  native-country         30162 non-null  object
     14  Listing of attributes  30162 non-null  object
    dtypes: int64(6), object(9)
    memory usage: 3.7+ MB


#### I. 레이블 인코딩(train)


```python
Label_adult_data= adult_data.copy()
```


```python
Label_adult_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>eduaction-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>Listing of attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32556</th>
      <td>27</td>
      <td>Private</td>
      <td>257302</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32557</th>
      <td>40</td>
      <td>Private</td>
      <td>154374</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32558</th>
      <td>58</td>
      <td>Private</td>
      <td>151910</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32559</th>
      <td>22</td>
      <td>Private</td>
      <td>201490</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32560</th>
      <td>52</td>
      <td>Self-emp-inc</td>
      <td>287927</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>15024</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
<p>30162 rows × 15 columns</p>
</div>




```python
from sklearn.preprocessing import LabelEncoder

features = Label_adult_data[['workclass','education','marital-status','occupation','relationship','race','sex','native-country','Listing of attributes']]
for feature in features:
    LE = LabelEncoder()
    LE = LE.fit(Label_adult_data[feature])
    Label_adult_data[feature] = LE.transform(Label_adult_data[feature])
    
```


```python
Label_adult_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>eduaction-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>Listing of attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>5</td>
      <td>77516</td>
      <td>9</td>
      <td>13</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>4</td>
      <td>83311</td>
      <td>9</td>
      <td>13</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>2</td>
      <td>215646</td>
      <td>11</td>
      <td>9</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>2</td>
      <td>234721</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>2</td>
      <td>338409</td>
      <td>9</td>
      <td>13</td>
      <td>2</td>
      <td>9</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32556</th>
      <td>27</td>
      <td>2</td>
      <td>257302</td>
      <td>7</td>
      <td>12</td>
      <td>2</td>
      <td>12</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32557</th>
      <td>40</td>
      <td>2</td>
      <td>154374</td>
      <td>11</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32558</th>
      <td>58</td>
      <td>2</td>
      <td>151910</td>
      <td>11</td>
      <td>9</td>
      <td>6</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32559</th>
      <td>22</td>
      <td>2</td>
      <td>201490</td>
      <td>11</td>
      <td>9</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32560</th>
      <td>52</td>
      <td>3</td>
      <td>287927</td>
      <td>11</td>
      <td>9</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>15024</td>
      <td>0</td>
      <td>40</td>
      <td>38</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>30162 rows × 15 columns</p>
</div>



#### I. 레이블 인코딩(test)


```python
Label_adult_test= adult_test.copy()
```


```python
Label_adult_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>eduaction-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>Listing of attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>Private</td>
      <td>226802</td>
      <td>11th</td>
      <td>7</td>
      <td>Never-married</td>
      <td>Machine-op-inspct</td>
      <td>Own-child</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>Private</td>
      <td>89814</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Farming-fishing</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>Local-gov</td>
      <td>336951</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Protective-serv</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44</td>
      <td>Private</td>
      <td>160323</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>7688</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>34</td>
      <td>Private</td>
      <td>198693</td>
      <td>10th</td>
      <td>6</td>
      <td>Never-married</td>
      <td>Other-service</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16275</th>
      <td>33</td>
      <td>Private</td>
      <td>245211</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>16276</th>
      <td>39</td>
      <td>Private</td>
      <td>215419</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Divorced</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>16278</th>
      <td>38</td>
      <td>Private</td>
      <td>374983</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>16279</th>
      <td>44</td>
      <td>Private</td>
      <td>83891</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Divorced</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>5455</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K.</td>
    </tr>
    <tr>
      <th>16280</th>
      <td>35</td>
      <td>Self-emp-inc</td>
      <td>182148</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>United-States</td>
      <td>&gt;50K.</td>
    </tr>
  </tbody>
</table>
<p>15060 rows × 15 columns</p>
</div>




```python
from sklearn.preprocessing import LabelEncoder

features = Label_adult_data[['workclass','education','marital-status','occupation','relationship','race','sex','native-country','Listing of attributes']]
for feature in features:
    LE = LabelEncoder()
    LE = LE.fit(Label_adult_test[feature])
    Label_adult_test[feature] = LE.transform(Label_adult_test[feature])
    
```


```python
Label_adult_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>eduaction-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>Listing of attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>2</td>
      <td>226802</td>
      <td>1</td>
      <td>7</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>37</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>2</td>
      <td>89814</td>
      <td>11</td>
      <td>9</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>37</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>1</td>
      <td>336951</td>
      <td>7</td>
      <td>12</td>
      <td>2</td>
      <td>10</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>37</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44</td>
      <td>2</td>
      <td>160323</td>
      <td>15</td>
      <td>10</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>7688</td>
      <td>0</td>
      <td>40</td>
      <td>37</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>34</td>
      <td>2</td>
      <td>198693</td>
      <td>0</td>
      <td>6</td>
      <td>4</td>
      <td>7</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>37</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16275</th>
      <td>33</td>
      <td>2</td>
      <td>245211</td>
      <td>9</td>
      <td>13</td>
      <td>4</td>
      <td>9</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>37</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16276</th>
      <td>39</td>
      <td>2</td>
      <td>215419</td>
      <td>9</td>
      <td>13</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>37</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16278</th>
      <td>38</td>
      <td>2</td>
      <td>374983</td>
      <td>9</td>
      <td>13</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>37</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16279</th>
      <td>44</td>
      <td>2</td>
      <td>83891</td>
      <td>9</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>5455</td>
      <td>0</td>
      <td>40</td>
      <td>37</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16280</th>
      <td>35</td>
      <td>3</td>
      <td>182148</td>
      <td>9</td>
      <td>13</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>37</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>15060 rows × 15 columns</p>
</div>

## V. 분류기(Decision Tree, Random Forest, Logistic Regression, SVM, GBM) 및 평가 지표


```python
y_train = Label_adult_data['Listing of attributes']
X_train = Label_adult_data.drop('Listing of attributes',axis =1)
y_test = Label_adult_test['Listing of attributes']
X_test = Label_adult_test.drop('Listing of attributes',axis =1)
```


```python
X_train, X_test, y_train, y_test = train_test_split(adult_data, cancer.target,
                                                   test_size = 0.2, random_state = 156)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [65], in <cell line: 1>()
    ----> 1 X_train, X_test, y_train, y_test = train_test_split(adult_data, cancer.target,
          2                                                    test_size = 0.2, random_state = 156)


    NameError: name 'train_test_split' is not defined



```python
from sklearn.metrics import accuracy_score
# 판독기
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


# 결정트리, Random Forest, 로지스틱 회귀를 윟란 사이킷런 Classifier
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()
gb_clf = GradientBoostingClassifier(random_state=0)

# DecisionTreeClassifier 학습/예측/평가
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
print('DescisionTreeClassifier 정확도: {0: .4f}'.format(accuracy_score(y_test, dt_pred)))

# RandomForestClassifier 학습/예측/평가
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
print('RandomForestClassifier 정확도: {0: .4f}'.format(accuracy_score(y_test, rf_pred)))

# LogisticRegressionClassifier 학습/예측/평가
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
print('RogisticRegressionClassifier 정확도: {0: .4f}'.format(accuracy_score(y_test, lr_pred)))

# GBMVlassifeir
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)
print('GBMClassifier 정확도: {0: .4f}'.format(accuracy_score(y_test, gb_pred)))
```


```python
from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix, f1_score 

def get_clf_eval(y_test , pred):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1:{3:.4f}'.format(accuracy, precision, recall, f1))

```


```python
get_clf_eval(y_test,dt_pred)
```


```python
get_clf_eval(y_test,rf_pred)
```


```python
get_clf_eval(y_test,lr_pred)
```


```python
get_clf_eval(y_test,gb_pred)
```

## 1st. GBM 성능
정확도 : 0.8629  
정밀도 : 0.7940  
재현율 : 0.5970  
F1-Score : 0.6816

## 2st. Random Forest
정확도 : 0.8505  
정밀도 : 0.7385  
재현율 : 0.6059  
F1-Score : 0.6657


```python

```


```python
y_train = adult_data_dummies['Listing of attributes']
X_train = adult_data_dummies.drop('Listing of attributes',axis =1)
y_test = adult_test_dummies['Listing of attributes']
X_test = adult_test_dummies.drop('Listing of attributes',axis =1)
```


```python

```


```python

```
