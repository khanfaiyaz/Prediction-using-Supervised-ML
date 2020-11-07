# GRIP - The Spark Foundation
## Data Science & Business Analytics Tasks
## Task 1: Prediction using Supervised ML 
## By Faiyaz Khan

1. Predict the percentage of an student based on the no. of study hours. 2. This is a simple linear regression task as it involves just 2 variables. 3. You can use R, Python, SAS Enterprise Miner or any other tool 4. Data can be found at http://bit.ly/w-data 5. What will be predicted score if a student studies for 9.25 hrs/ day?


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
from sklearn import model_selection
from sklearn import linear_model
```


```python
df=pd.read_csv("task1dataset.csv")
print("Load the data")
df
```

    Load the data
    




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
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.5</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.5</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.5</td>
      <td>20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9.2</td>
      <td>88</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.5</td>
      <td>60</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.3</td>
      <td>81</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.7</td>
      <td>25</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7.7</td>
      <td>85</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5.9</td>
      <td>62</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4.5</td>
      <td>41</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3.3</td>
      <td>42</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.1</td>
      <td>17</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8.9</td>
      <td>95</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.9</td>
      <td>24</td>
    </tr>
    <tr>
      <th>18</th>
      <td>6.1</td>
      <td>67</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7.4</td>
      <td>69</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2.7</td>
      <td>30</td>
    </tr>
    <tr>
      <th>21</th>
      <td>4.8</td>
      <td>54</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3.8</td>
      <td>35</td>
    </tr>
    <tr>
      <th>23</th>
      <td>6.9</td>
      <td>76</td>
    </tr>
    <tr>
      <th>24</th>
      <td>7.8</td>
      <td>86</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape

```




    (25, 2)




```python
df.columns
```




    Index(['Hours', 'Scores'], dtype='object')




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 25 entries, 0 to 24
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   Hours   25 non-null     float64
     1   Scores  25 non-null     int64  
    dtypes: float64(1), int64(1)
    memory usage: 528.0 bytes
    


```python
df.describe()
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
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>25.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.012000</td>
      <td>51.480000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.525094</td>
      <td>25.286887</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.100000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.700000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.800000</td>
      <td>47.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.400000</td>
      <td>75.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.200000</td>
      <td>95.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby(['Hours'])['Scores'].mean()
```




    Hours
    1.1    17.0
    1.5    20.0
    1.9    24.0
    2.5    25.5
    2.7    27.5
    3.2    27.0
    3.3    42.0
    3.5    30.0
    3.8    35.0
    4.5    41.0
    4.8    54.0
    5.1    47.0
    5.5    60.0
    5.9    62.0
    6.1    67.0
    6.9    76.0
    7.4    69.0
    7.7    85.0
    7.8    86.0
    8.3    81.0
    8.5    75.0
    8.9    95.0
    9.2    88.0
    Name: Scores, dtype: float64



# Exploring the dataset


```python
plt.scatter(df['Hours'], df['Scores'], color='Red',marker='o')
plt.title("Hours Vs Scores")
plt.xlabel("Hours studied")
plt.ylabel("Percentage Scoreed")
plt.show()
```


![png](output_11_0.png)



```python
df.corr()
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
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Hours</th>
      <td>1.000000</td>
      <td>0.976191</td>
    </tr>
    <tr>
      <th>Scores</th>
      <td>0.976191</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.lmplot(x="Hours",y="Scores", data=df)
plt.title("Plotting the regression line")
#sns.regplot(x="Hours", y="Scores", data=df)
```




    Text(0.5, 1, 'Plotting the regression line')




![png](output_13_1.png)


From the graph above, we can see that there is a **positive linear relationship** between the *number of hours* studied and the *scores obtained*. We can say that with the increase of Hours studied(x), there is an increase in the scores obtained(y).

# Dividing the data into attributes(inputs) and labels (outputs)


```python
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```


```python
X
```




    array([[2.5],
           [5.1],
           [3.2],
           [8.5],
           [3.5],
           [1.5],
           [9.2],
           [5.5],
           [8.3],
           [2.7],
           [7.7],
           [5.9],
           [4.5],
           [3.3],
           [1.1],
           [8.9],
           [2.5],
           [1.9],
           [6.1],
           [7.4],
           [2.7],
           [4.8],
           [3.8],
           [6.9],
           [7.8]])




```python
y
```




    array([21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30,
           24, 67, 69, 30, 54, 35, 76, 86], dtype=int64)



# Splitting the dataset into the Training set and Test set


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
```

# Training the Simple Linear Regression model on the Training set


```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



# Predicting the Test set results


```python
y_pred = regressor.predict(X_test)
```


```python
y_pred
```




    array([17.04289179, 33.51695377, 74.21757747, 26.73351648, 59.68164043,
           39.33132858, 20.91914167, 78.09382734, 69.37226512])




```python
# Comparing Actual vs Predicted
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df1
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
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>17.042892</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27</td>
      <td>33.516954</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
      <td>74.217577</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>26.733516</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62</td>
      <td>59.681640</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35</td>
      <td>39.331329</td>
    </tr>
    <tr>
      <th>6</th>
      <td>24</td>
      <td>20.919142</td>
    </tr>
    <tr>
      <th>7</th>
      <td>86</td>
      <td>78.093827</td>
    </tr>
    <tr>
      <th>8</th>
      <td>76</td>
      <td>69.372265</td>
    </tr>
  </tbody>
</table>
</div>



# Visualising the Training set results


```python
# PLotting the training set
plt.scatter(X_train,y_train, color='blue')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title('(Trainig set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```


![png](output_28_0.png)


# Visualising the Test set results


```python
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('(Testing set)')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Scored')
plt.show()
```


![png](output_30_0.png)



```python
# Checking the correlations
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu",annot_kws={'fontsize':12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```


![png](output_31_0.png)


# Visualizing the differences between actual Scores and predicted Scores


```python
plt.scatter(y_test,y_pred,c='r')
plt.plot(y_test,y_pred,c='b')
plt.xlabel("Prices")
plt.ylabel("Predicted Score")
plt.title("Score vs Predicted Score")
plt.show()
```


![png](output_33_0.png)


# What will be predicted score if a student studies for 9.25 hrs/ day?

## Prediction through our model


```python
Hours = np.array([[9.25]])
predict=regressor.predict(Hours)
print("No of Hours = {}".format(Hours))
print("Predicted Score = {}".format(predict[0]))
```

    No of Hours = [[9.25]]
    Predicted Score = 92.14523314523314
    

# Checking accuracy of our model


```python
print("Train : ",regressor.score(X_train,y_train)*100)
print("Test : ",regressor.score(X_test,y_test)*100)
```

    Train :  95.01107277744313
    Test :  95.5570080138813
    

# Finding mean absolute error, r^2 score error and Mean Squared Error


```python
from sklearn import metrics  
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print('Mean absolute error:', metrics.mean_absolute_error(y_test, regressor.predict(X_test))) 
print('r^2 score error:',r2_score(y_test, regressor.predict(X_test)))
print('Mean squared error: ',mean_squared_error(y_test, regressor.predict(X_test)))
```

    Mean absolute error: 4.691397441397438
    r^2 score error: 0.955570080138813
    Mean squared error:  25.463280738222547
    

## Mean absolute error: 4.691397441397446 which is quite accurate model for predicting the result

# Thank you

