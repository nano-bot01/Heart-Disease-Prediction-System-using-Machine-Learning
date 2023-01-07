# Heart-Disease-Prediction-System-using-Logistic-Regression
A Heart Disease Prediction System built on machine learning 


## Principle 

This prediction system is based on ECG data on heart diseases of patients

#### What is ECG ??


<p align="center">
  <img width="650" height="400" src="https://user-images.githubusercontent.com/78251168/211028900-e320780a-23d4-44f8-962f-65348841b4ee.jpg">
</p>


An electrocardiogram (ECG) is a quick test that can be used to examine the electrical activity and rhythm of your heart.
The electrical signals that your heart beats out each time it beats are picked up by sensors that are affixed to your skin.
A machine records these signals, and a doctor examines them to see whether they are odd.

### Workflow of model

  - Data collection 
  - Split Features and Target set
  - Train-Test split
  - Model Training
  - Model Evaluation
  - Predicting Results



## Data collection 

[Dataset Link](https://drive.google.com/file/d/1CEql-OEexf9p02M5vCC1RDLXibHYE9Xz/view?usp=drivesdk)

## Dependencies

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## Split Features and Target set

```
X = heart_data.drop(columns = 'target', axis = 1)
X.head()
# now X contains table without target column which will help for training the dataset
```

## Train Test Split

```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, stratify = Y, random_state = 3 )
```
* here we have test data size is 20 percent of total data which is evenly distributed with degree of randomness = 3

## Model Training 

```
model = LogisticRegression()
model.fit(X_train.values, Y_train)
```

## Model Evaluation 

<p align="center">
  <img width="650" height="400" src="https://user-images.githubusercontent.com/78251168/211057178-3b209f44-9e51-4a6b-819b-019c9f4ddb10.png">
</p>


```
# accuracy of traning data
# accuracy function measures accuracy of model

X_train_prediction = model.predict(X_train.values)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("The accuracy of training data : ", training_data_accuracy)
```

```
# accuracy of test data

X_test_prediction = model.predict(X_test.values)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("The accuracy of test data : ", test_data_accuracy)
```

## Predicting Results

#### Steps : 

  - take input data
  - Process the data, change into array 
  - reshape data as single element in array 
  - predict output using predict function 
  - output the value

```
# input feature values
input_data = (42,1,0,136,315,0,1,125,1,1.8,1,0,1)

# change the input data into a numpy array 
input_data_as_numpy_array = np.array(input_data)

# reshape the array to predict data for only one instance
reshaped_array = input_data_as_numpy_array.reshape(1,-1)
```

### Printing Results

```
# predicting the result and printing it

prediction = model.predict(reshaped_array)

print(prediction)

if(prediction[0] == 0):
    print("The Patient has a healthy heart 💛💛💛💛")

else:
    print("The Patient has an unhealthy heart 💔💔💔💔")
```

## Notations of predicted output: 

  - [0] : means patient has a healthy heart 💛💛💛💛
  - [1] : means patient has a unhealthy heart 💔💔💔💔
  
  
## Contributor 

#### [Ankit Nainwal](https://github.com/nano-bot01)
- [Twitter](https://twitter.com/Anku___)
- [LinkedIn](https://www.linkedin.com/in/ankit-nainwal1/?original_referer=)
- [Hashnode](https://hashnode.com/@ankitnainwal)
- [Instagram](https://www.instagram.com/the.ankit.nainwal/)



### Please ⭐⭐⭐. 
