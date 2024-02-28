# EXPERIMENT 01:Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural network regression models learn complex relationships between input variables and continuous outputs through interconnected layers of neurons. By iteratively adjusting parameters via forward and backpropagation, they minimize prediction errors. Their effectiveness hinges on architecture design, regularization, and hyperparameter tuning to prevent overfitting and optimize performance.

### Architecture:
This neural network architecture comprises two hidden layers with ReLU activation functions, each having 5 and 3 neurons respectively, followed by a linear output layer with 1 neuron. The input shape is a single variable, and the network aims to learn and predict continuous outputs.

## Neural Network Model

![image](https://github.com/sangeethak15-AI/basic-nn-model/assets/93992063/53c8431d-4d46-49de-9fea-9df909720a67)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Sangeetha.K
### Register Number:212221230085
```python

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('dlee').sheet1

rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df=df.astype({'Input':'float'})
df=df.astype({'Output':'float'})
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = df[['Input']].values
y = df[['Output']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

model=Sequential([
    #Hidden ReLU Layers
    Dense(units=5,activation='relu',input_shape=[1]),
    Dense(units=3,activation='relu'),
    #Linear Output Layer
    Dense(units=1)
])

model.compile(optimizer='rmsprop',loss='mse')
model.fit(X_train1,y_train,epochs=4000)

loss= pd.DataFrame(model.history.history)
loss.plot()

X_test1 =Scaler.transform(X_test)
model.evaluate(X_test1,y_test)

X_n1=[[5]]
X_n1_1=Scaler.transform(X_n1)
model.predict(X_n1_1)



```
## Dataset Information

![image](https://github.com/sangeethak15-AI/basic-nn-model/assets/93992063/c7338dce-1c3a-43d0-8b49-f68004b6cfca)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/sangeethak15-AI/basic-nn-model/assets/93992063/c3758ce1-abc2-408e-86cc-4f5a3868d38b)

### Epoch Training:

![image](https://github.com/sangeethak15-AI/basic-nn-model/assets/93992063/07f85d06-cbee-4a58-87c3-66e16046be96)

### Test Data Root Mean Squared Error

![image](https://github.com/sangeethak15-AI/basic-nn-model/assets/93992063/d393c400-10ca-4a38-84d6-778f8a1d7b06)


### New Sample Data Prediction

![image](https://github.com/sangeethak15-AI/basic-nn-model/assets/93992063/af4b38e7-18cd-41a2-80fa-91f454a7518f)


## RESULT

Thus a basic neural network regression model for the given dataset is written and executed successfully.
