import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

data = pd.read_csv("/Users/kraj200/Downloads/AAPL44.csv")

data["onlyindex"] = data.index

data["AvgPerDay"] = (data["High"] + data["Low"]) / 2

maxday = (data.loc[data.AvgPerDay == data.AvgPerDay.max(),["Date"]]) #["Date"][74]
minday = (data.loc[data.AvgPerDay == data.AvgPerDay.min(),["Date"]]) #["Date"][171]

data["Close"]=pd.to_numeric(data.Close,errors='coerce')
data = data.dropna()
trainData = data.iloc[:,4:5].values

sc = MinMaxScaler(feature_range=(0,1))
trainData = sc.fit_transform(trainData)

X_train = []
y_train = []

for i in range (60,1149): #60 : timestep // 1149 : length of the data
    X_train.append(trainData[i-60:i,0]) 
    y_train.append(trainData[i,0])

X_train,y_train = np.array(X_train),np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1)) #adding the batch_size axis

model = Sequential()

model.add(LSTM(units=100, return_sequences = True, input_shape =(X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units=1))
model.compile(optimizer='adam',loss="mean_squared_error")

hist = model.fit(X_train, y_train, epochs=10, batch_size = 32, verbose=2)

#testData = pd.read_csv("/Users/kraj200/Downloads/AAPL44.csv")
testData = pd.read_csv("/Users/kraj200/Downloads/Google_train_data.csv")
testData["Close"]=pd.to_numeric(testData.Close,errors='coerce')
testData = testData.dropna()
testData = testData.iloc[:,4:5]
y_test = testData.iloc[60:,0:].values 
#input array for the model
inputClosing = testData.iloc[:,0:].values 
inputClosing_scaled = sc.transform(inputClosing)
X_test = []
length = len(testData)
timestep = 60
for i in range(timestep,length):  
    X_test.append(inputClosing_scaled[i-timestep:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

y_pred = model.predict(X_test)

predicted_price = sc.inverse_transform(y_pred)

plt.figure("Welcome to figure 1")
ax = sns.scatterplot(x='onlyindex', y='Close', legend='full', data=data)
ax = sns.regplot(x='onlyindex', y='Close', data=data, scatter=False, ax=ax.axes, order=99, line_kws={"color": "purple"})
ay = sns.regplot(x='onlyindex',y='Close',data=data,scatter=False, line_kws={"color": "blue"})

plt.figure("Welcome to figure 2")
plt.plot(y_test, color = 'red', label = 'Actual Stock Price')
plt.plot(predicted_price, color = 'green', label = 'Predicted Stock Price')
plt.title('Apple stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

real_data = [inputClosing_scaled[len(inputClosing_scaled)+1-timestep:len(inputClosing_scaled+1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = sc.inverse_transform(prediction)
print(f"Prediction: {prediction}")

#print(f"Predicted Price: {predicted_price}")

# model.save("neural_net500epochs.h5")