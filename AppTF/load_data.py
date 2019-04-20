import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler


#load training data set from excel file
training_data_df=pd.read_excel("../data/gameTranindata.xlsx",dtype=float)

#pull out columns for X(data to train with) and  Y (value to predict)
X_training=training_data_df.drop('total_earnings',axis=1).values
Y_training=training_data_df[['total_earnings']].values

#load testing data set from excel file
test_data_df=pd.read_excel("../data/gameTestdata.xlsx",dtype=float)

#pull out columns for X (data to train with) and Y (value to predict)
X_testing=test_data_df.drop('total_earnings',axis=1).values
Y_testing=test_data_df[['total_earnings']].values


#all data needs to be scaled to a small rang like 0 to 1 for the neural
# network to work well. create scalers for the inputs and outputs
X_scaler=MinMaxScaler(feature_range=(0,1))#使每个feaure值在0到1之间
Y_scaler=MinMaxScaler(feature_range=(0,1))

#scale both the training inputs and outputs
X_scaled_training=X_scaler.fit_transform(X_training)
Y_scaled_training=Y_scaler.fit_transform(Y_training)


#it's very important that the training and test data are scaled with the same sacler
X_scaled_testing=X_scaler.transform(X_testing)
Y_scaled_testing=Y_scaler.transform(Y_testing)



print(X_scaled_testing.shape)
print(Y_scaled_testing.shape)
print("note:Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0],Y_scaler.min_[0]))