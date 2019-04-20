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

# define model parameters
learning_rate = 0.001  # 学习率
training_epochs = 100  # 迭代次数
display_step = 5  # 每5次迭代展示一次

# define how many inputs and outputs are in our neural network
number_of_inputs = 9
number_of_outputs = 1

# define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# section one: define the layers of the neural network itself

# input layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

# layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name="weights1", shape=[number_of_inputs, layer_1_nodes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="baiases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="baiases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="baiases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# output layer
with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights4", shape=[layer_3_nodes, number_of_outputs],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="baiases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

# section two: define the cost function of the neural network that will measure prediction

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# section three: define the optimizer function that will be run to optimize the neural network

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# initialize a session so that we can run tensorflow  operations
with tf.Session() as session:
    # run the global variable initializer to initialize all variables and layers of
    session.run(tf.global_variables_initializer())

    # create log file writers to record training progress.
    # we'll store training and testing log data separately
    training_writer = tf.summary.FileWriter("data/logs/training", session.graph)
    testing_writer = tf.summary.FileWriter("data/logs/testing", session.graph)

    # run the optimizer over and over to train the network
    # one epoch is one full run through the training data set.
    for epoch in range(training_epochs):

        # feed in the training data and do one step of neural network tranining
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

        # print the current training status to the screen
        # print("training pass:{}".format(epoch))

        # every 5 training steps,log our progress
        # if epoch % 5 == 0:
        training_cost= session.run([cost],
                                                          feed_dict={X: X_scaled_training, Y: Y_scaled_training})
        testing_cost= session.run([cost],
                                                        feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})
        # if epoch % 5 == 0:
        #     training_cost, training_summary = session.run([cost, summary],
        #                                                   feed_dict={X: X_scaled_training, Y: Y_scaled_training})
        #     testing_cost, testing_summary = session.run([cost, summary],
        #                                                 feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

            # write the current training staus to the log files(which we can visai)
            # training_writer.add_summary(training_summary, epoch)
            # testing_writer.add_summary(testing_summary, epoch)

        print(epoch, training_cost, testing_cost)

    # training is now complete!
    print("training is complete")

    # print the last traning_cost and testing_cost
    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

    print("finanl traing cost:{}".format(final_training_cost))
    print("final testing cost:{}".format(final_testing_cost))

