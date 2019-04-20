import os
import tensorflow as tf

#turn off tensorflow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#define computational graph
X =tf.placeholder(tf.float32,name="X")
Y =tf.placeholder(tf.float32,name="Y")

addition =tf.add(X,Y)

#create the session
with tf.Session() as session:
    result = session.run(addition,feed_dict={X:[1,2,10],Y:[4,2,10]})
    print(result)