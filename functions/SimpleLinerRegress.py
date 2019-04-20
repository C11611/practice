import numpy as np 

class SimpleLinearRegression1:
	def __int__(self):
		"""初始化Simple SimpleLinearRegression """
		self.a_=None
		self.b_=None

	def fit(self,x_train,y_train):
		assert x_train.ndim==1,\
		"simple linear regressor can only solve singel feature training data."
		assert len(x_train)==len(y_train),\
		"the size of  x_train must be equal to the size of y_trian"

		x_mean=np.mean(x_train)
		y_mean=np.mean(y_train)

		num=0.0
		d=0.0
		for x,y in zip(x_train,y_train):
			num+=(x-x_mean)*(y-y_mean)
			d+=(x-x_mean)**2


		self.a_=num/d
		self.b_=y_mean -self.a_*x_mean

		return self

	def predict(self,x_predict):
		