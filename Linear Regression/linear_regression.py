import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Lin_reg:
	def __init__(self,dataset,var1,var2):
		self.df=pd.read_csv(dataset)
		self.xtemp=self.df[var1]
		self.t=[]
		self.x=self.xtemp[0:10000]
		self.ytemp=self.df[var2]
		self.y=self.ytemp[0:10000]
		self.s=200
		self.alpha=0.001
		self.theta0=0
		self.theta1=np.random.rand(1,1)[0][0]
		self.ypredicted=[]

	def initialplot(self,title,xlabel,ylabel):
		temp=np.random.randint(len(self.x),size=self.s)
		for i in temp:
			plt.scatter(self.x[i]*100,self.y[i]*100)
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()
	
	def lin_reg(self,choice,epochs,iters):
		#Build the model
		print("Initial Random Theta1: ",self.theta1)
		print("Initial Random Theta0: ",self.theta0)

		print("MODEL TRAINING...")

		j=0
		while(j<epochs):
			#y=mx+c
			mx=np.dot(self.theta1,self.x)
			self.ypredicted=mx+self.theta0
			
			#calculate the cost i.e MSE
			hx_y=self.ypredicted-self.y
			self.mse=np.sum(np.power(hx_y,2))/(2*len(self.x))
			if((j+1)%100==0 or j==0):
				if(choice==True):
					print("\t Iterations completed: ",j+1,"\n\tMSE: ",self.mse,"\n")
				else:
					print("\tTraining Model.........",j+1," iterations completed\n")
			
			#Gradient descent to reduce cost
			for i in range(0,iters):
				self.theta0=self.theta0-(self.alpha/len(self.x))*np.sum(hx_y)
				self.theta1=self.theta1-(self.alpha/len(self.x))*np.sum((hx_y)*self.x)
			j=j+1
		print("Final MSE: ",self.mse,'\n')
		print("Final Theta1: ",self.theta1,'\n')
		print("Final Theta0: ",self.theta0,'\n')


	def test(self,title,xlabel,ylabel,x,y):
		temp=np.random.randint(len(self.x),size=(1,50))
		xtest=[self.x[i] for i in temp]
		yactual=[self.y[i]*100 for i in temp]
		ytest=np.dot(self.theta1,xtest)+self.theta0
		self.t=temp

		for i in range(0,len(xtest)):
			plt.scatter(xtest[i]*100,yactual[i],color='r')
			plt.plot(xtest[i]*100,ytest[i]*100,color='b')
		plt.scatter(x*100,y*100,color='g')
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

#print(df.keys())
#print(df.head())

#sb.set(font_scale=1.5)
#temp=sb.heatmap(df.corr())
#temp=sb.heatmap(np.random.rand(10,10))

if __name__=='__main__':
	linreg=Lin_reg('OnlineNewsPopularity.csv',' n_unique_tokens',' n_non_stop_unique_tokens')
	linreg.initialplot("Initial Plot",' n_unique_tokens',' n_non_stop_unique_tokens')
	linreg.lin_reg(True,800,200)
	#linreg.test('After Training Model',' n_unique_tokens',' n_non_stop_unique_tokens')
	print('\n############# MODEL BUILT ##############\n')
	print('\tY=',linreg.theta1,' * X + ',linreg.theta0)

	choice=True
	while(choice):
		x=float(input("Enter The Value of x (0-100): "))
		x=x/100
		y=(linreg.theta1*x)+linreg.theta0
		print('\tY=',y)

		linreg.test('After Training Model',' n_unique_tokens',' n_non_stop_unique_tokens',x,y)

		c=input("Exit? 'Y'  : ")
		if(c=='Y' or c=='y'):
			choice=False












	



