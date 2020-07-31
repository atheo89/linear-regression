import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def cost_function(x,y,theta):
    
    a = 1/(2*m)
    b = np.sum(((x@theta)-y)**2)
    j = (a)*(b)
    return j

def gradient(x,y,theta):
    
    alpha = 0.00001
    iteration = 2000
    #gradient descend algorithm
    J_history = np.zeros([iteration, 1]);
    for iter in range(0,2000):
        
        error = (x @ theta) -y
        temp0 = theta[0] - ((alpha/m) * np.sum(error*x[:,0]))
        temp1 = theta[1] - ((alpha/m) * np.sum(error*x[:,1]))
        theta = np.array([temp0,temp1]).reshape(2,1)
        J_history[iter] = (1 / (2*m) ) * (np.sum(((x @ theta)-y)**2))   #compute J value for each iteration 

    return theta, J_history

path = "Data\\HappinessAlcoholConsumption_dataset.csv"
df= pd.read_csv(path)

# initial data analysis (IDA)
print(df.head())

shape = df.shape
print("The dimensions of the dataframe are: ",shape)
dataTypeSeries = df.dtypes 
print('Data type of each column of Dataframe :')
print(dataTypeSeries)


# look for any notable trends between all pairs of variables
sns.pairplot(df)
plt.show()

# filter only the required variables
A = df[['Beer_PerCapita','HappinessScore']]

# Convert the pandas data frame in to numpy array 
matrix = np.array(A.values,'float')

#x = 'Beer_PerCapita',y = 'HappinessScore'
X = matrix[:,0]
y = matrix[:,1]

#feature normalization
# input variable divided by maximum value among input values in x
# actually do is compressing all our input variable in to smaller and similar magnitude so that later computation will be faster and efficient .
X = X/(np.max(X)) 

plt.plot(X,y,'bo')
plt.ylabel('Happiness Score')
plt.xlabel('Alcohol consumption')
plt.legend(['Happiness Score'])
plt.title('Alcohol_Vs_Happiness')
plt.grid()
plt.show()

#initialising parameter
m = np.size(y)
X = X.reshape([122,1])
x = np.hstack([np.ones_like(X),X])
theta = np.zeros([2,1])
print(theta,'\n',m)

print(cost_function(x,y,theta))
theta , J = gradient(x,y,theta)
print(theta)
print(J)

#plot linear fit for our theta
plt.plot(X,y,'bo')
plt.plot(X,x@theta,'-')
plt.axis([0,1,3,7])
plt.ylabel('Happiness Score')
plt.xlabel('Alcohol consumption')
plt.legend(['HAPPY','LinearFit'])
plt.title('Alcohol_Vs_Happiness')
plt.grid()
plt.show()

# Let’s predict for new input value
predict1 = [1,(164/np.max(matrix[:,0]))] @ theta #normalising the input value, 1 is for intercept term so not need to normalise
print(predict1)
