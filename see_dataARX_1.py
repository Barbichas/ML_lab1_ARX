#27.9.2024
#For Machine Learning labs
#Recursive linear regression for discrete time SLIT

import matplotlib.pyplot as plt
import numpy as np   #manipulating numpy arrays
import pandas as pd  #statistics fucntions
import seaborn as sns
from sklearn.linear_model import Ridge    #ridge regression
from sklearn.linear_model import Lasso    #lasso regression
from sklearn.model_selection import GridSearchCV, KFold   #cross validation
import warnings #for the heavy cross validation optimization computations
        

u_test_file =  "u_test.npy"
u_train_file = "u_train.npy"
y_train_file = "output_train.npy"


u_test  = np.load(u_test_file)
u_train = np.load(u_train_file)
y_train1 = np.load(y_train_file)


#### plotting the data  raw ###############
plt.plot(u_train ,label = "u_input",color='blue')
plt.plot(y_train1, label = "y_output",color='red')
plt.xlabel('Sample index')
plt.ylabel('Signal')
plt.title('Training Data Raw')
plt.legend()
plt.figure()

## frequency analysis suggests a low-pass filter  ##
u_train_FFT = np.fft.fft(u_train)
y_train_FFT = np.fft.fft(y_train1)
plt.plot(u_train_FFT ,label = "Input Spectrum",color='blue')
plt.plot(y_train_FFT, label = "Output Spectrum",color='red')
plt.xlabel('Sample index')
plt.ylabel('Signal')
plt.title('Training Data Raw Spectrum')
plt.legend()
plt.figure()


#############################################################
############    Global variables                             ###############
###########     You may assume n < 10, m < 10, d < 10        ###############
###########  best is n=9 , m=9 , d= 6
#############################################################

n = 5               #n>=0 , n of output points in buffer
m = 1              #m>=0  ,n of input points in buffer minus 1
d = 6               #d>=0  ,difference in indexes of last(most recent) output and input in buffer plus 1
N = len(y_train1)   #total number of output points in traininig data set
print("Number of data points is " + str(N))
p = 0               #number of features for linear regression

##############################################################
#############  Regressor function           ##################
def regressor(k,n,m,d,output_vector,input_vector):

    phi = []
    aux = output_vector[k-n:k]
    phi.extend(aux[::-1])

    aux = input_vector[k-d-m : k-d +1]
    phi.extend(aux[::-1])
    
    return phi

def trim_for_regressor(n , m , d, out_y, in_u):
    p = max([n,m+d]) #number of features for linear regression
    X_train = np.array([regressor(k,n,m,d,out_y,in_u) for k in range(p,N)])
    y_train = out_y[p:N]

    return X_train , y_train , p

###############################################################
###### linear regression - minimize SSE           #############
## beta = (XtX)^-1 Xt Y                           #############
###############################################################
def linear_regression(X,y):
    XtX =np.matmul(np.transpose(X),X)
    XtX_inv = np.linalg.inv(XtX)
    beta = np.matmul(XtX_inv,np.matmul(np.transpose(X),y )) 
    #print("Linear regression coefficients = " +str(beta))

    y_prediction = np.matmul(X , beta)
    SquaredErrors = (y - y_prediction )**2
    SSE = np.sum(SquaredErrors)

    r2 = 1- SSE/(np.sum(y**2))
    #print("r2 with simple linear regression= " + str(r2))

    return beta , r2

#########  Function for checking model stability      #########
def is_stable(ni , mi, di, beta):
    #poles at the roots of the polynomial
    # z^n + a1 z^(n-1) + a2 z^(n-2) +...+ an
    coefs = [1]
    if n >= 1:
        coefs = coefs + [-1* k for k in beta[0:n] ]
    poles = np.roots(coefs)
    for k in poles:
        if abs(k) >= 1:
            return 0
    return 1

X_train , y_train , p = trim_for_regressor(n ,m ,d ,y_train1,u_train)

print("y-> "+ str(y_train.shape))
print("x-> "+ str(X_train.shape))

plt.scatter(X_train[:, 0], y_train, color='blue',s = 1)
if(X_train.shape[1] > 1 ):
    plt.scatter(X_train[:, 1], y_train, color='green',s = 1)
if(X_train.shape[1] > 2 ):
    plt.scatter(X_train[:, 2], y_train, color='red',s = 1)
if(X_train.shape[1] > 3 ):
    plt.scatter(X_train[:, 3], y_train, color='orange',s = 1)
if(X_train.shape[1] > 4 ):
    plt.scatter(X_train[:, 4], y_train, color='purple',s = 1)
plt.xlabel('X train')
plt.ylabel('Y train')
plt.title('Training data(many dimensions overlaped)')
plt.figure()



#########  Finding parameters n ,m and d ######################
if (0):
    best_r2 = 0
    for ni in range(1,10):
        for mi in range(1,10):
            for di in range(1,10):
                print("Checking parameters-> n=" + str(ni) + " ,m= " + str(mi) + " ,d=" + str(di))
                Xi , yi , pi = trim_for_regressor(ni , mi, di ,y_train1, u_train)
                betai , r2i = linear_regression(Xi,yi)
                r2i = r2i * is_stable(ni , mi ,di , betai)
                if (r2i == 0):
                    print("Unstable!! -> n=" + str(ni) + " ,m= " + str(mi) + " ,d=" + str(di) + "!!!!!!!!!")
                if ( r2i > best_r2 ):
                    best_r2 = r2i
                    n = ni
                    m = mi
                    d = di
                    p = pi
                    beta = betai
    print("Best parameters-> n=" + str(n) + " ,m= " + str(m) + " ,d=" + str(d))
    print("Simple linear regression r2= " +str(best_r2))




X_train , y_train , p = trim_for_regressor(n ,m ,d ,y_train1,u_train)
beta , r2 = linear_regression(X_train, y_train)

#############    function to get predictions(non normalized)  #############################

def prediction(X):

    #apply simple linear model
    y = np.matmul(X,beta)

    return y

#############    plotting training prediction(same model will be used for test data)  ##############################
y_train_prediction = prediction(X_train)
r2_train = 1- (np.sum((y_train_prediction-y_train)**2))/(np.sum((y_train-np.mean(y_train))**2))
print("r2 for train is " +str(r2_train))
plt.scatter(y_train, prediction(X_train), color='blue',s = 1)
plt.xlabel('Y_train')
plt.ylabel('Y_train_predicted(non normalised)')
plt.title('Training Data Prediction')
plt.figure()

### plotting error on training histogram non normalised
Errors = y_train - y_train_prediction
counts, bin_edges = np.histogram(Errors, bins=100)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color='blue')
plt.xlabel('Error size')
plt.ylabel('Error count')
plt.title('Error histogram linear regression on training dataset')
plt.figure()

###################        producing prediction iterating with regressor   ###########
# y(k) = phi(k) * beta
y_test_prediction = []
k = 0
while (k < p):
    y_test_prediction.append(0)
    k+=1

while (k<len(u_test)):
    reg = regressor(k,n,m,d,y_test_prediction,u_test)
    #print("regressor(" + str(k) + ")= " + str(reg)) 
    aux = np.dot(reg, beta )
    y_test_prediction.append(aux)
    k += 1

####################       plotting test prediction    #######################################
plt.plot(u_test ,label = "u_input",color='blue')
plt.plot(y_test_prediction, label = "y_output",color='red')
plt.xlabel('Sample index')
plt.ylabel('Signal')
plt.title('Testing Data Prediction')
plt.legend()
plt.figure()

## frequency analysis suggests a low-pass filter  ##
u_train_FFT = np.fft.fft(u_test)
y_train_FFT = np.fft.fft(y_test_prediction)
plt.plot(u_train_FFT ,label = "Input Spectrum",color='blue')
plt.plot(y_train_FFT, label = "Output Spectrum",color='red')
plt.xlabel('Sample index')
plt.ylabel('Signal')
plt.title('Testing Data Predicted Spectrum')
plt.legend()


plt.show()
