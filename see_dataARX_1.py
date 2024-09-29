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
y_train = np.load(y_train_file)


#### plotting the data  raw ###############
plt.plot(u_train ,label = "u_input",color='blue')
plt.plot(y_train, label = "y_output",color='red')
plt.xlabel('Sample index')
plt.ylabel('Signal')
plt.title('Training Data Raw')
plt.legend()
plt.figure()

## frequency analysis suggests a low-pass filter  ##
u_train_FFT = np.fft.fft(u_train)
y_train_FFT = np.fft.fft(y_train)
plt.plot(u_train_FFT ,label = "Input Spectrum",color='blue')
plt.plot(y_train_FFT, label = "Output Spectrum",color='red')
plt.xlabel('Sample index')
plt.ylabel('Signal')
plt.title('Training Data Raw Spectrum')
plt.legend()
plt.figure()


#############################################################
############    create X_train with regressors phi(k)        ###############
###########     You may assume n < 10, m < 10, d < 10        ###############
#############################################################

n = 8 #n>=0 , n of output points in buffer
m = 10 #m>=0  ,n of input points in buffer minus 1
d = 7 #d>=0  ,difference in indexes of last(most recent) output and input in buffer plus 1
N = len(y_train) #total number of output points in traininig data set
print("Number of data points is " + str(N))

##############################################################
#############  Regressor function           ##################
def regressor(k,output_vector,input_vector):

    phi = []
    aux = output_vector[k-n:k]
    phi.extend(aux[::-1])

    aux = input_vector[k-d-m : k-d +1]
    phi.extend(aux[::-1])
    
    return phi

p = max([n,m+d]) #number of features for linear regression


X_train = np.array([regressor(k,y_train,u_train) for k in range(p,N)])
y_train = y_train[p:N]

print("y-> "+ str(y_train.shape))
print("x-> "+ str(X_train.shape))

print()
plt.scatter(X_train[:, 0], y_train, color='blue',s = 1)
if(X_train.shape[1] > 1 ):
    plt.scatter(X_train[:, 1], y_train, color='green',s = 1)
if(X_train.shape[1] > 2 ):
    plt.scatter(X_train[:, 2], y_train, color='red',s = 1)
if(X_train.shape[1] > 3 ):
    plt.scatter(X_train[:, 3], y_train, color='orange',s = 1)
if(X_train.shape[1] > 4 ):
    plt.scatter(X_train[:, 4], y_train, color='purple',s = 1)
plt.xlabel('X train without outliers')
plt.ylabel('Y train without outliers')
plt.title('Training data(many dimensions overlaped)')
plt.figure()

####       Normalize       #################################
X_train_means = np.mean(X_train,axis = 0)    #Important for finale!
X_train_centered = X_train - X_train_means
X_train_centered_maxs = np.max(np.abs(X_train_centered), axis=0)  # Important for finale!
X_train_normalised = X_train_centered / X_train_centered_maxs
X_train_normalised_std_devs = np.std(X_train_centered, axis=0 ) #Important for finale!
X_train_normalised = X_train_normalised/ X_train_normalised_std_devs

y_train_mean = np.mean(y_train)          #Important for finale!
y_train_centered = y_train - y_train_mean
y_train_centered_max = np.max(y_train_centered)
y_train_normalised = y_train_centered / y_train_centered_max
y_train_normalised_std_dev = np.std(y_train_centered) #use standard deviation to normalise gaussian noise
y_train_normalised = y_train_normalised/ y_train_normalised_std_dev

#### plotting the data  normalised ###############
plt.scatter(X_train_normalised[:, 0], y_train_normalised, color='blue',s = 1)
if(X_train.shape[1] > 1 ):
    plt.scatter(X_train_normalised[:, 1], y_train_normalised, color='green',s = 1)
if(X_train.shape[1] > 2 ):
    plt.scatter(X_train_normalised[:, 2], y_train_normalised, color='red',s = 1)
if(X_train.shape[1] > 3 ):
    plt.scatter(X_train_normalised[:, 3], y_train_normalised, color='orange',s = 1)
if(X_train.shape[1] > 4 ):
    plt.scatter(X_train_normalised[:, 4], y_train_normalised, color='purple',s = 1)
plt.xlabel('X train normalised')
plt.ylabel('Y train normalised')
plt.title('Training Data Centered and Normalised')
plt.figure()

###############################################################
###### linear regression - minimize SSE           #############
## beta = (XtX)^-1 Xt Y                           #############
###############################################################

XtX =np.matmul(np.transpose(X_train_normalised),X_train_normalised)
XtX_inv = np.linalg.inv(XtX)
beta = np.matmul(XtX_inv,np.matmul(np.transpose(X_train_normalised),y_train_normalised )) 
print("Linear regression coefficients = " +str(beta))

y_train_normalised_prediction = np.matmul(X_train_normalised , beta)
SquaredErrors = (y_train_normalised - y_train_normalised_prediction )**2
SSE = np.sum(SquaredErrors)

r2 = 1- SSE/(np.sum(y_train_normalised**2))
print("r2 with simple linear regression= " + str(r2))

#### plotting the predictions normalised still ###############
plt.scatter(y_train_normalised, y_train_normalised_prediction, color='blue',s = 1)
plt.xlabel('Y_train')
plt.ylabel('Y_train_predicted')
plt.title('Training Data Prediction')
plt.figure()


### plotting error histogram for normalised predicition
Errors = y_train_normalised - y_train_normalised_prediction
counts, bin_edges = np.histogram(Errors, bins=100)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color='blue')
plt.xlabel('Error size')
plt.ylabel('Error count')
plt.title('Error histogram simple linear regression (normalised)')
plt.figure()

'''
##########################################################
##########           Ridge            ####################
##########################################################
rdg_alphas = []
rdg_scores = []
rdg_betas  = []
alpha_values = np.logspace(-4,0.5,100)
for a in alpha_values:
    rdg_alphas.append(a)
    rdg = Ridge(alpha = a)
    rdg.fit(X_train_normalised,y_train_normalised)
    rdg_scores.append( rdg.score(X_train_normalised,y_train_normalised) )
    rdg_betas.append(rdg.coef_)

plt.scatter(rdg_alphas,[b[0] for b in rdg_betas], color='blue',s = 1)
plt.scatter(rdg_alphas,[b[1] for b in rdg_betas], color='red',s = 1)
plt.scatter(rdg_alphas,[b[2] for b in rdg_betas], color='green',s = 1)
plt.scatter(rdg_alphas,[b[3] for b in rdg_betas], color='orange', s = 1)
plt.scatter(rdg_alphas,[b[4] for b in rdg_betas], color='purple',s = 1)
plt.xlabel('ridge_alphas')
plt.ylabel('Ridge coefficients')
plt.title('Ridge coefficients')
plt.xscale("symlog")
plt.yscale("symlog")
plt.figure()

# encontrar o melhor ridge com cross validation alpha
#o melhor k é 19
find_best_partition_k = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Suppress all warnings
    if find_best_partition_k == 1:
        # ver qual o numero otimo de particoes no crossvalidation(deu sempre o mesmo valor)
        print("Lengthy calculations for ridge.")
        ks_rdg_cv = np.linspace(2,30,29)
        best_r2s_rdg_cv = []
        for k in ks_rdg_cv:
            if k % 5 == 0:
                print("Evaluating k = " + str(k))
            rdg = Ridge()
            rdg_cv = GridSearchCV(rdg, param_grid={'alpha': alpha_values}, cv= int(k), scoring='r2')
            rdg_cv.fit(X_train_normalised, y_train_normalised)
            best_r2s_rdg_cv.append(rdg_cv.best_score_)
            k += 1
        plt.plot(ks_rdg_cv, best_r2s_rdg_cv, color='blue')
        plt.xlabel('Number of data partitions')
        plt.ylabel('Best r² found for various ridge alphas')
        plt.title('Finding best number of partitions for ridge cross validation')
        plt.figure()

#final ridge model
rdg = Ridge()
best_rdg = rdg
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Suppress all warnings
    print("Finding best ridge")
    rdg_cv = GridSearchCV(rdg, param_grid={'alpha': alpha_values}, cv= 19, scoring='r2')
    rdg_cv.fit(X_train_normalised, y_train_normalised)
    best_rdg = rdg_cv.best_estimator_


##########################################################
##########           Lasso            ####################
##########################################################
#see how the coefficients evolve
lss_alphas = []
lss_scores = []
lss_betas  = []
alpha_values = np.logspace(-10,-2,50)
for a in alpha_values:
    lss_alphas.append(a)
    lss = Lasso(alpha = a)
    lss.fit(X_train_normalised,y_train_normalised)
    lss_scores.append( lss.score(X_train_normalised,y_train_normalised) )
    lss_betas.append(abs(lss.coef_))

plt.scatter(lss_alphas,[b[0] for b in lss_betas], color='blue',s = 1)
plt.scatter(lss_alphas,[b[1] for b in lss_betas], color='red',s = 1)
plt.scatter(lss_alphas,[b[2] for b in lss_betas], color='green',s = 1)  #green is the weakest link
plt.scatter(lss_alphas,[b[3] for b in lss_betas], color='orange', s = 1)
plt.scatter(lss_alphas,[b[4] for b in lss_betas], color='purple',s = 1)
plt.xscale("log")
plt.yscale("log")
plt.xlabel('lasso_alphas')
plt.ylabel('Lasso coeficients')
plt.title('Lasso coefficients')
plt.figure()

#best is k = 18
find_best_partition_k = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Suppress all warnings
    if find_best_partition_k == 1:
        # checking the best number of partitions, k
        print("Lengthy calculations for lasso.")
        ks_lss_cv = np.linspace(2,30,29)
        best_r2s_lss_cv = []
        for k in ks_lss_cv:
            if k % 5 == 0:
                print("Evaluating k = " + str(k))
            lss = Lasso()
            lss_cv = GridSearchCV(lss, param_grid={'alpha': alpha_values}, cv= int(k), scoring='r2')
            lss_cv.fit(X_train_normalised, y_train_normalised)
            best_r2s_lss_cv.append(lss_cv.best_score_)
            k += 1
        plt.plot(ks_lss_cv, best_r2s_lss_cv, color='blue')
        plt.xlabel('Number of data partitions')
        plt.ylabel('Best r² found for various lasso alphas')
        plt.title('Finding best number of partitions for lasso cross validation')
        plt.figure()

#final lasso model
lss = Lasso()
best_lss = lss
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Suppress all warnings
    print("Finding best lasso.")
    lss_cv = GridSearchCV(lss, param_grid={'alpha': alpha_values}, cv= 18, scoring='r2')
    lss_cv.fit(X_train_normalised, y_train_normalised)
    best_lss = lss_cv.best_estimator_
'''

#############    function to get predictions(non normalized)  #############################

def prediction(X):
    #normalise X
    X = X - X_train_means
    X = X / X_train_centered_maxs
    X = X/ X_train_normalised_std_devs
    #apply simple linear model
    y = np.matmul(X,beta)
    #apply ridge regression model(cross validated)
    #y = best_rdg.predict(X)
    #apply lasso regression model(cross validated)
    #y = best_lss.predict(X)
    #denormalise y
    y = y * y_train_normalised_std_dev
    y = y * y_train_centered_max
    y = y + y_train_mean
    return y

#############    plotting prediction(same model will be used for test data)  ##############################
y_train_prediction = prediction(X_train)
r2_train = 1- (np.sum((y_train_prediction-y_train)**2))/(np.sum((y_train-np.mean(y_train))**2))
print("r2 for train is " +str(r2_train))
plt.scatter(y_train, prediction(X_train), color='blue',s = 1)
plt.xlabel('Y_train')
plt.ylabel('Y_train_predicted(non normalised)')
plt.title('Training Data Prediction')
plt.figure()

### plotting error histogram non normalised
Errors = y_train - y_train_prediction
counts, bin_edges = np.histogram(Errors, bins=100)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color='blue')
plt.xlabel('Error size')
plt.ylabel('Error count')
plt.title('Error histogram best linear regression(non normalised)')
plt.figure()

###################        producing prediction iterating instead of matrix operations   ###########
# y(k) = phi(k) * beta
y_test_prediction = []
k = 0
while (k < p):
    y_test_prediction.append(0)
    k+=1

while (k<len(u_test)):
    reg = regressor(k,y_test_prediction,u_test)
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
plt.figure()


plt.show()
