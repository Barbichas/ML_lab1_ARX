#27.9.2024
#For Machine Learning labs
#Recursive linear regression for discrete time SLIT

import matplotlib.pyplot as plt
import numpy as np   #manipulating numpy arrays
import pandas as pd  #statistics fucntions
import seaborn as sns
from sklearn.model_selection import cross_val_score  #cross validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge    #ridge regression
from sklearn.linear_model import Lasso    #lasso regression
from sklearn.model_selection import GridSearchCV, KFold   #cross validation search
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

n = 9               #n>=0 , n of output points in buffer
m = 9              #m>=0  ,n of input points in buffer minus 1
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

####  n , m d, parameters were chosen
X_train , y_train , p = trim_for_regressor(n ,m ,d ,y_train1,u_train)

##############      Cross validation for simple regression   #######################
if(0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        print("Lengthy calculations for simple linear regression.")
        # Different values of k for k-fold cross-validation
        ks_lin_cv = np.linspace(2, 50, 49)
        best_r2s_lin_cv = []
        
        for k in ks_lin_cv:
            if k % 5 == 0:
                print("Evaluating k = " + str(int(k)))
            
            # Initialize the simple linear regression model
            lin_reg = LinearRegression()
            
            # Perform k-fold cross-validation and store the average R² score
            r2_scores = cross_val_score(lin_reg, X_train, y_train, cv=int(k), scoring='r2')
            best_r2s_lin_cv.append(np.mean(r2_scores))
        
        # Plot the results
        plt.plot(ks_lin_cv, best_r2s_lin_cv, color='blue')
        plt.xlabel('Number of data partitions (k)')
        plt.ylabel('Average r2 score')
        plt.title('Finding best number of partitions for linear regression cross-validation')
        plt.figure()

'''
##########################################################
##########           Ridge            ####################
##########################################################
rdg_alphas = []
rdg_scores = []
rdg_betas  = []
alpha_values = np.logspace(-4,0.5,10)

#checking adequate range for alpha
if(1):
    alpha_values = np.logspace(-4,7,1000)
    for a in alpha_values:
        rdg_alphas.append(a)
        rdg = Ridge(alpha = a)
        rdg.fit(X_train,y_train)
        rdg_scores.append( rdg.score(X_train,y_train) )
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


# find best ridge, need alpha and k partitions
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Suppress all warnings
    if (0):
        alpha_values = np.logspace(-4,7,10)
        print("Lengthy calculations for ridge.")
        ks_rdg_cv = np.linspace(2,10,9)
        best_r2s_rdg_cv = []
        for k in ks_rdg_cv:
            if k % 5 == 0:
                print("Evaluating k = " + str(k))
            rdg = Ridge()
            rdg_cv = GridSearchCV(rdg, param_grid={'alpha': alpha_values}, cv= int(k), scoring='r2')
            rdg_cv.fit(X_train, y_train)
            best_r2s_rdg_cv.append(rdg_cv.best_score_)
            k += 1
        plt.plot(ks_rdg_cv, best_r2s_rdg_cv, color='blue')
        plt.xlabel('Number of data partitions')
        plt.ylabel('Best r² found for various ridge alphas')
        plt.title('Finding best number of partitions for ridge cross validation')
        plt.figure()

#final ridge model, best k = 7!!!!!!!!!
rdg = Ridge()
best_rdg = rdg
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Suppress all warnings
    print("Finding best ridge")
    rdg_cv = GridSearchCV(rdg, param_grid={'alpha': alpha_values}, cv= 7, scoring='r2')
    rdg_cv.fit(X_train, y_train)
    best_rdg = rdg_cv.best_estimator_

'''


##########################################################
##########           Lasso            ####################
##########################################################
#see how the coefficients evolve
lss_alphas = []
lss_scores = []
lss_betas  = []
alpha_values = np.logspace(-10,-2,10)

if(1):
    alpha_values = np.logspace(-10,5,1000)
    for a in alpha_values:
        lss_alphas.append(a)
        lss = Lasso(alpha = a)
        lss.fit(X_train,y_train)
        lss_scores.append( lss.score(X_train,y_train) )
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
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Suppress all warnings
    if (0):
        # checking the best number of partitions, k
        print("Lengthy calculations for lasso.")
        ks_lss_cv = np.linspace(2,10,9)
        best_r2s_lss_cv = []
        for k in ks_lss_cv:
            if k % 5 == 0:
                print("Evaluating k = " + str(k))
            lss = Lasso()
            lss_cv = GridSearchCV(lss, param_grid={'alpha': alpha_values}, cv= int(k), scoring='r2')
            lss_cv.fit(X_train, y_train)
            best_r2s_lss_cv.append(lss_cv.best_score_)
            k += 1
        plt.plot(ks_lss_cv, best_r2s_lss_cv, color='blue')
        plt.xlabel('Number of data partitions')
        plt.ylabel('Best r² found for various lasso alphas')
        plt.title('Finding best number of partitions for lasso cross validation')
        plt.figure()

#final lasso model best k=7!!!!!!!!!!!!!!!
lss = Lasso()
best_lss = Lasso()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Suppress all warnings
    print("Finding best lasso.")
    alpha_values = np.logspace(-10,-2,10)
    lss_cv = GridSearchCV(lss, param_grid={'alpha': alpha_values}, cv= 7, scoring='r2')
    lss_cv.fit(X_train, y_train)
    best_lss = lss_cv.best_estimator_
##########################################################################################



#############    function to get predictions(non normalized)  #############################
beta , r2 = linear_regression(X_train , y_train)
def prediction(X):

    #apply simple linear model
    #y = np.matmul(X,beta)
    #apply ridge regression model(cross validated)
    #y = best_rdg.predict(X)
    #apply lasso regression model(cross validated)
    y = best_lss.predict(X)

    return y

#############    plotting training prediction(same model will be used for test data)  ##############################
y_train_prediction = prediction(X_train)
r2_train = 1- (np.sum((y_train_prediction-y_train)**2))/(np.sum((y_train-np.mean(y_train))**2))
print("r2 for train is " +str(r2_train))
plt.scatter(y_train, y_train_prediction, color='blue',s = 1)
plt.xlabel('Y_train')
plt.ylabel('Y_train_predicted')
plt.title('Training Data Prediction vs True')
plt.figure()

plt.plot(y_train, color='blue',label="Original")
plt.plot(y_train_prediction, color = 'red',label= "Prediction")
plt.xlabel('Index')
plt.ylabel('Signal')
plt.title('Training Data Prediction')
plt.legend()
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
