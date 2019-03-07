import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pch

def getData(X,Y):
    my_X = np.genfromtxt(X,delimiter=',');
    my_Y = np.genfromtxt(Y,delimiter=',');
    return my_X,my_Y;

def normalize_data(X):
    #Used to normalize the data such that mean is 0 and the standard deviation is 1
    mu_X1 = np.mean(X[:,0:1]);
    sigma_X1 = np.std(X[:,0:1]);
    mu_X2 = np.mean(X[:,1:2]);
    sigma_X2 = np.std(X[:,1:2]);
    norm_X1 = np.array((X[:,0:1] - mu_X1)/sigma_X1);
    norm_X2 = np.array((X[:,1:2] - mu_X2)/sigma_X2);
    return mu_X1,mu_X2,sigma_X1,sigma_X2,norm_X1,norm_X2;

def calculateH(theta,X):
    val = theta @ np.transpose(X);
    sigm = 1/(1+np.exp(-1*val));
    return sigm;

def hessian(theta,X):
    h_theta = calculateH(theta,X);
    P = h_theta*(1-h_theta);
    W = np.diag(P.reshape(len(P[0]),));
    H = (np.transpose(X) @ W) @ X;
    return H; #always negative due to concavity of the log likelihood function.

def dell(Y,X,theta):
    h_theta = calculateH(theta,X);
    V = np.transpose(X) @ (Y - h_theta.reshape(Y.shape[0],));
    return V.reshape(-1,1);

def log_likelihood(Y,X,theta):
    Y= Y.reshape(1,Y.shape[0]);
    h_theta = calculateH(theta,X);
    ll_theta = Y*np.log(h_theta) + (1-Y) * np.log((1-h_theta));
    return np.sum(ll_theta);

def logistic_regression(my_X,my_Y,n):
    theta = np.zeros((1,n+1));
    N = len(my_X);
    pred = np.zeros(N);
    X = np.c_[ my_X , np.ones(N) ]
    ll_theta = log_likelihood(my_Y,X,theta);
    while(1):
        H_inv = np.linalg.inv(hessian(theta,X))
        V = dell(my_Y,X,theta);
        theta += np.transpose(H_inv @ V);
        ll_theta_new = log_likelihood(my_Y,X,theta);
        if(abs(ll_theta-ll_theta_new) < 10**-10):
            ll_theta = ll_theta_new
            break;
        ll_theta = ll_theta_new;
    for i in range(0,N):
        pred[i] = -1*(theta[0][0]*X[i][0]+ theta[0][2])/theta[0][1];
    return theta,pred;

def plot(norm_X,norm_X1,norm_X2,my_Y,pred):
    colors = ["r" if cls else "y" for cls in my_Y]
    plt.figure(figsize=(10,10))
    plt.title("Fig 3- Logistic Regression ",fontsize = 20)
    plt.scatter(norm_X[:,0],norm_X[:,1],c = colors,s = 20);
    line, = plt.plot(norm_X1,pred.reshape(norm_X1.shape[0],1),color="black",label = "Separator");
    plt.xlabel("Feature-1")
    plt.ylabel("Feature-2")
    class_0 = pch.Patch(color = "red", label = "Class-0")
    class_1 = pch.Patch(color = "yellow", label = "Class-1")
    plt.legend(handles = [class_0,class_1,line]);
    plt.savefig("log.png")
    plt.show();

def main(argv):
    if(len(argv)<2):
        sys.stderr.write('Insufficient command line arguments. Please enter - <name.py><X_file><Y_file>');
        sys.exit(1); #command line input error
    X_data = argv[0];
    Y_data = argv[1];
    my_X,my_Y = getData(X_data,Y_data);
    mu_X1,mu_X2,sigma_X1,sigma_X2,norm_X1,norm_X2 = normalize_data(my_X);
    norm_X = np.c_[norm_X1,norm_X2]
    theta,pred = logistic_regression(norm_X,my_Y,2);
    plot(norm_X,norm_X1,norm_X2,my_Y,pred);

if __name__ == "__main__":
    main(sys.argv[1:]);
