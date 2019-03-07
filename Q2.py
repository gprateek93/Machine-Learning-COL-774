import sys
import numpy as np
import matplotlib.pyplot as plt

def getData(X,Y):
    my_X = np.genfromtxt(X,delimiter=',');
    my_Y = np.genfromtxt(Y,delimiter=',');
    return my_X,my_Y;

def normalize_data(X):
    #Used to normalize the data such that mean is 0 and the standard deviation is 1
    mu_X = np.mean(X);
    sigma_X = np.std(X);
    norm_X = np.array((X - mu_X)/sigma_X);
    return mu_X, sigma_X, norm_X;

def get_weights(my_X,X_instance,tau):
    W = np.exp(-1*((X_instance-my_X)**2)/(2*(tau**2))); #calculate the weight matrix for each instance
    W = np.diag(W);
    return W;

def normal_equation_l(my_X,my_Y):
    X = np.c_[my_X,np.ones(len(my_X))]
    theta = ((np.linalg.inv((np.transpose(X) @ X))@np.transpose(X))@my_Y)
    pred_Y = theta @ np.transpose(X)
    return pred_Y

def normal_equation_w(my_X,my_Y,tau):
    m = np.min(my_X);
    n = np.max(my_X);
    nX = np.arange(m,n,0.0005); #for a smooth function
    M = len(nX);
    N = len(my_X);
    X = np.c_[ my_X , np.ones(N) ];
    pred_Y = np.zeros(M);
    for i in  range(0,M):
        my_W = get_weights(my_X,nX[i],tau);
        theta = ((((np.linalg.inv((np.transpose(X) @ my_W) @ X))@np.transpose(X))@my_W)@my_Y);
        pred_Y[i] = theta[0]*nX[i] + theta[1];
    return pred_Y,nX;

def plot_linear(my_X,pred_Y,my_Y):
    #for plotting the prediction
    plt.figure(figsize=(10,10))
    plt.title("Fig 2a - Linear regression hypothesis for given data",fontsize = 20)
    data,= plt.plot(my_X,my_Y,'b.',label = "Data");
    line,= plt.plot(my_X,pred_Y,color = "red",label = "Hypothesis");
    plt.legend(handles = [data,line])
    plt.savefig("2a.png")
    plt.show(block = False)

def plot(nX,pred_Y,my_X,my_Y):
    plt.figure(figsize=(10,10))
    plt.title("Fig 2b - Weighted linear regression hypothesis for given data",fontsize = 20)
    data,= plt.plot(my_X,my_Y,'b.',label = "Data");
    line,= plt.plot(nX,pred_Y,color = "red",label = "Hypothesis");
    plt.legend(handles = [data,line])
    plt.savefig("2b_08.png")
    plt.show()

def main(argv):
    if(len(sys.argv)<3):
        sys.stderr.write('Insufficient command line arguments. Please enter - <name.py><X_file><Y_file><tau>')
        sys.exit(1)
    X_data = argv[0];
    Y_data = argv[1];
    tau = float(argv[2]);
    my_X,my_Y = getData(X_data,Y_data);
    mu,sigma,norm_X = normalize_data(my_X)
    pred_Y = normal_equation_l(norm_X,my_Y);
    plot_linear(norm_X,pred_Y,my_Y)
    pred_Y,nX = normal_equation_w(norm_X,my_Y,tau);
    plot(nX,pred_Y, norm_X,my_Y);

if __name__ == "__main__":
    main(sys.argv[1:]);
