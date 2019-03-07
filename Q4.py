import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pch

def getData(X,Y):
    my_X = np.genfromtxt(X,dtype= (int,int));
    my_Y = np.genfromtxt(Y,dtype = "str");
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

def bernoulli_var(Y):
    #change the Y labels into integer values of 0 and 1
    Z = np.where(Y == "Alaska",0,1);
    return Z;

def calculate_linear_gda_param(X,Y):
    Y_0 = np.where(Y == 0,1,0);
    Y_1 = np.where(Y == 1,1,0);
    phi = np.sum(Y)/len(Y);
    mu_0 = ((Y_0.reshape(1,len(Y))) @ X)/np.sum(Y_0);
    mu_1 = ((Y_1.reshape(1,len(Y))) @ X)/np.sum(Y_1);
    E_0 = np.transpose(X-mu_0) @ np.diag(Y_0)
    E_1 = np.transpose(X-mu_1) @ np.diag(Y_1)
    E = (E_0+E_1) @ np.transpose(E_0+E_1)/len(Y)
    return phi,mu_0,mu_1,E;

def get_linear_separator(phi,mu_0,mu_1,E,X):
    E_inv = np.linalg.inv(E)
    theta_1 = 2*(mu_0-mu_1) @ E_inv
    theta_0 = mu_1 @ E_inv @ np.transpose(mu_1) - mu_0 @ E_inv @ np.transpose(mu_0) + 2*np.log((1-phi)/phi)
    pred = -1*(theta_1[0][0]*X[:,0:1] + theta_0)/theta_1[0][1]
    return theta_1,theta_0,pred;

def calculate_quadratic__gda_param(X,Y):
    Y_0 = np.where(Y == 0,1,0);
    Y_1 = np.where(Y == 1,1,0);
    phi = np.sum(Y)/len(Y);
    mu_0 = ((Y_0.reshape(1,len(Y))) @ X)/np.sum(Y_0);
    mu_1 = ((Y_1.reshape(1,len(Y))) @ X)/np.sum(Y_1);
    E_0 = (np.transpose(X-mu_0) @ np.diag(Y_0) @ (X-mu_0))/np.sum(Y_0)
    E_1 = (np.transpose(X-mu_1) @ np.diag(Y_1) @ (X-mu_1))/np.sum(Y_1)
    return phi,mu_0,mu_1,E_0,E_1;

def get_quadratic_separator(phi,mu_0,mu_1,E_0,E_1,X):
    E_0_inv = np.linalg.inv(E_0)
    E_1_inv = np.linalg.inv(E_1)
    theta_2 = E_1_inv - E_0_inv
    theta_1 = 2 * (mu_0 @ E_0_inv - mu_1 @ E_1_inv)
    theta_0 = mu_1 @ E_1_inv @ np.transpose(mu_1) - mu_0 @ E_0_inv @ np.transpose(mu_0) + np.log(np.linalg.det(E_1)/np.linalg.det(E_0))+2*np.log((1-phi)/phi)
    m = np.min(X[:,0])
    n = np.max(X[:,0])
    m1 = -8 #np.min(X[:,1])
    n1 = np.max(X[:,1])
    R,S = np.mgrid[m:n:100j,m1:n1:100j]
    T = np.c_[R.flatten(),S.flatten()];
    pred = np.zeros(10000);
    for i in range(0,10000):
        pred[i] = -1* (T[i] @ theta_2 @ np.transpose(T[i]) + theta_1 @ np.transpose(T[i]) + theta_0);
    pred = pred.reshape(R.shape)
    return R,S,pred;

def plot_scatter(X,Y):
    colors = ["r" if cls else "y" for cls in Y]
    plt.scatter(X[:,0:1], X[:,1:2], c=colors)
    class_0 = pch.Patch(color = "red")
    class_1 = pch.Patch(color = "yellow")
    plt.xlabel("Growth ring diameters of Salmons in Fresh Water",fontsize = 15)
    plt.ylabel("Growth ring diameters of Salmons in Marine Water",fontsize = 15)
    plt.legend([class_0,class_1],['Canada', 'Alaska'],fontsize = 15,loc = "upper left")
    plt.show()

def plot_linear(X,Y,Z):
    colors = ["r" if cls else "y" for cls in Y]
    plt.scatter(X[:,0:1], X[:,1:2], c=colors)
    class_0 = pch.Patch(color = "red")
    class_1 = pch.Patch(color = "yellow")
    line, = plt.plot(X[:,0],Z);
    plt.xlabel("Growth ring diameters of Salmons in Fresh Water",fontsize = 15)
    plt.ylabel("Growth ring diameters of Salmons in Marine Water",fontsize = 15)
    plt.legend([class_0,class_1,line],['Canada', 'Alaska', 'Linear Separator'],fontsize = 15,loc = "upper left")
    plt.show()

def plot(X,Y,Z,R,S,quad):
    X_0 = np.where(Y == 0);
    X_1 = np.where(Y == 1);
    plt.figure(figsize=(10,10))
    colors = ["r" if cls else "y" for cls in Y]
    plt.title("Fig 4- Plot of Salmons in Alaska and Canada based on the growth ring diameters in fresh and marine waters",fontsize =13)
    plt.scatter(X[:,0:1], X[:,1:2], c=colors)
    class_0 = pch.Patch(color = "red")
    class_1 = pch.Patch(color = "yellow")
    line, = plt.plot(X[:,0],Z);
    CS = plt.contour(R,S,quad,[0])
    h,_ = CS.legend_elements() #making CS iterable
    plt.xlabel("Growth ring diameters of Salmons in Fresh Water",fontsize = 15)
    plt.ylabel("Growth ring diameters of Salmons in Marine Water",fontsize = 15)
    plt.legend([class_0,class_1,line,h[0]],['Canada', 'Alaska', 'Linear Separator', 'Quadratic Separator'],fontsize = 15,loc = "upper left")
    plt.savefig("gda.png")
    plt.show();

def main(argv):
    if(len(argv)<3):
        sys.stderr.write('Insufficient command line arguments. Please enter - <name.py><X_file><Y_file><mode>');
        sys.exit(1); #command line input error
    X_data  = argv[0]
    Y_data  = argv[1]
    mode  = int(argv[2])
    my_X,my_Y = getData(X_data,Y_data)
    Z = bernoulli_var(my_Y);
    mu_X1,mu_X2,sigma_X1,sigma_X2,norm_X1,norm_X2 = normalize_data(my_X);
    norm_X = np.c_[norm_X1,norm_X2];
    if(mode == 0):
        phi,mu_0,mu_1,E = calculate_linear_gda_param(norm_X,Z)
        A,B,C = get_linear_separator(phi,mu_0,mu_1,E,norm_X)
        plot_scatter(norm_X,Z)
        plot_linear(norm_X,Z,C)
    else:
        phi,mu_0,mu_1,E = calculate_linear_gda_param(norm_X,Z)
        A,B,C = get_linear_separator(phi,mu_0,mu_1,E,norm_X)
        phi,mu_0,mu_1,E_0,E_1 = calculate_quadratic__gda_param(norm_X,Z)
        R,S,pred_quad = get_quadratic_separator(phi,mu_0,mu_1,E_0,E_1,norm_X)
        plot(norm_X,Z,C,R,S,pred_quad);

if __name__ == "__main__":
    main(sys.argv[1:]);
