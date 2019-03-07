import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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

def calculateH(theta,X):
    return theta @ np.transpose(X);

def mean_squared_error(Y,h):
    loss = np.array(Y-h);
    mse = np.sum(loss**2)/(2*Y.shape[0])
    return mse;

def gradient_descent(my_X,my_Y,eta,error):
    #Batch gradient descent algorithm
    norm_X = my_X;
    theta_0 = [0];
    theta_1 = [0];
    j_theta = [];
    theta = np.zeros((1,2));
    N = len(my_X);
    my_X = np.c_[ my_X , np.ones(N) ] #add the intercept term in all the examples.
    h_theta = calculateH(theta, my_X);
    loss = mean_squared_error(my_Y,h_theta);
    j_theta.append(loss);
    while(1):
        gradient = (my_Y - h_theta) @ my_X;
        theta = theta + (eta * gradient)/N;
        theta_0.append(theta[0][0]);
        theta_1.append(theta[0][1]);
        h_theta = calculateH(theta,my_X);
        new_loss = mean_squared_error(my_Y,h_theta);
        if(abs(new_loss - loss) < error): #stopping criteria
            loss = new_loss;
            j_theta.append(loss);
            break;
        loss = new_loss;
        j_theta.append(loss);
    return theta,theta_0,theta_1,j_theta;

def plot_linear(my_X,my_Y,theta):
    #for plotting the prediction
    norm_X = my_X;
    N = len(my_X);
    my_X = np.c_[ my_X , np.ones(N) ];
    h_theta = calculateH(theta,my_X); #Calculate the final prediction
    plt.figure(figsize=(10,10))
    plt.title("Fig 1a- Prediction of density of wine based on their acidity ",fontsize =20)
    data, = plt.plot(norm_X,my_Y,'r.',color = "blue", label = "Data");
    line, = plt.plot(norm_X,h_theta.reshape(100,),color = "red", label = "Hypothesis");
    plt.xlabel("Normalized acidity of Wine");
    plt.ylabel("Density of Wine");
    plt.legend(handles=[data, line]);
    plt.show(block = False);

def surface_plot(X,Y,par,theta,time):
    N = len(X);
    X = np.c_[X,np.ones(N)]
    theta_0 , theta_1 = np.mgrid[theta[0][0]-1:theta[0][0]+1:150j , theta[0][1]-1:theta[0][1]+1:150j]
    theta = np.c_[theta_0.flatten() , theta_1.flatten()]
    J = np.zeros(theta.shape[0])
    for i in range(0,theta.shape[0]):
        h_theta = calculateH(theta[i],X)
        J[i] = mean_squared_error(Y,h_theta)
    J = J.reshape(theta_0.shape)
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)
    plt.title("Fig 1b- Demonstration of the convergence of batch gradient descent over a surface plot of loss function w.r.t theta parameters",fontsize =11)
    ax.plot_surface(theta_0, theta_1, J,alpha = 0.5,cmap = cm.PuRd)
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'$J(\theta)$')
    time *= 10**3
    par = par.T

    def update_lines(num,par,lines):
        lines.set_data(par[0:2,:num])
        lines.set_3d_properties(par[2,:num])
        lines.set_marker("o")
        return lines
    lines, = [ax.plot(par[0,:], par[1,:], par[2,:],'b-',markersize = 5)[0]]
    anim = animation.FuncAnimation(fig, update_lines, len(par[0]), fargs=(par,lines),
                                   interval=time, repeat = True)
    plt.show(block = False)
    return anim

def plot_contour(X,Y,par,theta,time):
    N = len(X);
    X = np.c_[X,np.ones(N)]
    theta_0 , theta_1 = np.mgrid[theta[0][0]-1:theta[0][0]+1:150j , theta[0][1]-1:theta[0][1]+1:150j]
    theta = np.c_[theta_0.flatten() , theta_1.flatten()]
    J = np.zeros(theta.shape[0])
    for i in range(0,theta.shape[0]):
        h_theta = calculateH(theta[i],X)
        J[i] = mean_squared_error(Y,h_theta)
    L = J;
    J = J.reshape(theta_0.shape)
    fig = plt.figure(figsize=(10,10))
    plt.title("Fig 1c- Demonstration of the convergence of batch gradient descent over a contour plot sketched between theta parameters",fontsize = 10)
    plt.contour(theta_0, theta_1,J,levels = np.sort(par[:,2]))
    plt.gca().set_xlim(-1,1)
    plt.gca().set_ylim(-0,2)
    par = np.transpose(par)
    ax = plt.gca()
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    time = time * (10**3)
    def update_lines(num,lines):
        lines.set_data(par[0:2,:num])
        lines.set_marker("o")
        return lines
    lines = [ax.plot(par[0,:], par[1,:],'r-',markersize = 5)[0]]
    anim = animation.FuncAnimation(fig, update_lines, frames = len(par[0]), fargs=(lines),
                                   interval=time, repeat = True)
    plt.show()
    return anim

def main(argv):
    if(len(argv)<4):
        sys.stderr.write('Insufficient command line arguments. Please enter - <name.py><X_file><Y_file><learning rate><time_gap>');
        sys.exit(1); #command line input error
    X_data = argv[0]
    Y_data = argv[1]
    eta = float(argv[2])
    time = float(argv[3])
    my_X,my_Y = getData(X_data,Y_data);
    mu,sigma,norm_X = normalize_data(my_X); #Step 1 = normalize the data
    theta,theta_0,theta_1,j_theta = gradient_descent(norm_X,my_Y,eta,10**-12); #Step 2 = Apply Batch Gradient Descent on normalized data
    N = len(j_theta)
    par = np.c_[np.array(theta_0).reshape(N,1),np.array(theta_1).reshape(N,1),np.array(j_theta).reshape(N,1)]
    plot_linear(norm_X,my_Y,theta); #Step 3 = Plot the Graph
    surf = surface_plot(norm_X,my_Y,par,theta,time);
    cont = plot_contour(norm_X,my_Y,par,theta,time);
    return surf,cont

if __name__ == "__main__":
    surf,cont = main(sys.argv[1:]);
