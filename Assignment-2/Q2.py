from svmclass import *
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def best_crossvalid_param(train_labels,train_features,test_labels,test_features):
    x_train,x_val,y_train,y_val = train_test_split(train_features,train_labels,test_size = 0.1)
    cs = [1e-5,1e-3,1,5,10]
    test_acc = []
    val_acc = []
    for c in cs:
        print(c)
        prob = svm_problem(y_train, x_train)
        p = str(c)
        r = '-s 0 -t 2 -c '+p+' -g 0.05'
        param = svm_parameter(r)
        m = svm_train(prob,param)
        t_labels, t_acc, t_vals = svm_predict(test_labels, test_features, m)
        v_labels, v_acc, v_vals = svm_predict(y_val, x_val, m)
        print(t_acc[0],v_acc[0])
        test_acc.append(t_acc[0])
        val_acc.append(v_acc[0])
    cs = np.log(np.array(cs))
    test_acc = np.array(test_acc)
    val_acc = np.array(val_acc)
    return cs,test_acc,val_acc

def plot_c(cs=[],test_acc=[],val_acc=[]):
    fig = plt.figure()
    plt.xlabel("LogC")
    plt.ylabel("Accuracy")
    plt.plot(cs,test_acc,label = 'Test Set Accuracy')
    plt.plot(cs,val_acc,label = 'Validation Set Accuracy')
    plt.legend(loc = 'upper left')

def main(args):
    trainfile = args[0]
    testfile = args[1]
    part = args[2]
    mode = args[3]
    train_data = pd.read_csv(trainfile, header = None)
    test_data = pd.read_csv(testfile, header = None)
    if(part == 'binary_class' and mode == 'a'):
        train_features,train_labels = preprocess_data(train_data)
        test_features,test_labels = preprocess_data(test_data)
        bin_svm = SVM_binaryclass('l')
        bin_svm.train(train_features,train_labels)
        acc,pred,dist = bin_svm.predict(test_features,test_labels)
        print("The accuracy for linear kernel in binary classification is ",acc*100)
    elif(part == 'binary_class' and mode == 'b'):
        train_features,train_labels = preprocess_data(train_data)
        test_features,test_labels = preprocess_data(test_data)
        bin_svm = SVM_binaryclass('g',0.05)
        bin_svm.train(train_features,train_labels)
        acc,pred,dist = bin_svm.predict(test_features,test_labels)
        print("The accuracy for gaussian kernel in binary classification is ",acc*100)
    elif(part == 'binary_class' and mode == 'c'):
        train_features,train_labels = preprocess_data(train_data)
        test_features,test_labels = preprocess_data(test_data)
        print("The accuracy for linear kernel in libsvm is \n")
        libsvm('l',train_features,train_labels,test_features,test_labels)
        print("The accuracy for gaussian kernel in libsvm is \n")
        libsvm('g',train_features,train_labels,test_features,test_labels)
    elif(part == 'multi_class' and mode == 'a'):
        svm_dict = train_multiclass('svm_dictionary.pkl',train_data,0.05)
        acc,true,pred = predict_multiclass(svm_dict,0.05,test_data)
        print("The accuracy my model is ",acc*100)
    elif(part == 'multi_class' and mode =='b'):
        train_features,train_labels = preprocess_data(train_data,'m')
        test_features,test_labels = preprocess_data(test_data,'m')
        print("The accuracy for gaussian kernel in libsvm is \n")
        libsvm('g',train_features,train_labels,test_features,test_labels)
    elif(part == 'multi_class' and mode == 'c'):
        svm_dict = train_multiclass('svm_dictionary.pkl',train_data,0.05)
        acc,true,pred = predict_multiclass(svm_dict,0.05,test_data)
        print("The confusion matrix for the model is ",confusion_matrix(true,pred))
    elif(part == 'multi_class' and mode == 'd'):
        train_features,train_labels = preprocess_data(train_data,'m')
        test_features,test_labels = preprocess_data(test_data,'m')
        cs,test_acc,val_acc = best_crossvalid_param(train_labels,train_features,test_labels,test_features)
        plot_c(cs,test_acc,val_acc)
    else:
        print("Wrong part number")

if __name__ == "__main__":
    if(len(sys.argv)) !=5:
        sys.stderr("Wrong formal of command line. Enter like <file.py><trainfile.csv><tesfile.csv><binary_class or multi_class><part number>")
    else:
        main(sys.argv[1:])
