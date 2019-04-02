import nodeclass as nc
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class DecisionTree:
    def __init__(self,node):
        #print('Decision Tree with a root created')
        self.root = node
        self.total_nodes = 0
    
    def train(self,root,data,mode = 'non-median'):
        #print(data.shape)
        if len(data[data[:,-1] == 1]) == len(data[:,-1]):
            root.assign_label(1)
            return 
        elif len(data[data[:,-1] == 0]) == len(data[:,-1]):
            root.assign_label(0)
            return
        elif data.shape[1] == 1: #corner case
            if len(data[data[:,-1] == 0]) > len(data[data[:,-1] == 1]):
                root.assign_label(0)
            else:
                root.assign_label(1)
            return 

        else:
            if len(data[data[:,-1] == 0]) > len(data[data[:,-1] == 1]):
                root.assign_label(0)
            else:
                root.assign_label(1)
            split_attribute = root.best_attribute(data,mode)
            #print(split_attribute)
            if(root.inf_gain < 1e-10):
                return
            if(mode == 'median'):
                #print(root.inf_gain)
                #print(split_attribute)
                med = np.median(data[:,split_attribute])
                root.median = med
                #print(med)
                left_data = np.array(data[data[:,split_attribute]<=med])
                print(data.shape,left_data.shape)
                right_data = np.array(data[data[:,split_attribute]>med])
                left_child = nc.DecisionTreeNode()
                right_child = nc.DecisionTreeNode()
                root.children[0] = left_child
                root.children[1] = right_child
                self.train(left_child,left_data,mode)
                self.train(right_child,right_data,mode)
            else:
                values = np.unique(data[:,split_attribute])
                for val in values:
                    val_data = np.array(data[data[:,split_attribute] == val])
                    #np.delete(val_data,split_attribute,1)
                    child = nc.DecisionTreeNode()
                    root.children[val] = child
                    self.train(child,val_data,mode)
            return

    def set_node_count(self):#,root):
        #bfs
        queue = []
        queue.insert(0,self.root)
        self.total_nodes += 1
        queue[-1].nodes_till_now = self.total_nodes
        while len(queue) != 0:
            top = queue[-1]
            #print(top.nodes_till_now)
            queue.pop()
            if top and top.children:
                for child in top.children.values():
                    self.total_nodes+=1
                    child.nodes_till_now = self.total_nodes
                    queue.insert(0,child)

        #dfs
        '''if(root):
            self.total_nodes += 1
            root.nodes_till_now = self.total_nodes
            for child in root.children.values():
                self.set_node_count(child)'''

    def predictUtil(self,root,data,y_pred,mode = 'non-median'):
        '''queue = []
        queue.insert(0,self.root)
        while(len(queue)!=0):
            top = queue[-1]
            queue.pop()
            y_pred.append(top.label)
            if(top == target):
                return
            if top.children:
                for child in top.children.values():
                    queue.insert(0,child)'''
    
        if root:# and self.flag == 0:
            y_pred.append(root.label)
            '''if(root == target):
                self.flag = 1
                return '''
            if root.children:
                val = data[root.split_attr]
                if(mode == 'median'):
                    if val <= root.median:
                        self.predictUtil(root.children[0],data,y_pred,mode)
                    else:
                        self.predictUtil(root.children[1],data,y_pred,mode)
                else:
                    if(val in root.children):
                        self.predictUtil(root.children[val],data,y_pred,mode)

    
    def predict(self,data,mode = 'non-median'):
        num_of_nodes_max = self.height(self.root)
        #print(num_of_nodes_max)
        acc = np.empty((1,3))
        yfinal = []
        for row in data:
            y_pred = []
            self.predictUtil(self.root,row,y_pred,mode)
            yfinal.append(y_pred[-1])
            y_true = row[-1]
            ##print(y_pred)
            # y_true = np.array(y_true)
            # y_pred = np.array(y_pred)
            # row_acc = np.where(y_pred == y_true,1,0)
            # np.append(acc,row_acc.reshape(1,-1),axis = 0)
        #     #print(acc.shape)
        # #print(acc.shape)
        # #acc = np.array(acc)
        # total = len(acc)
        
        '''print(total,num_of_nodes_max)
        acc = np.sum(acc,0) / total
        nodes = range(1,num_of_nodes+1)
        plt.plot(nodes,acc)'''
        acc = accuracy_score(data[:,-1],yfinal)
        print(acc*100)

    #def predict(self,data,root):

    def height(self,root):
        if(root):
            if root.children:
                m = 0
                for child in root.children.values():
                    h = self.height(child)
                    if h>m:
                        m = h
                return m+1
            else:
                return 1

    '''def predict(self,data,acc_history):
        #bfs
        queue = []
        queue.insert(0,self.root)
        while(len(queue) != 0):
            top = queue[-1]
            queue.pop()
            if top.children:
                for child in top.children.values():
                    queue.insert(0,child)
            yfinal = []
            for row in data:
                y_pred = []
                self.predictUtil(queue[0],row,y_pred)
                yfinal.append(y_pred[-1])
            acc = accuracy_score(data[:,-1],yfinal)
            print(acc*100, queue[0].nodes_till_now)
            acc_history.append((queue[0].nodes_till_now,acc*100))


        #dfs
        if(root and target):
            yfinal = []
            for row in data:
                y_pred = []
                self.flag = 0
                self.predictUtil(root,target,row,y_pred)
                yfinal.append(y_pred[-1])
            acc = accuracy_score(data[:,-1],yfinal)
            print(acc*100, target.nodes_till_now)
            acc_history.append((target.nodes_till_now,acc*100))
            if target.children:
                for child in target.children.values():
                    self.predict(data,root,child,acc_history)'''

    def traverse(self,root):
        if(root):
            if root.children:
                for child in root.children:
                    self.traverse(root.children[child])
            #else:
                #print(root.label)
    
    def plot_this(self,history):
        plt.plot(*zip(*history))
        plt.savefig('plot.png')
        plt.show()

    def play_with_library(self,data_train,data_test,data_val):
        xtrain = data_train[:,:-1]
        xtest = data_test[:,:-1]
        xval = data_val[:,:-1]
        
        ytrain = data_train[:,-1]
        ytest = data_test[:,-1]
        yval = data_val[:,-1]

        '''best_depth = -1
        best_sample_spit = -1
        best_leaf_split = -1
        best_acc = -1
        for depth in range(1,27):
            for sample_split in range(2,200,20):
                for leaf_split in range(2,200,20):
                    tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = depth, min_samples_split = sample_split, min_samples_leaf = leaf_split)
                    tree = tree.fit(xtrain,ytrain)
                    y_pred = tree.predict(xval)
                    acc = accuracy_score(yval,y_pred)
                    if acc > best_acc:
                        best_acc = acc
                        best_depth = depth
                        best_leaf_split = leaf_split
                        best_sample_spit = sample_split
        print(best_acc*100, best_depth, best_sample_spit, best_leaf_split)

        #train accuracy with best parameters
        tree_tr = DecisionTreeClassifier(criterion = 'entropy', max_depth = best_depth, min_samples_split = best_sample_spit, min_samples_leaf = best_leaf_split)
        tree_tr = tree_tr.fit(xtrain,ytrain)
        y_pred = tree.predict(xtrain)
        tr_acc = accuracy_score(ytrain,y_pred)
        print("Train accuracy with best parameters is ", tr_acc)

        #test accuracy with best parameters 
        tree_ts = DecisionTreeClassifier(criterion = 'entropy', max_depth = best_depth, min_samples_split = best_sample_spit, min_samples_leaf = best_leaf_split)
        tree_ts = tree_ts.fit(xtrain,ytrain)
        y_pred = tree.predict(xtest)
        ts_acc = accuracy_score(ytest,y_pred)
        print("Train accuracy with best parameters is ", ts_acc)'''
        tree_tr = DecisionTreeClassifier(criterion = 'entropy')
        tree_tr = tree_tr.fit(xtrain,ytrain)
        y_pred = tree_tr.predict(xtrain)
        tr_acc = accuracy_score(ytrain,y_pred)
        print("Test accuracy with best parameters is ", tr_acc)

    def random_forest(self,data_train,data_test,data_val):
        xtrain = data_train[:,:-1]
        xtest = data_test[:,:-1]
        xval = data_val[:,:-1]
        
        ytrain = data_train[:,-1]
        ytest = data_test[:,-1]
        yval = data_val[:,-1]

        forest = RandomForestClassifier(criterion = 'entropy')
        forest = forest.fit(xtrain,ytrain)
        y_pred = forest.predict(xtrain)
        y_pred2 = forest.predict(xtest)
        acc_tr = accuracy_score(ytrain,y_pred)
        acc_ts = accuracy_score(ytest,y_pred2)
        print("Training accuracy is ", acc_tr*100)
        print("Teat accuracy is ",acc_ts*100)