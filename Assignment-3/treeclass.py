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
        self.train_history = {}
        self.test_history = {}
        self.val_history = {}
    
    def train(self,root,data,train_data,test_data,valdata,mode = 'non-median'):
        #print(data.shape)
        self.total_nodes+=1
        print(self.total_nodes)
        
        if len(data[data[:,-1] == 1]) == len(data[:,-1]):
            root.assign_label(1)
            '''train_acc = self.predict(train_data,mode)
            test_acc = self.predict(test_data,mode)
            val_acc = self.predict(valdata,mode)
            self.train_history[self.total_nodes] = train_acc
            self.test_history[self.total_nodes] = test_acc
            self.val_history[self.total_nodes] = val_acc'''
            return 
        elif len(data[data[:,-1] == 0]) == len(data[:,-1]):
            root.assign_label(0)
            '''train_acc = self.predict(train_data,mode)
            test_acc = self.predict(test_data,mode)
            val_acc = self.predict(valdata,mode)
            self.train_history[self.total_nodes] = train_acc
            self.test_history[self.total_nodes] = test_acc
            self.val_history[self.total_nodes] = val_acc'''
            return
        elif data.shape[1] == 1: #corner case
            if len(data[data[:,-1] == 0]) > len(data[data[:,-1] == 1]):
                root.assign_label(0)
            else:
                root.assign_label(1)
            '''train_acc = self.predict(train_data,mode)
            test_acc = self.predict(test_data,mode)
            val_acc = self.predict(valdata,mode)
            self.train_history[self.total_nodes] = train_acc
            self.test_history[self.total_nodes] = test_acc
            self.val_history[self.total_nodes] = val_acc'''
            return 

        else:
            if len(data[data[:,-1] == 0]) > len(data[data[:,-1] == 1]):
                root.assign_label(0)
            else:
                root.assign_label(1)
            split_attribute = root.best_attribute(data,mode)
            '''train_acc = self.predict(train_data,mode)
            test_acc = self.predict(test_data,mode)
            val_acc = self.predict(valdata,mode)
            self.train_history[self.total_nodes] = train_acc
            self.test_history[self.total_nodes] = test_acc
            self.val_history[self.total_nodes] = val_acc'''
            #print(split_attribute)
            if(root.inf_gain < 1e-10):
                return
            if(mode == 'median'):
                #print(root.inf_gain)
                print(split_attribute)
                med = np.median(data[:,split_attribute])
                #print(med)
                root.median = med
                '''train_acc = self.predict(train_data,mode)
                test_acc = self.predict(test_data,mode)
                val_acc = self.predict(valdata,mode)
                self.train_history[self.total_nodes] = train_acc
                self.test_history[self.total_nodes] = test_acc
                self.val_history[self.total_nodes] = val_acc'''
                #print(med)
                left_data = np.array(data[data[:,split_attribute]<=med])
                #print(data.shape,left_data.shape)
                right_data = np.array(data[data[:,split_attribute]>med])
                left_child = nc.DecisionTreeNode()
                right_child = nc.DecisionTreeNode()
                root.children[0] = left_child
                root.children[1] = right_child
                self.train(left_child,left_data,train_data,test_data,valdata,mode)
                self.train(right_child,right_data,train_data,test_data,valdata,mode)
            else:
                values = np.unique(data[:,split_attribute])
                for val in values:
                    val_data = np.array(data[data[:,split_attribute] == val])
                    #np.delete(val_data,split_attribute,1)
                    child = nc.DecisionTreeNode()
                    root.children[val] = child
                    self.train(child,val_data,train_data,test_data,valdata,mode)
            return

    def node_count(self):#,root):
        #bfs
        queue = []
        total_nodes = 0
        queue.insert(0,self.root)
        total_nodes += 1
        #queue[-1].nodes_till_now = self.total_nodes
        while len(queue) != 0:
            top = queue[-1]
            #print(top.nodes_till_now)
            queue.pop()
            if top and top.children:
                for child in top.children.values():
                    total_nodes+=1
                    #child.nodes_till_now = self.total_nodes
                    queue.insert(0,child)
        return total_nodes

    def predictUtil(self,root,data,y_pred,mode = 'non-median'):
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
        #print(num_of_nodes_max)
        acc = np.empty((1,3))
        yfinal = []
        for row in data:
            y_pred = []
            self.predictUtil(self.root,row,y_pred,mode)
            yfinal.append(y_pred[-1])
        #print(yfinal)
        acc = accuracy_score(data[:,-1],yfinal)
        print(acc*100)
        return acc*100

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

    def traverse(self,root):
        if(root):
            if root.children:
                for child in root.children:
                    self.traverse(root.children[child])
            #else:
                #print(root.label)
    
    def plot_this(self):
        plt.figure()
        plt.xlabel('Number of nodes')
        plt.ylabel('Accuracy in percentage')
        plt.plot(self.train_history.keys(),self.train_history.values(),label = 'Train accuracy')
        plt.plot(self.test_history.keys(),self.test_history.values(),label = 'Test accuracy')
        plt.plot(self.val_history.keys(),self.val_history.values(),label = 'Validation accuracy')
        plt.legend(loc = 'upper left')
        #plt.plot(*zip(*history))
        plt.savefig('plot.png')
        plt.show()

    def play_with_library(self,data_train,data_test,data_val):
        xtrain = data_train[:,:-1]
        xtest = data_test[:,:-1]
        xval = data_val[:,:-1]
        
        ytrain = data_train[:,-1]
        ytest = data_test[:,-1]
        yval = data_val[:,-1]

        depth_var = {}
        sample_split_var = {}
        leaf_split_var = {}
        best_depth = 0
        best_leaf_split = 0
        best_sample_split = 0
        best_acc = -1
        for depth in range(1,27):
            tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = depth)
            tree = tree.fit(xtrain,ytrain)
            y_pred = tree.predict(xval)
            acc = accuracy_score(yval,y_pred)
            if(acc>best_acc):
                best_acc = acc
                best_depth = depth
            depth_var[depth] = acc*100
        best_acc = -1
        for sample_split in range(2,200):
            tree = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = sample_split)
            tree = tree.fit(xtrain,ytrain)
            y_pred = tree.predict(xval)
            acc = accuracy_score(yval,y_pred)
            if(acc>best_acc):
                best_acc = acc
                best_sample_split = sample_split
            sample_split_var[sample_split] = acc*100
        best_acc = -1
        for leaf_split in range(2,200):
            tree = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = leaf_split)
            tree = tree.fit(xtrain,ytrain)
            y_pred = tree.predict(xval)
            acc = accuracy_score(yval,y_pred)
            if(acc>best_acc):
                best_acc = acc
                best_leaf_split = leaf_split
            leaf_split_var[leaf_split] = acc*100
        
        plt.figure()
        plt.xlabel('depth')
        plt.ylabel('accuracy in percentage')
        plt.plot(depth_var.keys(),depth_var.values())
        plt.show()

        plt.figure()
        plt.xlabel('Min sample split')
        plt.ylabel('accuracy in percentage')
        plt.plot(sample_split_var.keys(),sample_split_var.values())
        plt.show()

        plt.figure()
        plt.xlabel('Min samples leaf')
        plt.ylabel('accuracy in percentage')
        plt.plot(leaf_split_var.keys(),leaf_split_var.values())
        plt.show()

        print(best_depth,best_sample_split,best_leaf_split)

        tree_tr = DecisionTreeClassifier(criterion = 'entropy', max_depth = best_depth, min_samples_split = best_sample_split, min_samples_leaf = best_leaf_split)
        tree_tr = tree_tr.fit(xtrain,ytrain)
        y_pred = tree_tr.predict(xtrain)
        tr_acc = accuracy_score(ytrain,y_pred)
        print("Train accuracy with best parameters is ", tr_acc)

        tree_ts = DecisionTreeClassifier(criterion = 'entropy', max_depth = best_depth, min_samples_split = best_sample_split, min_samples_leaf = best_leaf_split)
        tree_ts = tree_ts.fit(xtrain,ytrain)
        y_pred = tree_ts.predict(xtest)
        ts_acc = accuracy_score(ytest,y_pred)
        print("Train accuracy with best parameters is ", ts_acc)

        tree_tv = DecisionTreeClassifier(criterion = 'entropy', max_depth = best_depth, min_samples_split = best_sample_split, min_samples_leaf = best_leaf_split)
        tree_tv = tree_tv.fit(xtrain,ytrain)
        y_pred = tree_tv.predict(xtest)
        tv_acc = accuracy_score(ytest,y_pred)
        print("Train accuracy with best parameters is ", tv_acc)


    def random_forest(self,data_train,data_test,data_val):
        xtrain = data_train[:,:-1]
        xtest = data_test[:,:-1]
        xval = data_val[:,:-1]
        
        ytrain = data_train[:,-1]
        ytest = data_test[:,-1]
        yval = data_val[:,-1]

        best_n_estimator = 0
        best_bootstrap = 0
        best_max_features = 5
        best_acc = -1

        n_est_var = {}
        bootstrap_var = {}
        max_feature_var = {}

        for estimator in range(1,40):
            print(estimator)
            forest = RandomForestClassifier(criterion = 'entropy',n_estimators= estimator)
            forest = forest.fit(xtrain,ytrain)
            y_pred = forest.predict(xval)
            acc = accuracy_score(yval,y_pred)
            n_est_var[estimator] = acc*100
            if(acc>best_acc):
                best_acc = acc
                best_n_estimator = estimator
        print("done")
        best_acc = -1

        best_acc = -1

        boot = [True,False]
        for strap in boot:
            forest = RandomForestClassifier(criterion = 'entropy',bootstrap =strap)
            forest = forest.fit(xtrain,ytrain)
            y_pred = forest.predict(xval)
            acc = accuracy_score(yval,y_pred)
            if strap == True:
                bootstrap_var[1] = acc*100
            else:
                bootstrap_var[0] = acc*100
            if(acc>best_acc):
                best_acc = acc
                if strap == True:
                    best_bootstrap = 1
                else:
                    best_bootstrap = 0

        print(best_n_estimator,best_max_features,best_bootstrap)

        plt.figure()
        plt.xlabel("n_estimators")
        plt.ylabel("accuracies in percentage")
        plt.plot(n_est_var.keys(),n_est_var.values())
        plt.show()

        plt.figure()
        plt.xlabel("Bootstrap 0 for False and 1 for True")
        plt.ylabel("accuracies in percentage")
        plt.plot(bootstrap_var.keys(),bootstrap_var.values())
        plt.show()

        if best_bootstrap == 1:
            best_bootstrap = True
        else:
            best_bootstrap = False

        forest_tr = RandomForestClassifier(criterion = 'entropy',n_estimators = best_n_estimator, max_features= best_max_features, bootstrap= best_bootstrap)
        forest_tr = forest_tr.fit(xtrain,ytrain)
        y_pred_tr = forest_tr.predict(xtrain)
        y_pred_ts = forest_tr.predict(xtest)
        y_pred_tv = forest_tr.predict(xval)

        acc_tr = accuracy_score(ytrain,y_pred_tr)
        acc_ts = accuracy_score(ytest,y_pred_ts)
        acc_tv = accuracy_score(yval,y_pred_tv)
        print("Training accuracy is ", acc_tr*100)
        print("Test accuracy is ",acc_ts*100)
        print("val accuracy is ",acc_tv*100)
        
    
    def get_list_of_nodes(self):
        nodes = []
        queue = []
        queue.insert(0,self.root)
        while(len(queue) != 0):
            top = queue[-1]
            queue.pop()
            nodes.append(top)
            if top.children:
                for child in top.children.values():
                    queue.insert(0,child)
        return nodes

    def post_pruning(self,train_data,test_data,val_data):
        self.train_history = {}
        self.test_history = {}
        self.val_history = {}
        nodes = self.get_list_of_nodes()
        nodes.reverse()
        while True:
            best_node_to_prune = None
            best_accuracy_gained = -1
            total_nodes =self.node_count()
            acc_before = self.predict(val_data)
            train_acc = self.predict(train_data)
            test_acc = self.predict(test_data)
            self.train_history[total_nodes] = train_acc
            self.test_history[total_nodes] = test_acc
            self.val_history[total_nodes] = acc_before
            for node in nodes:
                if node.children:
                    acc_b = self.predict(val_data)
                    my_children = node.children
                    node.children = []
                    acc_a = self.predict(val_data)
                    node.children = my_children
                    acc_gained = acc_a - acc_b
                    if acc_gained > best_accuracy_gained:
                        best_node_to_prune = node
                        best_accuracy_gained = acc_gained
            if best_node_to_prune.children:
                best_node_to_prune.children = []
            acc_after = self.predict(val_data)
            if(acc_after - acc_before) < 1e-10:
                break
        total_nodes =self.node_count()
        acc_before = self.predict(val_data)
        train_acc = self.predict(train_data)
        test_acc = self.predict(test_data)
        print(train_acc,test_acc,acc_before)
        self.train_history[total_nodes] = train_acc
        self.test_history[total_nodes] = test_acc
        self.val_history[total_nodes] = acc_before


                    