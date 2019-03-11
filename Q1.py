import utils
import numpy as np
import json
from sklearn.metrics import accuracy_score
def make_dictionary(filename,num_classes=5):
    reviews = utils.json_reader(filename)
    dictionary = {}
    total_words_per_class = np.zeros([num_classes])
    for review in reviews:
        words = review['text'].split(' ')
        total_words_per_class[int(review['stars'])-1]+=len(words)
        for word in words:
            if word not in dictionary:
                dictionary[word] = np.ones([num_classes])#laplace smoothening in numerator
                dictionary[word][int(review['stars'])-1]+=1
            else:
                dictionary[word][int(review['stars'])-1]+=1
    total_words_per_class = total_words_per_class + len(dictionary) #laplace smoothening in denominator
    return dictionary,total_words_per_class
dictionary,total_words = make_dictionary('train.json',5)
def get_parameters(dictionary,total_words):
    param = {}
    for key in dictionary:
        prob = np.log(np.divide(dictionary[key],total_words))
        param[key] = prob
    return param
param = get_parameters(dictionary,total_words)
def predict(filename,num_classes,parameter,total_words):
    reviews = utils.json_reader(filename)
    ground_truth = []
    predicted = []
    for review in reviews:
        ground_truth.append(int(review['stars']))
        words = review['text'].split(' ')
        prob_per_class = np.zeros([num_classes])
        for word in words:
            if word not in parameter:
                parameter[word] = np.log(np.divide(np.ones([num_classes]),total_words))
            prob_per_class = np.add(prob_per_class,parameter[word])
        predicted.append(np.argmax(prob_per_class)+1)
    return ground_truth, predicted

true,pred = predict('test.json',5,param,total_words)

accuracy = accuracy_score(true,pred)

print(accuracy)
