import utils
import numpy as np
import json
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from random import randint
import os
import pickle
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import sys
def ngrams(s,n):
    return list(zip(*([s[i:] for i in range(n)])))
def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(sentence.lower())

    nltk_tagged = nltk.pos_tag(tokens)

    word_tag = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    lemm_words = []
    for word, tag in word_tag:
        if tag is None:
            lemm_words.append(word)
        else:
            lemm_words.append(lemmatizer.lemmatize(word, tag))
    return lemm_words

def make_dictionary(filename,num_classes=5,mode='a'):
    class_count = np.zeros([num_classes])
    reviews = utils.json_reader(filename)
    dictionary = {}
    total_words_per_class = np.zeros([num_classes])
    m = 0 #total number of examples/reviews
    for review in reviews:
        m+=1
        class_count[int(review['stars'])-1]+=1
        if mode == 'a' or mode == 'b' or mode == 'c':
            words = review['text'].split(' ') #No feature extraction
        elif mode == 'd':
            words = utils.getStemmedDocuments(review['text']) #With stemming and removal of stopwords
        elif mode == 'e1':
            words = review['text']
            words = utils.getStemmedDocuments(review['text'])
            words = ngrams(words,2) #Using bigrams on stemming . Can also be used for n-grams.
        elif mode == 'e2':
            words = review['text']
            words = lemmatize(words) #using lemmatizing
            words = ngrams(words,2)  #Using bigrams on lemmatizing . Can also be used for n-grams.

        total_words_per_class[int(review['stars'])-1]+=len(words)
        for word in words:

            if word not in dictionary:
                dictionary[word] = np.ones([num_classes])#laplace smoothening in numerator
                dictionary[word][int(review['stars'])-1]+=1
            else:
                dictionary[word][int(review['stars'])-1]+=1
    total_words_per_class = total_words_per_class + len(dictionary) #laplace smoothening in denominator
    return dictionary,total_words_per_class,class_count,m

def get_parameters(dictionary,total_words,class_count,total_examples,pickle_file):
    if os.path.exists(pickle_file):
        inp = open(pickle_file,'rb')
        parameter = pickle.load(inp)
        param = parameter[0]
        total_words = parameter[1]
        phi_y = parameter[2]
        if pickle_file == 'parameters_naive_lemmatize_bigram.pkl':
            phi_y = parameter[1]
            total_words = parameter[2]
        inp.close()
    else:
        param = {}
        phi_y = np.log(class_count/total_examples)
        for key in dictionary:
            prob = np.log(np.divide(dictionary[key],total_words))
            param[key] = prob
        parameter = [param,total_words,phi_y]
        output = open(pickle_file,'wb')
        pickle.dump(parameter,output)
        output.close()
    return param,total_words,phi_y

def predict(filename,num_classes,parameter,phi_y,total_words,mode):
    reviews = utils.json_reader(filename)
    ground_truth = []
    predicted = []
    for review in reviews:
        ground_truth.append(int(review['stars']))
        if mode == 'a' or mode == 'd' or mode == 'e1' or mode == 'e2':
            if mode == 'a':
                words = review['text'].split(' ')
            elif mode == 'd':
                words = utils.getStemmedDocuments(review['text'])
            elif mode == 'e1':
                words = utils.getStemmedDocuments(review['text'])
                words = ngrams(words,2)
            elif mode == 'e2':
                words = review['text']
                words = lemmatize(words)
                words = ngrams(words,2)
            prob_per_class = np.zeros([num_classes])
            for word in words:
                if word not in parameter:
                    parameter[word] = np.log(np.divide(np.ones([num_classes]),total_words))
                prob_per_class = np.add(prob_per_class,parameter[word])
            prob_per_class = np.add(prob_per_class,phi_y)
            predicted.append(np.argmax(prob_per_class)+1)
        elif mode == 'b1':
            predicted.append(randint(1,5))
        elif mode == 'b2':
            predicted.append(np.argmax(total_words)+1)
    return ground_truth, predicted

def f1(true,pred,mode):
    if mode == 'None':
        acc = f1_score(true,pred,average = None)
    else:
        acc = f1_score(true,pred,average = 'macro')
    return acc

def main(args):
    print("here")
    train_file = args[0]
    test_file = args[1]
    mode = args[2]
    dictionary = {}
    total_words = np.zeros([5])
    class_count = np.zeros([5])
    total_examples = 0
    if(mode == 'a'):
        if os.path.exists("parameters_naive.pkl") == 0:
            dictionary,total_words,class_count,total_examples = make_dictionary(train_file,5,mode)
        param,total_words,phi_y = get_parameters(dictionary,total_words,class_count,total_examples,'parameters_naive.pkl')
        true,pred = predict(test_file,5,param,phi_y,total_words,mode)
        accuracy = accuracy_score(true,pred)
        print("The test accuracy is ",accuracy*100)
        true,pred = predict(train_file,5,param,phi_y,total_words,mode)
        accuracy = accuracy_score(true,pred)
        print("The train accuracy is ",accuracy*100)

    elif(mode == 'b'):
        if os.path.exists("parameters_naive.pkl") == 0:
            dictionary,total_words,class_count,total_examples = make_dictionary(train_file,5,'a')
        param,total_words,phi_y = get_parameters(dictionary,total_words,class_count,total_examples,'parameters_naive.pkl')
        true,pred = predict(test_file,5,param,phi_y,total_words,'b1')
        accuracy = accuracy_score(true,pred)
        print("The random guessing accuracy is ",accuracy*100)
        true,pred = predict(train_file,5,param,phi_y,total_words,'b2')
        accuracy = accuracy_score(true,pred)
        print("The majority class guessing is ",accuracy*100)
    elif(mode == 'c'):
        if os.path.exists("parameters_naive.pkl") == 0:
            dictionary,total_words,class_count,total_examples = make_dictionary(train_file,5,'a')
        param,total_words,phi_y = get_parameters(dictionary,total_words,class_count,total_examples,'parameters_naive.pkl')
        true,pred = predict(test_file,5,param,phi_y,total_words,'a')
        conf = confusion_matrix(true,pred)
        print("The confusion matrix is ",conf)
    elif(mode == 'd'):
        if os.path.exists("parameters_naive_stemming.pkl") == 0:
            dictionary,total_words,class_count,total_examples = make_dictionary(train_file,5,mode)
        param,total_words,phi_y = get_parameters(dictionary,total_words,class_count,total_examples,'parameters_naive_stemming.pkl')
        true,pred = predict(test_file,5,param,phi_y,total_words,mode)
        accuracy = accuracy_score(true,pred)
        print("The test accuracy is ",accuracy*100)
    elif(mode == 'e'):
        if os.path.exists("parameters_naive_stemming_bigram.pkl") == 0:
            dictionary,total_words,class_count,total_examples = make_dictionary(train_file,5,'e1')
        param,total_words,phi_y = get_parameters(dictionary,total_words,class_count,total_examples,'parameters_naive_stemming_bigram.pkl')
        true,pred = predict(test_file,5,param,phi_y,total_words,'e1')
        accuracy = accuracy_score(true,pred)
        print("The test accuracy is in the case of stemming with bigram ",accuracy*100)
        if os.path.exists("parameters_naive_lemmatize_bigram.pkl") == 0:
            dictionary,total_words,class_count,total_examples = make_dictionary(train_file,5,'e2')
        param,total_words,phi_y = get_parameters(dictionary,total_words,class_count,total_examples,'parameters_naive_lemmatize_bigram.pkl')
        true,pred = predict(test_file,5,param,phi_y,total_words,'e2')
        accuracy = accuracy_score(true,pred)
        print("The test accuracy is in the case of lemmatizatioin with bigram ",accuracy*100)
    elif(mode == 'f'):
        param,total_words,phi_y = get_parameters(dictionary,total_words,class_count,total_examples,'parameters_naive_lemmatize_bigram.pkl')
        true,pred = predict(test_file,5,param,phi_y,total_words,'e2')
        f1_matrix = f1(true,pred,'None')
        macro_f1 = f1(true,pred,'macro')
        print("The F1 score matrix is ",f1_matrix)
        print("The macro F1 score is ", macro_f1)
    elif(mode == 'g'):
        if os.path.exists("parameters_naive_lemmatize_bigram.pkl") == 0:
            dictionary,total_words,class_count,total_examples = make_dictionary(train_file,5,'e2')
        param,total_words,phi_y = get_parameters(dictionary,total_words,class_count,total_examples,'parameters_naive_lemmatize_bigram.pkl')
        true,pred = predict(test_file,5,param,phi_y,total_words,'e2')
        accuracy = accuracy_score(true,pred)
        print("The test set accuracy is ",accuracy*100)
    else:
        print("Wrong part number entered")

if __name__ == "__main__":
    print("here")
    if len(sys.argv) != 4:
        sys.stderr("Wrong format for command line arguments. Follow <file.py><train.json><test.json><part_number> ")
    main(sys.argv[1:])
