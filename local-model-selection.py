import submission
import helper
import pickle
import re
import pandas
import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score

##### params
training_data = helper.read_data('./asset/training_data.txt')
classifier_path = './asset/classifier.dat'
n_fold = 5

def transform_training_data(data):
    after_transformed = []
    non_decimal = re.compile('\d')
    for d in data:
        after_transformed.append(non_decimal.sub('', d))
    return after_transformed
def get_stress(data):
    result = []
    for d in data:
        index = 1
        d = d.split(':')[1]
        for u in d.split(' '):
            if re.match('^[A-Za-z]+[0-9]$', u):
                if u.endswith('1'):
                    result.append(index)
                    break
                index += 1
    return result

kf = KFold(n_splits = n_fold, shuffle=True)
training_data = np.array(training_data)

train_error_total = 0
test_error_total = 0

for train, test in kf.split(training_data):
    submission.train(training_data[train], classifier_path)
    
    # get correct answer
    ground_truth_train = get_stress(training_data[train])
    ground_truth_test  = get_stress(training_data[test])

    # transform data from training format to test data
    train_x = transform_training_data(training_data[train])
    test_x  = transform_training_data(training_data[test])

    # get prediction
    train_prediction = submission.test(train_x, classifier_path)
    test_prediction  = submission.test(test_x, classifier_path)
    train_score = f1_score(ground_truth_train, train_prediction, average='micro')
    test_score  = f1_score(ground_truth_test, test_prediction, average='micro')
    train_error_total += train_score
    test_error_total += test_score

    print('*****************')
    print('traing f1_score : \t{}'.format(train_score))
    print('test   f1_score : \t{}'.format(test_score))
print('===================')
print('avg. training f1_score :\t{}'.format(train_error_total/n_fold))
print('avg. testing  f1_score :\t{}'.format(test_error_total/n_fold))
