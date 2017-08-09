from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import re
import pickle

def isConsonant(u):
    consonant = ['P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N',
                'NG', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
    try:
        consonant.index(u)
    except:
        return False;
    else:
        return True;

def line_parser(line, isTest):
    data = {}
    data['wd'] = line.split(':')[0]
    data['len'] = len(data['wd'])
    #data['prefix_2'] = data['wd'][0:1]
    #data['prefix_3'] = data['wd'][0:2]
    #data['prefix_4'] = data['wd'][0:3]
    
    #data['suffix_2'] = data['wd'][-2:]
    #data['suffix_3'] = data['wd'][-3:]
    #data['suffix_4'] = data['wd'][-4:]

    specialSuffix = ['TION', 'BLE', 'ING', 'ED', 'IVE']
    for s in specialSuffix:
        if data['wd'].endswith(s):
            data[s] = 1
    data['ph'] = line.split(':')[1]
    data['total_num_ph'] = len(data['ph'].split(' '))
    data['v_num'] = 0
    index = 1
    temp = 0
    for u in data['ph'].split(' '):
        if True == isTest:
            if isConsonant(u):
                continue;
            else:
                data['v_num'] += 1
                #data[u] = index
                data[str(index)] = u
                index += 1
        else:
             if re.match('^[A-Za-z]+[0-9]$', u):
                #print(u)
                data['v_num'] += 1
                #data[u[:-1]] = index
                data[str(index)] = u[:-1]
                if u.endswith('1'):
                    temp = index
                index += 1
    #print(temp)
    ph = data['ph'].split(' ')
    index = 1
    s = 0
    for i in range(0, len(ph)):
        if isConsonant(ph[i]) == True:
            s += 1
        else:
            data['consonant' + str(index)] = s
            index += 1
            s = 0
    data.pop('ph')
    data.pop('wd')
    if True == isTest:
        return data
    else:
        return data, temp

################# training #################

def train(training_data, classifier_file):# do not change the heading of the function
    features = []
    for data in training_data:
        features.append(line_parser(data, False))

    encoder = LabelEncoder()
    vectorizer = DictVectorizer(dtype=int, sparse=False)
    x, y = list(zip(*features))
    x = vectorizer.fit_transform(x)
    y = encoder.fit_transform(y)

    #nb = MultinomialNB(alpha=2)
    #nb = svm.SVC()
    nb = KNeighborsClassifier()
    nb.fit(x, y)
    f = open(classifier_file, 'wb')

    obj = {'nb':nb, 'encoder':encoder, 'vectorizer':vectorizer}
    pickle.dump(obj, f)
    f.close()
################# testing #################

def test(test_data, classifier_file):# do not change the heading of the function
    f = open(classifier_file, 'rb')
    obj = pickle.load(f)
    f.close()
    features = []
    for data in test_data:
        features.append(line_parser(data, True))
    features = obj['vectorizer'].transform(features)
    result = obj['nb'].predict(features)
    result = obj['encoder'].inverse_transform(result)
    return result.tolist()
# first trial  : 0.7670
# second trial : 0.8540
# third trial  : 0.8890