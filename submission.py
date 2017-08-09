from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import re
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

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

    ### 特征：字长
    data['len'] = len(data['wd'])

    ### 特征：特殊后缀
    specialSuffix = ['TION', 'BLE', 'ING', 'ED', 'IVE']
    for s in specialSuffix:
        if data['wd'].endswith(s):
            data[s] = 1
    data['ph'] = line.split(':')[1]

    ### 特征：音标数
    data['total_num_ph'] = len(data['ph'].split(' '))

    ### 特征：元音数
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

    ### 特征：每个元音之前的辅音数
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

    ### 特征：元音距离占比密度

    ### 特征：

    ### 删除辅助属性
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
    nb = RandomForestClassifier(n_estimators=50)
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