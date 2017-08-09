# 大数据挖掘与处理-课程设计报告

## 一、小组成员

| 学号       | 姓名   | Name          |
| -------- | :--- | ------------- |
| 15331310 | 吴博文  | Bob Wu        |
|          | 杨竣然  | Jonathan Yang |
|          |      | Yifeng Peng   |

## 二、feature的选择

面对这个项目，我们一开始显得毫无头绪，因为我们都没有专业的语言学背景，不能知道单词的重读音节会受到哪些因素的影响。所以我们只好进行一些实验性的工作。

结合课堂所讲，根据我们所理解的那些机器学习的模型，所选取的feature一定要与结果（target）有很强的相关性，也就是按照这个feature进行分类以后，所得到的结果会比较有偏向性，集中某一个或者某几个值，那么这就是一个有价值的feature

所以最终我们利用的feature有：单词长度、元音个数、每个元音的排序、元音前辅音的个数、单词是否有出现特定的后缀（tion、ble、ive、ing、ed等）

我们发现单词长度其实十分影响重读的音节，因为单词比较长通常重读考前，因为如果重读靠后，读起来比较困难。

然后单词的前后缀是十分重要的特征，例如有TION后缀的单词，通常重读最后一个音节。

另外英语单词每个音节应该是需要包含辅音和元音的，就好像汉语拼音一样，所以元音前辅音的个数应该也是有作用的，另外一些辅音和元音的组合也是比较常用于重读。

元音个数是个十分重要的feature，一个简单的例子就是如果一个单词只有一个元音，那么只能得到结果为1。同时经过统计，四音节单词多重读第二音节，这个与词长抓住的信息点相似。

关于这方面的研究记录在附录

## 三、模型的选择

关于模型的选择说实话我们也没什么思路，主要以试为主

## 四、本地对于模型的评价与提升

使用交叉检验进行本地测试（cross-validation)，使用sklearn中的KFold，令k=5进行测试，基本可以得到可信度极高的结果。

```python
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

from sklearn.grid_search import GridSearchCV

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

```



但是评价一个模型是否优秀不能只看test-error， 还需要考虑training-error，不然很可能出现过拟合（over-fitting)的情况。

## 五、心得体会

虽然这次的项目并没有涉及到十分复杂的机器学习算法，主要是使用sklearn中已经实现好了的模型。但是也感觉学到了很多，知道要想用好这些机器学习的库也是需要了解其中一些基本的原理，这样才能选好feature。

也体会到了，对于数据挖掘，数据预处理其实是比较繁重的工作，他也十分影响模型的效果。做课程设计的过程中，我们发现改变模型其实对结果影响不大，对结果提升比较大的是feature的选取和对数据的预处理。这也是这次项目中所学到最多的。



## 六、附录

### 选取feature的研究过程（截取自notebook）



```python
import numpy as np
import pandas as pd
import sklearn as sklearn
import matplotlib as plot
import re as re
from util import *
import decimal
```


```python
import helper
```


```python
train_data = helper.read_data("./asset/training_data.txt")
```


```python
train_data[1:5]
```




    ['PURVIEW:P ER1 V Y UW2',
     'HEHIR:HH EH1 HH IH0 R',
     'MUSCLING:M AH1 S AH0 L IH0 NG',
     'NONPOISONOUS:N AA0 N P OY1 Z AH0 N AH0 S']




```python
train_data[1]
```




    'PURVIEW:P ER1 V Y UW2'



# Suffix


```python
suffix = ['ABLE', 'MENT', 'TIVE', 'EE', 'ERT', 'TION', 'SION', 'ITY', 'EST']
for s in suffix:
    sum_suffix = 0
    for data in train_data:
        word = data.split(':')[0]
        if word.endswith(s):
            sum_suffix += 1
    print(s + ': ' + str(sum_suffix))


```

    ABLE: 187
    MENT: 145
    TIVE: 95
    EE: 194
    ERT: 191
    TION: 373
    SION: 81
    ITY: 113
    EST: 183



```python
for s in suffix:
    target_arr = []
    for data in train_data:
        if data.split(':')[0].endswith(s):
            target_arr.append(data)
    print(conditional_res_test(target_arr, 'Suffix - ' + s))

```

    ## Suffix - ABLE ##
    stress: [0, 90, 96, 1, 0, 0]
     top_rate = 0.51
    stress_rv: [0, 0, 10, 139, 38, 0]
     top_rate_rv = 0.74
    ---------
    
    ## Suffix - MENT ##
    stress: [0, 40, 93, 12, 0, 0]
     top_rate = 0.64
    stress_rv: [0, 5, 89, 45, 6, 0]
     top_rate_rv = 0.61
    ---------
    
    ## Suffix - TIVE ##
    stress: [0, 28, 54, 13, 0, 0]
     top_rate = 0.57
    stress_rv: [0, 0, 56, 31, 8, 0]
     top_rate_rv = 0.59
    ---------
    
    ## Suffix - EE ##
    stress: [0, 109, 62, 23, 0, 0]
     top_rate = 0.56
    stress_rv: [0, 65, 93, 36, 0, 0]
     top_rate_rv = 0.48
    ---------
    
    ## Suffix - ERT ##
    stress: [0, 175, 15, 1, 0, 0]
     top_rate = 0.92
    stress_rv: [0, 15, 172, 4, 0, 0]
     top_rate_rv = 0.9
    ---------
    
    ## Suffix - TION ##
    stress: [0, 22, 120, 231, 0, 0]
     top_rate = 0.62
    stress_rv: [0, 0, 365, 4, 4, 0]
     top_rate_rv = 0.98
    ---------
    
    ## Suffix - SION ##
    stress: [0, 11, 56, 14, 0, 0]
     top_rate = 0.69
    stress_rv: [0, 0, 74, 2, 5, 0]
     top_rate_rv = 0.91
    ---------
    
    ## Suffix - ITY ##
    stress: [0, 29, 84, 0, 0, 0]
     top_rate = 0.74
    stress_rv: [0, 0, 2, 111, 0, 0]
     top_rate_rv = 0.98
    ---------
    
    ## Suffix - EST ##
    stress: [0, 161, 21, 1, 0, 0]
     top_rate = 0.88
    stress_rv: [0, 12, 118, 53, 0, 0]
     top_rate_rv = 0.64
    ---------



# Word Length & Vowel Amount


```python
max_word_length = 0
min_word_length = 100
for data in train_data:
    word_length = len(data.split(':')[0])
    max_word_length = max(word_length, max_word_length)
    min_word_length = min(word_length, min_word_length)
print('max: ' + str(max_word_length) + ', min: ' + str(min_word_length))
```

    max: 17, min: 1



```python
for data in train_data:
    if len(data.split(':')[0]) == max_word_length:
        print('max: ' + data)
    if len(data.split(':')[0]) == min_word_length:
        print('min: ' + data)
```

    max: STRAIGHTFORWARDLY:S T R EY2 T F AO1 R W ER0 D L IY0
    min: W:D AH1 B AH0 L Y UW0



```python
max_pa = 0
min_pa = 100
for data in train_data:
    pa = len(data.split(':')[1].split(' '))
    max_pa = max(pa, max_pa)
    min_pa = min(pa, min_pa)
print('max:', max_pa, 'min: ', min_pa)
```

    max: 14 min:  2



```python
for wl in range(min_word_length, max_word_length + 1):
    for pa in range(min_pa, max_pa + 1):
        arr = []
        for data in train_data:
            t_pa = len(data.split(':')[1].split(' '))
            t_wl = len(data.split(':')[0])
            if t_pa == pa & t_wl == wl:
                arr.append(data)
        if len(arr) != 0:
            print(conditional_res_test(arr, 'wl: ' + str(wl) + ', pa: ' + str(pa)))
                
```

    ## wl: 2, pa: 2 ##
    stress: [0, 3, 0, 0, 0, 0]
     top_rate = 1.0
    stress_rv: [0, 0, 3, 0, 0, 0]
     top_rate_rv = 1.0
    ---------
    
    ## wl: 2, pa: 3 ##
    stress: [0, 1, 0, 0, 0, 0]
     top_rate = 1.0
    stress_rv: [0, 0, 1, 0, 0, 0]
     top_rate_rv = 1.0
    ---------
    
    ## wl: 2, pa: 6 ##
    stress: [0, 3, 0, 0, 0, 0]
     top_rate = 1.0
    stress_rv: [0, 0, 3, 0, 0, 0]
     top_rate_rv = 1.0
    ---------
    
    ## wl: 2, pa: 7 ##
    stress: [0, 1, 0, 0, 0, 0]
     top_rate = 1.0
    stress_rv: [0, 0, 1, 0, 0, 0]
     top_rate_rv = 1.0
    ---------
    
    ## wl: 2, pa: 10 ##
    stress: [0, 3, 0, 0, 0, 0]
     top_rate = 1.0
    stress_rv: [0, 0, 3, 0, 0, 0]
     top_rate_rv = 1.0
    ---------
    
    ## wl: 2, pa: 11 ##
    stress: [0, 1, 0, 0, 0, 0]
     top_rate = 1.0
    stress_rv: [0, 0, 1, 0, 0, 0]
     top_rate_rv = 1.0
    ---------
    
    ## wl: 2, pa: 14 ##
    stress: [0, 3, 0, 0, 0, 0]
     top_rate = 1.0
    stress_rv: [0, 0, 3, 0, 0, 0]
     top_rate_rv = 1.0
    ---------
    
    ## wl: 3, pa: 3 ##
    stress: [0, 77, 9, 0, 0, 0]
     top_rate = 0.9
    stress_rv: [0, 9, 77, 0, 0, 0]
     top_rate_rv = 0.9
    ---------
    
    ## wl: 3, pa: 7 ##
    stress: [0, 70, 5, 0, 0, 0]
     top_rate = 0.93
    stress_rv: [0, 5, 70, 0, 0, 0]
     top_rate_rv = 0.93
    ---------
    
    ## wl: 3, pa: 11 ##
    stress: [0, 77, 9, 0, 0, 0]
     top_rate = 0.9
    stress_rv: [0, 9, 77, 0, 0, 0]
     top_rate_rv = 0.9
    ---------
    
    ## wl: 4, pa: 4 ##
    stress: [0, 4039, 480, 1, 0, 0]
     top_rate = 0.89
    stress_rv: [0, 443, 4036, 41, 0, 0]
     top_rate_rv = 0.89
    ---------
    
    ## wl: 4, pa: 5 ##
    stress: [0, 2223, 229, 1, 0, 0]
     top_rate = 0.91
    stress_rv: [0, 211, 2220, 22, 0, 0]
     top_rate_rv = 0.91
    ---------
    
    ## wl: 4, pa: 6 ##
    stress: [0, 2354, 253, 0, 0, 0]
     top_rate = 0.9
    stress_rv: [0, 229, 2358, 20, 0, 0]
     top_rate_rv = 0.9
    ---------
    
    ## wl: 4, pa: 7 ##
    stress: [0, 785, 69, 0, 0, 0]
     top_rate = 0.92
    stress_rv: [0, 61, 786, 7, 0, 0]
     top_rate_rv = 0.92
    ---------
    
    ## wl: 4, pa: 12 ##
    stress: [0, 4039, 480, 1, 0, 0]
     top_rate = 0.89
    stress_rv: [0, 443, 4036, 41, 0, 0]
     top_rate_rv = 0.89
    ---------
    
    ## wl: 4, pa: 13 ##
    stress: [0, 2223, 229, 1, 0, 0]
     top_rate = 0.91
    stress_rv: [0, 211, 2220, 22, 0, 0]
     top_rate_rv = 0.91
    ---------
    
    ## wl: 4, pa: 14 ##
    stress: [0, 2354, 253, 0, 0, 0]
     top_rate = 0.9
    stress_rv: [0, 229, 2358, 20, 0, 0]
     top_rate_rv = 0.9
    ---------
    
    ## wl: 5, pa: 5 ##
    stress: [0, 4759, 733, 6, 0, 0]
     top_rate = 0.87
    stress_rv: [0, 491, 4731, 276, 0, 0]
     top_rate_rv = 0.86
    ---------
    
    ## wl: 5, pa: 7 ##
    stress: [0, 2253, 363, 1, 0, 0]
     top_rate = 0.86
    stress_rv: [0, 188, 2320, 109, 0, 0]
     top_rate_rv = 0.89
    ---------
    
    ## wl: 5, pa: 13 ##
    stress: [0, 4759, 733, 6, 0, 0]
     top_rate = 0.87
    stress_rv: [0, 491, 4731, 276, 0, 0]
     top_rate_rv = 0.86
    ---------
    
    ## wl: 6, pa: 6 ##
    stress: [0, 6011, 2235, 60, 0, 0]
     top_rate = 0.72
    stress_rv: [0, 698, 6627, 979, 2, 0]
     top_rate_rv = 0.8
    ---------
    
    ## wl: 6, pa: 7 ##
    stress: [0, 2461, 1033, 20, 0, 0]
     top_rate = 0.7
    stress_rv: [0, 234, 2883, 396, 1, 0]
     top_rate_rv = 0.82
    ---------
    
    ## wl: 6, pa: 14 ##
    stress: [0, 6011, 2235, 60, 0, 0]
     top_rate = 0.72
    stress_rv: [0, 698, 6627, 979, 2, 0]
     top_rate_rv = 0.8
    ---------
    
    ## wl: 7, pa: 7 ##
    stress: [0, 1509, 1209, 150, 2, 0]
     top_rate = 0.53
    stress_rv: [0, 170, 1953, 738, 9, 0]
     top_rate_rv = 0.68
    ---------
    
    ## wl: 8, pa: 8 ##
    stress: [0, 3402, 2015, 799, 12, 0]
     top_rate = 0.55
    stress_rv: [0, 264, 2920, 2743, 301, 0]
     top_rate_rv = 0.47
    ---------
    
    ## wl: 8, pa: 9 ##
    stress: [0, 1800, 1197, 466, 9, 0]
     top_rate = 0.52
    stress_rv: [0, 129, 1698, 1504, 141, 0]
     top_rate_rv = 0.49
    ---------
    
    ## wl: 8, pa: 10 ##
    stress: [0, 2205, 1617, 634, 6, 0]
     top_rate = 0.49
    stress_rv: [0, 183, 2323, 1787, 169, 0]
     top_rate_rv = 0.52
    ---------
    
    ## wl: 8, pa: 11 ##
    stress: [0, 888, 861, 343, 4, 0]
     top_rate = 0.42
    stress_rv: [0, 74, 1198, 767, 57, 0]
     top_rate_rv = 0.57
    ---------
    
    ## wl: 8, pa: 12 ##
    stress: [0, 3329, 2000, 792, 10, 0]
     top_rate = 0.54
    stress_rv: [0, 261, 2903, 2672, 295, 0]
     top_rate_rv = 0.47
    ---------
    
    ## wl: 8, pa: 13 ##
    stress: [0, 1743, 1182, 459, 7, 0]
     top_rate = 0.51
    stress_rv: [0, 126, 1682, 1448, 135, 0]
     top_rate_rv = 0.5
    ---------
    
    ## wl: 8, pa: 14 ##
    stress: [0, 2134, 1602, 627, 4, 0]
     top_rate = 0.49
    stress_rv: [0, 180, 2306, 1718, 163, 0]
     top_rate_rv = 0.53
    ---------
    
    ## wl: 9, pa: 9 ##
    stress: [0, 909, 705, 362, 4, 0]
     top_rate = 0.46
    stress_rv: [0, 45, 784, 895, 256, 0]
     top_rate_rv = 0.45
    ---------
    
    ## wl: 9, pa: 11 ##
    stress: [0, 508, 491, 251, 2, 0]
     top_rate = 0.41
    stress_rv: [0, 27, 575, 513, 137, 0]
     top_rate_rv = 0.46
    ---------
    
    ## wl: 9, pa: 13 ##
    stress: [0, 870, 696, 357, 4, 0]
     top_rate = 0.45
    stress_rv: [0, 43, 780, 867, 237, 0]
     top_rate_rv = 0.45
    ---------
    
    ## wl: 10, pa: 10 ##
    stress: [0, 453, 498, 368, 10, 0]
     top_rate = 0.37
    stress_rv: [0, 35, 541, 515, 238, 0]
     top_rate_rv = 0.41
    ---------
    
    ## wl: 10, pa: 11 ##
    stress: [0, 192, 223, 177, 2, 0]
     top_rate = 0.38
    stress_rv: [0, 16, 249, 238, 91, 0]
     top_rate_rv = 0.42
    ---------
    
    ## wl: 10, pa: 14 ##
    stress: [0, 440, 495, 365, 9, 0]
     top_rate = 0.38
    stress_rv: [0, 33, 539, 509, 228, 0]
     top_rate_rv = 0.41
    ---------
    
    ## wl: 11, pa: 11 ##
    stress: [0, 70, 106, 91, 2, 0]
     top_rate = 0.39
    stress_rv: [0, 11, 109, 99, 50, 0]
     top_rate_rv = 0.41
    ---------
    
    ## wl: 12, pa: 12 ##
    stress: [0, 38, 80, 73, 1, 0]
     top_rate = 0.42
    stress_rv: [0, 1, 88, 72, 31, 0]
     top_rate_rv = 0.46
    ---------
    
    ## wl: 12, pa: 13 ##
    stress: [0, 19, 47, 29, 1, 0]
     top_rate = 0.49
    stress_rv: [0, 1, 36, 42, 17, 0]
     top_rate_rv = 0.44
    ---------
    
    ## wl: 12, pa: 14 ##
    stress: [0, 29, 65, 64, 0, 0]
     top_rate = 0.41
    stress_rv: [0, 0, 77, 59, 22, 0]
     top_rate_rv = 0.49
    ---------
    
    ## wl: 13, pa: 13 ##
    stress: [0, 1, 7, 14, 2, 0]
     top_rate = 0.58
    stress_rv: [0, 2, 14, 7, 1, 0]
     top_rate_rv = 0.58
    ---------
    
    ## wl: 14, pa: 14 ##
    stress: [0, 0, 4, 4, 0, 0]
     top_rate = 0.5
    stress_rv: [0, 0, 4, 4, 0, 0]
     top_rate_rv = 0.5
    ---------



# Word Length


```python
for wl in range(min_word_length, max_word_length + 1):
    arr = []
    for data in train_data:
        t_wl = len(data.split(':')[0])
        if t_wl == wl:
            arr.append(data)
    if len(arr) != 0:
        print(len(arr))
        print(conditional_res_test(arr, 'wl: ' + str(wl)))
```

    1
    ## wl: 1 ##
    stress: [0, 1, 0, 0, 0, 0]
     top_rate = 1.0
    stress_rv: [0, 0, 0, 1, 0, 0]
     top_rate_rv = 1.0
    ---------
    
    7
    ## wl: 2 ##
    stress: [0, 4, 3, 0, 0, 0]
     top_rate = 0.57
    stress_rv: [0, 3, 4, 0, 0, 0]
     top_rate_rv = 0.57
    ---------
    
    102
    ## wl: 3 ##
    stress: [0, 83, 8, 11, 0, 0]
     top_rate = 0.81
    stress_rv: [0, 17, 82, 2, 1, 0]
     top_rate_rv = 0.8
    ---------
    
    1084
    ## wl: 4 ##
    stress: [0, 988, 93, 3, 0, 0]
     top_rate = 0.91
    stress_rv: [0, 84, 989, 11, 0, 0]
     top_rate_rv = 0.91
    ---------
    
    4678
    ## wl: 5 ##
    stress: [0, 4094, 582, 2, 0, 0]
     top_rate = 0.88
    stress_rv: [0, 380, 4166, 132, 0, 0]
     top_rate_rv = 0.89
    ---------
    
    9620
    ## wl: 6 ##
    stress: [0, 7699, 1898, 23, 0, 0]
     top_rate = 0.8
    stress_rv: [0, 905, 8066, 648, 1, 0]
     top_rate_rv = 0.84
    ---------
    
    11006
    ## wl: 7 ##
    stress: [0, 7891, 2910, 203, 2, 0]
     top_rate = 0.72
    stress_rv: [0, 1012, 8429, 1550, 15, 0]
     top_rate_rv = 0.77
    ---------
    
    8996
    ## wl: 8 ##
    stress: [0, 5913, 2526, 553, 4, 0]
     top_rate = 0.66
    stress_rv: [0, 730, 5721, 2435, 110, 0]
     top_rate_rv = 0.64
    ---------
    
    6609
    ## wl: 9 ##
    stress: [0, 4084, 1850, 671, 4, 0]
     top_rate = 0.62
    stress_rv: [0, 377, 3350, 2599, 283, 0]
     top_rate_rv = 0.51
    ---------
    
    4173
    ## wl: 10 ##
    stress: [0, 2328, 1210, 623, 12, 0]
     top_rate = 0.56
    stress_rv: [0, 169, 1655, 1971, 378, 0]
     top_rate_rv = 0.47
    ---------
    
    2226
    ## wl: 11 ##
    stress: [0, 1104, 670, 440, 12, 0]
     top_rate = 0.5
    stress_rv: [0, 78, 739, 1048, 361, 0]
     top_rate_rv = 0.47
    ---------
    
    1004
    ## wl: 12 ##
    stress: [0, 423, 321, 247, 13, 0]
     top_rate = 0.42
    stress_rv: [0, 24, 345, 456, 179, 0]
     top_rate_rv = 0.45
    ---------
    
    370
    ## wl: 13 ##
    stress: [0, 158, 110, 99, 3, 0]
     top_rate = 0.43
    stress_rv: [0, 8, 111, 169, 82, 0]
     top_rate_rv = 0.46
    ---------
    
    91
    ## wl: 14 ##
    stress: [0, 33, 32, 24, 2, 0]
     top_rate = 0.36
    stress_rv: [0, 3, 26, 42, 20, 0]
     top_rate_rv = 0.46
    ---------
    
    30
    ## wl: 15 ##
    stress: [0, 11, 8, 9, 2, 0]
     top_rate = 0.37
    stress_rv: [0, 2, 9, 8, 11, 0]
     top_rate_rv = 0.37
    ---------
    
    2
    ## wl: 16 ##
    stress: [0, 0, 0, 2, 0, 0]
     top_rate = 1.0
    stress_rv: [0, 0, 2, 0, 0, 0]
     top_rate_rv = 1.0
    ---------
    
    1
    ## wl: 17 ##
    stress: [0, 0, 1, 0, 0, 0]
     top_rate = 1.0
    stress_rv: [0, 0, 0, 1, 0, 0]
     top_rate_rv = 1.0
    ---------



# Phonetic Amount


```python
for pa in range(min_pa, max_pa + 1):
    arr = []
    for data in train_data:
        t_pa = len(data.split(':')[1].split(' '))
        if t_pa == pa:
            arr.append(data)
    if len(arr) != 0:
        print(conditional_res_test(arr, 'pa: ' + str(pa)))
```

    ## pa: 2 ##
    stress: [0, 6, 1, 0, 0, 0]
     top_rate = 0.86
    stress_rv: [0, 1, 6, 0, 0, 0]
     top_rate_rv = 0.86
    ---------
    
    ## pa: 3 ##
    stress: [0, 391, 56, 0, 0, 0]
     top_rate = 0.87
    stress_rv: [0, 52, 395, 0, 0, 0]
     top_rate_rv = 0.88
    ---------
    
    ## pa: 4 ##
    stress: [0, 4107, 511, 3, 0, 0]
     top_rate = 0.89
    stress_rv: [0, 475, 4104, 42, 0, 0]
     top_rate_rv = 0.89
    ---------
    
    ## pa: 5 ##
    stress: [0, 9115, 1568, 19, 0, 0]
     top_rate = 0.85
    stress_rv: [0, 1144, 9007, 550, 1, 0]
     top_rate_rv = 0.84
    ---------
    
    ## pa: 6 ##
    stress: [0, 9057, 2858, 125, 0, 0]
     top_rate = 0.75
    stress_rv: [0, 1044, 9137, 1851, 8, 0]
     top_rate_rv = 0.76
    ---------
    
    ## pa: 7 ##
    stress: [0, 5977, 2789, 455, 4, 0]
     top_rate = 0.65
    stress_rv: [0, 633, 5481, 3032, 79, 0]
     top_rate_rv = 0.59
    ---------
    
    ## pa: 8 ##
    stress: [0, 3476, 2071, 807, 12, 0]
     top_rate = 0.55
    stress_rv: [0, 271, 2987, 2802, 306, 0]
     top_rate_rv = 0.47
    ---------
    
    ## pa: 9 ##
    stress: [0, 1790, 1330, 682, 13, 0]
     top_rate = 0.47
    stress_rv: [0, 103, 1478, 1717, 517, 0]
     top_rate_rv = 0.45
    ---------
    
    ## pa: 10 ##
    stress: [0, 669, 651, 475, 14, 0]
     top_rate = 0.37
    stress_rv: [0, 42, 680, 726, 361, 0]
     top_rate_rv = 0.4
    ---------
    
    ## pa: 11 ##
    stress: [0, 185, 287, 239, 7, 0]
     top_rate = 0.4
    stress_rv: [0, 23, 298, 262, 135, 0]
     top_rate_rv = 0.42
    ---------
    
    ## pa: 12 ##
    stress: [0, 39, 81, 77, 1, 0]
     top_rate = 0.41
    stress_rv: [0, 1, 92, 73, 32, 0]
     top_rate_rv = 0.46
    ---------
    
    ## pa: 13 ##
    stress: [0, 2, 15, 24, 3, 0]
     top_rate = 0.55
    stress_rv: [0, 3, 25, 14, 2, 0]
     top_rate_rv = 0.57
    ---------
    
    ## pa: 14 ##
    stress: [0, 0, 4, 4, 0, 0]
     top_rate = 0.5
    stress_rv: [0, 0, 4, 4, 0, 0]
     top_rate_rv = 0.5
    ---------



# Vowel Amount


```python
max_va = 0
min_va = 100
for data in train_data:
    va = line_parser(data)['v_num']
    max_va = max(va, max_va)
    min_va = min(va, min_va)
print('max:', max_va, 'min: ', min_va)
```

    max: 4 min:  2



```python
for va in range(min_va, max_va + 1):
    arr = []
    for data in train_data:
        t_va = line_parser(data)['v_num']
        if t_va == va:
            arr.append(data)
    if len(arr) != 0:
        print(conditional_res_test(arr, 'va: ' + str(va)))
```

    ## va: 2 ##
    stress: [0, 24435, 3184, 0, 0, 0]
     top_rate = 0.88
    stress_rv: [0, 3184, 24435, 0, 0, 0]
     top_rate_rv = 0.88
    ---------
    
    ## va: 3 ##
    stress: [0, 8938, 6903, 554, 0, 0]
     top_rate = 0.55
    stress_rv: [0, 554, 6903, 8938, 0, 0]
     top_rate_rv = 0.55
    ---------
    
    ## va: 4 ##
    stress: [0, 1441, 2135, 2356, 54, 0]
     top_rate = 0.39
    stress_rv: [0, 54, 2356, 2135, 1441, 0]
     top_rate_rv = 0.39
    ---------



# Vowel Ratio


```python
max_vr = 0
min_vr = 1
for data in train_data:
    vr = line_parser(data)['vr']
    max_vr = max(vr, max_vr)
    min_vr = min(vr, min_vr)
print('max: ', max_vr, 'min: ', min_vr)
```

    max:  0.29 min:  0.08



```python
for vr in [x / 100.0 for x in range(int(min_vr*100), int(max_vr*100) + 1, 1)]:
    arr = []
    for data in train_data:
        t_vr = line_parser(data)['vr']
        if t_vr == vr:
            arr.append(data)
    if len(arr) != 0:
        print(conditional_res_test(arr, 'vr: ' + str(vr)))
```

    ## vr: 0.08 ##
    stress: [0, 5, 1, 0, 0, 0]
     top_rate = 0.83
    stress_rv: [0, 1, 5, 0, 0, 0]
     top_rate_rv = 0.83
    ---------
    
    ## vr: 0.09 ##
    stress: [0, 64, 10, 0, 0, 0]
     top_rate = 0.86
    stress_rv: [0, 9, 65, 0, 0, 0]
     top_rate_rv = 0.88
    ---------
    
    ## vr: 0.1 ##
    stress: [0, 395, 52, 0, 0, 0]
     top_rate = 0.88
    stress_rv: [0, 32, 405, 10, 0, 0]
     top_rate_rv = 0.91
    ---------
    
    ## vr: 0.11 ##
    stress: [0, 1725, 240, 21, 0, 0]
     top_rate = 0.87
    stress_rv: [0, 171, 1752, 63, 0, 0]
     top_rate_rv = 0.88
    ---------
    
    ## vr: 0.12 ##
    stress: [0, 4964, 1057, 93, 3, 0]
     top_rate = 0.81
    stress_rv: [0, 573, 4831, 702, 11, 0]
     top_rate_rv = 0.79
    ---------
    
    ## vr: 0.13 ##
    stress: [0, 6144, 1360, 220, 1, 0]
     top_rate = 0.8
    stress_rv: [0, 871, 5836, 950, 68, 0]
     top_rate_rv = 0.76
    ---------
    
    ## vr: 0.14 ##
    stress: [0, 4712, 1948, 483, 9, 0]
     top_rate = 0.66
    stress_rv: [0, 339, 3942, 2610, 261, 0]
     top_rate_rv = 0.55
    ---------
    
    ## vr: 0.15 ##
    stress: [0, 7456, 1941, 495, 17, 0]
     top_rate = 0.75
    stress_rv: [0, 989, 7204, 1302, 414, 0]
     top_rate_rv = 0.73
    ---------
    
    ## vr: 0.16 ##
    stress: [0, 2208, 2010, 555, 8, 0]
     top_rate = 0.46
    stress_rv: [0, 159, 1989, 2329, 304, 0]
     top_rate_rv = 0.49
    ---------
    
    ## vr: 0.17 ##
    stress: [0, 1691, 901, 637, 12, 0]
     top_rate = 0.52
    stress_rv: [0, 97, 1848, 1000, 296, 0]
     top_rate_rv = 0.57
    ---------
    
    ## vr: 0.18 ##
    stress: [0, 4403, 1928, 136, 1, 0]
     top_rate = 0.68
    stress_rv: [0, 475, 4694, 1277, 22, 0]
     top_rate_rv = 0.73
    ---------
    
    ## vr: 0.19 ##
    stress: [0, 173, 248, 217, 3, 0]
     top_rate = 0.39
    stress_rv: [0, 6, 274, 305, 56, 0]
     top_rate_rv = 0.48
    ---------
    
    ## vr: 0.2 ##
    stress: [0, 492, 388, 18, 0, 0]
     top_rate = 0.55
    stress_rv: [0, 21, 442, 434, 1, 0]
     top_rate_rv = 0.49
    ---------
    
    ## vr: 0.21 ##
    stress: [0, 9, 55, 30, 0, 0]
     top_rate = 0.59
    stress_rv: [0, 0, 37, 50, 7, 0]
     top_rate_rv = 0.53
    ---------
    
    ## vr: 0.22 ##
    stress: [0, 327, 45, 0, 0, 0]
     top_rate = 0.88
    stress_rv: [0, 45, 326, 0, 1, 0]
     top_rate_rv = 0.88
    ---------
    
    ## vr: 0.23 ##
    stress: [0, 40, 32, 3, 0, 0]
     top_rate = 0.53
    stress_rv: [0, 3, 32, 40, 0, 0]
     top_rate_rv = 0.53
    ---------
    
    ## vr: 0.24 ##
    stress: [0, 0, 1, 2, 0, 0]
     top_rate = 0.67
    stress_rv: [0, 0, 2, 1, 0, 0]
     top_rate_rv = 0.67
    ---------
    
    ## vr: 0.27 ##
    stress: [0, 0, 4, 0, 0, 0]
     top_rate = 1.0
    stress_rv: [0, 0, 4, 0, 0, 0]
     top_rate_rv = 1.0
    ---------



# Distance of First 2 Vowel


```python

```


```python

```


```python

```


```python

```

# Distance of Last 2 Vowel


```python

```


```python

```


```python

```

# No Condition


```python
print(conditional_res_test(train_data, 'No Condition'))
```

    ## No Condition ##
    stress: [0, 34814, 12222, 2910, 54, 0]
     top_rate = 0.7
    stress_rv: [0, 3792, 33694, 11073, 1441, 0]
     top_rate_rv = 0.67
    ---------




```python

```