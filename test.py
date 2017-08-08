import helper
import submission

training_data = helper.read_data('./asset/training_data.txt')
classifier_path = './asset/classifier.dat'
submission.train(training_data, classifier_path)

test_data = helper.read_data('./asset/tiny_test.txt')
prediction = submission.test(test_data, classifier_path)
print(prediction)
print(type(prediction))
print(type(prediction[0]))
from sklearn.metrics import f1_score
ground_truth = [1, 1, 2, 1]
print(f1_score(ground_truth, prediction, average='micro'))