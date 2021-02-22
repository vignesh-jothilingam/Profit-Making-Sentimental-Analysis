from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import time


print('Reading Files -- as Dataframe')
train = pd.read_csv(r"trainning_dataset.csv")
train = train.dropna(axis=0, subset=['Label'])
test = pd.read_csv(r"testing_dataset.csv")
test = test.dropna(axis=0, subset=['Label'])




vectorizer = TfidfVectorizer(
                             min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True
                             )



train_vectors = vectorizer.fit_transform(train['Content'])
test_vectors = vectorizer.transform(test['Content'])



print('Trainning and Testing')
classifier_linear = svm.SVC(kernel='linear')
print('Trainning Started')
t0 = time.time()
classifier_linear.fit(train_vectors, train['Label'])
t1 = time.time()


print('Testing Started')
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
print('Calculating trainning and testing times')
time_linear_train = t1-t0
time_linear_predict = t2-t1



print('Saving Model')
pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))
pickle.dump(classifier_linear, open('classifier.sav', 'wb'))
print("Models Saved Successfully")




print("Report for the Model training")
print("Training time: %fs \nPrediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(test['Label'], prediction_linear, output_dict=True)
print('positive report:      \n', report['pos'])
print('negative report:     \n ', report['neg'])
