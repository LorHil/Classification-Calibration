import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
#from pycalib.models import CalibratedModel, IsotonicCalibration

if __name__ == '__main__':
    dataframe = pd.read_csv('train.csv')
    dataframe2 = pd.read_csv('test.csv')

    Y_train = dataframe["Label"]
    X_train = dataframe['EmailText']

    x_test = dataframe2['EmailText']

    Y_train = Y_train.map({'ham': 0, 'spam': 1})

    # SMALL PREPROCESSING
    X_train = X_train.map(lambda x: x.lower())
    X_train = X_train.str.replace('[^\w\s]', '')

    x_test = x_test.map(lambda x: x.lower())
    x_test = x_test.str.replace('[^\w\s]', '')

    #print(x_train)

    # EXTRACTING FEATURES
    count_vect = CountVectorizer()
    features = count_vect.fit_transform(X_train)
    x_test = count_vect.transform(x_test)

    # Take the Term Frequency Inverse Document Frequency instead of simple word-counting
    #transformer = TfidfTransformer().fit(features)
    #features = transformer.transform(features)

    # Split dataset to be able to evaluate the model
    # in random train and test sets
    x_train, x_validate, y_train, y_validate = train_test_split(features, Y_train, test_size=0.1, random_state=80)
    print(np.mean(y_validate))
    # Model
    model = MultinomialNB() # Laplace smoothening parameter is 1 by default (0 = no smoothening)
    model.fit(x_train, y_train)
    prob_pos_model = model.predict_proba(x_validate)[:, 1]

    # EVALUATING THE MODEL
    predicted = model.predict(x_validate)

    print("The accuracy is: ", np.mean(predicted == y_validate))
    print(confusion_matrix(y_validate, predicted))

    #features_test = count_vect.transform(x_validate)
    #print("The accuracy of the model is: ", model.score(features_test,y_validate))

    # PLOTTING
    #print(features)
    
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_validate, predicted))

    # CALIBRATION
    model_isotonic = CalibratedClassifierCV(model, method='isotonic')
    model_isotonic.fit(x_train, y_train)

    predicted = model_isotonic.predict(x_validate)

    print("The accuracy when calibrated is: ", np.mean(predicted == y_validate))
    print(confusion_matrix(y_validate, predicted, labels=[0, 1]))

    prob_pos_isotonic = model_isotonic.predict_proba(x_validate)[:, 1]
    #print(prob_pos_isotonic)
    #print(len(prob_pos_isotonic))

    model_sigmoid = CalibratedClassifierCV(model, method='sigmoid')
    model_sigmoid.fit(x_train, y_train)
    prob_pos_sigmoid = model_sigmoid.predict_proba(x_validate)[:, 1]

    print("Brier score losses: (the smaller the better)")
    model_score = brier_score_loss(y_validate, prob_pos_model)
    print("No calibration: %1.3f" % model_score)

    model_isotonic_score = brier_score_loss(y_validate, prob_pos_isotonic)
    print("With isotonic calibration: %1.3f" % model_isotonic_score)

    model_sigmoid_score = brier_score_loss(y_validate, prob_pos_sigmoid)
    print("With Sigmoid calibration: %1.3f" % model_sigmoid_score)
    #print(y_validate)
    plt.figure()
    order = np.lexsort((prob_pos_model, ))
    plt.plot(prob_pos_model[order], 'r', label='No calibration (%1.3f)' % model_score)
    plt.plot(prob_pos_isotonic[order], 'g', linewidth=3, label='Isotonic calibration (%1.3f)' % model_isotonic_score)
    plt.plot(prob_pos_sigmoid[order], 'b', linewidth=2, label='Sigmoid calibration (%1.3f)' % model_sigmoid_score)
    #plt.plot(y_validate[order], 'k', linewidth=3, label='Empirical')
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Instances sorted according to predicted probability "
           "(uncalibrated GNB)")
    plt.ylabel("P(Label = 'spam')")
    plt.legend(loc="upper left")
    plt.title("Multinomial Naïve Bayes probabilities")

    #plt.show()

    # USING THE CLASSIFIER ON THE UNKNOWN DATA
    prob_spam_isotonic = model_isotonic.predict_proba(x_test)[:, 1]
    print(prob_spam_isotonic)
    with open('scores.txt', 'w') as textfile:
        # given titles to the columns
        for i in prob_spam_isotonic:
            textfile.write(str(i))
            textfile.write('\n')

    textfile.close()

