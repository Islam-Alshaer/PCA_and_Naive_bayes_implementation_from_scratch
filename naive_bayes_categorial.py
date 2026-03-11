#we are going to implement a naive bayes classifier for both categorical and numerical data
#we will use gaussian naive bayes for numerical data and multinomial naive bayes for categorical data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


class NaiveBayesCategorial:
    #y' = arg max P(y) * P(x1|y) * P(x2|y) * ... * P(xn|y)
    def predict(self, X_test):
        y_pred = []
        for example in X_test.values:
            y_pred.append(self.predict_example(example))
        return np.array(y_pred)

    def predict_example(self, example):
        winner = ''
        winner_score = 0
        for y_class in self.classes:
            score = 1
            score *= self.priors[y_class]
            for i in range(len(self.feature_names)):
                feature = self.feature_names[i]
                x_class = example[i]
                score *= self.likelihoods[feature][x_class][y_class]

            if score > winner_score:
                winner_score = score
                winner = y_class

        return winner





    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
        else:
            self.feature_names = np.arange(X.shape[1])
            X = pd.DataFrame(X)

        self.classes = np.unique(y)
        self.priors = {}
        for y_class in self.classes:
            self.priors[y_class] = np.mean(y == y_class)

        #calculate likelihoods for each feature for each feature class for each target class
        self.likelihoods = {}
        for feature in self.feature_names:
            self.likelihoods[feature] = {}
            for x_class in np.unique(X[feature]):
                self.likelihoods[feature][x_class] = {}
                for y_class in self.classes:
                    self.likelihoods[feature][x_class][y_class] = self.calculate_likelihood(X, y, feature, x_class, y_class)
        #we could calculate it only at prediction, but we give priority to prediction speed




    def calculate_likelihood(self, X, y, feature, x_class, y_class):
        #p(a|b) = p(a and b) / p(b)
        #in this case p(x_class|y_class) = p(x_class and y_class) / p(y_class)
        p_y = (y == y_class)
        p_x_and_y = ((X[feature] == x_class) & p_y).sum()
        p_y_class = p_y.sum()
        return p_x_and_y / p_y_class


def test_fit(my_model):
    datatype = np.dtype([('feature1', 'U10'), ('feature2', 'U10')])
    X = np.array([('red', 'small'), ('red', 'large'), ('blue', 'small'), ('blue', 'large')], dtype=datatype)
    y = np.array(['class1', 'class1', 'class2', 'class2'])
    my_model.fit(X, y)
    print('classes: ', my_model.classes)
    print('priors: ', my_model.priors)
    print('likelihoods: ', my_model.parameters)

def test_calculate_likelihood(my_model):
    datatype = np.dtype([('feature1', 'U10'), ('feature2', 'U10')])
    X = np.array([('red', 'small'), ('red', 'large'), ('blue', 'small'), ('blue', 'large')], dtype=datatype)
    y = np.array(['class1', 'class1', 'class2', 'class2'])

    feature = 'feature1'
    x_class = 'red'
    y_class = 'class1'
    print('test 1: ', my_model.calculate_likelihood(X, y, feature, x_class, y_class))

    #test 2
    X = np.array([('red', 'small'), ('red', 'large'), ('red', 'large'), ('blue', 'small')], dtype=datatype)
    y = np.array(['class1', 'class1', 'class1', 'class1'])
    feature = 'feature1'
    x_class = 'red'
    y_class = 'class1'
    print('test 2: ', my_model.calculate_likelihood(X, y, feature, x_class, y_class))


def test_predict(my_model):

    #load a dataset for test
    data = pd.read_csv('car_evaluation.csv')

    #train test split
    X = data.drop('class', axis=1)
    y = data['class'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    my_model.fit(X_train, y_train)
    y_pred = my_model.predict(X_test)

    #use built in sklearn categorial naive bayes for comparison
    model = CategoricalNB()

    #prepare data for sklearn model
    enc = OrdinalEncoder()
    X_train_enc = enc.fit_transform(X_train)
    lenc = LabelEncoder()
    y_train_enc = lenc.fit_transform(y_train)

    model.fit(X_train_enc, y_train_enc)

    X_test_enc = enc.transform(X_test)
    y_pred_sklearn = model.predict(X_test_enc)

    #compare my NB and their NB
    y_pred_sklearn = lenc.inverse_transform(y_pred_sklearn)
    print('My NB similarity to sklearn: ', np.mean(y_pred == y_pred_sklearn))


if __name__ == '__main__':
    nb = NaiveBayesCategorial()
    test_calculate_likelihood(nb)
    # test_fit(nb)
    test_predict(nb)











