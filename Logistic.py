import numpy as np
import math

class LogisticRegression:
    def __init__(self, learning_rate=0.05, iterations=1000): #default values?
        self.x_train = None
        self.y_train = None
        self.coef = None
        self.intercept = None
        self.iterations = iterations
        self.learning_rate = learning_rate

    def fit(self, x, y):
        if x.ndim != 2:
            print("Needed 2-dim array")
            return
        self.x_train = x
        self.y_train = y
        self.gradient_descent()

    def sigmoid(self, a):
        return 1 / (1 + math.exp(-a))

    def gradient_descent(self):
        n_samples, n_features = self.x_train.shape

        b_curr = 0
        m_curr = np.zeros(n_features)

        for iter in range(self.iterations):
            y_predicted = np.zeros(n_samples)
            for s in range(n_samples):
                a = 0
                for f in range(n_features):
                    a += m_curr[f] * self.x_train[s][f]
                a += b_curr
                y_predicted[s] = round(self.sigmoid(a))


            for k in range(n_features):
                val = 0
                for i in range(n_samples):
                    a = 0
                    for j in range(n_features):
                        a += m_curr[j] * self.x_train[i][j]
                    a += b_curr
                    val += (self.sigmoid(a) - self.y_train[i]) * self.x_train[i][k]
                val /= n_samples
                m_curr[k] -= self.learning_rate * val

            e = 0
            for s in range(n_samples):
                e += y_predicted[s] - self.y_train[s]
            e /= n_samples

            b_curr -= e * self.learning_rate

        self.coef = m_curr
        self.intercept = b_curr
        
    def print_weights(self):
        print("Coef:", self.coef)
        print("Intercept:", self.intercept)

    def predict(self, x):
        n_samples, n_features = x.shape
        y_pred = np.zeros(n_samples)

        for s in range(n_samples):
            a = 0
            for f in range(n_features):
                a += self.coef[f] * x[s][f]
            a += self.intercept
            y_pred[s] = round(self.sigmoid(a))
        print('Result:', y_pred)
        return y_pred

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        k = 0

        for i in range(len(y_test)):
            if y_pred[i] == y_test[i]:
                k += 1
        score = k / len(y_test)
        print("Accuracy:", score)

    def predict_proba(self, x_test):
        n_samples, n_features = x_test.shape
        y_pred = np.zeros([n_samples,2])

        for s in range(n_samples):
            a = 0
            for f in range(n_features):
                a += self.coef[f] * x_test[s][f]
            a += self.intercept
            y_pred[s][0] = self.sigmoid(a)
            y_pred[s][1] = 1 - y_pred[s][0]

        print(y_pred)

    def print_df(self):
        print('x: \n')
        print(self.x_train)
        print('y: \n')
        print(self.y_train)
    
    def confusion_matrix(self, y_test, y_pred):
        TP = TN = FP = FN = 0
        for i in range(len(y_test)):
            if y_pred[i] == 1:
                if y_pred[i] == y_test[i]:
                    TP += 1
                else:
                    FP += 1
            elif y_pred[i] == y_test[i]:
                TN += 1
            else: 
                FN += 1

        return TP, TN, FP, FN

