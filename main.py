import pandas as pd
import numpy as np
import argparse
import yaml
from Logistic import LogisticRegression
from data_preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
from csv import DictWriter

def read_params():
    parser = argparse.ArgumentParser(description='Model Parameters')
    parser.add_argument('-p', '--params', type=str, help='model_params.yaml path')
    args = parser.parse_args()

    params = {}
    with open(args.params) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

        for i in range(len(data)):
            for key, value in data[i].items():
                params[key] = value
    return params

def prepare_data(train_df, test_df):
    x_train, x_test, y_train, y_test = train_test_split(train_df.iloc[:, 2:], train_df.iloc[:, 1], test_size=0.2, random_state=0, shuffle=False)

    x_train = np.array(x_train).astype(int)
    y_train = np.array(y_train).astype(int)
    x_test = np.array(x_test).astype(int)
    y_test = np.array(y_test).astype(int)

    return x_train, x_test, y_train, y_test

def predict_save(test_df, model, path):
    x_pred = np.array(test_df.iloc[:, 1:]).astype(int)
    y_pred = model.predict(x_pred)

    index = np.array(test_df['PassengerId']).astype(int)
    predicted_df = pd.DataFrame(y_pred, index)
    print(predicted_df)
    predicted_df.to_csv(path)

def save_results(y_true, y_predicted, params):
    precision = metrics.precision_score(y_true, y_predicted)
    recall = metrics.precision_score(y_true, y_predicted)
    f1 = metrics.f1_score(y_true, y_predicted)
    accuracy = metrics.accuracy_score(y_true, y_predicted)

    result = {
        'Model': params['model_name'],
        'Learning Rate': params['learning_rate'],
        'Iterations': params['iterations'],
        'Precision Score': precision,
        'Recall Score': recall,
        'F1 Score': f1,
        'Accuracy': accuracy
    }

    headers = result.keys()

    with open(params['results'], 'a', newline='') as f:
        writer_obj = DictWriter(f, fieldnames=headers)
        writer_obj.writerow(result)
        f.close()


if __name__ == '__main__':
    params = read_params()

    train_df, test_df = Preprocess(params['train_df'], params['test_df'])
    
    x_train, x_test, y_train, y_test = prepare_data(train_df, test_df)

    model = LogisticRegression(params['learning_rate'], params['iterations'])
    model.fit(x_train, y_train)
    model.print_weights()

    y_predicted = model.predict(x_test)
    model.score(x_test, y_test)
    
    predict_save(test_df, model, params['predicted'])
    
    save_results(y_test, y_predicted, params)

