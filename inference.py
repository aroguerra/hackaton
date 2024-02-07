import pickle
import numpy as np
import pandas as pd
from flask import Flask, request

app = Flask(__name__)


def load_model(model_path):
    """
    Function to load the predictive model with pickle
    :param model_path: path to model pkl
    :type model_path: str
    :return: pkl model
    """
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    return loaded_model


def perform_preprocess(df):
    practice_sport_mapping = {'never': 0, 'sometimes': 1, 'regularly': 2}
    df['PracticeSport_encoded'] = df['PracticeSport'].map(practice_sport_mapping)

    # Assuming an ordinal mapping for 'WklyStudyHours'
    wkly_study_hours_mapping = {'< 5': 0, '5 - 10': 1, '> 10': 2}
    df['WklyStudyHours_encoded'] = df['WklyStudyHours'].map(wkly_study_hours_mapping)

    parent_educ_mapping = {
        'some high school': 1,
        'high school': 2,
        'some college': 3,
        "associate's degree": 4,
        "bachelor's degree": 5,
        "master's degree": 6
    }

    df['ParentEduc_Ordinal'] = df['ParentEduc'].map(parent_educ_mapping)

    df['TestPrep_Binary'] = df['TestPrep'].map({'none': 0, 'completed': 1})

    df['LunchType_Binary'] = df['LunchType'].map({'standard': 1, 'free/reduced': 0})

    df['activities'] = df['activities'].map({'yes': 1, 'no': 0})

    df['IsFirstChild'] = df['IsFirstChild'].map({'yes': 1, 'no': 0})

    df['TransportMeans'] = df['TransportMeans'].map({'school_bus': 1, 'private': 0})



    parent_status_mapping = {
        'widowed': 1,
        'single': 2,
        'divorced': 3,
        "married": 4,
    }

    df['ParentMaritalStatus_encoded'] = df['ParentMaritalStatus'].map(parent_status_mapping)

    df = df.drop(columns=['WklyStudyHours', 'PracticeSport', 'ParentEduc', 'LunchType', 'TestPrep', 'ParentMaritalStatus'])

    return df


def perform_inference(model, X_test_path, saved_predictions_path):
    """
    Function to validate if the model saved and loaded on inference has the same predictions as
    the model in train.py
    :param model: model pkl
    :type model: pkl
    :param X_test_path: path to test set
    :type X_test_path: str
    :param saved_predictions_path: path to predictions csv of the model
    :type saved_predictions_path: str
    """
    X_test = pd.read_csv(X_test_path)
    y_predictions_inference = model.predict(X_test)
    saved_predictions = np.loadtxt(saved_predictions_path, delimiter=',', skiprows=1)

    if np.array_equal(y_predictions_inference, saved_predictions):
        print("Predictions match.")
    else:
        print("Predictions do not match.")


@app.route('/predict_grades', methods=['GET'])
def predict_grade():
    response = {
        'IsFirstChild': request.args.get('IsFirstChild'),
        'activities': request.args.get('activities'),
        'freetime': float(request.args.get('freetime')),
        'NrSiblings': float(request.args.get('NrSiblings')),
        'PracticeSport': request.args.get('PracticeSport'),
        'WklyStudyHours': request.args.get('WklyStudyHours'),
        'ParentEduc': request.args.get('ParentEduc'),
        'TestPrep': request.args.get('TestPrep'),
        'LunchType': request.args.get('LunchType'),
        'TransportMeans': request.args.get('TransportMeans'),
        'ParentMaritalStatus': request.args.get('ParentMaritalStatus'),
    }

    # mock_response = {
    #     'IsFirstChild': 1,
    #     'activities': 'yes',
    #     'freetime': 3,
    #     'NrSiblings': 2,
    #     'PracticeSport': 'regularly',
    #     'WklyStudyHours': '5 - 10',
    #     'ParentEduc': "bachelor's degree",
    #     'TestPrep': 'completed',
    #     'LunchType': 'standard',
    #     'TransportMeans': 'private',
    #     'ParentMaritalStatus': 'married',
    # }
    print(response)

    # # Creating a DataFrame from the response dictionary
    df2 = pd.DataFrame([response])
    # print(df2)

    # is_first_child = float(request.args.get('IsFirstChild'))
    # activities = float(request.args.get('activities'))
    # freetime = float(request.args.get('freetime'))
    # nr_siblings = float(request.args.get('NrSiblings'))
    # practice_sport_encoded = float(request.args.get('PracticeSport'))
    # wkly_study_hours_encoded = float(request.args.get('WklyStudyHours_'))
    # parent_educ_ordinal = float(request.args.get('ParentEduc'))
    # test_prep_binary = float(request.args.get('TestPrep'))
    # lunch_type_binary = float(request.args.get('LunchType'))
    # transport_means_private = float(request.args.get('TransportMeans'))
    # parent_marital_status_divorced = float(request.args.get('ParentMaritalStatus'))

    #input_data = [is_first_child, activities, freetime, nr_siblings, practice_sport_encoded,
    #            wkly_study_hours_encoded, parent_educ_ordinal, test_prep_binary, lunch_type_binary,
    #               transport_means_private, parent_marital_status_divorced]

    prepro_data = perform_preprocess(df2)
    prepro_data_list = prepro_data.values.tolist()

    prediction_math = model_math.predict(prepro_data_list)[0]
    prediction_reading = model_reading.predict(prepro_data_list)[0]
    prediction_writing = model_writing.predict(prepro_data_list)[0]
    # return

    # print(model_math.predict([[1.0, 0.0, 3.0, 7.0, 1.0, 2.0, 1.0, 6.0, 0.0, 1.0, 0.0]])[0])
    # print(model_reading.predict([[1.0, 0.0, 5.0, 7.0, 1.0, 2.0, 1.0, 6.0, 0.0, 1.0, 4.0]])[0])
    # print(model_writing.predict([[0.0, 0.0, 4.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0]])[0])
    #print(prediction_math, prediction_reading, prediction_writing)
    return [prediction_math, prediction_reading, prediction_writing]


@app.route('/test', methods=['GET'])
def test():
    return 'heyyyy'


if __name__ == "__main__":
    #predict_grade()
    model_path_math = 'studmodel_math.pkl'
    model_path_reading = 'studmodel_reading.pkl'
    model_path_writing = 'studmodel_writing.pkl'
    X_test_path = 'X_test.csv'

    model_math = load_model(model_path_math)
    model_reading = load_model(model_path_reading)
    model_writing = load_model(model_path_writing)

    saved_predictions_path = 'preds_math.csv'
    saved_predictions_path = 'preds_read.csv'
    saved_predictions_path = 'preds_write.csv'

    # perform_inference(model_math, X_test_path, saved_predictions_path)
    # perform_inference(model_reading, X_test_path, saved_predictions_path)
    # perform_inference(model_writing, X_test_path, saved_predictions_path)


    #app.run(host='0.0.0.0', port=5001, debug=True)
    app.run(host='0.0.0.0', port=8080, debug=True)


# testing example
# http://localhost:5001/predict_churn?is_male=1&num_inters=0&late_on_payment=1&age=1&years_in_contract=3
# http://localhost:5001/predict_grades?IsFirstChild=1&activities=yes&freetime=3&NrSiblings=2&PracticeSport=regularly&WklyStudyHours=5%20-%2010&ParentEduc=bachelor's%20degree&TestPrep=completed&LunchType=standard&TransportMeans=private&ParentMaritalStatus=married
# http://localhost:5001/predict_grades?IsFirstChild=1&activities=yes&freetime=3&NrSiblings=2&PracticeSport=regularly&WklyStudyHours=5%20-%2010&ParentEduc=bachelor's%20degree&TestPrep=completed&LunchType=standard&TransportMeans=private&ParentMaritalStatus=married
# http://localhost:5001/predict_grades?IsFirstChild=yes&activities=no&freetime=3&NrSiblings=7&PracticeSport=never&WklyStudyHours=5%20-%2010&ParentEduc=bachelor's%20degree&TestPrep=completed&LunchType=standard&TransportMeans=private&ParentMaritalStatus=widowed