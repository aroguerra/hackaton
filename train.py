import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from inference import perform_preprocess
from sklearn.ensemble import AdaBoostRegressor

if __name__ == "__main__":
    df = pd.read_csv('combined_shuffled_data.csv')
    TARGET = ['MathScore', 'ReadingScore', 'WritingScore']
    FEATS = list(set(df.columns) - set(TARGET))

    X_train, X_test, y_train, y_test = train_test_split(df[FEATS], df[TARGET],
                                                        test_size=0.2,
                                                        random_state=42)

    X_train_enc = perform_preprocess(X_train)
    X_test_enc = perform_preprocess(X_test)

    y_train_MathScore = y_train['MathScore']
    y_train_ReadingScore = y_train['ReadingScore']
    y_train_WritingScore = y_train['WritingScore']

    y_test_MathScore = y_test['MathScore']
    y_test_ReadingScore = y_test['ReadingScore']
    y_test_WritingScore = y_test['WritingScore']

    regression_model_ada_math = AdaBoostRegressor(n_estimators=100)
    regression_model_ada_math.fit(X_train_enc, y_train_MathScore)
    y_pred_ada_math = regression_model_ada_math.predict(X_test_enc)

    regression_model_ada_read = AdaBoostRegressor(n_estimators=100)
    regression_model_ada_read.fit(X_train_enc, y_train_ReadingScore)
    y_pred_ada_read = regression_model_ada_read.predict(X_test_enc)

    regression_model_ada_write = AdaBoostRegressor(n_estimators=100)
    regression_model_ada_write.fit(X_train_enc, y_train_WritingScore)
    y_pred_ada_write = regression_model_ada_write.predict(X_test_enc)

    X_test_enc.to_csv('X_test.csv', index=False)

    np.savetxt('preds_math.csv', y_pred_ada_math, delimiter=',', header='prediction', comments='')
    np.savetxt('preds_read.csv', y_pred_ada_read, delimiter=',', header='prediction', comments='')
    np.savetxt('preds_write.csv', y_pred_ada_write, delimiter=',', header='prediction', comments='')

    with open('studmodel_math.pkl', 'wb') as model_file:
        pickle.dump(regression_model_ada_math, model_file)

    with open('studmodel_reading.pkl', 'wb') as model_file:
        pickle.dump(regression_model_ada_read, model_file)

    with open('studmodel_writing.pkl', 'wb') as model_file:
        pickle.dump(regression_model_ada_write, model_file)

    print("Model saved successfully.")
