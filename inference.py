from pycaret.regression import *
import pandas as pd

loaded_model = load_model('insurance_regression_model')

new_data = pd.DataFrame({
    'age': [35],
    'sex': ['male'],
    'bmi': [28],
    'children': [2],
    'smoker': ['no'],
    'region': ['southeast']
})

# Make predictions
predictions = predict_model(loaded_model, data=new_data)
print(predictions['prediction_label'].values)