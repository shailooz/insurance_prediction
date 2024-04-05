from pycaret.datasets import get_data
data = get_data('insurance')

from pycaret.regression import *

reg_setup = setup(data, target='charges', session_id=42)

best_model = compare_models()


final_model = finalize_model(best_model)

save_model(final_model, 'insurance_regression_model')