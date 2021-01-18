from m5 import __data__

import pandas as pd

calendar = pd.read_csv(__data__ / "calendar.csv")
sell_prices = pd.read_csv(__data__ / "sell_prices.csv")
sales_train_eval = pd.read_csv(__data__ / "sales_train_evaluation.csv")
sales_train_valid = pd.read_csv(__data__ / "sales_train_validation.csv")
sample_submission = pd.read_csv(__data__ / "sample_submission.csv")
