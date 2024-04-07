from datetime import datetime

class TrainerOptions:
    def __init__(self):
        self.financial_statements = 4
        self.training_test_split_date = datetime(2019, 1, 1)
        self.forecast_days = 15