from datetime import datetime

class TrainerOptions:
    def __init__(self):
        self.history_days = 30
        self.forecast_days = 30
        self.training_test_split_date = datetime(2019, 1, 1)