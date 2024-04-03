from pricedata import PriceData

class Ticker:
    def __init__(self):
        self.financial_statements = None
        self.key_ratios = None
        self.price_data = None

    def add_price_data(self, row):
        if row[0] == "Date" or any(filter(lambda x: x == "null", row)):
            return
        if self.price_data is None:
            self.price_data = []
        price_data = PriceData(row)
        self.price_data.append(price_data)