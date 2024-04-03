from rowreader import RowReader

class PriceData:
    def __init__(self, row):
        reader = RowReader(row)
        self.date = reader.get_date()
        self.open = reader.get_float()
        self.high = reader.get_float()
        self.low = reader.get_float()
        self.close = reader.get_float()
        self.adjusted_close = reader.get_float()
        self.volume = reader.get_int()