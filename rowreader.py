from datetime import datetime

class RowReader:
    def __init__(self, row):
        self.row = row
        self.index = 0

    def get_string(self):
        string = self.row[self.index]
        self.index += 1
        return string

    def get_float(self):
        string = self.get_string()
        return float(string)

    def get_int(self):
        string = self.get_string()
        return int(string)

    def get_date(self):
        string = self.get_string()
        return datetime.strptime(string, "%Y-%m-%d")