import os
from pathlib import Path
from torchdata.datapipes.iter import IterableWrapper, FileOpener
import configuration
from ticker import Ticker

class Trainer:
	def __init__(self):
		self.tickers = {}

	def run(self):
		self.load_data_sets()

	def load_data_sets(self):
		self.load_financial_statements()
		self.load_key_ratios()

	def load_financial_statements(self):
		json_files = self.parse_json_files("FinancialStatements")
		for path_and_json in json_files:
			name, financial_statements = self.get_path_and_json_name(path_and_json)
			financial_statements = filter(Trainer.is_10_q_filing, financial_statements)
			ticker = self.get_ticker(name)
			ticker.financial_statements = financial_statements

	def load_key_ratios(self):
		json_files = self.parse_json_files("KeyRatios")
		for path_and_json in json_files:
			name, key_ratios = self.get_path_and_json_name(path_and_json)
			ticker = self.get_ticker(name)
			ticker.key_ratios = key_ratios

	def get_path_and_json_name(self, path_and_json):
		path = Path(path_and_json[0])
		return path.stem, path_and_json[1]

	def read_directory(self, directory):
		path = os.path.join(configuration.DATA_PATH, directory)
		files = os.listdir(path)
		full_paths = map(lambda f: os.path.join(path, f), files)
		return full_paths

	def parse_json_files(self, directory):
		paths = self.read_directory(directory)
		iterable = IterableWrapper(paths)
		file_opener = FileOpener(iterable, mode="b")
		json_files = file_opener.parse_json_files()
		return json_files

	def get_ticker(self, name):
		if name not in self.tickers:
			self.tickers[name] = Ticker()
		return self.tickers[name]

	@staticmethod
	def is_10_q_filing(financial_statement):
		def is_valid_source(target_key):
			if target_key not in financial_statement:
				return False
			statement = financial_statement[target_key]
			source_key = "source"
			if statement is None or source_key not in statement:
				return False
			return statement[source_key] == "10-Q"
		targets = [
			"balanceSheets",
			"cashFlow",
			"incomeStatement"
		]
		valid = True
		for target in targets:
			valid = valid and is_valid_source(target)
		return valid
