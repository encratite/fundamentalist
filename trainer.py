import os
import time
import csv
import json
import sys
import sortedcontainers
import pathlib
import datetime
import itertools
import torch
import tqdm
import multiprocessing
import configuration
from normalizer import Normalizer
from transform import TransformOutput

class Trainer:
	USE_GPU = True
	RUN_LOADER_TEST = False
	USE_CPROFILE = False

	FINANCIAL_STATEMENTS = "FinancialStatements"
	KEY_RATIOS = "KeyRatios"
	PRICE_DATA = "PriceData"

	def __init__(self, options):
		self._options = options
		self._training_input = []
		self._test_input = []
		self._training_labels = []
		self._test_labels = []
		self._backtesting_data = []

	def run(self):
		self._load_datasets()

	def _load_datasets(self):
		print("Loading datasets")
		time1 = time.perf_counter()

		paths = self._read_directory(Trainer.KEY_RATIOS)
		self._training_input = []
		self._test_input = []
		self._training_labels = []
		self._test_labels = []
		self._backtesting_data = []
		if Trainer.RUN_LOADER_TEST:
			paths = list(paths)[:50]

		pool = multiprocessing.Pool(processes=configuration.LOADER_THREADS)
		results = pool.map(self._run_thread, paths)
		pool.close()
		pool.join()

		for result in results:
			if result is None:
				continue
			training_input, training_labels, test_input, test_labels, backtesting_data = result
			self._training_input.extend(training_input)
			self._training_labels.extend(training_labels)
			self._test_input.extend(test_input)
			self._test_labels.extend(test_labels)
			self._backtesting_data.extend(backtesting_data)
			self._backtesting_data = sorted(self._backtesting_data, key=lambda x: x[0])

		time2 = time.perf_counter()
		self._print_perf_counter("Loaded datasets", time1, time2)

		self._normalize()

		if not Trainer.USE_CPROFILE:
			self._train_model()

	def _run_thread(self, key_ratios_path):
		ticker = self._get_ticker_name(key_ratios_path)
		price_data_path = self._get_data_file_path(Trainer.PRICE_DATA, ticker, "csv")
		if not os.path.isfile(key_ratios_path) or not os.path.isfile(price_data_path):
			return None
		key_ratios = self._load_key_ratios(key_ratios_path)
		price_data = self._load_price_data(price_data_path)
		output = self._generate_datasets(ticker, key_ratios, price_data)
		return output

	def _print_perf_counter(self, description, start, end):
		print(f"{description} in {end - start:0.1f} s")

	def _get_ticker_name(self, file_path):
		path = pathlib.Path(file_path)
		return path.stem

	def _load_key_ratios(self, path):
		with open(path) as file:
			key_ratios = json.load(file)
		company_metrics = key_ratios["companyMetrics"]
		company_metrics = filter(lambda x: x["fiscalPeriodType"][0] == "Q", company_metrics)
		company_metrics = map(self._add_end_date, company_metrics)
		company_metrics = filter(lambda x: x[0] is not None, company_metrics)
		features = list(map(lambda x: (x[0], self._get_key_ratio_features(x[1])), company_metrics))
		return features

	def _get_key_ratio_features(self, metrics):
		def add_values(keys, dictionary):
			for key in keys:
				value = dictionary.get(key, 0)
				if not isinstance(value, int) and not isinstance(value, float):
					value = 0
				limit = 1e3
				if value > limit:
					# print(f"Invalid value for key \"{key}\": {value}")
					value = limit
				values.append(value)

		values = []

		keys = [
			"revenuePerShare",
			"earningsPerShare",
			"freeCashFlowPerShare",
			"bookValuePerShare",
			"revenueGrowthRate",
			"grossMargin",
			"operatingMargin",
			"netMargin",
			"roe",
			"roic",
			"debtToEquityRatio",
			"financialLeverage",
			"quickRatio",
			"currentRatio",
			"debtToEbitda",
			"assetTurnover",
			"receivableTurnover",
			"returnOnAssetCurrent",
			"priceToSalesRatio",
			"priceToEarningsRatio",
			"priceToCashFlowRatio",
			"priceToBookRatio",
			"evEbitda",
			"earningsGrowthRate",
			"freeCashFlowGrowthRate",
			"bookValueGrowthRate"
		]
		add_values(keys, metrics)

		return values


	def _load_price_data(self, path):
		def get_float(i, row):
			cell = row[i]
			if cell == "null":
				return None
			return float(cell)

		price_data = sortedcontainers.SortedKeyList(key=lambda x: x[0])
		with open(path) as csv_file:
			reader = csv.reader(csv_file, delimiter=",")
			iter_reader = iter(reader)
			next(iter_reader)
			for row in iter_reader:
				date = self._get_datetime(row[0])
				open_price = get_float(1, row)
				close_price = get_float(4, row)
				if open_price is None or close_price is None:
					continue
				price_data.add((date, open_price, close_price))
		return price_data

	def _get_price(self, date, price_data):
		past = date - datetime.timedelta(days=4)
		iter = price_data.irange_key(past, date)
		previous = None
		for p in iter:
			price_date, open, close = p
			current = (open, close)
			if price_date == date:
				return current
			previous = current
		return previous

	def _generate_datasets(self, ticker, key_ratios, price_data):
		def get_performance(past, future):
			return future / past - 1.0

		training_input = []
		training_labels = []
		test_input = []
		test_labels = []
		backtesting_data = []

		for sample in key_ratios:
			current_date, features = sample
			past_date = current_date - datetime.timedelta(days=self._options.history_days)
			future_date = current_date + datetime.timedelta(days=self._options.forecast_days)

			current_price = self._get_price(current_date, price_data)
			past_price = self._get_price(past_date, price_data)
			future_price = self._get_price(future_date, price_data)
			if current_price is None or past_price is None or future_price is None:
				continue

			past_open, past_close = past_price
			current_open, current_close = current_price
			future_open, future_close = future_price

			past_zero = past_open == 0 or past_close == 0
			current_zero = current_open == 0 or current_close == 0
			future_zero = future_open == 0 or future_close == 0
			if past_zero or current_zero or future_zero:
				continue

			performance = get_performance(past_open, current_open)
			features.append(performance)
			label = get_performance(current_open, future_close)
			if current_date < self._options.training_test_split_date:
				training_input.append(features)
				training_labels.append(label)
			else:
				test_input.append(features)
				test_labels.append(label)
				backtest = (ticker, current_date, features, label)
				backtesting_data.append(backtest)
		return training_input, training_labels, test_input, test_labels, backtesting_data

	def _read_directory(self, directory):
		path = os.path.join(configuration.DATA_PATH, directory)
		files = os.listdir(path)
		full_paths = map(lambda f: os.path.join(path, f), files)
		return full_paths

	def _get_data_file_path(self, directory, ticker, extension):
		return os.path.join(configuration.DATA_PATH, directory, f"{ticker}.{extension}")

	def _get_datetime(self, string):
		if "/" in string:
			year = int(string[6:10])
			month = int(string[0:2])
			day = int(string[3:5])
		else:
			year = int(string[0:4])
			month = int(string[5:7])
			day = int(string[8:10])
		return datetime.datetime(year, month, day)

	def _get_end_date(self, metrics):
		date_string = metrics.get("fiscalPeriodEndDate", None)
		if date_string is None:
			return None
		return self._get_datetime(date_string)

	def _add_end_date(self, metrics):
		date = self._get_end_date(metrics)
		return (date, metrics)

	def _normalize(self):
		print("Normalizing data")
		time1 = time.perf_counter()

		normalizer = Normalizer()
		for datapoint in itertools.chain(self._training_input, self._test_input):
			normalizer.reset()
			for feature in datapoint:
				normalizer.handle_value(feature)

		normalizer.normalize(self._training_input)
		normalizer.normalize(self._test_input)

		time2 = time.perf_counter()
		self._print_perf_counter("Normalized data", time1, time2)

	def _train_model(self):
		print("Creating tensors")
		device = "cuda" if Trainer.USE_GPU else "cpu"

		training_input = torch.tensor(self._training_input, device=device)
		test_input = torch.tensor(self._test_input, device=device)
		training_labels = torch.tensor(self._training_labels, device=device)
		test_labels = torch.tensor(self._test_labels, device=device)

		self._training_input = []
		self._test_input = []
		self._training_labels = []
		self._test_labels = []

		features = training_input.shape[1]
		model = torch.nn.Sequential(
			torch.nn.Linear(features, 1)
		)
		model.to(device)

		loss_function = torch.nn.L1Loss()
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)

		epochs = 1000
		batch_size = 256 * 1024
		batch_start = torch.arange(0, len(training_input), batch_size)

		test_loss = 0

		print(f"Commencing training with {training_input.shape[0]} data points and {features} features")
		time1 = time.perf_counter()
		for epoch in range(1, epochs + 1):
			model.train()
			with tqdm.tqdm(batch_start, unit="batch", mininterval=0) as bar:
				bar.set_description(f"Epoch {epoch}")
				for start in bar:
					offset = start + batch_size
					input_batch = training_input[start:offset]
					label_batch = training_labels[start:offset]
					prediction = torch.flatten(model(input_batch))
					loss = loss_function(prediction, label_batch)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					bar.set_postfix(training_loss=float(loss), test_loss=test_loss)
			model.eval()
			test_prediction = torch.flatten(model(test_input))
			test_loss = float(loss_function(test_prediction, test_labels))
		time2 = time.perf_counter()
		self._print_perf_counter("Trained and evaluated model", time1, time2)
		self._backtest(model)

	def _backtest(self, model):
		print("Performing backtest")
		initial_capital = 100000
		money = initial_capital
		portfolio_stocks = 5
		fee = 10
		now = self._backtesting_data[0][1]
		while money > 20000:
			future = now + datetime.timedelta(days=self._options.forecast_days)
			available = list(filter(lambda x: x[1] >= now and x[1] < future, self._backtesting_data))
			if len(available) == 0:
				break
			print(f"Updating portfolio on {now} with ${money:.2f} in the bank")
			rated = []
			for backtest in available:
				ticker, current_date, features, label = backtest
				input_tensor = torch.tensor(features, device="cuda")
				rating = float(torch.flatten(model(input_tensor)))
				entry = (ticker, current_date, rating, label)
				rated.append(entry)
			rated = sorted(rated, key=lambda x: x[2], reverse=True)
			count = min(portfolio_stocks, len(rated))
			buy = rated[0:count]
			stake = money / count
			for entry in buy:
				ticker, current_date, rating, label = entry
				gain = stake * label
				if label > 0:
					print(f"Gained ${gain:.2f} by investing in {ticker} (rating {rating:.4f})")
				else:
					print(f"Lost ${abs(gain):.2f} by investing in {ticker} (rating {rating:.4f})")
				money += gain
			money -= count * fee
			now = future
		print(f"Finished backtest with ${money:.2f} in the bank ({money / initial_capital - 1.0:+.2%})")