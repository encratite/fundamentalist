import os
import time
import csv
import json
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

	def run(self):
		self._load_datasets()

	def _load_datasets(self):
		print("Loading datasets")
		time1 = time.perf_counter()

		paths = self._read_directory(Trainer.FINANCIAL_STATEMENTS)
		self._training_input = []
		self._test_input = []
		self._training_labels = []
		self._test_labels = []
		if Trainer.RUN_LOADER_TEST:
			paths = list(paths)[:50]

		pool = multiprocessing.Pool(processes=configuration.LOADER_THREADS)
		results = pool.map(self._run_thread, paths)
		pool.close()
		pool.join()

		for result in results:
			if result is None:
				continue
			training_input, training_labels, test_input, test_labels = result
			self._training_input.extend(training_input)
			self._training_labels.extend(training_labels)
			self._test_input.extend(test_input)
			self._test_labels.extend(test_labels)

		time2 = time.perf_counter()
		self._print_perf_counter("Loaded datasets", time1, time2)

		self._normalize()
		if not Trainer.USE_CPROFILE:
			self._train_model()

	def _run_thread(self, financial_statements_path):
		ticker = self._get_ticker_name(financial_statements_path)
		# print(f"Processing \"{ticker}\"")
		key_ratios_path = self._get_data_file_path(Trainer.KEY_RATIOS, ticker, "json")
		price_data_path = self._get_data_file_path(Trainer.PRICE_DATA, ticker, "csv")
		if not os.path.isfile(key_ratios_path) or not os.path.isfile(price_data_path):
			return None
		fundamental_data = self._load_financial_statements(financial_statements_path)
		if fundamental_data is None:
			return None
		# self._load_key_ratios(key_ratios_path)
		price_data = self._load_price_data(price_data_path)
		output = self._split_data(fundamental_data, price_data)
		return output

	def _print_perf_counter(self, description, start, end):
		print(f"{description} in {end - start:0.1f} s")

	def _get_ticker_name(self, file_path):
		path = pathlib.Path(file_path)
		return path.stem

	def _load_financial_statements(self, path):
		with open(path) as file:
			financial_statements = json.load(file)
		financial_statements = filter(self._is_10_q_filing, financial_statements)
		financial_statements = map(self._add_source_date, financial_statements)
		financial_statements = filter(lambda x: x[0] is not None, financial_statements)
		financial_statements = sorted(financial_statements, key=lambda fs: fs[0])
		count = self._options.financial_statements
		if len(financial_statements) < count:
			return None
		dated_features = []
		for f in financial_statements:
			date, financial_statement = f
			features = self._get_financial_statement_features(financial_statement)
			dated_features.append((date, features))
		output = []
		for i in range(max(len(dated_features) - count, 0)):
			j = i + count
			batch = dated_features[i:j]
			batch_date = batch[-1][0]
			new_data = (batch_date, list(map(lambda x: x[1], batch)))
			output.append(new_data)
		return output

	def _load_key_ratios(self, path):
		with open(path) as file:
			key_ratios = json.load(file)
		pass

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

	def _split_data(self, fundamental_data, price_data):
		training_input = []
		training_labels = []
		test_input = []
		test_labels = []
		for sample in fundamental_data:
			batch_date, features = sample
			future_date = batch_date + datetime.timedelta(days=self._options.forecast_days)
			price1 = self._get_price(batch_date, price_data)
			price2 = self._get_price(future_date, price_data)
			if price1 is None or price2 is None:
				continue
			open1, close1 = price1
			open2, close2 = price2
			if open1 == 0 or close1 == 0 or open2 == 0 or close2 == 0:
				continue
			label = close2 / open1 - 1.0
			if batch_date < self._options.training_test_split_date:
				training_input.append(features)
				training_labels.append(label)
			else:
				test_input.append(features)
				test_labels.append(label)
		return training_input, training_labels, test_input, test_labels

	def _read_directory(self, directory):
		path = os.path.join(configuration.DATA_PATH, directory)
		files = os.listdir(path)
		full_paths = map(lambda f: os.path.join(path, f), files)
		return full_paths

	def _get_data_file_path(self, directory, ticker, extension):
		return os.path.join(configuration.DATA_PATH, directory, f"{ticker}.{extension}")

	def _is_10_q_filing(self, financial_statement):
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

	def _get_datetime(self, string):
		year = int(string[0:4])
		month = int(string[5:7])
		day = int(string[8:10])
		return datetime.datetime(year, month, day)

	def _get_source_date(self, financial_statement):
		source_date_string = financial_statement["balanceSheets"].get("sourceDate", None)
		if source_date_string is None:
			return None
		return self._get_datetime(source_date_string)

	def _add_source_date(self, financial_statement):
		source_date = self._get_source_date(financial_statement)
		return (source_date, financial_statement)

	def _get_financial_statement_features(self, financial_statement):
		def get_dict(key, dictionary):
			return dictionary.get(key, {})

		def add_values(keys, dictionary):
			for key in keys:
				value = dictionary.get(key, 0)
				values.append(value)

		balance_sheets = financial_statement["balanceSheets"]
		cash_flow = financial_statement["cashFlow"]
		income_statement = financial_statement["incomeStatement"]

		current_assets = get_dict("currentAssets", balance_sheets)
		long_term_assets = get_dict("longTermAssets", balance_sheets)
		current_liabilities = get_dict("currentLiabilities", balance_sheets)
		long_term_liabilities = get_dict("longTermLiabilities", balance_sheets)
		equity = get_dict("equity", balance_sheets)

		financing = get_dict("financing", cash_flow)
		investing = get_dict("investing", cash_flow)
		operating = get_dict("operating", cash_flow)

		expense = get_dict("expense", income_statement)
		income = get_dict("income", income_statement)
		revenue = get_dict("revenue", income_statement)
		cash = get_dict("cash", income_statement)

		values = []

		current_assets_keys = [
			"accountsReceivableTradeNet",
			"totalReceivablesNet",
			"totalInventory",
			"otherCurrentAssetsTotal",
			"totalCurrentAssets",
			"propertyPlantEquipmentTotalGross",
			"accumulatedDepreciationTotal",
			"intangiblesNet",
			"longTermInvestments",
			"totalAssets",
			"shortTermInvestments",
			"cashAndShortTermInvestments",
			"cashEquivalents",
			"goodwillNet",
			"accountsReceivableTradeGross",
			"receivablesOther",
			"inventoriesFinishedGoods",
			"inventoriesWorkInProgress",
			"inventoriesRawMaterials",
			"otherCurrentAssets",
			"buildingsGross",
			"landImprovementsGross",
			"machineryEquipmentGross",
			"otherPropertyPlantEquipmentGross",
			"intangiblesGross",
			"accumulatedIntangibleAmortization",
			"ltInvestmentAffiliateCompanies",
			"otherLongTermAssets",
			"otherLongTermAssetsTotal",
			"payableAccrued",
			"accruedExpenses",
			"propertyPlantEquipmentTotalNet"
		]
		add_values(current_assets_keys, current_assets)

		long_term_assets_keys = [
			"totalOperatingLeasesSupplemental",
			"totalCurrentAssetsLessInventory",
			"quickRatio",
			"currentRatio",
			"netDebtInclPrefStockMinInterest",
			"tangibleBookValueCommonEquity",
			"tangibleBookValuePerShareCommonEq"
		]
		add_values(long_term_assets_keys, long_term_assets)

		current_liabilities_keys = [
			"otherCurrentLiabilitiesTotal",
			"totalCurrentLiabilities",
			"longTermDebt",
			"capitalLeaseObligations",
			"totalDebt",
			"totalLiabilities",
			"notesPayableShortTermDebt",
			"currentPortOfLTDebtCapitalLeases",
			"incomeTaxesPayable",
			"otherCurrentLiabilities",
			"totalLongTermDebt",
			"otherLongTermLiabilities",
			"otherPayables",
			"otherLiabilitiesTotal",
			"accountsPayable"
		]
		add_values(current_liabilities_keys, current_liabilities)

		long_term_liabilities_keys = [
			"longTermDebtMaturingWithin1Year",
			"longTermDebtMaturingInYear2",
			"longTermDebtMaturingInYear3",
			"longTermDebtMaturingInYear4",
			"longTermDebtMaturingInYear5",
			"longTermDebtMaturingIn2Or3Years",
			"longTermDebtMaturingIn4Or5Years",
			"longTermDebtMaturingInYear6AndBeyond",
			"totalLongTermDebtSupplemental",
			"operatingLeasePaymentsDueInYear1",
			"operatingLeasePaymentsDueInYear2",
			"operatingLeasePaymentsDueInYear3",
			"operatingLeasePaymentsDueInYear4",
			"operatingLeasePaymentsDueInYear5",
			"operatingLeasePymtsDuein2Or3Years",
			"operatingLeasePymtsDuein45Years",
			"operatingLeasePaymentsDueInYear6AndBeyond"
		]
		add_values(long_term_liabilities_keys, long_term_liabilities)

		equity_keys = [
			"commonStockTotal",
			"additionalPaidInCapital",
			"retainedEarningsAccumulatedDeficit",
			"unrealizedGainLoss",
			"totalEquity",
			"totalLiabilitiesShareholdersEquity",
			"totalCommonSharesOutstanding",
			"commonStock",
			"otherComprehensiveIncome",
			"otherEquityTotal",
			"totalEquityMinorityInterest",
			"sharesOutstandingCommonStockPrimaryIssue",
			"treasurySharesCommonStockPrimaryIssue",
			"accumulatedIntangibleAmortSuppl",
			"translationAdjustment"
		]
		add_values(equity_keys, equity)

		financing_keys = [
			"totalCashDividendsPaid",
			"issuanceRetirementOfDebtNet",
			"cashFromFinancingActivities",
			"otherFinancingCashFlow",
			"financingCashFlowItems",
			"cashDividendsPaidCommon",
			"repurchaseRetirementOfCommon",
			"commonStockNet",
			"issuanceRetirementOfStockNet",
			"longTermDebtIssued",
			"longTermDebtReduction",
			"longTermDebtNet"
		]
		add_values(financing_keys, financing)

		investing_keys = [
			"cashFromInvestingActivities",
			"capitalExpenditures",
			"purchaseOfFixedAssets",
			"saleMaturityOfInvestment",
			"purchaseOfInvestments",
			"otherInvestingCashFlow",
			"otherInvestingCashFlowItemsTotal",
		]
		add_values(investing_keys, investing)

		operating_keys = [
			"netIncomeStartingLine",
			"depreciationDepletion",
			"changesInWorkingCapital",
			"cashFromOperatingActivities",
			"netChangeInCash",
			"foreignExchangeEffects",
			"otherNonCashItems",
			"nonCashItems",
			"accountsReceivable",
			"inventories",
			"otherLiabilities",
			"depreciationSupplemental",
			"netCashBeginningBalance",
			"netCashEndingBalance",
			"payableAccrued"
		]
		add_values(operating_keys, operating)

		expense_keys = [
			"grossProfit",
			"sellingGeneralAdminExpensesTotal",
			"unusualExpenseIncome",
			"researchDevelopment",
			"restructuringCharge",
			"interestExpenseSupplemental",
			"depreciationSupplemental",
			"amortizationOfIntangiblesSupplemental",
			"stockBasedCompensationSupplemental",
			"rentalExpenseSupplemental",
			"researchDevelopmentExpSupplemental",
			"sellingGeneralAdminExpenses",
			"laborRelatedExpense",
			"totalOperatingExpense"
		]
		add_values(expense_keys, expense)

		income_keys = [
			"incomeAvailableToComExclExtraOrd",
			"incomeAvailableToComInclExtraOrd",
			"provisionForIncomeTaxes",
			"netIncomeBeforeTaxes",
			"otherNet",
			"netIncomeBeforeExtraItems",
			"minorityInterest",
			"operatingIncome",
			"interestIncExpNetNonOpTotal",
			"netInterestIncome",
			"interestInvestIncomeNonOperating",
			"otherNonOperatingIncomeExpense",
			"interestExpenseNonOperating",
			"incomeInclExtraBeforeDistributions",
			"normalizedIncomeBeforeTaxes",
			"incomeTaxExImpactOfSpeciaItems",
			"normalizedIncomeAfterTaxes",
			"normalizedIncAvailToCom",
			"netIncomeAfterTaxes",
			"investmentIncomeNonOperating",
			"effectOfSpecialItemsOnIncomeTaxes",
			"interestExpenseNetNonOperating",
			"bankTotalRevenue"
		]
		add_values(income_keys, income)

		revenue_keys = [
			"opsCommonStockPrimaryIssue",
			"costOfRevenueTotal",
			"netSales",
			"costOfRevenue",
			"basicNormalizedEPS",
			"dilutedNormalizedEPS",
			"grossMargin",
			"operatingMargin",
			"normalizedEBIT",
			"dilutedWeightedAverageShares",
			"dilutedEPSExcludingExtraOrdItems",
			"basicWeightedAverageShares",
			"basicEPSExcludingExtraordinaryItems",
			"basicEPSIncludingExtraordinaryItems",
			"normalizedEBITDA",
			"dilutedEPSIncludingExtraOrdItems",
			"totalRevenue"
		]
		add_values(revenue_keys, revenue)

		cash_keys = [
			"investing",
			"financing",
			"total"
		]
		add_values(cash_keys, cash)

		return values

	def _normalize(self):
		def get_chain():
			return itertools.chain(self._training_input, self._test_input)

		def flatten(input):
			output = []
			for element in input:
				output.extend(element)
			return output

		def flatten_input(dataset):
			return list(map(flatten, dataset))

		print("Normalizing data")
		time1 = time.perf_counter()

		normalizer = Normalizer()
		for datapoint in get_chain():
			for features in datapoint:
				normalizer.reset()
				for feature in features:
					normalizer.handle_value(feature)
		for datapoint in get_chain():
			normalizer.normalize(datapoint)
		self._training_input = flatten_input(self._training_input)
		self._test_input = flatten_input(self._test_input)

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
			torch.nn.LSTMCell(features, 256),
			TransformOutput(),
			torch.nn.ReLU(),
			torch.nn.LSTMCell(256, 128),
			TransformOutput(),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 1)
		)
		model.to(device)

		loss_function = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters())

		epochs = 50
		batch_size = 256 * 1024
		batch_start = torch.arange(0, len(training_input), batch_size)

		test_loss = 0

		print(f"Commencing training for {features} features")
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