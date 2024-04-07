import os
import sys
import time
import csv
import json
from pathlib import Path
from datetime import datetime, timedelta
import torch
from torch import Tensor
import torch.nn as nn
import tqdm
import configuration
from normalization import NormalizationTracker

class Trainer:
	FINANCIAL_STATEMENTS = "FinancialStatements"
	KEY_RATIOS = "KeyRatios"
	PRICE_DATA = "PriceData"

	def __init__(self, options):
		self._options = options
		self._new_data = []
		self._training_input = []
		self._test_input = []
		self._training_labels = []
		self._test_labels = []
		self._normalization_tracker = NormalizationTracker()

	def run(self):
		self._load_datasets()

	def _load_datasets(self):
		print("Loading datasets")
		self._normalization_tracker = NormalizationTracker()
		start = time.perf_counter()
		paths = self._read_directory(Trainer.FINANCIAL_STATEMENTS)
		self._new_data = []
		self._training_input = []
		self._test_input = []
		self._training_labels = []
		self._test_labels = []
		for financial_statements_path in paths:
			ticker = self._get_ticker_name(financial_statements_path)
			print(f"Processing \"{ticker}\"")
			key_ratios_path = self._get_data_file_path(Trainer.KEY_RATIOS, ticker, "json")
			price_data_path = self._get_data_file_path(Trainer.PRICE_DATA, ticker, "csv")
			if not os.path.isfile(key_ratios_path) or not os.path.isfile(price_data_path):
				continue
			has_data = self._load_financial_statements(financial_statements_path)
			if not has_data:
				continue
			# self._load_key_ratios(key_ratios_path)
			price_data = self._load_price_data(price_data_path)
			self._process_new_data(price_data)
		end = time.perf_counter()
		self._normalization_tracker.normalize(self._training_input)
		self._normalization_tracker.normalize(self._test_input)
		print(f"Loaded datasets in {end - start:0.1f} s")
		self._train_model()

	def _get_ticker_name(self, file_path):
		path = Path(file_path)
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
			return False
		dated_features = []
		for f in financial_statements:
			date, financial_statement = f
			features = self._get_financial_statement_features(financial_statement)
			dated_features.append((date, features))
		for i in range(max(len(dated_features) - count, 0)):
			j = i + count
			batch = dated_features[i:j]
			batch_date = batch[-1][0]
			output = (batch_date, list(map(lambda x: x[1], batch))[0])
			self._new_data.append(output)
		return True

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

		price_data = []
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
				price_data.append((date, open_price, close_price))
		return price_data

	def _get_price(self, date, price_data):
		previous = None
		for p in price_data:
			price_date, open, close = p
			current = (open, close)
			if price_date == date:
				return current
			elif price_date > date:
				return previous
			previous = current
		return None

	def _process_new_data(self, price_data):
		for data in self._new_data:
			batch_date, features = data
			future_date = batch_date + timedelta(days=self._options.forecast_days)
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
				self._training_input.append(features)
				self._training_labels.append(label)
			else:
				self._test_input.append(features)
				self._test_labels.append(label)
		self._new_data = []

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
		return datetime(year, month, day)

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
				self._normalization_tracker.handle_value(value)
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

		self._normalization_tracker.reset()

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

	def _train_model(self):
		training_input = Tensor(self._training_input)
		test_input = Tensor(self._test_input)
		training_labels = Tensor(self._training_labels)
		test_labels = Tensor(self._test_labels)

		self._training_input = []
		self._test_input = []
		self._training_labels = []
		self._test_labels = []

		features = training_input.shape[1]
		gru_hidden_size = 128
		model = nn.Sequential(
			nn.GRUCell(features, gru_hidden_size),
			nn.ReLU(),
			nn.Linear(gru_hidden_size, 1)
		)

		loss_function = nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters())

		epochs = 50
		batch_size = 8
		batch_start = torch.arange(0, len(training_input), batch_size)

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
					bar.set_postfix(mse=float(loss))
			model.eval()
		prediction = torch.flatten(model(test_input))
		loss = float(loss_function(prediction, test_labels))
		print(f"MSE with test data: {loss:.3f}")