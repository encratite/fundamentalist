import os
import time
import csv
import json
from pathlib import Path
from datetime import datetime
from torch import Tensor
import configuration

class Trainer:
	FINANCIAL_STATEMENTS = "FinancialStatements"
	KEY_RATIOS = "KeyRatios"
	PRICE_DATA = "PriceData"

	def __init__(self, options):
		self._options = options
		self._financial_statement_tensors = []

	def run(self):
		self._load_datasets()

	def _load_datasets(self):
		print("Loading datasets")
		start = time.perf_counter()
		paths = self._read_directory(Trainer.FINANCIAL_STATEMENTS)
		self._financial_statement_tensors = []
		for financial_statements_path in paths:
			ticker = self._get_ticker_name(financial_statements_path)
			print(f"Processing \"{ticker}\"")
			key_ratios_path = self._get_data_file_path(Trainer.KEY_RATIOS, ticker, "json")
			price_data_path = self._get_data_file_path(Trainer.PRICE_DATA, ticker, "csv")
			if not os.path.isfile(key_ratios_path) or not os.path.isfile(price_data_path):
				continue
			valid = self._load_financial_statements(financial_statements_path)
			self._load_key_ratios(key_ratios_path)
			self._load_price_data(price_data_path)
		end = time.perf_counter()
		print(f"Loaded datasets in {end - start:0.1f} s")

	def _get_ticker_name(self, file_path):
		path = Path(file_path)
		return path.stem

	def _load_financial_statements(self, path):
		with open(path) as file:
			financial_statements = json.load(file)
		financial_statements = filter(self._is_10_q_filing, financial_statements)
		financial_statements = map(self._add_source_date, financial_statements)
		financial_statements = filter(lambda fs: fs[0] is not None, financial_statements)
		financial_statements = sorted(financial_statements, key=lambda fs: fs[0])
		count = self._options.financial_statements
		if len(financial_statements) < count:
			return False
		financial_statements = financial_statements[-count:]
		for f in financial_statements:
			date, financial_statement = f
			self._get_financial_statement_tensor(financial_statement)
		return True

	def _load_key_ratios(self, path):
		with open(path) as file:
			key_ratios = json.load(file)
		pass

	def _load_price_data(self, path):
		with open(path) as csv_file:
			reader = csv.reader(csv_file, delimiter=",")
			for row in reader:
				pass

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

	def _get_financial_statement_tensor(self, financial_statement):
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

		tensor = Tensor(values)
		self._financial_statement_tensors.append(tensor)