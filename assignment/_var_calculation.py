import pandas as pd # type: ignore
import numpy as np # type: ignore
import datetime as dt
from typing import Callable, Tuple
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('VaR Calculation')

class VarCalc:

	def _init_(self):
		pass

	@staticmethod
	def read_csv(filepath: str) -> pd.DataFrame:
		"""Utility function to generate timeseries dataframe from assignment data"""
		df = pd.read_csv(filepath_or_buffer=filepath, delimiter='\t')
		df['date'] = df.date.apply(dt.datetime.strptime, args=['%d/%m/%Y'])
		df.set_index('date', drop=True, inplace=True)

		return df

	@staticmethod
	def _add_shifted_time_column(df: pd.DataFrame) -> pd.DataFrame:
		"""
		Adds a new column with dates shifted backward
		:param df: instrument timeseries
		:return: instrument_timeseries dataframe with shifted column
		"""
		df['time_shift'] = df[df.columns[0]].shift(-1)

		return df

	@staticmethod
	def log_shift(time0_value: float, time1_value: float, horizon_days: float) -> float:
		"""
		Used to calculate a log return for an instrument based on single day data. sqrt(horizon) is then used to estimate
		the N day expectation of this return
		:param time0_value: value at start of the period
		:param time1_value: value at the end of the period
		:param horizon_days: days to estimate return for based on single day of data
		:return: float shift amount
		"""
		return np.exp(np.log(time1_value / time0_value) * np.sqrt(horizon_days)) - 1

	@staticmethod
	def _calculate_shift_return(df: pd.DataFrame, horizon_days: float, shift_function: Callable) -> pd.DataFrame:
		"""
		Applies the shift logic over all rows of the instrument timeseries
		:param df: instrument times series dataframe
		:param horizon_days: days for which the return should be estimated
		:param shift_function: the python callable used to calculate the shift
		:return: instrument time series dataframe with new 'shifted_change' column
		"""
		# the shifted value (row.iloc[1]) corresponds to the previous day's value
		df['shifted_change'] = df.apply(lambda row: shift_function(time0_value=row.iloc[1],
																time1_value=row.iloc[0],
																horizon_days=horizon_days), axis=1)
		return df

	@staticmethod
	def _calculate_pnl_for_shift(df: pd.DataFrame, portfolio_value: float) -> pd.DataFrame:
		"""
		Multiply the estimated daily return by the value of the instrument in the portfolio
		:param df: instrument time series dataframe
		:param portfolio_value: Value of the instrument in the portfolio
		:return: instrument time series dataframe with new 'pnl_vector' column
		"""
		df['pnl_vector'] = df['shifted_change'] * portfolio_value

		return df

	
	def _calculate_instrument_pnl_vector(self, instrument_timeseries: pd.DataFrame, portfolio_value: float,
										return_function: Callable, horizon_days: float) -> pd.DataFrame:
		"""
		Calculates the PnL for the given portfolio given the historical returns found in the instrument time series
		:param instrument_timeseries: a dataframe with business date index containing instrument values on those days
		:param portfolio_value: Value of the instrument in the portfolio
		:param return_function: the function used to calculate the instruments return
		:param horizon_days: number of days for which we estimate the return based on the one-day return
		:return: instrument_timeseries dataframe with new 'pnl_vector' column
		"""
		instrument_timeseries = self._add_shifted_time_column(instrument_timeseries)
		instrument_timeseries = self._calculate_shift_return(instrument_timeseries, horizon_days=horizon_days,
														shift_function=return_function)
		return self._calculate_pnl_for_shift(df=instrument_timeseries, portfolio_value=portfolio_value)

	@staticmethod
	def _calculate_99_var_from_total_pnls(total_pnl_vector: pd.Series) -> float:
		"""
		Performs a .99 confidence level VaR calculation -> (0.4 * second worst PnL) + (0.6 * third worst PnL)

		Assumes total_pnl_vector contains 260 days of pnls values

		:param total_pnl_vector: the series of historical returns for the instrument in the portfolio
		:return: float VaR value
		"""
		sorted_returns = total_pnl_vector.sort_values()
		return sorted_returns.iloc[1] * 0.4 + sorted_returns.iloc[2] * 0.6


	def calculate_var(self, calculation_config: list[Tuple[pd.DataFrame, float, float, Callable]]) -> float:
		"""
		Performs a .99 confidence level VaR calculation for all instruments in a given portfolio.

		instrument time series should contain 260 days of instruments prices.

		Configured using a calculation config, e.g.:

		calculation_config = [(instrument 1 timeseries, horizon_days, instrument 1 value in portfolio,
							instrument 1 shift function),
							(instrument 2 timeseries, horizon_days, instrument 2 value in portfolio,
							instrument 2 shift function), ...]

		:param calculation_config: List of tuples providing data and calculation methodology
		:return: a value for VaR for the configured portfolio
		"""
		logger.info(f'Received calculation config containing {len(calculation_config)} instruments')
		portfolio_total_pnl_vector = pd.Series(index=calculation_config[0][0].index, data=0)
		for (timeseries, _horizon_days, portfolio_value, return_function) in calculation_config:
			logger.info(f'calculating return pnls for instrument using N={_horizon_days}, portfolio_value={portfolio_value}'
						f', return_function={return_function.__name__}')
			component_pnl = self._calculate_instrument_pnl_vector(instrument_timeseries=timeseries,
															portfolio_value=portfolio_value,
															return_function=return_function,
															horizon_days=_horizon_days)

			portfolio_total_pnl_vector += component_pnl['pnl_vector']

		return self._calculate_99_var_from_total_pnls(total_pnl_vector=portfolio_total_pnl_vector)