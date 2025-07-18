import unittest
import pandas as pd # type: ignore
from _var_calculation import VarCalc


class TestCalc(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.var_calculator = VarCalc()

        """Reading the input file"""
        self.df = pd.read_csv("C:/Users/Nammal/Desktop/assignment/assignment/data_var_calculation/ccy_rates.txt", delimiter="\t", parse_dates=["date"], index_col="date")
    

        """Setting the input values"""
    def test_calculate_var(self):
        ccy1= 153084.81
        ccy2 = 95891.51
        horizon_days = 1

        """Defining calculation config"""
        calculation_config = [
            (self.df[['ccy-1']], horizon_days, ccy1, self.var_calculator.log_shift),
            (self.df[['ccy-2']], horizon_days, ccy2, self.var_calculator.log_shift)
        ]

        """ Calling the calculate_var method and passing configs as parameter"""
        calculated_var = self.var_calculator.calculate_var(calculation_config)
        print(f"\nCalculated 1-Day VaR: {calculated_var:.2f}")

        self.expected_var = -13572.73 #expected value

        """ Comapring the calculated and expected value """
        self.assertAlmostEqual(calculated_var, self.expected_var, places=2,
                               msg=f"Expected around {self.expected_var}, but got {calculated_var}")


if __name__ == '__main__':
    unittest.main()
