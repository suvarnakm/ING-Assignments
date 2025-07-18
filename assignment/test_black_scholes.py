import unittest
from black_scholes import black_scholes_price, CallPut, InvalidPriceParametersException

class TestBlackScholes(unittest.TestCase):
    """ Testing base cases """
    def test_call(self):
        price = black_scholes_price(
            forward=19.04367,
            strike=17,
            time_to_expiry=0.460,
            volatility=0.3,
            interest_rate=0.005,
            call_put=CallPut.CALL
        )
        expected_price = 2.70 
        self.assertAlmostEqual(price, expected_price, places=2)
    def test_put(self):
        price = black_scholes_price(
            forward=19.04367,
            strike=17,
            time_to_expiry=0.460,
            volatility=0.3,
            interest_rate=0.005,
            call_put=CallPut.PUT
        )
        expected_price = 0.66
        self.assertAlmostEqual(price, expected_price, places=2)

    """ Testing strike =0"""
    def test_strike_zero_call(self):
        result = black_scholes_price(19.04367, 0, 0.460, 0.3, 0.005, CallPut.CALL)
        self.assertEqual(result, 19.04367)
    def test_strike_zero_put(self):
        result = black_scholes_price(19.04367, 0, 0.460, 0.3, 0.005, CallPut.PUT)
        self.assertEqual(result, 0)

    """ Testing time_to_expiry =0"""
    def test_time_to_expiry_call(self):
        result = black_scholes_price(19.04367, 17, 0, 0.3, 0.005, CallPut.CALL)
        self.assertEqual(result, max(0, 19.04367 - 17))
    def test_time_to_expiry_put(self):
        result = black_scholes_price(19.04367, 17, 0, 0.3, 0.005, CallPut.PUT)
        self.assertEqual(result, max(0, 17 - 19.04367))

    """ Testing strike< 0"""
    def test_negative_strike_raises(self):
        with self.assertRaises(InvalidPriceParametersException) as e:
            black_scholes_price(19.04367, -1, 0.460, 0.3, 0.005, CallPut.CALL)
        print("Exception Message:", e.exception)
    
    """ Testing forward<= 0"""
    def test_zero_forward_raises(self):
        with self.assertRaises(InvalidPriceParametersException) as e:
            black_scholes_price(0, 17, 0.460, 0.3, 0.005, CallPut.CALL)
        print("Exception Message:", e.exception)
    def test_negative_forward_raises(self):
        with self.assertRaises(InvalidPriceParametersException) as e:
            black_scholes_price(-1, 17, 0.460, 0.3, 0.005, CallPut.CALL)
        print("Exception Message:", e.exception)

    """ Testing volatility<= 0"""
    def test_zero_volatility_raises(self):
        with self.assertRaises(InvalidPriceParametersException)as e:
            black_scholes_price(19.04367, 17, 0.460, 0.0, 0.005, CallPut.CALL)
        print("Exception Message:", e.exception)
    def test_negative_volatility_raises(self):
        with self.assertRaises(InvalidPriceParametersException) as e:
            black_scholes_price(19.04367, 17, 0.460, -1, 0.005, CallPut.CALL)
        print("Exception Message:", e.exception)
    
if __name__ == "__main__":
    unittest.main(exit=False)