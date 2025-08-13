import numpy as np

class RiskManager:
    def __init__(self, max_drawdown=0.2, max_volatility=0.05):
        self.max_drawdown = max_drawdown  # e.g., 20%
        self.max_volatility = max_volatility  # e.g., 5%
        self.portfolio_values = []
        self.violations = []

    def update(self, portfolio_value):
        self.portfolio_values.append(portfolio_value)
        self.check_risk()

    def check_risk(self):
        if len(self.portfolio_values) < 2:
            return
        values = np.array(self.portfolio_values)
        # Drawdown
        peak = np.max(values)
        trough = np.min(values)
        drawdown = (peak - trough) / peak if peak > 0 else 0
        # Volatility (std of returns)
        returns = np.diff(values) / values[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0
        # Check thresholds
        if drawdown > self.max_drawdown:
            self.violations.append(f"Drawdown exceeded: {drawdown:.2%}")
            print(f"[RISK] Drawdown exceeded: {drawdown:.2%}")
        if volatility > self.max_volatility:
            self.violations.append(f"Volatility exceeded: {volatility:.2%}")
            print(f"[RISK] Volatility exceeded: {volatility:.2%}")

    def check_manipulation(self, actions):
        # Placeholder: flag excessive order frequency or spoofing
        if len(actions) > 10 and np.std(actions[-10:]) == 0:
            self.violations.append("Potential manipulative trading detected (repetitive actions)")
            print("[ETHICS] Potential manipulative trading detected (repetitive actions)")

    def get_violations(self):
        return self.violations 