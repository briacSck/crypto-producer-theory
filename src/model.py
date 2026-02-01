"""
Crypto Producer Theory: Payment Method Choice Model

This module implements the producer profit maximization model from:
Noel, Thomas & Sockalingum, Briac (2024). "Crypto Producer Theory"
UC Berkeley INFO 134/234.

The model analyzes firm decisions to accept Bitcoin, cash, and/or card payments
considering transaction costs, network effects, and Bitcoin volatility.
"""

import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class PaymentMethod(Enum):
    """Payment method types"""
    CASH = "A"
    CARD = "B"
    CASH_AND_CARD = "C"
    BITCOIN = "D"
    BITCOIN_AND_CASH = "E"
    BITCOIN_AND_CARD = "F"
    ALL_THREE = "G"


@dataclass
class ModelParameters:
    """
    Parameters for the producer model.

    Attributes:
        a_k: Willingness to pay for payment method k (dict by method)
        c_k: Marginal cost for payment method k (dict by method)
        gamma: Probability that Bitcoin volatility is favorable (0 ≤ γ ≤ 1)
        N_k: Number of consumers using payment method k (dict)
        M_k: Number of firms accepting payment method k (dict)
    """
    a_k: Dict[str, float]
    c_k: Dict[str, float]
    gamma: float
    N_k: Optional[Dict[str, float]] = None
    M_k: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Validate parameters"""
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")

        # Check all payment methods have parameters
        methods = [m.value for m in PaymentMethod]
        for method in methods:
            if method not in self.a_k:
                raise ValueError(f"Missing willingness to pay for method {method}")
            if method not in self.c_k:
                raise ValueError(f"Missing marginal cost for method {method}")


@dataclass
class EquilibriumResult:
    """
    Equilibrium solution for a payment method scenario.

    Attributes:
        method: Payment method identifier (A-G)
        q_star: Equilibrium quantity
        p_star: Equilibrium price
        profit_star: Equilibrium profit (or expected profit)
        is_volatile: Whether scenario includes Bitcoin (uses expected profit)
    """
    method: str
    q_star: float
    p_star: float
    profit_star: float
    is_volatile: bool

    def __repr__(self):
        profit_type = "E[π]" if self.is_volatile else "π"
        return (f"Equilibrium({self.method}): q*={self.q_star:.4f}, "
                f"p*={self.p_star:.4f}, {profit_type}*={self.profit_star:.4f}")


class ProducerModel:
    """
    Producer profit maximization model with multiple payment methods.

    Implements equations (1) and (2) from the paper for scenarios without
    and with Bitcoin volatility, respectively.
    """

    # Set X: Methods without Bitcoin (no volatility)
    METHODS_WITHOUT_VOLATILITY = {"A", "B", "C"}

    # Set Y: Methods with Bitcoin (volatility considerations)
    METHODS_WITH_VOLATILITY = {"D", "E", "F", "G"}

    def __init__(self, params: ModelParameters):
        """
        Initialize producer model.

        Args:
            params: ModelParameters object with costs, WTP, and volatility
        """
        self.params = params

    def demand(self, p: float, a_k: float) -> float:
        """
        Inverse demand function: q = a_k - p

        Args:
            p: Price
            a_k: Willingness to pay parameter

        Returns:
            Quantity demanded
        """
        return max(0, a_k - p)

    def inverse_demand(self, q: float, a_k: float) -> float:
        """
        Demand function: p = a_k - q

        Args:
            q: Quantity
            a_k: Willingness to pay parameter

        Returns:
            Price
        """
        return a_k - q

    def profit_without_volatility(self, q: float, a_k: float, c_k: float) -> float:
        """
        Profit function for scenarios WITHOUT Bitcoin (Equation 1).

        π^k_j = (a^k - q^k)q^k - c^k q^k

        Args:
            q: Quantity
            a_k: Willingness to pay
            c_k: Marginal cost

        Returns:
            Profit
        """
        p = self.inverse_demand(q, a_k)
        return p * q - c_k * q

    def expected_profit_with_volatility(self, q: float, a_k: float, 
                                       c_k: float, gamma: float) -> float:
        """
        Expected profit function for scenarios WITH Bitcoin (Equation 2).

        E(π^k_j) = γ[(a^k - q^k)q^k - c^k q^k] - (1 - γ)c^k q^k

        With probability γ: normal profit
        With probability (1-γ): lose revenue, only pay costs

        Args:
            q: Quantity
            a_k: Willingness to pay
            c_k: Marginal cost
            gamma: Probability of favorable volatility

        Returns:
            Expected profit
        """
        p = self.inverse_demand(q, a_k)
        normal_profit = p * q - c_k * q
        crash_loss = c_k * q

        return gamma * normal_profit - (1 - gamma) * crash_loss

    def solve_equilibrium_without_volatility(self, method: str) -> EquilibriumResult:
        """
        Solve equilibrium for scenarios A, B, C (no Bitcoin).

        FOC: ∂π^k/∂q^k = 0 ⟺ a^k - 2q^k - c^k = 0

        Solution:
            q* = (a^k - c^k) / 2
            p* = (a^k + c^k) / 2
            π* = ((a^k - c^k) / 2)^2

        Args:
            method: Payment method identifier (A, B, or C)

        Returns:
            EquilibriumResult object
        """
        if method not in self.METHODS_WITHOUT_VOLATILITY:
            raise ValueError(f"Method {method} should use volatility solver")

        a_k = self.params.a_k[method]
        c_k = self.params.c_k[method]

        # Analytical solution from FOC
        q_star = (a_k - c_k) / 2
        p_star = (a_k + c_k) / 2
        profit_star = ((a_k - c_k) / 2) ** 2

        return EquilibriumResult(
            method=method,
            q_star=q_star,
            p_star=p_star,
            profit_star=profit_star,
            is_volatile=False
        )

    def solve_equilibrium_with_volatility(self, method: str) -> EquilibriumResult:
        """
        Solve equilibrium for scenarios D, E, F, G (with Bitcoin).

        FOC: ∂E(π^k)/∂q^k = 0 ⟺ γ[a^k - 2q^k - c^k] - (1-γ)c^k = 0

        Solution:
            q* = (γa^k - c^k) / (2γ)
            p* = (γa^k + c^k) / (2γ)
            E[π*] = ((γa^k - c^k) / 2)^2 - (1-γ)c^k[(γa^k - c^k) / (2γ)]

        Args:
            method: Payment method identifier (D, E, F, or G)

        Returns:
            EquilibriumResult object
        """
        if method not in self.METHODS_WITH_VOLATILITY:
            raise ValueError(f"Method {method} should use non-volatility solver")

        a_k = self.params.a_k[method]
        c_k = self.params.c_k[method]
        gamma = self.params.gamma

        # Handle edge case: gamma = 0
        if gamma == 0:
            # When γ=0, expected profit → -∞ for any q>0
            # Optimal is q=0 (don't produce)
            return EquilibriumResult(
                method=method,
                q_star=0.0,
                p_star=a_k,
                profit_star=-np.inf,
                is_volatile=True
            )

        # Analytical solution from FOC
        numerator = gamma * a_k - c_k

        # Check if solution is feasible (positive quantity)
        if numerator <= 0:
            return EquilibriumResult(
                method=method,
                q_star=0.0,
                p_star=a_k,
                profit_star=0.0,
                is_volatile=True
            )

        q_star = numerator / (2 * gamma)
        p_star = (gamma * a_k + c_k) / (2 * gamma)

        # Calculate expected profit
        profit_component_1 = (numerator / 2) ** 2
        profit_component_2 = (1 - gamma) * c_k * (numerator / (2 * gamma))
        expected_profit = profit_component_1 - profit_component_2

        return EquilibriumResult(
            method=method,
            q_star=q_star,
            p_star=p_star,
            profit_star=expected_profit,
            is_volatile=True
        )

    def solve_all_equilibria(self) -> Dict[str, EquilibriumResult]:
        """
        Solve equilibria for all 7 payment method scenarios.

        Returns:
            Dictionary mapping method (A-G) to EquilibriumResult
        """
        results = {}

        # Solve methods without volatility (Set X: A, B, C)
        for method in self.METHODS_WITHOUT_VOLATILITY:
            results[method] = self.solve_equilibrium_without_volatility(method)

        # Solve methods with volatility (Set Y: D, E, F, G)
        for method in self.METHODS_WITH_VOLATILITY:
            results[method] = self.solve_equilibrium_with_volatility(method)

        return results

    def compare_scenarios(self, method_1: str, method_2: str) -> Dict[str, float]:
        """
        Compare profits between two payment method scenarios.

        Args:
            method_1: First payment method
            method_2: Second payment method

        Returns:
            Dictionary with profit difference and relative advantage
        """
        result_1 = (self.solve_equilibrium_without_volatility(method_1) 
                   if method_1 in self.METHODS_WITHOUT_VOLATILITY
                   else self.solve_equilibrium_with_volatility(method_1))

        result_2 = (self.solve_equilibrium_without_volatility(method_2)
                   if method_2 in self.METHODS_WITHOUT_VOLATILITY
                   else self.solve_equilibrium_with_volatility(method_2))

        profit_diff = result_2.profit_star - result_1.profit_star

        return {
            f"profit_{method_1}": result_1.profit_star,
            f"profit_{method_2}": result_2.profit_star,
            "profit_difference": profit_diff,
            "method_2_better": profit_diff > 0
        }

    def bitcoin_adoption_condition(self, non_bitcoin_method: str = "C",
                                   bitcoin_method: str = "G") -> Dict[str, any]:
        """
        Analyze conditions under which Bitcoin adoption is profitable.

        Solves inequality from Section 2.3:
        π^k* ≤ E[π^k]*

        Args:
            non_bitcoin_method: Baseline method without Bitcoin (default: C)
            bitcoin_method: Method with Bitcoin (default: G)

        Returns:
            Dictionary with adoption analysis
        """
        comparison = self.compare_scenarios(non_bitcoin_method, bitcoin_method)

        # Get parameters for analysis
        a_non = self.params.a_k[non_bitcoin_method]
        c_non = self.params.c_k[non_bitcoin_method]
        a_btc = self.params.a_k[bitcoin_method]
        c_btc = self.params.c_k[bitcoin_method]
        gamma = self.params.gamma

        return {
            "should_adopt_bitcoin": comparison["method_2_better"],
            "profit_gain": comparison["profit_difference"],
            "gamma": gamma,
            "interpretation": self._interpret_adoption(gamma, comparison["method_2_better"])
        }

    def _interpret_adoption(self, gamma: float, is_profitable: bool) -> str:
        """Generate human-readable interpretation of adoption decision"""
        if gamma < 0.3:
            vol_desc = "very high volatility"
        elif gamma < 0.7:
            vol_desc = "moderate volatility"
        else:
            vol_desc = "low volatility"

        decision = "adopt" if is_profitable else "reject"

        return (f"With {vol_desc} (γ={gamma:.2f}), producer should "
                f"{decision} Bitcoin as payment method")

    def gamma_threshold_analysis(self, non_bitcoin_method: str = "C",
                                 bitcoin_method: str = "G",
                                 gamma_range: Tuple[float, float] = (0.01, 0.99),
                                 n_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Analyze how profitability changes across gamma values.

        Useful for finding critical thresholds where Bitcoin adoption
        becomes profitable.

        Args:
            non_bitcoin_method: Baseline without Bitcoin
            bitcoin_method: Method with Bitcoin
            gamma_range: Range of gamma values to test
            n_points: Number of points to evaluate

        Returns:
            Dictionary with gamma values, profits, and differences
        """
        gammas = np.linspace(gamma_range[0], gamma_range[1], n_points)
        profits_non_btc = []
        profits_btc = []

        original_gamma = self.params.gamma

        for g in gammas:
            # Temporarily update gamma
            self.params.gamma = g

            # Calculate profits
            result_non = self.solve_equilibrium_without_volatility(non_bitcoin_method)
            result_btc = self.solve_equilibrium_with_volatility(bitcoin_method)

            profits_non_btc.append(result_non.profit_star)
            profits_btc.append(result_btc.profit_star)

        # Restore original gamma
        self.params.gamma = original_gamma

        profit_diffs = np.array(profits_btc) - np.array(profits_non_btc)

        # Find threshold where Bitcoin becomes profitable
        threshold_idx = np.where(profit_diffs > 0)[0]
        gamma_threshold = gammas[threshold_idx[0]] if len(threshold_idx) > 0 else None

        return {
            "gamma_values": gammas,
            "profit_without_bitcoin": np.array(profits_non_btc),
            "profit_with_bitcoin": np.array(profits_btc),
            "profit_difference": profit_diffs,
            "bitcoin_profitable": profit_diffs > 0,
            "gamma_threshold": gamma_threshold
        }


def create_default_parameters(gamma: float = 0.7) -> ModelParameters:
    """
    Create default parameter set for quick testing.

    Assumes:
    - Higher willingness to pay for more convenient payment methods
    - Bitcoin has lower marginal costs (no processor fees) but volatility risk
    - Card networks have highest processing fees

    Args:
        gamma: Probability of favorable Bitcoin volatility

    Returns:
        ModelParameters object
    """
    return ModelParameters(
        a_k={
            "A": 100.0,  # Cash only
            "B": 105.0,  # Card only (higher WTP for convenience)
            "C": 110.0,  # Cash + Card
            "D": 95.0,   # Bitcoin only (lower due to adoption friction)
            "E": 108.0,  # Bitcoin + Cash
            "F": 112.0,  # Bitcoin + Card
            "G": 115.0   # All three (maximum flexibility)
        },
        c_k={
            "A": 5.0,    # Cash handling costs
            "B": 8.0,    # Card processing fees (2-3%)
            "C": 6.5,    # Average of both
            "D": 2.0,    # Bitcoin: low transaction fees
            "E": 3.5,    # Average with cash
            "F": 5.0,    # Average with card
            "G": 5.5     # Average of all three
        },
        gamma=gamma
    )


if __name__ == "__main__":
    # Example usage
    print("Crypto Producer Theory Model")
    print("=" * 50)

    # Create model with default parameters
    params = create_default_parameters(gamma=0.8)
    model = ProducerModel(params)

    # Solve all equilibria
    print("Equilibrium Solutions:")
    print("-" * 50)
    results = model.solve_all_equilibria()
    for method, result in sorted(results.items()):
        print(result)

    # Compare Bitcoin adoption
    print(" " + "=" * 50)
    print("Bitcoin Adoption Analysis:")
    print("-" * 50)
    adoption_analysis = model.bitcoin_adoption_condition(
        non_bitcoin_method="C",
        bitcoin_method="G"
    )
    print(f"Decision: {adoption_analysis['interpretation']}")
    print(f"Profit gain from Bitcoin: ${adoption_analysis['profit_gain']:.2f}")

    # Gamma threshold analysis
    print(" " + "=" * 50)
    print("Finding gamma threshold for Bitcoin adoption...")
    print("-" * 50)
    threshold_data = model.gamma_threshold_analysis()
    if threshold_data['gamma_threshold']:
        print(f"Bitcoin becomes profitable at γ ≥ {threshold_data['gamma_threshold']:.3f}")
    else:
        print("Bitcoin not profitable in tested range")
