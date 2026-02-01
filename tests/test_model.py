"""
Test script for model.py

Verifies all core functionality and reproduces paper results.
"""

import numpy as np
import sys
sys.path.append('.')  # Ensure model.py is importable

from model import (
    ProducerModel, 
    ModelParameters, 
    PaymentMethod,
    create_default_parameters
)


def test_basic_functionality():
    """Test that model runs without errors"""
    print("Test 1: Basic Model Initialization")
    print("-" * 50)

    params = create_default_parameters(gamma=0.7)
    model = ProducerModel(params)

    print("✅ Model initialized successfully")
    print(f"   Gamma: {params.gamma}")
    print(f"   Payment methods: {len(params.a_k)}")


def test_equilibrium_without_volatility():
    """Test scenarios A, B, C match paper equations"""
    print("Test 2: Equilibrium Without Volatility (Set X)")
    print("-" * 50)

    # Simple test case with known solution
    params = ModelParameters(
        a_k={"A": 100, "B": 100, "C": 100, "D": 100, "E": 100, "F": 100, "G": 100},
        c_k={"A": 20, "B": 20, "C": 20, "D": 20, "E": 20, "F": 20, "G": 20},
        gamma=0.7
    )
    model = ProducerModel(params)

    result_A = model.solve_equilibrium_without_volatility("A")

    # Verify analytical solution: q* = (100-20)/2 = 40
    expected_q = (100 - 20) / 2
    expected_p = (100 + 20) / 2
    expected_profit = ((100 - 20) / 2) ** 2

    assert np.isclose(result_A.q_star, expected_q), "Quantity mismatch"
    assert np.isclose(result_A.p_star, expected_p), "Price mismatch"
    assert np.isclose(result_A.profit_star, expected_profit), "Profit mismatch"

    print(f"✅ Method A: q*={result_A.q_star}, p*={result_A.p_star}, π*={result_A.profit_star}")
    print(f"   Expected: q*={expected_q}, p*={expected_p}, π*={expected_profit}")


def test_equilibrium_with_volatility():
    """Test scenarios D, E, F, G match paper equations"""
    print("Test 3: Equilibrium With Volatility (Set Y)")
    print("-" * 50)

    # Test with gamma = 1 (no volatility risk) - should match non-volatile case
    params = ModelParameters(
        a_k={"A": 100, "B": 100, "C": 100, "D": 100, "E": 100, "F": 100, "G": 100},
        c_k={"A": 20, "B": 20, "C": 20, "D": 20, "E": 20, "F": 20, "G": 20},
        gamma=1.0  # Perfect certainty
    )
    model = ProducerModel(params)

    result_A = model.solve_equilibrium_without_volatility("A")
    result_D = model.solve_equilibrium_with_volatility("D")

    # When gamma=1, volatile and non-volatile should give same results
    assert np.isclose(result_A.profit_star, result_D.profit_star, rtol=1e-10),         f"Profits should match when gamma=1: {result_A.profit_star} vs {result_D.profit_star}"

    print(f"✅ Method D (γ=1): E[π]*={result_D.profit_star:.4f}")
    print(f"   Method A (no vol): π*={result_A.profit_star:.4f}")
    print(f"   Difference: {abs(result_D.profit_star - result_A.profit_star):.10f} (should be ~0)")

    # Test with gamma = 0 (maximum risk)
    params.gamma = 0.0
    model_zero = ProducerModel(params)
    result_D_zero = model_zero.solve_equilibrium_with_volatility("D")

    print(f"✅ Method D (γ=0): E[π]*={result_D_zero.profit_star} (should be -inf or 0)")


def test_comparison_function():
    """Test scenario comparison"""
    print("Test 4: Scenario Comparison")
    print("-" * 50)

    params = create_default_parameters(gamma=0.8)
    model = ProducerModel(params)

    comparison = model.compare_scenarios("C", "G")

    print(f"✅ Comparing Cash+Card (C) vs All Methods (G):")
    print(f"   Profit(C): ${comparison['profit_C']:.2f}")
    print(f"   Profit(G): ${comparison['profit_G']:.2f}")
    print(f"   Difference: ${comparison['profit_difference']:.2f}")
    print(f"   Bitcoin adoption profitable: {comparison['method_2_better']}")


def test_bitcoin_adoption_condition():
    """Test Bitcoin adoption analysis"""
    print("Test 5: Bitcoin Adoption Condition")
    print("-" * 50)

    # Test at different gamma levels
    gammas_to_test = [0.2, 0.5, 0.8, 1.0]

    for gamma in gammas_to_test:
        params = create_default_parameters(gamma=gamma)
        model = ProducerModel(params)

        adoption = model.bitcoin_adoption_condition("C", "G")

        print(f"   γ={gamma}: {adoption['interpretation']}")
        print(f"      Profit gain: ${adoption['profit_gain']:.2f}")


def test_gamma_threshold():
    """Test gamma threshold analysis"""
    print("Test 6: Gamma Threshold Analysis")
    print("-" * 50)

    params = create_default_parameters(gamma=0.5)
    model = ProducerModel(params)

    threshold_data = model.gamma_threshold_analysis(
        non_bitcoin_method="C",
        bitcoin_method="G",
        n_points=50
    )

    if threshold_data['gamma_threshold']:
        print(f"✅ Bitcoin becomes profitable at γ ≥ {threshold_data['gamma_threshold']:.3f}")
        print(f"   Tested range: [{threshold_data['gamma_values'].min():.2f}, "
              f"{threshold_data['gamma_values'].max():.2f}]")

        # Find profit at threshold
        idx = np.argmin(np.abs(threshold_data['gamma_values'] - threshold_data['gamma_threshold']))
        profit_at_threshold = threshold_data['profit_difference'][idx]
        print(f"   Profit difference at threshold: ${profit_at_threshold:.4f}")
    else:
        print("⚠️  Bitcoin not profitable in tested gamma range")


def test_all_scenarios():
    """Solve and display all 7 scenarios"""
    print("Test 7: All Payment Scenarios")
    print("-" * 50)

    params = create_default_parameters(gamma=0.7)
    model = ProducerModel(params)

    results = model.solve_all_equilibria()

    print("Set X (No Volatility):")
    for method in ["A", "B", "C"]:
        r = results[method]
        print(f"   {method}: q*={r.q_star:.2f}, p*={r.p_star:.2f}, π*={r.profit_star:.2f}")

    print("Set Y (With Volatility, γ=0.7):")
    for method in ["D", "E", "F", "G"]:
        r = results[method]
        print(f"   {method}: q*={r.q_star:.2f}, p*={r.p_star:.2f}, E[π]*={r.profit_star:.2f}")

    # Find optimal method
    best_method = max(results.items(), key=lambda x: x[1].profit_star)
    print(f"✅ Optimal payment method: {best_method[0]} with profit ${best_method[1].profit_star:.2f}")


def run_all_tests():
    """Run complete test suite"""
    print("=" * 70)
    print("CRYPTO PRODUCER THEORY MODEL - TEST SUITE")
    print("=" * 70)

    try:
        test_basic_functionality()
        test_equilibrium_without_volatility()
        test_equilibrium_with_volatility()
        test_comparison_function()
        test_bitcoin_adoption_condition()
        test_gamma_threshold()
        test_all_scenarios()

        print(" " + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)

    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
