# Crypto Producer Theory: Payment Method Choice and Bitcoin Adoption Under Volatility Risk

[![Tests](https://github.com/briacSck/crypto-producer-theory/actions/workflows/tests.yml/badge.svg)](https://github.com/briacSck/crypto-producer-theory/actions/workflows/tests.yml)

This project develops a partial-equilibrium model of a firm's payment-method decision when Bitcoin is one of the available options. The central tension is straightforward: Bitcoin carries lower transaction costs than card networks, but it introduces crash risk — with probability `1 − γ` the firm produces, incurs costs, and receives nothing. We derive closed-form equilibria for seven payment scenarios and characterize the critical stability threshold `γ*` above which Bitcoin adoption is profit-maximizing.

## Motivation

El Salvador's 2021 Bitcoin Legal Tender Law forced firms to confront a decision that standard producer theory had not formally addressed: when a payment method can crash to zero, how should a profit-maximizing firm weigh its cost advantage against its volatility risk? This model provides a tractable answer.

## The Model

A single firm faces a linear inverse demand function and chooses output to maximize profit. Payment methods differ along two dimensions: demand intercept `a^k` (consumers self-select the methods they prefer) and marginal transaction cost `c^k`. Bitcoin additionally introduces *volatility risk*: with probability `γ` the firm receives full revenue; with probability `1 − γ` Bitcoin crashes to zero and the firm bears production costs with no corresponding revenue.

We consider seven payment scenarios organized into two sets:

| Set | Scenario | Payment methods accepted |
|-----|----------|--------------------------|
| X | A | Cash only |
| X | B | Card only |
| X | C | Cash + Card |
| Y | D | Bitcoin only |
| Y | E | Bitcoin + Cash |
| Y | F | Bitcoin + Card |
| Y | G | All three |

**Set X** (A–C) is deterministic: standard first-order conditions yield `π* = (a^k − c^k)² / 4`. **Set Y** (D–G) requires maximizing expected profit. Solving the FOC yields a closed form that nests the deterministic case cleanly:

```
E[π*] = (γa^k − c^k)² / (4γ)
```

At `γ = 1` this reduces exactly to the Set X formula. As `γ` falls, the effective demand intercept shrinks, capturing the volatility discount. The `1/γ` denominator means that even a modest crash probability can dominate a substantial cost advantage.

## Adoption Threshold

The adoption decision reduces to comparing `E[π^{G*}]` (Scenario G, all three methods) against `π^{C*}` (Scenario C, Cash + Card — the best non-Bitcoin alternative). There exists a unique threshold `γ*` at which the two are equal; Bitcoin adoption is profit-maximizing if and only if `γ > γ*`.

Under baseline parameters, **`γ* ≈ 0.9031`**: the firm requires Bitcoin to be crash-free at least 90% of the time for adoption to be rational. This threshold is consistent with the empirically low merchant adoption rates observed globally, and provides a formal rationale for why stablecoins and CBDCs — which by design push `γ` toward 1 — could substantially lower the barrier to crypto adoption.

## Getting Started

Start with the notebook for an interactive walkthrough of the equilibria and comparative statics:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_equilibrium_analysis.ipynb
```

To verify the equilibria numerically across the full range of γ values:

```bash
python -m pytest tests/test_model.py -v
```

The full mathematical derivations (first-order conditions, second-order checks, the adoption inequality) are in `docs/model_derivation.md`. The paper is in `paper/`.

## Repository Structure

```
src/model.py                             # Core model and equilibrium solver
tests/test_model.py                      # Unit tests, including intermediate-γ validation
notebooks/01_equilibrium_analysis.ipynb  # Interactive walkthrough
docs/model_derivation.md                 # Full derivations
tables/                                  # Computed results (CSV + LaTeX)
figures/                                 # Adoption threshold and phase diagrams
paper/                                   # Paper (TeX source + PDF)
```

## License

Code: [MIT](LICENSE). Paper and figures © 2024–2026 Noel & Sockalingum, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Status

- [x] Theoretical model and closed-form equilibria
- [x] Computational implementation and verified results
- [ ] ML work on empirical adoption data


