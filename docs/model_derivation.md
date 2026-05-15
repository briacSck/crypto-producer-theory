# Model Derivation: Producer Payment Choice

## Introduction

This document presents the formal derivation of a producer profit maximization model in economies where Bitcoin is legal tender (e.g., El Salvador). The model compares profits across seven payment acceptance situations, incorporating cryptocurrency volatility.

Based on: Noel, T. & Sockalingum, B. (2024, revised 2026). *Crypto Producer Theory: Payment Method Choice and Bitcoin Adoption Under Volatility Risk.*

---

## 1. Setup

### 1.1 Agents and Payment Methods

- **N**: Number of consumers
- **M**: Number of firms
- **Payment methods**: Bitcoin (crypto), Cash, Card (open banking network)

### 1.2 Consumer Types (7 Situations)

Consumers may use different combinations of payment methods. We denote:

| Letter | Payment Method(s) | Set |
|--------|------------------|-----|
| A | Cash only | X (no Bitcoin) |
| B | Card only | X |
| C | Cash + Card | X |
| D | Bitcoin only | Y (includes Bitcoin) |
| E | Bitcoin + Cash | Y |
| F | Bitcoin + Card | Y |
| G | All three methods | Y |

**Sets**:
- **X** = {A, B, C}: Situations without Bitcoin
- **Y** = {D, E, F, G}: Situations with Bitcoin

### 1.3 Network Effects

Consumer utility \( U^k_i \) for using payment method \( k \) is an **increasing function** of:
- \( N^k \): Number of consumers using method \( k \)
- \( M^k \): Number of firms accepting method \( k \)

**Implication**: Higher adoption → higher utility (network externality)

**Equilibrium condition** (not solved in this paper):
\\\
U^A_i = U^B_i = ... = U^G_i
\\\

Consumers choose the payment combination yielding maximum utility.

---

## 2. Producer Profit Functions

### 2.1 General Profit Structure

For a firm \( j \) accepting payment situation \( k \):

\\\
p^k_j = P^k(q^k) · q^k - c^k · q^k
\\\

Where:
- \( P^k \): Price charged (depends on payment method accepted)
- \( q^k \): Quantity sold
- \( c^k \): Marginal cost (payment-method specific; **no fixed costs assumed**)

**Assumption**: Price depends on payment method because different consumer types have different willingness to pay.

### 2.2 Demand Function

Linear inverse demand (situation-specific):

\\\
p^k = a^k - q^k
\\\

Where \( a^k \) = willingness to pay for consumers using method \( k \).

**Key**: Different payment users have different \( a^k \) values.

---

## 3. Volatility Modeling (Bitcoin Situations Only)

### 3.1 Volatility as Crash Risk

Bitcoin exhibits **price volatility**. Model this as:

- **γ**: Probability that Bitcoin price remains stable (no crash)
- **1 - γ**: Probability of Bitcoin crash

**Assumption**: If Bitcoin crashes, firm's profit = **-c^k · q^k** (pure loss of costs).

### 3.2 Expected Profit (Situations Y)

For situations including Bitcoin (\( k \in Y \)):

\\\
E(p^k_j) = γ · [P^k(q^k) · q^k - c^k · q^k] - (1 - γ) · c^k · q^k
\\\

Simplifying:

\\\
E(p^k_j) = γ · [(a^k - q^k) · q^k - c^k · q^k] - (1 - γ) · c^k · q^k
\\\

---

## 4. Solving the Model

### 4.1 Situations Without Bitcoin (Set X)

**Profit function**:

\\\
p^k_j = (a^k - q^k) · q^k - c^k · q^k
\\\

**First-order condition**:

\\\
∂p^k_j / ∂q^k = 0
⟹ a^k - 2q^k - c^k = 0
\\\

**Equilibrium**:

\\\
(q^k)* = (a^k - c^k) / 2
(p^k)* = (a^k + c^k) / 2
π^k* = [(a^k - c^k) / 2]²
\\\

**Standard result**: Profit depends quadratically on (willingness to pay - cost).

---

### 4.2 Situations With Bitcoin (Set Y)

**Expected profit function**:

\\\
E(p^k_j) = γ · [(a^k - q^k) · q^k - c^k · q^k] - (1 - γ) · c^k · q^k
\\\

**First-order condition**:

\\\
∂E(p^k_j) / ∂q^k = 0
⟹ γ · (a^k - 2q^k - c^k) - (1 - γ) · c^k = 0
\\\

**Equilibrium**:

\\\
(q^k)* = (γ · a^k - c^k) / (2γ)
(p^k)* = (γ · a^k + c^k) / (2γ)
E(π^k_j)* = (γ · a^k - c^k)² / (4γ)
\\\

**Key parameter**: Volatility probability **γ** directly affects optimal quantity and expected profit.

---

## 5. Comparing Bitcoin vs. Non-Bitcoin Situations

### 5.1 General Comparison

Compare profits:

\\\
π^{k*} ≤ E[π^k_j]*
\\\

For which values of **γ** is accepting Bitcoin more profitable?

---

### 5.2 Extreme Cases

#### **Case 1: γ = 1** (No volatility)

When γ = 1 (Bitcoin never crashes):

\\\
E(p^k_j)* = [(a^k - c^k) / 2]² - 0 = (p^k)*
\\\

**Result**: Profits are **equal**. No difference between accepting Bitcoin or not.

---

#### **Case 2: γ → 0** (Crash certainty)

As γ → 0, the optimal quantity:

\\\
q* = (γ · a^k - c^k) / (2γ) → -c^k / (2γ) → -∞
\\\

The loss term dominates:

\\\
(1 - γ) · c^k · q* → 1 · c^k · (-c^k / (2γ)) = -(c^k)² / (2γ) → -∞
\\\

**Result**: The unconstrained FOC gives q* = (γa^k - c^k)/(2γ) → -c^k/(2γ) → -∞
as γ → 0, implying E[π*] → -∞ in the unconstrained problem.

The code applies the non-negativity constraint q ≥ 0:

- **γ = 0 exactly**: the code returns E[π*] = -∞ (crash certainty, no production
  is feasible at positive output).
- **0 < γ ≤ c^k/a^k**: the numerator γa^k - c^k ≤ 0, so q* = 0 and E[π*] = 0
  (the constrained optimum; the firm produces nothing and earns zero).
- **γ > c^k/a^k**: the interior solution q* > 0 applies.

The -∞ result therefore applies only at γ = 0 exactly; for small positive γ the
code correctly returns 0, not -∞.

---

### 5.3 Interpretation

**As crash certainty increases (γ → 0), Bitcoin adoption becomes strictly dominated.**

At γ = 0, the firm would need to produce a negative quantity (q* → -∞) to satisfy the FOC, which is economically infeasible. Expected profit collapses to -∞. This is the correct economic result: a firm facing certain Bitcoin crashes earns unboundedly negative expected profit and will not participate.

**At γ = 1** (no crash risk), the stochastic term vanishes and the Set Y equilibrium reduces exactly to the deterministic Set X form. The code verifies this: at γ = 1, E[π^G*] = 2997.56 > π^C* = 2678.06, confirming that without volatility risk, Bitcoin's lower transaction cost advantage is fully realized.

**For general γ ∈ (0, 1)**, the critical threshold γ* ≈ 0.9031 (computed analytically from the quadratic derived in Section 2.3 of the paper) separates the regime where Bitcoin adoption is profitable (γ > γ*) from where it is not (γ < γ*).

**Limitations**:
- Consumer choice (N^k) is not fully endogenized; N^b is treated as exogenous
- The model captures only downside volatility (crashes); a symmetric treatment incorporating appreciation would likely lower γ*

---

## 6. Key Mechanisms (Information Economics Parallel)

This model exhibits features common to **information technologies**:

1. **Network effects**: Payment method value ↑ with adoption (like platforms)
2. **Low marginal costs**: Crypto transactions have lower fees than card networks
3. **Low switching costs**: Easy to switch between blockchains (low lock-in)
4. **Privacy tradeoffs**: Blockchain transparency vs. consumer privacy (privacy paradox)

---

## 7. Future Extensions (From Original Paper)

### Not Yet Solved:
1. **General inequality** for all γ ∈ [0,1] (only extremes solved)
2. **Consumer equilibrium**: Derive N^k endogenously from utility maximization
3. **Better volatility modeling**: Use financial economics (variance, option pricing)
4. **Detailed cost structure**: Distinguish fixed vs. variable costs by payment type

### Possible Extensions:
1. **Welfare analysis**: Social welfare across situations
2. **Price discrimination**: Firms charge different prices by payment method
3. **Econometric calibration**: Test model predictions on El Salvador data
4. **Dynamic model**: Repeated game with learning about volatility

---

## References

Noel, T. & Sockalingum, B. (2024, revised 2026). *Crypto Producer Theory: Payment Method Choice and Bitcoin Adoption Under Volatility Risk.*

El Salvador Bitcoin adoption (2021), BIS cryptocurrency research (2021), China e-CNY research (2023).
