# Model Derivation: Producer Payment Choice

## Introduction

This document presents the formal derivation of a producer profit maximization model in economies where Bitcoin is legal tender (e.g., El Salvador). The model compares profits across seven payment acceptance situations, incorporating cryptocurrency volatility.

Based on: Noel, T. & Sockalingum, B. (2024). *Crypto Producer Theory*. UC Berkeley INFO 134/234.

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

**Implication**: Higher adoption ? higher utility (network externality)

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

- **?**: Probability that Bitcoin price remains stable (no crash)
- **1 - ?**: Probability of Bitcoin crash

**Assumption**: If Bitcoin crashes, firm's profit = **-c^k · q^k** (pure loss of costs).

### 3.2 Expected Profit (Situations Y)

For situations including Bitcoin (\( k \in Y \)):

\\\
E(p^k_j) = ? · [P^k(q^k) · q^k - c^k · q^k] - (1 - ?) · c^k · q^k
\\\

Simplifying:

\\\
E(p^k_j) = ? · [(a^k - q^k) · q^k - c^k · q^k] - (1 - ?) · c^k · q^k
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
?p^k_j / ?q^k = 0 
? a^k - 2q^k - c^k = 0
\\\

**Equilibrium**:

\\\
(q^k)* = (a^k - c^k) / 2
(p^k)* = (a^k + c^k) / 2
(p^k)* = [(a^k - c^k) / 2]²
\\\

**Standard result**: Profit depends quadratically on (willingness to pay - cost).

---

### 4.2 Situations With Bitcoin (Set Y)

**Expected profit function**:

\\\
E(p^k_j) = ? · [(a^k - q^k) · q^k - c^k · q^k] - (1 - ?) · c^k · q^k
\\\

**First-order condition**:

\\\
?E(p^k_j) / ?q^k = 0 
? ? · (a^k - 2q^k - c^k) - (1 - ?) · c^k = 0
\\\

**Equilibrium**:

\\\
(q^k)* = (? · a^k - c^k) / (2?)
(p^k)* = (? · a^k + c^k) / (2?)
E(p^k_j)* = [(? · a^k - c^k) / 2]² - (1 - ?) · c^k · [(? · a^k - c^k) / (2?)]
\\\

**Key parameter**: Volatility probability **?** directly affects optimal quantity and expected profit.

---

## 5. Comparing Bitcoin vs. Non-Bitcoin Situations

### 5.1 General Comparison

Compare profits:

\\\
(p^k)* = E(p^k_j)*  ?
\\\

For which values of **?** is accepting Bitcoin more profitable?

---

### 5.2 Extreme Cases

#### **Case 1: ? = 1** (No volatility)

When ? = 1 (Bitcoin never crashes):

\\\
E(p^k_j)* = [(a^k - c^k) / 2]² - 0 = (p^k)*
\\\

**Result**: Profits are **equal**. No difference between accepting Bitcoin or not.

---

#### **Case 2: ? ? 0** (High volatility, frequent crashes)

For the volatility term:

\\\
(1 - ?) · c^k · [(? · a^k - c^k) / (2?)]
= (1 - ?) · c^k · [a^k/2 - c^k/(2?)]
? -(c^k)² / (2?)  as  ? ? 0
? -8
\\\

Plugging back:

\\\
(p^k)* = [(? · a^k - c^k) / 2]² + 8
\\\

**Result**: Expected profit with Bitcoin ? **+8**

---

### 5.3 Interpretation (Counterintuitive Finding)

**As volatility increases (? ? 0), Bitcoin situations become infinitely more profitable.**

**Why?** 

This is a **mathematical artifact** of the model's simplification, not economic reality. The model assumes:
- Crash ? firm absorbs full cost loss
- No upper bound on losses

**Economic interpretation**:
- The model captures that in high-volatility environments, **accepting Bitcoin can still be optimal** if:
  - Marginal costs are low
  - Willingness to pay (\( a^k \)) compensates for risk
  - Network effects (not fully solved here) increase demand

**Limitations** (acknowledged in original paper):
- Need better volatility modeling (financial economics methods)
- Need to bound losses realistically
- Consumer choice (N^k) not fully endogenized

---

## 6. Key Mechanisms (Information Economics Parallel)

This model exhibits features common to **information technologies**:

1. **Network effects**: Payment method value ? with adoption (like platforms)
2. **Low marginal costs**: Crypto transactions have lower fees than card networks
3. **Low switching costs**: Easy to switch between blockchains (low lock-in)
4. **Privacy tradeoffs**: Blockchain transparency vs. consumer privacy (privacy paradox)

---

## 7. Future Extensions (From Original Paper)

### Not Yet Solved:
1. **General inequality** for all ? ? [0,1] (only extremes solved)
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

Noel, T. & Sockalingum, B. (2024). *Crypto Producer Theory*. UC Berkeley INFO 134/234 Final Paper.

El Salvador Bitcoin adoption (2021), BIS cryptocurrency research (2021), China e-CNY research (2023).
