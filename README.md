# Crypto Producer Theory: Bitcoin Adoption Model

Economic model of cryptocurrency adoption as a payment-choice problem, with volatility.

## Overview

This repository implements a structural model where producers choose optimal payment mix (Bitcoin, cards, cash) considering:
- Revenue volatility from BTC price fluctuations
- Transaction fee differences across payment methods

Based on research paper from UC Berkeley INFO 134/234.

## Model

Producers maximize profit: p = revenue - costs - volatility_penalty  
Choice variables: (b, c, p) ? [0,1]³ (Bitcoin, card, cash acceptance)

Key mechanism: Adoption complementarity creates multiple equilibria.

## Repository Contents

- src/model.py: Core equations and equilibrium solver
- notebooks/: Simulations and comparative statics
- docs/model_derivation.md: Mathematical derivations
- papers/: Original research paper

## Status

WIP: Implementing computational version of theoretical model


