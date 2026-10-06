# DRO-CVaR-PPO — Distributionally Robust Dispatch
## Novelty Summary & Literature Grounding

---

## What DRO-CVaR-PPO Is

A Proximal Policy Optimization (PPO) agent whose training objective augments the standard
surrogate loss with a DRO-CVaR tail-risk penalty:

```
Policy Loss = -E[clipped surrogate] + lambda * DRO-CVaR_{0.90}(-R)

where DRO-CVaR_{1-alpha}(l) = CVaR_{1-alpha}(l) + sqrt(rho*) * sigma_tail / (1-alpha)

rho* = 0.0806  [calibrated from 4.03% coal-critical frequency]
lambda = 0.10   [tail-risk penalty weight]
```

State space: 14 features from India's real 18-state grid data (2017-2025).
Action: continuous dispatch fraction a in [0,1] per state-day.
Reward: economic gain - carbon penalty - under-dispatch grid stress.

---

## What Was Already Known (Prior Work)

### Reinforcement Learning for Power Systems

| Paper | Key Contribution | How DRO-CVaR-PPO Differs |
|---|---|---|
| **Schulman et al. (2017)** *Proximal Policy Optimization Algorithms*, arXiv | PPO algorithm: clipped surrogate objective. Standard in continuous control. | Our contribution is the DRO-CVaR augmentation of the PPO loss, not PPO itself. |
| **Mnih et al. (2015)** *Human-level control through deep reinforcement learning*, Nature | DQN for Atari games. Established deep RL for discrete control. | Discrete actions, no risk-awareness. |
| **Haarnoja et al. (2018)** *Soft Actor-Critic: Off-Policy Maximum Entropy DRL*, ICML | SAC with entropy regularisation for continuous control. | Entropy regularisation != risk regularisation. SAC's max-entropy does not specifically hedge tail scenarios from distributional shifts. |
| **Tan et al. (2022)** *Multi-Agent Reinforcement Learning for Industrial Control*, IEEE Trans. IE | MARL for industrial energy dispatch. Reward: profit maximisation. | No risk-awareness, no India-specific uncertainty model. |
| **Zhang et al. (2021)** *Deep Reinforcement Learning for Power System Operations*, Proc. IEEE | Survey of DRL for power systems (unit commitment, economic dispatch, voltage control). | Identifies risk-aware RL as open problem. RE-FUSED-V2 fills this gap for India. |

### Distributionally Robust Optimization

| Paper | Key Contribution | How RE-FUSED Uses It |
|---|---|---|
| **Duchi & Namkoong (2018)** *Learning Models with Uniform Performance*, Ann. Statistics | Closed-form DRO-CVaR for chi-squared ball: CVaR + sqrt(rho) * sigma_tail / (1-alpha). *Primary theoretical source.* | **Exact equation used in RE-FUSED loss.** |
| **Namkoong & Duchi (2017)** *Variance-based Regularization with Convex Objectives*, NeurIPS | Showed variance penalty = DRO with chi-squared uncertainty set. | Justifies using sqrt(rho) * sigma_tail form. |
| **Ben-Tal et al. (2009)** *Robust Optimization*, Princeton University Press | General framework for worst-case optimization over uncertainty sets. | General theory; Duchi & Namkoong give the specific closed form used. |
| **Wiesemann et al. (2014)** *Distributionally Robust Convex Optimization*, Operations Research | DRO with moment-based uncertainty sets. | Moment-based DRO requires estimating moments; chi-squared ball is simpler and calibratable from data. |
| **Esfahani & Kuhn (2018)** *Data-Driven Distributionally Robust Optimization*, Math. Programming | Wasserstein-ball DRO. Statistical convergence guarantees. | Wasserstein DRO has higher computational cost; chi-squared ball admits closed-form CVaR (used in RE-FUSED). |

### CVaR in Power Systems

| Paper | Key Contribution | RE-FUSED Extension |
|---|---|---|
| **Rockafellar & Uryasev (2000)** *Optimization of Conditional Value-at-Risk*, J. Risk | CVaR as coherent risk measure. LP formulation for CVaR minimisation. | Standard CVaR. RE-FUSED extends to DRO-CVaR specifically for India's coal-shock distributional jumps. |
| **Conejo et al. (2010)** *Decision Making Under Uncertainty in Electricity Markets*, Springer | CVaR-based stochastic programming for electricity market bidding. | Stochastic programming, not RL. No distributional shift from coal criticality. |

### Recent CVaR/DRO grid dispatch work and the closest global analogue (checked directly)

CVaR/DRO for grid dispatch is an active field as of 2025-2026 — the claims below were
checked against this recent work rather than assumed:

| Paper | Key Contribution | Relevance to RE-FUSED's claim |
|---|---|---|
| **(2026)** *CVaR-Guided Decision-Focused Learning and Risk-Triggered Re-Optimization for Two-Stage Robust Microgrid Operation*, arXiv:2604.16614 | CVaR-DFL with risk-triggered re-optimization for microgrid dispatch under extreme days. | Microgrid-scale, single-site. No forecast-conditioned gating, no India context. |
| **ScienceDirect (2022)** *Distributionally robust OPF in distribution network considering CVaR-averse voltage security* | CVaR-DRCC-OPF for voltage security in distribution networks, with GCN-DRL for real-time reactive dispatch. | Optimization-based (OPF), not policy-gradient RL. Confirms CVaR-DRO is established for grid risk; RE-FUSED's contribution is coupling it to PPO specifically. |
| **Nature Sci. Reports (2025)** *Credible capacity evaluation of virtual power plants considering wind and PV uncertainties* | Wasserstein-DRO + CVaR for VPP capacity planning, closed-loop with probabilistic forecasting. | Closest in spirit to RE-FUSED's forecast-uncertainty-to-dispatch coupling, but for VPP capacity planning, not real-time dispatch policy, and not India. |
| **Tang et al. (2026)** *Weather-Aware Reinforcement Learning for Power Grid Topology Control Under Extreme Events*, IET Smart Grid | SAC conditioned on an explicit weather encoding for topology control under extreme weather scenarios; DRO/CVaR/chance-constraints discussed as the broader framing this sits within. | Same conceptual family as RE-FUSED (context-conditioned RL for grid risk) but weather-conditioned, not carbon-conditioned, and not coupled to a forecasting gate. |
| **(2022)** *Confidence Estimation Transformer for Long-term Renewable Energy Forecasting in RL-based Power Grid Dispatching* (Conformer-RLpatching), arXiv:2204.04612 | Forecasting-confidence-conditioned RL dispatch, validated on China's real SG-126 grid simulator (State Grid Corp.), beating DDPG by 25.8% on a security score in a real competition benchmark. | **Closest global analogue to RE-FUSED's overall A-E pipeline** (forecast -> uncertainty -> dispatch RL, validated on a real grid). Not carbon-conditioned, not risk-tail-aware (uses confidence, not CVaR), not India. This should be cited as the precedent RE-FUSED extends, not omitted. |

**Revised claim (narrower, honest):** the individual pieces — DRO-CVaR theory, CVaR-RL,
and forecast-conditioned dispatch RL — are each independently established, including a
real-grid-validated forecast-to-dispatch RL system (Conformer-RLpatching, China). What
these do not combine is: (a) a carbon-signal-conditioned forecasting gate feeding (b) a
DRO-CVaR-augmented PPO objective, calibrated to (c) India's state-level coal-criticality
and monsoon-driven regime structure. That combination, not any single component, is
RE-FUSED's defensible novelty claim.

---

## What Is New in DRO-CVaR-PPO (India Application)

1. **India-Calibrated rho***: Prior DRO-RL work uses fixed or theoretically derived rho. RE-FUSED-V2 derives rho* = 0.0806 from empirical coal-critical frequency (4.03%), regime shift (monsoon RE gap), and price kurtosis — making it data-calibrated to India's specific tail risk sources.

2. **Coal-Critical Distributional Shift Framing**: Coal-critical events (state coal stock < 7 days) produce *discrete distributional jumps* — not captured by smooth Gaussian CVaR. The DRO framework explicitly hedges against these jumps via the chi-squared uncertainty ball. This framing is new to power systems DRO literature.

3. **End-to-End RL Integration**: forecast-conditioned dispatch RL has been validated on a
   real grid elsewhere (Conformer-RLpatching, China's SG-126 simulator — see table above),
   so "first ever" would be false. RE-FUSED-V2's distinguishing combination is DRO-CVaR
   (tail-risk, not confidence-based) integrated directly into the PPO training objective,
   conditioned on carbon signals via MCAG, and calibrated to India-specific coal-criticality
   and monsoon regime data — a combination not found in the papers reviewed for this work.

4. **sigma_reward as Stability Metric**: DRO dispatch is evaluated primarily on sigma_reward reduction (dispatch variance), not mean reward. This is motivated by India's DISCOM context: consistent procurement > high-variance expected gain. This evaluation framing differs from standard RL benchmarks.

---

## Key References for DRO-CVaR-PPO Section

**RL Foundations:**
1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. [PPO algorithm]
2. Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy DRL*. ICML 2018. [SAC baseline]

**DRO Theory:**
3. Duchi, J., Namkoong, H. (2018). *Learning Models with Uniform Performance*. Annals of Statistics. **[PRIMARY: closed-form DRO-CVaR equation]**
4. Namkoong, H., Duchi, J. (2017). *Variance-based Regularization with Convex Objectives*. NeurIPS 2017. [Variance-DRO equivalence]
5. Rockafellar, R.T., Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. J. Risk, 2(3), 21-41. [CVaR foundation]
6. Ben-Tal, A., El Ghaoui, L., Nemirovski, A. (2009). *Robust Optimization*. Princeton University Press. [General RO framework]

**Power Systems RL:**
7. Zhang, Z., Zhang, D., Qiu, R.C. (2020). *Deep reinforcement learning for power system applications: An overview*. CSEE J. Power & Energy Systems, 6(1), 213-225. [RL for power systems survey]
8. CEA (2024). *Coal Stock Position at Thermal Power Stations: Monthly Report*. Central Electricity Authority. [Coal criticality data source]

9. (2026). *CVaR-Guided Decision-Focused Learning and Risk-Triggered Re-Optimization for Two-Stage Robust Microgrid Operation*. arXiv:2604.16614.
10. (2022). *Distributionally robust OPF in distribution network considering CVaR-averse voltage security*. ScienceDirect.
11. (2025). *Credible capacity evaluation of virtual power plants considering wind and PV uncertainties*. Scientific Reports.
12. Tang et al. (2026). *Weather-Aware Reinforcement Learning for Power Grid Topology Control Under Extreme Events*. IET Smart Grid.
13. (2022). *Confidence Estimation Transformer for Long-term Renewable Energy Forecasting in Reinforcement Learning-based Power Grid Dispatching*. arXiv:2204.04612. **[Closest global analogue to RE-FUSED's A-E pipeline — real-grid validated on China's SG-126 simulator; cite explicitly as the precedent this work extends]**

