### How Does News Data Impacts Trading Decisions?

##### Repository Details:
This repository has the RL framework for performing futures trading in stock market.
The trading environment currently supports trading in near month contracts.
Discrete action space is supported for Q learning, SARSA and DQN. Continous action space is supported for PPO.
Continuous observation space is supported for DQN and PPO. Discrete obeservation space is supported for Q learning and SARSA.

1. deep_rl_trading: This folder consists of the PPO and DQN based RL models.
2. rl_trading: This folder consists of the Q learning and SARSA based RL models.

##### Data Source:
1. Price data: https://www.kaggle.com/datasets/nishanthsalian/indian-stock-index-1minute-data-2008-2020
2. News data: https://economictimes.indiatimes.com/archive.cms?from=mdr

##### Evaluation Metrics:
1. Total Profit
2. Return (%)
3. Maximum Drawdown
4. Volatility
5. Sharpe Ratio
6. Sortino Ratio

##### Prerequisite Libraries:
1. stable-baselines3==1.7.0
2. TA-Lib==0.4.25
3. rich==13.3.1
4. torch==1.13.0

##### Preliminary Reading:
1. Futures trading: https://zerodha.com/varsity/module/futures-trading/
2. Technical Indicator: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/overview
3. FinBERT: https://arxiv.org/abs/1908.10063
4. PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
5. DQN: https://arxiv.org/abs/1312.5602
6. Q Learning: https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/
7. SARSA: https://builtin.com/machine-learning/sarsa
