### Playing with News Context for Algorithmic Trading

##### Repository Details:
This repository has the RL framework for performing futures trading in stock market.
The trading environment currently supports trading in near month contracts.
Discrete action space is supported for DQN. Continous action space is supported for PPO.
Continuous observation space is supported for DQN and PPO.

1. context_aware_approach: This folder consists of models used in context_aware_approach. You will have to generate the embeddings for the LLM models apriori before using these models.
2. price_only_sentiment_aware_approach: This folder consists of models in price only approach and sentiment-aware approach.

##### Data Source:
1. Price data: https://www.kaggle.com/datasets/nishanthsalian/indian-stock-index-1minute-data-2008-2020
2. News data: https://economictimes.indiatimes.com/archive.cms?from=mdr

##### Evaluation Metrics:
1. Return (%)
2. Maximum Drawdown
3. Volatility

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
