# FinRL Imitation Learning

A curriculum learning approach is a promising approach for analyzing financial big data. Especially with alpha factors or smart investors trade logs, we can automate a workflow starting with imitating their strategies, and then using reinforcement learning method to further refine the results.

In complicated tasks such as Go and Atari games, imitation learning is often used to initialize deep neural networks that achieve human-level performance. Imitation learning involves training a model to imitate a human's behavior, typically using a dataset of expert demonstrations. This process provides a a starting point for further fine-tuning using reinforcement learning, which could learn through trial and error to find strategies that surpass human-level performance.

```
FinRL_Imitation_Learning
├── Data
|   ├──merged.csv
|   └──train_data.csv
|   └──trade_data.csv
└── StockPortfolioEnv.py
└── Stock_Selection.ipynb
└── Weight_Initialization.ipynb
└── Imitation_Sandbox.ipynb

```

## Progress
- [x] stock_selection
- [x] weight_initialization
- [x] imitation_sandbox
- [ ] reinforcement_learning

**1-Stock Selection**

Identify a pool of stocks that are not only favoured by retail investors, but also exist a high correlation between their trading preference and return rates. 

**2-Weight Initialization**

We construct action space, which will serve as data label. The action space is a critical component made of retailer trade logs. There are two key sources for action construction: MVO (mean-variance optimization) and retail investor perference.

**3-Imitation Sandbox**

We use a set of regression models, including linear models, trees, and neural networks, to analyze data. Our approach involves incrementally increasing the complexity of the models to evaluate their performance in predicting outcomes.

To ensure the reliability, we conduct a placebo test to evaluate the potential for information leakage. This involves feeding simulated data into our models to assess their performance in predicting outcomes that are not based on actual data. By doing so, we can ensure that our models are not biased by any unforeseen factors or hidden information that may have influenced the results.


## Performance
Table: Trading Performance of fitted models in train period.

|                | Linear | Random Forest | LSTM  | Neural Net | True position |
|----------------|--------|---------------|-------|------------|---------------------|
| Annual return  | 0.099  | 0.099         | 0.169 | 0.179      | 0.184               |
| Cumulative returns | 1.835  | 1.832         | 4.558 | 5.122      | 5.403               |
| Annual volatility | 0.246  | 0.246         | 0.249 | 0.248      | 0.249               |
| Sharpe ratio   | 0.509  | 0.508         | 0.751 | 0.790      | 0.803               |
| Max drawdown   | -0.509 | -0.509        | -0.421| -0.420     | -0.411              |
