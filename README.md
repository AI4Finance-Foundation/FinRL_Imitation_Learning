# FinRL Imitation Learning

A two-stage method for financial tasks

## File Structure

### **1-Data**		
+ **data:** daily price data of 11 selected XLK constitutes with technical indicators. Our picked sample stocks are {"QCOM", "ADSK", "FSLR", "MSFT", "AMD", "ORCL", "INTU", "WU", "LRCX", "TXN", "CSCO"} : Mean variance optimization weights available. Splitted into train and trade periods.

### **2-Demo & Development**
+ **Retail Market Order Imbalance:** historical files
+ **Stock_Selection:** data source and how we pick our stocks
+ **Weight_Initialization:** feature various weight allocation methods, such as mean-variance and rank-based methods
+ **Imitation_Sandbox:** supervised learning with our task 
+ **stats_test.py:** statistical tests on retail trade imbalance with return rates

### **3-Scripts**
+ **utils.py:** replay buffer implementation.
+ **TD3_BC.py:** TD3 implementation with behaviour cloning (BC) regularization.
+ **StockPortfolioEnv.py:** gym-style environment for asset allocation.
