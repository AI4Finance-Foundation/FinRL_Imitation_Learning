# FinRL Imitation Learning

A two-stage method for financial tasks

## File Structure

### **1-Data**		
+ **data:** daily price data of 11 selected XLK constitutes with technical indicators. Mean variance optimization weights available. 

### **2-Demo & Development**
+ **Retail Market Order Imbalance:** an end-to-end demo of the proposed two-stage method for asset allocation
+ **stats_test.py:** statistical tests on retail trade imbalance with return rates

### **3-Utilities**
+ **utils.py:** replay buffer implementation.
+ **TD3_BC.py:** TD3 implementation with behavior cloning (BC)regularization.
+ **StockPortfolioEnv.py:** gym-style environment for asset allocation.
