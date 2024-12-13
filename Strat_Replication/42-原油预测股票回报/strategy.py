import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class CrudeOilPredictor(bt.Strategy):
    def __init__(self):
        self.oil_data = self.datas[0]
        self.equity_data = self.datas[1]
        self.last_month = -1

    def next(self):
        current_month = self.data.datetime.date(0).month
        if current_month == self.last_month:
            return

        self.last_month = current_month
        oil_returns = (self.oil_data.close[0] / self.oil_data.close[-21]) - 1
        equity_returns = (self.equity_data.close[0] / self.equity_data.close[-21]) - 1

        # Perform regression
        X = np.array(oil_returns).reshape(-1, 1)
        y = np.array(equity_returns).reshape(-1, 1)
        regression = np.linalg.lstsq(X, y, rcond=None)[0]
        expected_return = regression[0][0] * oil_returns

        risk_free_rate = 0.0003  # 假设无风险利率为0.03%（每月）
        if expected_return > risk_free_rate:
            self.order_target_percent(target=1.0)
        else:
            self.order_target_percent(target=0.0)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

class PerformanceAnalyzer(bt.Analyzer):
    def __init__(self):
        self.rets = []

    def next(self):
        self.rets.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        returns = np.diff(self.rets) / self.rets[:-1]
        annual_return = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility

        cum_returns = np.maximum.accumulate(self.rets)
        drawdowns = (cum_returns - self.rets) / cum_returns
        max_drawdown = np.max(drawdowns)

        return {
            'portfolio_value': self.rets[-1],
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    
    oil_data = bt.feeds.PandasData(dataname=yf.download('CL=F', start="2020-01-01", end="2024-06-01"))
    equity_data = bt.feeds.PandasData(dataname=yf.download('SPY', start="2020-01-01", end="2024-06-01"))

    cerebro.adddata(oil_data, name='Oil')
    cerebro.adddata(equity_data, name='Equity')

    cerebro.addstrategy(CrudeOilPredictor)
    cerebro.broker.set_cash(1000000)
    cerebro.broker.setcommission(commission=0.0)  # 设置无交易佣金

    cerebro.addanalyzer(PerformanceAnalyzer, _name='performance')

    print('期初资产: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]
    print('回测区间: 2020-01-01 - 2024-06-01')
    print('期初资产: %.2f' % 1000000)
    print('期末资产: %.2f' % cerebro.broker.getvalue())

    performance = strat.analyzers.performance.get_analysis()
    print('年化收益: {:.2f}%'.format(performance['annual_return'] * 100))
    print('年化波动: {:.2f}%'.format(performance['annual_volatility'] * 100))
    print('最大回撤: {:.2f}%'.format(performance['max_drawdown'] * 100))
    print('夏普比: {:.2f}'.format(performance['sharpe_ratio']))

    cerebro.plot()
