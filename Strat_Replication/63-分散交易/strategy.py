import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    
    # 计算历史波动率（20天）
    hist['Returns'] = hist['Close'].pct_change()
    hist['Volatility'] = hist['Returns'].rolling(window=20).std() * np.sqrt(252)
    
    # 获取分析师目标价
    try:
        target_price = stock.info['targetMeanPrice']
    except:
        target_price = hist['Close'].mean()  # 如果无法获取目标价，使用平均价格作为替代
    
    # 计算当前价格与目标价之间的差异作为分歧度的代理
    hist['PriceDeviation'] = abs(hist['Close'] - target_price) / hist['Close']
    
    # 删除NaN值
    hist.dropna(inplace=True)
    
    return hist

class PandasDataWithDeviation(bt.feeds.PandasData):
    lines = ('PriceDeviation', 'Volatility',)
    params = (('PriceDeviation', -1), ('Volatility', -1),)

class DispersionTradingStrategy(bt.Strategy):
    params = (
        ('rebalance_months', [1, 4, 7, 10]),  # 每年1月、4月、7月和10月调整
        ('top_n', 5),  # 选择分歧度最高的前N只股票
    )

    def __init__(self):
        self.sp100 = self.getdatabyname('SPY')  # S&P 100 指数
        self.stocks = [d for d in self.datas if d._name != 'SPY']
        self.rebalance_date = None
        self.order = None  # 跟踪当前订单

    def next(self):
        # 检查是否有未完成的订单
        if self.order:
            return

        # 检查是否是调仓月份的第一个交易日
        if self.data.datetime.date(0).month in self.params.rebalance_months and self.data.datetime.date(0).day <= 5:
            self.rebalance_date = self.data.datetime.date(0)
            self.rebalance_portfolio()

    def rebalance_portfolio(self):
        self.log(f"Attempting to rebalance portfolio on {self.rebalance_date}")

        # 获取当前的价格偏离度和波动率数据
        deviation_data = self.get_deviation_data()
        
        if deviation_data is not None and not deviation_data.empty:
            self.log(f"Deviation data: {deviation_data}")
            
            # 按价格偏离度和波动率的组合指标对股票进行排序
            deviation_data['CombinedScore'] = deviation_data['PriceDeviation'] * deviation_data['Volatility']
            deviation_data = deviation_data.sort_values(by='CombinedScore', ascending=False)
            
            # 选择得分最高的股票
            high_deviation_stocks = deviation_data.head(self.params.top_n)['Ticker'].tolist()
            self.log(f"Selected stocks: {high_deviation_stocks}")

            # 清仓
            for data in self.datas:
                if self.getposition(data).size:
                    self.order = self.close(data=data)
                    self.log(f"Closing position for {data._name}")

            # 做空S&P 100指数
            self.order = self.sell(data=self.sp100)
            self.log(f"Shorting SPY")

            # 买入得分最高的股票
            weight = self.broker.getvalue() / len(high_deviation_stocks)
            for ticker in high_deviation_stocks:
                data = self.getdatabyname(ticker)
                size = int(weight / data.close[0])
                self.order = self.buy(data=data, size=size)
                self.log(f"Buy order placed for {ticker}, size: {size}")
        else:
            self.log("No valid deviation data available for rebalancing")

    def get_deviation_data(self):
        deviation_data = {'Ticker': [], 'PriceDeviation': [], 'Volatility': []}
        for stock in self.stocks:
            ticker = stock._name
            deviation_data['Ticker'].append(ticker)
            deviation_data['PriceDeviation'].append(stock.PriceDeviation[0])
            deviation_data['Volatility'].append(stock.Volatility[0])
        return pd.DataFrame(deviation_data)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')

class PerformanceAnalyzer(bt.Analyzer):
    def __init__(self):
        self.rets = []

    def next(self):
        self.rets.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        returns = np.diff(self.rets) / self.rets[:-1]
        annual_return = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

        # 最大回撤计算
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

    # S&P 100 成分股 (这里用部分股票代替，已删除 'FB')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'CMCSA', 'ADBE', 'CRM']

    start_date = "2020-01-01"
    end_date = "2024-07-01"

    # 添加S&P 100指数 (用SPY ETF代替)
    sp100_data = PandasDataWithDeviation(dataname=get_stock_data('SPY', start_date, end_date), name='SPY')
    cerebro.adddata(sp100_data)

    # 添加个股数据
    for ticker in tickers:
        data = PandasDataWithDeviation(dataname=get_stock_data(ticker, start_date, end_date), name=ticker)
        cerebro.adddata(data)

    cerebro.addstrategy(DispersionTradingStrategy)
    
    cerebro.broker.set_cash(1000000)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    cerebro.addanalyzer(PerformanceAnalyzer, _name='performance')

    print('Initial Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    performance = strat.analyzers.performance.get_analysis()
    print('Annual Return: {:.2f}%'.format(performance['annual_return'] * 100))
    print('Annual Volatility: {:.2f}%'.format(performance['annual_volatility'] * 100))
    print('Sharpe Ratio: {:.2f}'.format(performance['sharpe_ratio']))
    print('Maximum Drawdown: {:.2f}%'.format(performance['max_drawdown'] * 100))

    cerebro.plot()