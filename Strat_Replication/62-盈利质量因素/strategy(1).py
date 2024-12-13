import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_stock_financial(ticker, year):
    financials = yf.Ticker(ticker)
    balancesheet = financials.balance_sheet
    income_stmt = financials.income_stmt
    cashflow = financials.cashflow

    # 转置数据框以便访问
    balance_sheet = balancesheet.T
    income_statement = income_stmt.T
    cash_flow = cashflow.T

    # 确保所需字段存在
    required_fields = ['Total Assets', 'Total Liabilities', 'Total Stockholder Equity', 'Net Income', 'Operating Cash Flow']
    for field in required_fields:
        if field not in balance_sheet.columns and field not in income_statement.columns and field not in cash_flow.columns:
            print(f"Field '{field}' is missing for {ticker}")
            return None

    # 按年份筛选数据
    if year not in balance_sheet.index.year:
        print(f"Year {year} data is not available for {ticker}")
        return None

    # 获取指定年份的数据
    total_assets = balance_sheet.loc[balance_sheet.index.year == year, 'Total Assets'].iloc[0]
    total_liabilities = balance_sheet.loc[balance_sheet.index.year == year, 'Total Liabilities'].iloc[0]
    total_equity = balance_sheet.loc[balance_sheet.index.year == year, 'Total Stockholder Equity'].iloc[0]
    net_income = income_statement.loc[income_statement.index.year == year, 'Net Income'].iloc[0]
    operating_cash_flow = cash_flow.loc[cash_flow.index.year == year, 'Operating Cash Flow'].iloc[0]

    # 返回结果
    result = pd.DataFrame({
        'Total Assets': [total_assets],
        'Total Liabilities': [total_liabilities],
        'Total Equity': [total_equity],
        'Net Income': [net_income],
        'Operating Cash Flow': [operating_cash_flow]
    }, index=[year])

    return result

def calculate_quality_score(ticker, year):
    try:
        financial_data = get_stock_financial(ticker, year)
        if financial_data is None:
            return np.nan

        # 计算盈利质量指标
        cash_flow_to_earnings = financial_data['Operating Cash Flow'] / financial_data['Net Income']
        roe = financial_data['Net Income'] / financial_data['Total Equity']
        cf_to_assets = financial_data['Operating Cash Flow'] / financial_data['Total Assets']
        debt_to_assets = financial_data['Total Liabilities'] / financial_data['Total Assets']

        # 计算综合得分
        score = (cash_flow_to_earnings.iloc[0] + roe.iloc[0] + cf_to_assets.iloc[0] - debt_to_assets.iloc[0]) * 100

        return score

    except Exception as e:
        print(f"Error calculating quality score for {ticker}: {e}")
        return np.nan

class QualityStrategy(bt.Strategy):
    params = (
        ('rebalance_month', 6),  # 每年6月调整
    )

    def __init__(self):
        self.rebalance_date = None
        self.start_cash = self.broker.getvalue()

    def next(self):
        if self.data.datetime.date(0).month == self.params.rebalance_month and self.data.datetime.date(0).day == 30:
            self.rebalance_date = self.data.datetime.date(0)
            self.rebalance_portfolio()

    def rebalance_portfolio(self):
        self.log(f"Rebalancing Portfolio on {self.rebalance_date}")

        # 获取所有股票的盈利质量得分
        quality_scores = self.get_quality_scores()
        
        if quality_scores is not None:
            # 按盈利质量得分对股票进行排序
            quality_scores = quality_scores.sort_values(by='Quality_Score', ascending=False)
            
            # 选择得分最高的30%股票做多，得分最低的30%股票做空
            num_stocks = len(quality_scores)
            num_long = num_short = int(num_stocks * 0.3)
            long_stocks = quality_scores.head(num_long)['Ticker'].tolist()
            short_stocks = quality_scores.tail(num_short)['Ticker'].tolist()

            # 清仓
            for data in self.datas:
                if self.getposition(data).size:
                    self.close(data=data)

            # 做多高质量股票
            long_weight = self.broker.getvalue() / (2 * num_long)
            for ticker in long_stocks:
                data = self.getdatabyname(ticker)
                self.buy(data=data, size=long_weight / data.close[0])
                self.log(f"LONG {ticker}")

            # 做空低质量股票
            short_weight = self.broker.getvalue() / (2 * num_short)
            for ticker in short_stocks:
                data = self.getdatabyname(ticker)
                self.sell(data=data, size=short_weight / data.close[0])
                self.log(f"SHORT {ticker}")

    def get_quality_scores(self):
        year = self.rebalance_date.year - 1  # 使用上一年的财务数据
        quality_scores = {'Ticker': [], 'Quality_Score': []}
        for data in self.datas:
            ticker = data._name
            score = calculate_quality_score(ticker, year)
            quality_scores['Ticker'].append(ticker)
            quality_scores['Quality_Score'].append(score)
        return pd.DataFrame(quality_scores)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

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

    # 定义股票列表（这里使用示例股票，您可以扩展这个列表）
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'PYPL', 'INTC', 'CSCO', 'ORCL', 'IBM', 'QCOM', 'TXN', 'AMD', 'AVGO', 'MU', 'INTU', 'CRM', 'BABA', 'JD', 'BIDU', 'TCEHY', 'SHOP', 'SQ', 'UBER', 'LYFT', 'ZM', 'DOCU']


    # 添加股票数据
    for ticker in tickers:
        data = bt.feeds.PandasData(dataname=yf.download(ticker, start="2020-01-01", end="2024-06-01"))
        cerebro.adddata(data, name=ticker)

    cerebro.addstrategy(QualityStrategy)
    
    cerebro.broker.set_cash(1000000)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% 佣金

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