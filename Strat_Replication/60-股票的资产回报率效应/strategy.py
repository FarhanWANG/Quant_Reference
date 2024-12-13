import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np

# Define the strategy
class ROAStrategy(bt.Strategy):
    params = (('rebalance_month', 6),)  # Rebalance every June

    def __init__(self):
        self.rebalance_date = None
        self.start_cash = self.broker.getvalue()
        self.trade_log = []

    def next(self):
        if self.data.datetime.date(0).month == self.params.rebalance_month and self.data.datetime.date(0).day == 30:
            self.rebalance_date = self.data.datetime.date(0)
            self.rebalance_portfolio()

    def rebalance_portfolio(self):
        self.log(f"Rebalancing Portfolio on {self.rebalance_date}")

        # Get all stocks ROA data
        roa_data = self.get_roa_data()

        if roa_data is not None:
            # Sort stocks by ROA
            roa_data = roa_data.sort_values(by='ROA', ascending=False)

            # Select top and bottom deciles
            top_deciles = roa_data.head(int(len(roa_data) * 0.3))
            bottom_deciles = roa_data.tail(int(len(roa_data) * 0.3))

            # Clear positions
            for data in self.datas:
                position = self.getposition(data)
                if position.size:
                    if position.size > 0:
                        self.sell(data=data, size=position.size)
                        self.log_trade(data._name, 'SELL', position.size)
                    else:
                        self.buy(data=data, size=-position.size)
                        self.log_trade(data._name, 'BUY', -position.size)

            # Buy top deciles and short bottom deciles
            long_weight = self.broker.getvalue() / len(top_deciles)
            short_weight = self.broker.getvalue() / len(bottom_deciles)
            for ticker in top_deciles.index:
                data = self.getdatabyname(ticker)
                size = long_weight / data.close[0]
                self.buy(data=data, size=size)
                self.log_trade(ticker, 'BUY', size)

            for ticker in bottom_deciles.index:
                data = self.getdatabyname(ticker)
                size = short_weight / data.close[0]
                self.sell(data=data, size=size)
                self.log_trade(ticker, 'SELL', size)

    def get_roa_data(self):
        roa_data = {}
        for data in self.datas:
            ticker = data._name
            quarterly_financials = yf.Ticker(ticker).quarterly_financials
            quarterly_balance_sheet = yf.Ticker(ticker).quarterly_balance_sheet

            if 'Net Income' in quarterly_financials.index and 'Total Assets' in quarterly_balance_sheet.index:
                net_income = quarterly_financials.loc['Net Income']
                total_assets = quarterly_balance_sheet.loc['Total Assets']

                if not net_income.empty and not total_assets.empty:
                    roa = net_income / total_assets
                    roa_data[ticker] = roa.mean()

        return pd.DataFrame.from_dict(roa_data, orient='index', columns=['ROA'])

    def log(self, txt, data=None):
        dt = self.datas[0].datetime.date(0)
        if data is not None:
            symbol = data._name
            print(f'{dt}: {symbol}: {txt}')
        else:
            print(f'{dt}: {txt}')

    def log_trade(self, ticker, action, size):
        self.trade_log.append({
            'date': self.rebalance_date,
            'ticker': ticker,
            'action': action,
            'size': size
        })

# Performance Analyzer
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

# Run backtest
if __name__ == '__main__':
    cerebro = bt.Cerebro()

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'PYPL', 'INTC']

    for ticker in tickers:
        data = bt.feeds.PandasData(dataname=yf.download(ticker, start="2020-01-01", end="2024-06-01"))
        cerebro.adddata(data, name=ticker)

    cerebro.addstrategy(ROAStrategy)
    cerebro.broker.set_cash(1000000)
    cerebro.addsizer(bt.sizers.AllInSizer)
    cerebro.addanalyzer(PerformanceAnalyzer, _name='performance')

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

    performance = strat.analyzers.performance.get_analysis()
    print('Annual Return: {:.2f}%'.format(performance['annual_return'] * 100))
    print('Annual Volatility: {:.2f}%'.format(performance['annual_volatility'] * 100))
    print('Max Drawdown: {:.2f}%'.format(performance['max_drawdown'] * 100))
    print('Sharpe Ratio: {:.2f}'.format(performance['sharpe_ratio']))

    # print('Trade Log:')
    # for trade in strat.trade_log:
    #     print(trade)

    # cerebro.plot()


    img = cerebro.plot(style='line', plotdist=0.1, grid=True)
    img[0][0].savefig(f'Figure_0.png')
