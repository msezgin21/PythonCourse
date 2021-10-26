import random
import time

stockdict = {}


class Portfolio:
    def __init__(self):
        self.cash = 0
        self.stock = {}
        self.mutFu = {}
        self.history1 = {}


    def addCash(self, amount):
        self.cash += amount
        self.history1[f'addCash={amount}'] = time.time()

    def withdrawCash(self, amount):
        self.cash -= amount
        self.history1[f'withdrawCash={amount}'] = time.time()

    def buyStock(self, shares, stock1):
        self.cash -= stock1.price * shares
        if stock1.symbol in self.stock:
            self.stock[stock1.symbol] += shares
        else:
            self.stock[stock1.symbol] = shares
        self.history1[f'buyStock={shares}'] = time.time()

    def __str__(self):
        return f'cash: ${self.cash} \nstock: {self.stock} \nMutual funds: {self.mutFu}'

    def buyMutualFund(self, shares, mf):
        self.cash -= shares
        if mf.symbol in self.stock:
            self.mutFu[mf.symbol] += shares
        else:
            self.mutFu[mf.symbol] = shares
        self.history1[f'buyMutualFund={shares}'] = time.time()

    def sellMutualFund(self, symbol, shares):
        random_price = random.uniform(0.9, 1.2)
        self.cash += random_price * shares
        if symbol in self.mutFu:
            self.mutFu[symbol] -= shares
        else:
            self.mutFu[symbol] = shares
        self.history1[f'sellMutualFund={shares}'] = time.time()

    def sellStock(self, symbol,shares):
        random_price = random.uniform(0.5 * stockdict[symbol], 1.5 * stockdict[symbol])
        self.cash += random_price * shares
        if symbol in self.stock:
            self.stock[symbol] += shares
        else:
            self.stock[symbol] = shares
        self.history1[f'sellStock={shares}'] = time.time()

    def history(self):
        self.history1 = {k: v for k, v in sorted(self.history.items(), key=lambda item: item[1])}
        print(self.history1.keys())


class Stock:
    def __init__(self, price, symbol):
        self.price = price
        self.symbol = symbol
        stockdict[symbol] = price


class MutualFund:
    def __init__(self, symbol):
        self.symbol = symbol

    portfolio = Portfolio()
    portfolio.addCash(300.50)
    s = Stock(20, "HFH")
    portfolio.buyStock(5, s)
    mf1 = MutualFund("BRT")
    mf2 = MutualFund("GHT")
    portfolio.buyMutualFund(10.3, mf1)
    portfolio.buyMutualFund(2, mf2)
    print(portfolio)

    portfolio.sellMutualFund("BRT", 3)
    portfolio.sellStock("HFH", 1)
    portfolio.withdrawCash(50)
    portfolio.history()