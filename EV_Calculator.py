import pandas as pd
import numpy as np
import pytz
from scipy.stats import norm
import json
import requests
import concurrent.futures
from bs4 import BeautifulSoup
import sys
import os
import datetime
from datetime import datetime, timedelta
from pytz import timezone
from dateutil.relativedelta import relativedelta
import calendar
from math import e
import math
from pytz import timezone
import time
import pandas, requests, datetime, calendar
import finnhub
import numpy as np
from urllib.request import urlopen
import certifi
from tqdm import tqdm

#API Keys
market_data_api_key = "XXXXX" #https://www.marketdata.app, option chain data is a premium feature
finnhub_api_key = "XXXXX" #https://finnhub.io/pricing, API for this use is free 
fmp_api_key = "XXXXX" #https://site.financialmodelingprep.com/developer/docs/pricing, the dividend calendar is a premium feature

#Inputs
step = 0.1 #Affects the accuracy of the summations, needs to be even factor of 0.5 (Larger steps are significantly faster)
Earnings_Filter = True
Dividend_Filter = False
condor_prob_profit = 0.5 #Ideal POP is 50%-80%
spread_prob_profit = 0.55
min_alpha = 0.05 #Ideal Alpha is between 5%-20%
min_DTE = 25 #Ideal DTE is >20 days
max_DTE = 35 #After 40 DTE, algorithm can produce unrealistic results
minimum_volume = 50
minimum_open_interest = 50
max_bid_ask_spread = 0.15

#Lists
test = ["AAPL"]

List1 = ["AAL", "AAPL", "ABBV","ABNB", "ABT","ALK", "ACN", "ADBE","ADP","AIG", "AON","AMAT", "AMD", "AMZN", "APA", "ARKK", "AVGO", "AXP", "AZO","BA", "BABA", "BAC", "BBY","BMY",
                           "BLK","BIDU", "BP", "BX", "BYND", "C", "CAT","CCL", "CF", "CHWY", "CLF"]
                           
List2  = ["CMCSA", "COF", "COIN", "COP", "COST", "CRM", "CRWD", "CSCO", "CVS",
                           "CVX","CVNA", "CZR", "DAL", "DE","DELL","DG","DKS","DIA", "DIS", "DKNG", "DOCU", "DOW", "DVN", "EBAY", "EEM", "EFA", "EWZ", "EXPE", "F", "FCX", "FDX"]

List3 = ["FITB","FTNT", "FSLR","FSR", "FXI", "GDX", "GE", "GLD", "GM", "GOLD", "GOOG", "GOOGL", "GS", "HAL", "HD", "HPQ", "HYG", "IBM","IBKR","ICE", "INTU","INTC",
                           "IWM", "IYR", "JD", "JNJ", "JPM", "KMI", "KO", "KR", "KRE", "KSS", "LOW", "LQD","LLY", "LULU","LUV", "LCID","LVS", "MAR","MARA", "MCD", "MDLZ", "MET", "MU","MA"]

List4 =  ["META", "MGM", "MO", "MOS", "MPC", "MRK", "MRNA", "MRO", "MRVL", "MS", "MSFT", "NCLH", "NEM", "NET", "NFLX", "NKE",
                           "NIO","NOW","NVDA", "NYCB","ORCL", "OXY","OIH", "PEP", "PFE", "PG","PLTR","PLUG","PII", "PTON","PNC", "PYPL", "QCOM", "QQQ", "RCL", "RIG", "RIOT","RIVN", "ROKU", "SBUX", "SCHW", "SHOP"]

List5 = ["SLB", "SLV", "SNOW","SCHW","SCHD","SNPS", "SOUN","SMH", "SPY", "SPOT", "SQ", "T", "TD", "TFC","TGT", "TJX", "TLT", "TSLA", "TSM", "TTD", "TXN", "U", "UAL", "UBER", "UPS","UNH", "USB",
                           "USO", "V", "VFC", "VXX", "VZ", "WBA","WM", "WFC", "WMT", "X", "XOM", "XRT", "Z", "ZS"]

final_list = test #Enter list

#Dividend Filter
def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def get_div_calendar():
    from_date = datetime.datetime.now().strftime("%Y")+"-"+datetime.datetime.now().strftime("%m")+"-"+datetime.datetime.now().strftime("%d")
    temp_date = datetime.datetime.now()+relativedelta(months=+4)
    to_date = temp_date.strftime("%Y")+"-"+temp_date.strftime("%m")+"-"+temp_date.strftime("%d")
    url = ("https://financialmodelingprep.com/api/v3/stock_dividend_calendar?from=" + str(from_date) + "&to=" + str(to_date) + "&apikey="+str(fmp_api_key))
    response = get_jsonparsed_data(url)

    return response

def find_ex_dividend_date(data, ticker_symbol, days_ahead = 90):
    day_ahead = datetime.datetime.today().date()+relativedelta(days=days_ahead)
    unix_day_ahead = time.mktime(day_ahead.timetuple())
    ticker_symbol = ticker_symbol.strip().upper() 
    for entry in data:
        if entry.get('symbol', '').strip().upper() == ticker_symbol:
            x= entry.get('date')
            year = int(str("".join(x))[:4])
            month = int(str("".join(x))[5:7])
            day = int(str("".join(x))[8:10])
            date = datetime.datetime(year,month,day)
            unix_date = date.replace(tzinfo=pytz.utc).timestamp()
            return unix_date - 86400

    return  unix_day_ahead - 86400

#Risk free rate
def send_request(url: str) -> any:
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except:
        raise Exception('HTTP Error when scraping Fed website: ', sys.exc_info())

    return response

def scrape_3m_treasury_from_fed() -> float:
    """
    Scrapes the St Louis Fed website for the current 3m treasury yield.
    """

    url = 'https://fred.stlouisfed.org/series/DTB3'
    response = send_request(url)

    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        treasury_rate = soup.find("span", {"class": "series-meta-observation-value"}).text
    except:
        raise Exception('Error when parsing Fred website: ', sys.exc_info())

    return float(treasury_rate)

r = scrape_3m_treasury_from_fed() / 100

#Earnings filter
def finnhub_earnings(symbol):
    finnhub_client = finnhub.Client(api_key=finnhub_api_key)
    from_date = datetime.datetime.now()-relativedelta(months=1)
    temp_date = datetime.datetime.now()+relativedelta(months=+12)
    to_date = temp_date.strftime("%Y")+"-"+temp_date.strftime("%m")+"-"+temp_date.strftime("%d")
    
    data = finnhub_client.earnings_calendar(_from=from_date, to=to_date, symbol=symbol, international=False)
    list_of_dicts = data['earningsCalendar']
    dates = [item['date'] for item in list_of_dicts]

    return dates

def no_past_earnings(dates):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    if dates and dates[0] > today:
        try:
            dates[1]  
            return False
        except IndexError:
            return True
    return False



def stock_price(symbol):
    url = "https://api.marketdata.app/v1/stocks/quotes/" + symbol + "?token=" + market_data_api_key
    response = requests.get(url)
    response_json = json.loads(response.text)
    df = pd.DataFrame(response_json)
    if df.iloc[0]["mid"]:
        return float(df.iloc[0]["mid"])
    else:
        return float(df.iloc[0]["last"])

def bs_mod(S, K, T, r, sigma):
    T /= 365
    N = norm.cdf
    return 1 - (np.exp(-r*T)* N((np.log(S/K) + (r - sigma**2 / 2)*T) /  (sigma*np.sqrt(T))))

def Implied_Vol(symbol):
    today = datetime.datetime.today().date()
    url = "https://api.marketdata.app/v1/options/expirations/" + symbol + "?token=" + market_data_api_key
    response = requests.get(url)
    data = response.json()

    valid_expirations = [date for date in data["expirations"]
                         if 25 <= (datetime.datetime.strptime(date, "%Y-%m-%d").date() - today).days <= 35]

    #Find the expiration closest to 30 days if no valid expiration is found
    if not valid_expirations:
        differences = [(abs((datetime.datetime.strptime(date, "%Y-%m-%d").date() - today).days - 30), date)
                       for date in data["expirations"]]
        closest_to_30_days = sorted(differences, key=lambda x: x[0])[0][1]
        days_to_closest = (datetime.datetime.strptime(closest_to_30_days, "%Y-%m-%d").date() - today).days
        if days_to_closest < 30:
            #Find the nearest expiration greater than 30 days
            other_expiration = min([date for date in data["expirations"] if (datetime.datetime.strptime(date, "%Y-%m-%d").date() - today).days > 30], key=lambda date: (datetime.datetime.strptime(date, "%Y-%m-%d").date() - today).days)
        else:
            #Find the nearest expiration less than 30 days
            other_expiration = max([date for date in data["expirations"] if (datetime.datetime.strptime(date, "%Y-%m-%d").date() - today).days < 30], key=lambda date: (datetime.datetime.strptime(date, "%Y-%m-%d").date() - today).days)
        valid_expirations = [closest_to_30_days, other_expiration]

    ivs = []

    for expiration in valid_expirations:
        option_chain = "https://api.marketdata.app/v1/options/chain/" + symbol + "?expiration=" + str(expiration) + "&delta=" + str(0.5) + "&token=" + market_data_api_key
        chain_response = requests.get(option_chain)
        chain_data = json.loads(chain_response.text)

        # Filter out options with negative IV
        for i in range(len(chain_data['optionSymbol'])):
            if chain_data["iv"][i] and chain_data["iv"][i] > 0:
                ivs.append(chain_data["iv"][i])

   
    average_iv = sum(ivs) / len(ivs) if ivs else 0
    
    return average_iv



def Yang_zhang_volatility(symbol):
    end_date = datetime.datetime.today().date()
    start_date = end_date - timedelta(days=30)

   
    url = "https://api.marketdata.app/v1/stocks/candles/D/" + symbol + "?from=" + str(start_date) + "&to=" + str(end_date) + "&token=" + market_data_api_key
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame({
        'Open': data['o'],
        'High': data['h'],
        'Low': data['l'],
        'Close': data['c']
    })

    
    df['Log Return Open/Prev Close'] = np.log(df['Open'] / df['Close'].shift(1))
    df['Log Return Close/Open'] = np.log(df['Close'] / df['Open'])

    #Calculate Rogers-Satchell volatility
    df['RS'] = np.log(df['High']/df['Open'])*np.log(df['High']/df['Close']) + \
               np.log(df['Low']/df['Open'])*np.log(df['Low']/df['Close'])

    #Calculate the mean of log returns
    mean_log_return_open_prev_close = df['Log Return Open/Prev Close'].mean()
    mean_log_return_close_open = df['Log Return Close/Open'].mean()

    #Calculate individual components of the Yang-Zhang volatility
    df['Adj Overnight Volatility'] = (df['Log Return Open/Prev Close'] - mean_log_return_open_prev_close) ** 2
    df['Adj Open-to-Close Volatility'] = (df['Log Return Close/Open'] - mean_log_return_close_open) ** 2

    sigma_o_squared = df['Adj Overnight Volatility'].sum() / (len(df) - 1)
    sigma_c_squared = df['Adj Open-to-Close Volatility'].sum() / (len(df) - 1)
    sigma_rs = np.sqrt(df['RS'].sum() / (len(df)))

    
    alpha = 1.34  
    T = len(df)
    k = (alpha - 1) / (alpha + (T + 1) / (T - 1))

    
    yang_zhang_vol = np.sqrt(sigma_o_squared + k * sigma_c_squared + (1 - k) * sigma_rs**2) * np.sqrt(252)

    return yang_zhang_vol


def forecasted_vol(hv, iv):
    avg_spread = 1.15 #Average IV/HV spread for a stock (My placeholder value is 1.15)
    curr_spread = iv/hv
    constant = 5 #Constant to affect sensitivity of forecasted_vol from IV/HV spread value
    sigmoid = 1/(1+math.exp(constant * (-(curr_spread - avg_spread))))
    fv = (sigmoid*((hv))) + ((1-sigmoid)*(iv/avg_spread))

    return fv 


#Strategy Functions
def BearCallSpread_EV_Calc(strike1, strike2, price1, price2, stockp, daystoexp, stockiv):
    prof = (price1-price2)
    loss = (strike2-strike1)-(price1-price2)
    ev = ((prof)*bs_mod(stockp,strike1,daystoexp,r,stockiv))-((loss)*(1-bs_mod(stockp,strike2,daystoexp,r,stockiv)))
    ev1 = 0
    currPrice = 0
    slope = -1
    
    while currPrice + strike1 < strike2:
        probability = 1 - ((bs_mod(stockp,currPrice + strike1,daystoexp,r,stockiv)) +
                          (1 - bs_mod(stockp,currPrice + strike1 + step,daystoexp,r,stockiv)))
        ev1 += ((((currPrice + (step / 2))) * slope) + prof) * probability
        currPrice += step
        
    ev += ev1
    
    return ev/loss
    

def BullCallSpread_EV_Calc(strike1, strike2, price1, price2, stockp,daystoexp, stockiv):
    prof = (strike2-strike1)-(price1-price2)
    loss = (price1-price2)
    ev = ((1 - bs_mod(stockp,strike2,daystoexp,r,stockiv))*(prof))-((bs_mod(stockp,strike1,daystoexp,r,stockiv))*loss)
    ev1 = 0
    currPrice = 0
    
    while currPrice + strike1 < strike2:
        probability = 1 - ((bs_mod(stockp,currPrice + strike1,daystoexp,r,stockiv)) +
                          (1 - bs_mod(stockp,currPrice + strike1 + step,daystoexp,r,stockiv)))
        ev1 += ((((currPrice + (step / 2)))) - loss) * probability
        currPrice += step
    
    ev += ev1
    
    return ev/loss

def BearPutSpread_EV_Calc(strike1, strike2, price1, price2, stockp, daystoexp, stockiv):
    prof = (strike2-strike1)-(price2-price1)
    loss = (price2 - price1)
    ev = ((prof)*bs_mod(stockp,strike1,daystoexp,r,stockiv))-((loss)*(1-bs_mod(stockp,strike2,daystoexp,r,stockiv)))
    ev1 = 0
    currPrice = 0
    slope = -1
    
    while currPrice + strike1 < strike2:
        probability = 1 - ((bs_mod(stockp,currPrice + strike1,daystoexp,r,stockiv)) +
                          (1 - bs_mod(stockp,currPrice + strike1 + step,daystoexp,r,stockiv)))
        ev1 += ((((currPrice + (step / 2))) * slope) + prof) * probability
        currPrice += step
    ev += ev1

    return ev/loss

def BullPutSpread_EV_Calc(strike1, strike2, price1, price2, stockp, daystoexp, stockiv):
    prof = (price2-price1)
    loss = (strike2-strike1)-(price2-price1)
    ev = ((prof)*(1-bs_mod(stockp,strike2,daystoexp,r,stockiv)))-((loss)*(bs_mod(stockp,strike1,daystoexp,r,stockiv)))
    ev1 = 0
    currPrice = 0
    
    while currPrice + strike1 < strike2:
        probability = 1 - ((bs_mod(stockp,currPrice + strike1,daystoexp,r,stockiv)) +
                          (1 - bs_mod(stockp,currPrice + strike1 + step,daystoexp,r,stockiv)))
        ev1 += ((((currPrice + (step / 2)))) - loss) * probability
        currPrice += step
    
    ev += ev1

    return ev/loss

def IronCondor_EV_Calc(strikeP1, strikeP2, strikeC1, strikeC2, priceP1, priceP2, priceC1, priceC2, stockp, daystoexp, stockiv):
    prof = (priceP2 + priceC1) - (priceP1 + priceC2)
    left_loss = ((strikeP2-strikeP1)-(priceP2-priceP1)) - ((priceC1-priceC2))
    right_loss = ((strikeC2-strikeC1)-(priceC1-priceC2)) - ((priceP2-priceP1))
    risk = max(right_loss, left_loss)
    
    Prob_Prof = 1 - ((bs_mod(stockp,strikeP2,daystoexp,r,stockiv)) + (1-bs_mod(stockp,strikeC1,daystoexp,r,stockiv)))
    Prob_loss_left = bs_mod(stockp,strikeP1,daystoexp,r,stockiv)
    Prob_loss_right = (1-bs_mod(stockp,strikeC2,daystoexp,r,stockiv))
    ev = (prof*Prob_Prof) - ((left_loss*Prob_loss_left) + (right_loss*Prob_loss_right))
    evP = 0
    evC = 0
    currPriceP = 0
    currPriceC = 0
    slopeC = -1 #SlopeP = 1
    
    while currPriceP + strikeP1 < strikeP2:
        probability = 1 - ((bs_mod(stockp,currPriceP + strikeP1,daystoexp,r,stockiv)) +
                          (1 - bs_mod(stockp,currPriceP + strikeP1 + step,daystoexp,r,stockiv)))
        
        evP += ((((currPriceP + (step / 2)))) - left_loss) * probability
        currPriceP += step
    
    while currPriceC + strikeC1 < strikeC2:
        probability = 1 - ((bs_mod(stockp,currPriceC + strikeC1,daystoexp,r,stockiv)) +
                          (1 - bs_mod(stockp,currPriceC + strikeC1 + step,daystoexp,r,stockiv)))
        
        evC += ((((currPriceC + (step / 2))) * slopeC) + prof) * probability
        currPriceC += step
    
    ev += evP + evC

    return ev/risk

def InverseIronCondor_EV_Calc(strikeP1, strikeP2, strikeC1, strikeC2, priceP1, priceP2, priceC1, priceC2, stockp, daystoexp, stockiv):
    loss = ((priceP2 + priceC1)-(priceP1+priceC2))
    left_prof = ((strikeP2-strikeP1)-(priceP2-priceP1)) - ((priceC1-priceC2))
    right_prof = ((strikeC2-strikeC1)-(priceC1-priceC2)) - ((priceP2 - priceP1))
    Prob_prof_right = (1-bs_mod(stockp,strikeC2,daystoexp,r,stockiv))
    Prob_prof_left = bs_mod(stockp,strikeP1,daystoexp,r,stockiv)
    Prob_Loss = 1 - ((bs_mod(stockp,strikeP2,daystoexp,r,stockiv)) + (1-bs_mod(stockp,strikeC1,daystoexp,r,stockiv)))
    
    ev = ((left_prof*Prob_prof_left) + (right_prof*Prob_prof_right)) - (loss*Prob_Loss)
    evP = 0
    evC = 0
    currPriceP = 0
    currPriceC = 0
    slopeP = -1 #SlopeC = 1
    
    while currPriceP + strikeP1 < strikeP2:
        probability = 1 - ((bs_mod(stockp,currPriceP + strikeP1,daystoexp,r,stockiv)) +
                          (1 - bs_mod(stockp,currPriceP + strikeP1 + step,daystoexp,r,stockiv)))
        
        evP += ((((currPriceP + (step / 2))) * slopeP) + left_prof) * probability
        currPriceP += step
    
    while currPriceC + strikeC1 < strikeC2:
        probability = 1 - ((bs_mod(stockp,currPriceC + strikeC1,daystoexp,r,stockiv)) +
                          (1 - bs_mod(stockp,currPriceC + strikeC1 + step,daystoexp,r,stockiv)))
        
        evC += ((((currPriceC + (step / 2)))) - loss) * probability
        currPriceC += step
    
    ev += evP + evC

    return ev/loss

def calculate_EV_Spread(symbol):
    iv = Implied_Vol(symbol)
    Sheet = pd.DataFrame()
    stockp = stock_price(symbol)
    today = datetime.datetime.today().date()
    url = "https://api.marketdata.app/v1/options/expirations/" + symbol + "?token=" + market_data_api_key
    response = requests.get(url)
    data = response.json()
    num_days = max_DTE
    
    if Earnings_Filter:
        earnings = finnhub_earnings(symbol)
    
        if no_past_earnings(earnings):  
            restricted_date_earnings = earnings[0]
        else:
            restricted_date_earnings = datetime.datetime.now().strftime("%Y-%m-%d")
            
        restricted_date_earnings_dt = datetime.datetime.strptime(restricted_date_earnings, "%Y-%m-%d")
        today = datetime.datetime.now().date()
        days_to_expiration_earnings = (restricted_date_earnings_dt.date() - today).days
    
    if Dividend_Filter:
        dividends = find_ex_dividend_date(div_calendar, symbol)
        restricted_date_dividends = dividends
        restricted_date_div = datetime.datetime.utcfromtimestamp(restricted_date_dividends).date()
        days_to_expiration_dividends = (restricted_date_div - today).days
        
    if Earnings_Filter and Dividend_Filter:
        num_days = min(days_to_expiration_earnings,days_to_expiration_dividends,max_DTE)
        
    if Earnings_Filter:
        num_days = min(days_to_expiration_earnings,max_DTE)
    
    if Dividend_Filter:
        num_days = min(days_to_expiration_dividends, max_DTE)
        
    hv = Yang_zhang_volatility(symbol)
    fv =  forecasted_vol(hv,iv)
    
    if num_days > min_DTE:
        valid_expirations = [date for date in data["expirations"]
                            if min_DTE <= (datetime.datetime.strptime(date, "%Y-%m-%d").date() - today).days <= num_days]
        Sheet_hv = EV_Calculator(symbol,hv,valid_expirations,stockp)
        Sheet = pd.concat([Sheet, Sheet_hv], ignore_index=True)
    else: 
        valid_expirations = [date for date in data["expirations"]
                            if min_DTE <= (datetime.datetime.strptime(date, "%Y-%m-%d").date() - today).days <= max_DTE]
        Sheet_fv = EV_Calculator(symbol,fv,valid_expirations,stockp)
        Sheet = pd.concat([Sheet, Sheet_fv], ignore_index=True)
        
    return Sheet



def EV_Calculator(symbol,fv,valid_expirations,stockp):
    Sheet = pd.DataFrame()
    for expiration in valid_expirations:
        option_chain_url = "https://api.marketdata.app/v1/options/chain/" + symbol + "?expiration=" + str(expiration) + "&strikeLimit=" + str(14) + "&minVolume=" + str(minimum_volume) + "&minOpenInterest=" + str(minimum_open_interest) + "&maxBidASkSpread=" + str(max_bid_ask_spread) + "&token=" + market_data_api_key
        response = requests.get(option_chain_url)
        response_json = json.loads(response.text)
        df = pd.DataFrame({key: pd.Series(value) for key, value in response_json.items()})
        desired_columns = ["underlying", "side", "strike", "dte", "bid", "ask", "mid"]
        available_columns = [col for col in desired_columns if col in df.columns]
        df = df[available_columns]

        df_Call = pd.DataFrame(columns=df.columns)
        df_Put = pd.DataFrame(columns=df.columns)
        checkC = 0
        checkP = 0

        if "side" in df.columns:
            for index, row in df.iterrows():
                if row["side"] == "call":
                    df_Call.loc[index] = row
                elif row["side"] == "put":
                    df_Put.loc[index] = row
                    
        df_Call = df_Call.reset_index(drop=True)
        df_Put = df_Put.reset_index(drop=True)

        if len(df_Call.index) >= 2:
            checkC = 1
        if len(df_Put.index) >= 2:
            checkP = 1

        bear_call_spread_rows = []
        bull_call_spread_rows = []
        bear_put_spread_rows = []
        bull_put_spread_rows = []
        iron_condor_rows = []
        inverse_iron_condor_rows = []

        if checkC == 1:
            for i in range(len(df_Call)):
                for j in range(i + 1, min(i + 3,len(df_Call))):
                    strikeC1 = df_Call.iloc[i]["strike"]
                    strikeC2 = df_Call.iloc[j]["strike"]
                    priceC1 = df_Call.iloc[i]["mid"]
                    priceC1A = df_Call.iloc[i]["ask"]
                    priceC1B = df_Call.iloc[i]["bid"]
                    priceC2 = df_Call.iloc[j]["mid"]
                    priceC2A = df_Call.iloc[j]["ask"]
                    priceC2B = df_Call.iloc[j]["bid"]
                    days_to_exp = df_Call.iloc[i]["dte"]
                    xC = BearCallSpread_EV_Calc(strikeC1, strikeC2, priceC1, priceC2, stockp, days_to_exp, fv)
                    xC_prof_yC_loss = (priceC1-priceC2)
                    yC = BullCallSpread_EV_Calc(strikeC1, strikeC2, priceC1, priceC2, stockp, days_to_exp, fv)
                    breakeven = strikeC1 + xC_prof_yC_loss
                    xC_min = BearCallSpread_EV_Calc(strikeC1, strikeC2, priceC1B, priceC2A, stockp, days_to_exp, fv)
                    yC_min = BullCallSpread_EV_Calc(strikeC1, strikeC2, priceC1A, priceC2B, stockp, days_to_exp, fv)
                    xC_worst_fill = 100*(priceC1B-priceC2A)
                    xC_mid_fill = 100*(priceC1-priceC2)
                    yC_worst_fill = 100*(priceC1A-priceC2B)
                    yC_mid_fill = 100*(priceC1-priceC2)

                    if xC > min_alpha and (strikeC1 > stockp) and xC_min > 0 and bs_mod(stockp,breakeven,days_to_exp,r,fv) > spread_prob_profit:
                        bear_call_spread_rows.append({"Symbol": symbol, "Strategy": "Bear Call Spread", "Lower Strike": strikeC1, "Upper Strike": strikeC2,
                                            "Worst Fill Price": "$"+str(round(xC_worst_fill,4)), "Mid Fill Price": "$"+str(round(xC_mid_fill,4)),
                                            "Stock Price": stockp, "Days to Expiration": days_to_exp, "Alpha": str(round(100*xC,4))+"%","FV":str(round(100*fv,4))+"%"})
                    if yC > min_alpha and (strikeC2 < stockp) and yC_min > 0 and (1- bs_mod(stockp,breakeven,days_to_exp,r,fv)) > spread_prob_profit:
                        bull_call_spread_rows.append({"Symbol": symbol, "Strategy": "Bull Call Spread", "Lower Strike": strikeC1,"Upper Strike": strikeC2,
                                        "Worst Fill Price": "$"+str(round(yC_worst_fill,4)), "Mid Fill Price": "$"+str(round(yC_mid_fill,4)),
                                            "Stock Price": stockp, "Days to Expiration": days_to_exp, "Alpha": str(round(100*yC,4))+"%","FV":str(round(100*fv,4))+"%"})
        if checkP == 1:
            for k in range(len(df_Put)):
                for l in range(k + 1, min(k + 3, len(df_Put))):
                    strikeP1 = df_Put.iloc[k]["strike"]
                    strikeP2 = df_Put.iloc[l]["strike"]
                    priceP1 = df_Put.iloc[k]["mid"]
                    priceP1A = df_Put.iloc[k]["ask"]
                    priceP1B = df_Put.iloc[k]["bid"]
                    priceP2 = df_Put.iloc[l]["mid"]
                    priceP2A = df_Put.iloc[l]["ask"]
                    priceP2B = df_Put.iloc[l]["bid"]
                    days_to_exp = df_Put.iloc[k]["dte"]
                    xP = BearPutSpread_EV_Calc(strikeP1, strikeP2, priceP1, priceP2, stockp, days_to_exp, fv)
                    xP_profit_yP_loss = (strikeP2-strikeP1)-(priceP2-priceP1)
                    breakeven = strikeP1 + xP_profit_yP_loss
                    yP = BullPutSpread_EV_Calc(strikeP1, strikeP2, priceP1, priceP2, stockp, days_to_exp, fv)
                    xP_min = BearPutSpread_EV_Calc(strikeP1, strikeP2, priceP1B, priceP2A, stockp, days_to_exp, fv)
                    yP_min = BullPutSpread_EV_Calc(strikeP1, strikeP2, priceP1A, priceP2B, stockp, days_to_exp, fv)
                    xP_worst_fill = 100*(priceP2A-priceP1B)
                    xP_mid_fill = 100*(priceP2-priceP1)
                    yP_worst_fill = 100*(priceP2B-priceP1A)
                    yP_mid_fill = 100*(priceP2-priceP1)

                    if xP > min_alpha  and (strikeP1 > stockp) and xP_min > 0 and bs_mod(stockp,breakeven,days_to_exp,r,fv) > spread_prob_profit:
                        bear_put_spread_rows.append({"Symbol": symbol, "Strategy": "Bear Put Spread", "Lower Strike": strikeP1,"Upper Strike": strikeP2,
                                            "Worst Fill Price": "$"+str(round(xP_worst_fill,4)), "Mid Fill Price": "$"+str(round(xP_mid_fill,4)),
                                            "Stock Price": stockp, "Days to Expiration": days_to_exp, "Alpha": str(round(100*xP,4))+"%","FV":str(round(100*fv,4))+"%"})
                    if yP > min_alpha and (strikeP2 < stockp) and yP_min > 0 and (1-bs_mod(stockp,breakeven,days_to_exp,r,fv) > spread_prob_profit) > spread_prob_profit:
                        bull_put_spread_rows.append({"Symbol": symbol, "Strategy": "Bull Put Spread", "Lower Strike": strikeP1,"Upper Strike": strikeP2,
                                            "Worst Fill Price": "$"+str(round(yP_worst_fill,4)), "Mid Fill Price": "$"+str(round(yP_mid_fill,4)),
                                            "Stock Price": stockp, "Days to Expiration": days_to_exp, "Alpha": str(round(100*yP,4))+"%","FV":str(round(100*fv,4))+"%"})
            if checkC == 1 and checkP == 1:
                for i in range(len(df_Call)):
                    for j in range(i + 1, min(i + 3, len(df_Call))):
                        strikeC1 = df_Call.iloc[i]["strike"]
                        strikeC2 = df_Call.iloc[j]["strike"]
                        priceC1 = df_Call.iloc[i]["mid"]
                        priceC1A = df_Call.iloc[i]["ask"]
                        priceC1B = df_Call.iloc[i]["bid"]
                        priceC2 = df_Call.iloc[j]["mid"]
                        priceC2A = df_Call.iloc[j]["ask"]
                        priceC2B = df_Call.iloc[j]["bid"]
                        call_spread = strikeC2 - strikeC1

                        for k in range(len(df_Put)):
                            for l in range(k + 1, min(k + 3, len(df_Put))):
                                strikeP1 = df_Put.iloc[k]["strike"]
                                strikeP2 = df_Put.iloc[l]["strike"]
                                priceP1 = df_Put.iloc[k]["mid"]
                                priceP1A = df_Put.iloc[k]["ask"]
                                priceP1B = df_Put.iloc[k]["bid"]
                                priceP2 = df_Put.iloc[l]["mid"]
                                priceP2A = df_Put.iloc[l]["ask"]
                                priceP2B = df_Put.iloc[l]["bid"]
                                put_spread = strikeP2 - strikeP1

                                if call_spread != put_spread:
                                    continue

                                days_to_exp = df_Call.iloc[i]["dte"]

                                ev_C = IronCondor_EV_Calc(strikeP1, strikeP2, strikeC1, strikeC2, priceP1, priceP2, priceC1, priceC2, stockp, days_to_exp, fv)
                                ev_C_min = IronCondor_EV_Calc(strikeP1, strikeP2, strikeC1, strikeC2, priceP1A, priceP2B, priceC1B, priceC2A, stockp, days_to_exp, fv)
                                C_worst_fill = 100*((priceP2B + priceC1B) - (priceP1A + priceC2A))
                                C_mid_fill = 100*((priceP2 + priceC1) - (priceP1 + priceC2))
                                ev_IC = InverseIronCondor_EV_Calc(strikeP1, strikeP2, strikeC1, strikeC2, priceP1, priceP2, priceC1, priceC2, stockp, days_to_exp, fv)
                                ev_IC_min = InverseIronCondor_EV_Calc(strikeP1, strikeP2, strikeC1, strikeC2, priceP1B, priceP2A, priceC1A, priceC2B, stockp, days_to_exp, fv)
                                IC_worst_fill = 100*(((strikeP2-strikeP1)-(priceP2A-priceP1B)) - ((priceC1A-priceC2B)))
                                IC_mid_fill = 100*(((strikeP2-strikeP1)-(priceP2-priceP1)) - ((priceC1-priceC2)))
                                C_prof = (priceP2 + priceC1) - (priceP1 + priceC2)
                                left_breakeven = strikeP2 - C_prof
                                right_breakeven = strikeC1 + C_prof
                                IC_prob_prof = ((bs_mod(stockp,left_breakeven,days_to_exp,r,fv)) + (1-bs_mod(stockp,right_breakeven,days_to_exp,r,fv)))
                                C_prob_prof = 1 - ((bs_mod(stockp,left_breakeven,days_to_exp,r,fv)) + (1-bs_mod(stockp,right_breakeven,days_to_exp,r,fv)))

                                if ev_C > min_alpha  and stockp > ((strikeP2 + strikeP1)/2) and stockp < ((strikeC1 + strikeC2)/2) and ev_C_min > 0 and C_prob_prof > condor_prob_profit and stockp < strikeC1 and stockp > strikeP2:
                                    iron_condor_rows.append({
                                        "Symbol": symbol,
                                        "Strategy": "Iron Condor",
                                        "Call Lower Strike": strikeC1,
                                        "Call Upper Strike": strikeC2,
                                        "Put Lower Strike": strikeP1,
                                        "Put Upper Strike": strikeP2,
                                        "Worst Fill Price": "$"+str(round(C_worst_fill,4)),
                                        "Mid Fill Price": "$"+str(round(C_mid_fill,4)),
                                        "Stock Price": stockp,
                                        "Days to Expiration": days_to_exp,
                                        "Alpha": str(round(100*ev_C,4))+"%"
                                        ,"FV":str(round(100*fv,4))+"%"
                                    })

                                if ev_IC > min_alpha  and stockp > ((strikeP2 + strikeP1)/2) and stockp < ((strikeC1 + strikeC2)/2) and ev_IC_min > 0 and IC_prob_prof > condor_prob_profit:
                                        inverse_iron_condor_rows.append({
                                        "Symbol": symbol,
                                        "Strategy": "Inverse Iron Condor",
                                        "Call Lower Strike": strikeC1,
                                        "Call Upper Strike": strikeC2,
                                        "Put Lower Strike": strikeP1,
                                        "Put Upper Strike": strikeP2,
                                        "Worst Fill Price": "$"+str(round(IC_worst_fill,4)),
                                        "Mid Fill Price": "$"+str(round(IC_mid_fill,4)),
                                        "Stock Price": stockp,
                                        "Days to Expiration": days_to_exp,
                                        "Alpha": str(round(100*ev_IC,4))+"%"
                                        ,"FV":str(round(100*fv,4))+"%"
                                    })


        if bear_call_spread_rows:
            bear_call_spread_df = pd.DataFrame(bear_call_spread_rows)
            Sheet = pd.concat([Sheet, bear_call_spread_df], ignore_index=True)
        if bull_call_spread_rows:
            bull_call_spread_df = pd.DataFrame(bull_call_spread_rows)
            Sheet = pd.concat([Sheet, bull_call_spread_df], ignore_index=True)
        if bear_put_spread_rows:
            bear_put_spread_df = pd.DataFrame(bear_put_spread_rows)
            Sheet = pd.concat([Sheet, bear_put_spread_df], ignore_index=True)
        if bull_put_spread_rows:
            bull_put_spread_df = pd.DataFrame(bull_put_spread_rows)
            Sheet = pd.concat([Sheet, bull_put_spread_df], ignore_index=True)
        if iron_condor_rows:
            iron_condor_df = pd.DataFrame(iron_condor_rows)
            Sheet = pd.concat([Sheet, iron_condor_df], ignore_index=True)
        if inverse_iron_condor_rows:
            inverse_iron_condor_df = pd.DataFrame(inverse_iron_condor_rows)
            Sheet = pd.concat([Sheet, inverse_iron_condor_df], ignore_index=True)

    return Sheet

def EV_Sheet_Spread(symbol_list):
    EV_Sheet = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(calculate_EV_Spread, symbol_list)
        
        for result in tqdm(results, total=len(symbol_list), desc="Processing Symbols"):
            EV_Sheet.append(result)

    df_EV = pd.concat(EV_Sheet, ignore_index=True)
    
    return df_EV



if Dividend_Filter:
    div_calendar = get_div_calendar()
    
df = EV_Sheet_Spread(final_list)
df['Alpha'] = df['Alpha'].str.replace('%', '').astype(float)

highest_alpha_df = df.loc[df.groupby(['Symbol', 'Strategy'])['Alpha'].idxmax()]
df_spreads = highest_alpha_df.drop(columns=["Call Lower Strike", "Call Upper Strike", "Put Lower Strike", "Put Upper Strike"]).dropna(subset=["Lower Strike", "Upper Strike"])
df_spreads = df_spreads[['Symbol', 'Strategy', 'Lower Strike', 'Upper Strike', 'Worst Fill Price', 'Mid Fill Price', 'Stock Price', 'Days to Expiration', 'Alpha', 'FV']]
df_condors = highest_alpha_df.drop(columns=["Lower Strike", "Upper Strike"]).dropna(subset=["Call Lower Strike", "Call Upper Strike", "Put Lower Strike", "Put Upper Strike"])

print("Spreads DataFrame:")
print(df_spreads)

print("\nCondors DataFrame:")
print(df_condors)

