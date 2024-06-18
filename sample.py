import pandas as pd
import numpy as np
import streamlit as st
import scipy.optimize as optimize
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

#Curva de juros Pré que servira de base para os titulos

cdi_gen = np.random.uniform(0, 0.20)
cdif = [cdi_gen]

for i in range(9):
    cdi_gen = cdi_gen * (1 + np.random.uniform(-0.5, 0.5))
    cdif.append(cdi_gen)


t_anos = 10

anos = [pd.Timestamp(year=i+2024, month=1, day=1) for i in range(t_anos)]
cdi = pd.DataFrame({"CDI Futuro" : cdif, "Ano" : anos})
cdi['Ano'] = pd.to_datetime(cdi['Ano'])
cdi['Ano'] = cdi['Ano'].dt.year.astype(str)



#Funcoao para calculo do YTM
def bond_ytm(price, par, T, coup, freq=2, guess=0.05):
    freq = float(freq)
    periods = T*freq
    coupon = coup/100.*par/freq
    dt = [(i+1)/freq for i in range(int(periods))]
    ytm_func = lambda y: sum([coupon/(1+y/freq)**(freq*t) for t in dt]) + par/(1+y/freq)**(freq*T) - price
    return optimize.newton(ytm_func, guess)
        
        
    

#Funcao para calculo do valor presente
def bond_price(dias, ytm):
    price = 1000/(1+(ytm/100))**(dias/252)
    return price