import pandas as pd
import numpy as np
import streamlit as st
import scipy.optimize as optimize
import matplotlib.pyplot as plt

#Funcoes que serao ultilizadas

def bond_ytm(price, par, T, coup, freq=2, guess=0.05):
    freq = float(freq)
    periods = T*freq
    coupon = coup/100.*par/freq
    dt = [(i+1)/freq for i in range(int(periods))]
    ytm_func = lambda y: sum([coupon/(1+y/freq)**(freq*t) for t in dt]) + par/(1+y/freq)**(freq*T) - price
        
        
        
    return optimize.newton(ytm_func, guess)


def bond_price(par, T, ytm, coup, freq=2):
    freq = float(freq)
    periods = T*freq
    coupon = (coup/100)*par/freq
    dt = [(i+1)/freq for i in range(int(periods))]
    price = sum([coupon/(1+ytm/freq)**(freq*t) for t in dt]) + \
            par/(1+ytm/freq)**(freq*T)
    return price



def bond_mod_duration(price, par, T, coup, freq, dy=0.01):
    ytm = bond_ytm(price, par, T, coup, freq)
    
    ytm_minus = ytm - dy    
    price_minus = bond_price(par, T, ytm_minus, coup, freq)
    
    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)
    
    mduration = (price_minus-price_plus)/(2*price*dy)
    return mduration 



def bond_convexity(price, par, T, coup, freq, dy=0.01):
    ytm = bond_ytm(price, par, T, coup, freq)

    ytm_minus = ytm - dy    
    price_minus = bond_price(par, T, ytm_minus, coup, freq)
    
    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)
    
    convexity = (price_minus+price_plus-2*price)/(price*dy**2)
    return convexity



class bonds:
    def __init__(self):
        self.par_value = 1000
        self.coupon = np.random.randint(low=0, high= self.par_value*0.05)
        self.maturity = np.random.randint(2024, 2033)
        self.vencimento = self.maturity - 2023
        ano_maturidade = str(self.maturity)
        
        for index, row in cdi.iterrows():
            if row['Ano'] == ano_maturidade:
                self.ytm_init = row['CDI Futuro']
                break
        
        self.durationma

    
        self.pv = bond_price(self.par_value, self.vencimento,self.ytm_init , self.coupon)
        
    def risco_buy(self):
        teste = np.random.randint(low=0, high= 3)
        if teste == 0:
            self.ytm_init = self.ytm_init + np.random.uniform(low=0, high= 0.01)
        elif teste == 1:
            self.ytm_init = self.ytm_init + np.random.uniform(low=0.01, high= 0.05)
        elif teste == 2:
            self.ytm_init = self.ytm_init + np.random.uniform(low=0.05, high= 0.1)

    def risco_sell(self):
        teste = np.random.randint(low=0, high= 3)
        if teste == 0:
            self.ytm_init = self.ytm_init - np.random.uniform(low=0, high= self.ytm_init*0.1)
        elif teste == 1:
            self.ytm_init = self.ytm_init - np.random.uniform(low=self.ytm_init*0.1, high= self.ytm_init*0.2)
        elif teste == 2:
            self.ytm_init = self.ytm_init - np.random.uniform(low=self.ytm_init*0.2, high= self.ytm_init*0.3)








st.title("Simulação de Titulos")

#Geracao das taxas futuras ultilizadas na criacao dos titulos
cdi_gen = np.random.uniform(0, 0.20)

cdif = [cdi_gen]

for i in range(9):
    cdi_gen = cdi_gen * (1 + np.random.uniform(-0.2, 0.2))
    cdif.append(cdi_gen)


t_anos = 10

anos = [pd.Timestamp(year=i+2024, month=1, day=1) for i in range(t_anos)]
cdi = pd.DataFrame({"CDI Futuro" : cdif, "Ano" : anos})
cdi['Ano'] = pd.to_datetime(cdi['Ano'])
cdi['Ano'] = cdi['Ano'].dt.year.astype(str)



#Grafico da curva de juros
st.subheader("Curva de CDI Futuro")

st.line_chart(cdi.set_index('Ano'), y="CDI Futuro")

#Geracao de titulos a partir da curva de juros, com diferentes niveis de risco





col1, col2 = st.columns(2)
with col1:
    st.subheader("Carteira de titulos ativos")
  
with col2:
    st.subheader("Carteira de titulos passivos")
   


