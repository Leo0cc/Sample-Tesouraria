import pandas as pd
import numpy as np
import streamlit as st
import scipy.optimize as optimize
import matplotlib.pyplot as plt

#Funcoes que serao ultilizadas
st.set_page_config(layout="wide")
#Funcoao para calculo do YTM
def bond_ytm(price, par, T, coup, freq=2, guess=0.05):
    freq = float(freq)
    periods = T*freq
    coupon = coup/100.*par/freq
    dt = [(i+1)/freq for i in range(int(periods))]
    ytm_func = lambda y: sum([coupon/(1+y/freq)**(freq*t) for t in dt]) + par/(1+y/freq)**(freq*T) - price
        
        
        
    return optimize.newton(ytm_func, guess)

#Funcao para calculo do valor presente
def bond_price(par, T, ytm, coup, freq=2):
    freq = float(freq)
    periods = T*freq
    coupon = (coup/100)*par/freq
    dt = [(i+1)/freq for i in range(int(periods))]
    price = sum([coupon/(1+ytm/freq)**(freq*t) for t in dt]) + \
            par/(1+ytm/freq)**(freq*T)
    return price

#Funcao para calculo da Duration de Macaulay
def bond_macaulay_d(par,ytm, coup, T):
    periods = T * 2
    coupon_rate= (coup/100)/2
    ytm_semi = ytm/2
    pv = bond_price(par,T,ytm, coup)
    pvc = par*coupon_rate*((1-(1/(1+ytm_semi)**periods))/ytm_semi)
    H = pvc/pv
    
    
    dura = (( 1 + ytm_semi ) / ytm_semi ) * H + (( ytm_semi - coupon_rate ) / ytm_semi) * periods * (1-H)
    return dura/2 





#Funcao para calculo da Convexidade
def bond_convexity(price, par, T, coup, freq=2, dy=0.01):
    ytm = bond_ytm(price, par, T, coup, freq)

    ytm_minus = ytm - dy    
    price_minus = bond_price(par, T, ytm_minus, coup, freq)
    
    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)
    
    convexity = (price_minus+price_plus-2*price)/(price*dy**2)
    return convexity


#Funcao para calculo da Duration Efetiva
def effective_duration(price, par, T, coup, freq=2, dy=0.01):
    ytm = bond_ytm(price, par, T, coup)
    
    ytm_minus = ytm - dy   
    price_minus = bond_price(par, T, ytm_minus, coup, freq)
    
    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)
    
    mduration = (price_minus-price_plus)/(2*price*dy)
    return mduration

#Definicao dos parametros para criacao dos titulos
class bonds:
    def __init__(self):
        self.par_value = 1000
        self.coupon = np.random.randint(low=0, high= 15)
        self.maturity = np.random.randint(2024, 2033)
        self.vencimento = self.maturity - 2023
        ano_maturidade = str(self.maturity)
        
        
        
        
        for index, row in cdi.iterrows():
            if row['Ano'] == ano_maturidade:
                self.ytm_init = row['CDI Futuro']
                break
        
        self.pv = bond_price(self.par_value, self.vencimento,self.ytm_init , self.coupon)
        self.macaulay = bond_macaulay_d(self.par_value,self.ytm_init, self.coupon, self.vencimento)
        self.efecdu = effective_duration(self.pv, self.par_value, self.vencimento, self.coupon)
        self.convex = bond_convexity(self.pv, self.par_value,self.vencimento ,self.coupon )
#Definindo diferentes spreads e niveis de risco
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
        
#Gerando o DataFrame para carteira de titulos ativos
def create_bonds_buy():
    data = []
    for i in range(len(cdi["CDI Futuro"])):
        bond = bonds()
        bond.risco_buy()
        title = f"Title {i+1}"
        pv = round(bond.pv, 2)
        ytm = round(bond.ytm_init*100, 2)
        vencimento = (bond.vencimento)
        cupom = round(bond.coupon, 2)
        effec = (bond.efecdu)
        convex = (bond.convex)
        duration = (bond.macaulay)
        data.append({"Nome": title, "Valor Presente": pv, "YTM" : ytm, "Vencimento" : vencimento, "Taxa de Cupom" : cupom, "Duration Efetiva": effec, "Convexidade":convex, "Duration": duration})
    df = pd.DataFrame(data)
    df.set_index("Nome", inplace=True)
    return df.sort_values(by="Vencimento", ascending=True)


#Gerando o DataFrame para carteira de titulos passivos
def create_bonds_sell():
    data = []
    for i in range(len(cdi["CDI Futuro"])):
        bond = bonds()
        bond.risco_sell()
        title = f"Title {i+1}"
        pv = round(bond.pv, 2)
        ytm = round(bond.ytm_init*100, 2)
        vencimento = (bond.vencimento)
        cupom = (bond.coupon)
        effec = (bond.efecdu)
        convex = (bond.convex)
        duration = (bond.macaulay)
        data.append({"Nome": title, "Valor Presente": pv, "YTM" : ytm, "Vencimento" : vencimento, "Taxa de Cupom" : cupom, "Duration Efetiva": effec, "Convexidade":convex, "Duration": duration})
    df = pd.DataFrame(data)
    df.set_index("Nome", inplace=True)
    return df.sort_values(by="Vencimento", ascending=True)

#Contabilizando a posicao total da carteira

def  position_total(assets):
    carteira = []
    pv = assets["Valor Presente"].sum()
    
    return 


#Formatacao dos dados no Streamlit
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

#Geracao de titulos a partir da curva de juros


assets = create_bonds_buy()
assets_portfolio = position_total(assets)

liabilities = create_bonds_sell()
liabilities_portfolio = position_total(liabilities)

def  position_total(assets):
    carteira = []
    pv = assets["Valor Presente"].sum()
    return pv

st.write(position_total(assets))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Carteira de titulos ativos")
    st.dataframe(assets)
  
with col2:
    st.subheader("Carteira de titulos passivos")
    st.dataframe(liabilities)

   


