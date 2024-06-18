import pandas as pd
import numpy as np
import streamlit as st
import scipy.optimize as optimize
import matplotlib.pyplot as plt

#Formatacao dos dados no Streamlit
st.set_page_config(layout="wide")

st.title("Simulação de Portfolio")

#Geracao das taxas futuras ultilizadas na criacao dos titulos
cdif = [0.10, 0.105,0.107,0.11, 0.115, 0.12, 0.13, 0.15, 0.17 ]
prazos = [1,30,180,360, 540, 720, 1080,1800,3600]

cdi = pd.DataFrame({"CDI Futuro" : cdif, "Vencimento em dias" : prazos})


#Grafico da curva de juros
juros = st.checkbox("Ver curva de juros base")
if  juros:
    st.subheader("Curva de CDI Futuro")

    st.line_chart(cdi.set_index("Vencimento em dias"), y="CDI Futuro")

#Funcoes que serao ultilizadas para criar os bonds do portfolio

#Funcoao para calculo do YTM
def bond_ytm(dias, price, fv):

    ytm= (fv/price)
    ytm_periodo = ((ytm)**(360/dias))-1
        
        
    return ytm_periodo


#Funcao para calculo do valor presente
def bond_price(dias, ytm,fv):
    price = fv/(1+ytm)**(dias/360)
    return price

##Funcao para calculo do valor futuro
def bond_fv(dias, ytm, pu):
    price = float(pu*(1+ytm)**(dias/360))
    return price

#Definicao dos parametros para criacao dos titulos
class bonds:

    titulos = []

    def __init__(self, tipo):
        self.coupon = 0
        self.vencimento = np.random.choice(prazos)
        self.caracteristica = tipo
        if self.caracteristica == "Passivo":
            self.notional = 1000
            self.ytm_init = float((cdi[cdi["Vencimento em dias"] == self.vencimento]["CDI Futuro"].iloc[0])*1.1)
            self.pv = bond_price(self.vencimento,self.ytm_init,self.notional)
            bonds.titulos.append(self.pv)
        
        elif self.caracteristica == "Ativo" and bonds.titulos:
            self.ytm_init = float(cdi[cdi["Vencimento em dias"] == self.vencimento]["CDI Futuro"].iloc[0]*1.5)
            self.pv = bonds.titulos[-1]
            self.notional = bond_fv(self.vencimento, self.ytm_init, self.pv)
            
        


#Geracao o portfolio a partir da curva de juros
def create_bonds():
    data = []
    
    for i in range(50):
        bond = bonds("Passivo")
        vencimento = bond.vencimento
        ytm = bond.ytm_init
        pv = round(bond.pv, 2)
        notional = bond.notional
        caracteristica = "Passivo"
        title = f"Liability {i+1}"
        data.append({"Nome": title, "Notional": notional, "Valor Presente": pv, "YTM": ytm*100, "Vencimento": vencimento, "Caracteristica": caracteristica})
        bond = bonds("Ativo")
        vencimento = bond.vencimento
        ytm = bond.ytm_init
        pv = round(bond.pv, 2)
        notional = round(bond.notional, 2)
        caracteristica = "Ativo"
        title = f"Ativo {i+1}"
        data.append({"Nome": title, "Notional": notional, "Valor Presente": pv, "YTM": ytm*100, "Vencimento": vencimento, "Caracteristica": caracteristica})

    

    df = pd.DataFrame(data)
    df.set_index("Nome", inplace=True)
    return df.sort_values(by="Vencimento", ascending=True)


portfolio = create_bonds()

df_ativo = portfolio[portfolio["Caracteristica"] == "Ativo"]
df_ativo.rename(columns= {"Notional" :"Assets"}, inplace=True)
dfa = df_ativo.groupby("Vencimento")["Assets"].sum()

df_passivo = portfolio[portfolio["Caracteristica"] == "Passivo"]
df_passivo.rename(columns={"Notional":"Liabilities"}, inplace=True)
dfp = df_passivo.groupby("Vencimento")["Liabilities"].sum()

df_notional =  pd.concat([dfa, dfp], axis=1)
df_notional["Interest-rate gap"] = df_notional["Assets"] - df_notional["Liabilities"]
df_notional["Cumulative gap"] = df_notional["Interest-rate gap"].cumsum()

#Tabela do portfolio de titulos
titulos = st.checkbox("Ver portfolio sintetico")
if titulos:
    st.subheader("Portfolio sintetico")
    st.dataframe(df_notional)















   


