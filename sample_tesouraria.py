import pandas as pd
import numpy as np
import streamlit as st
import scipy.optimize as optimize
import matplotlib.pyplot as plt






st.title("Simulação de Titulos")

#Geracao das taxas futuras ultilizadas na criacao dos titulos
cdi_gen = np.random.uniform(0, 0.20)

cdif = [cdi_gen]

for i in range(9):
    cdi_gen = cdi_gen * (1 + np.random.uniform(-0.1, 0.1))
    cdif.append(cdi_gen)


t_anos = 10

anos = ["Ano " + str(i+1) for i in range( t_anos )]

cdi = pd.DataFrame({"CDI Futuro" : cdif, "Ano" : anos})
cdi = cdi.set_index("Ano")



#Grafico da curva de juros
st.subheader("Curva de CDI Futuro")

st.line_chart(cdi)




col1, col2 = st.columns(2)
with col1:
    st.subheader("Carteira de titulos ativos")
  
with col2:
    st.subheader("Carteira de titulos passivos")
   


