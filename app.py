import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Carregar os modelos treinados
modelo_p = joblib.load('modelo_prazo.pkl')
modelo_c = joblib.load('modelo_custo.pkl')

st.title("Protótipo: Estimativa de Prototipagem")
st.write("Insira as características da peça para prever Prazo e Custo.")

# 2. Interface de entrada (Ajuste os campos conforme as colunas do seu modelo)
# Exemplo: se o seu modelo usa QUANTITY e Step_Outsourcing, coloque-os aqui
qtde = st.number_input("Quantidade", min_value=1, value=1)
step_outsourcing = st.selectbox("É Outsourcing?", [0, 1]) # 0=Não, 1=Sim
descricao = st.text_area("Descrição Técnica da Peça")

if st.button("Gerar Estimativa"):
    # 3. Criar o DataFrame de entrada (Deve ter exatamente as mesmas colunas do treino)
    # ATENÇÃO: Adicione aqui todas as colunas que o seu modelo X espera
    input_data = pd.DataFrame({
        'QUANTITY': [qtde],
        'Step_Outsourcing': [step_outsourcing]
        # Inclua aqui as outras colunas que aparecem no seu gráfico de importância!
    })
    
    # 4. Previsão
    pred_prazo = np.expm1(modelo_p.predict(input_data))
    pred_custo = np.expm1(modelo_c.predict(input_data))
    
    # 5. Exibição
    st.success(f"Prazo Estimado: {pred_prazo[0]:.1f} dias")
    st.info(f"Custo Estimado: R$ {pred_custo[0]:.2f}")