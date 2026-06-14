import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# 1. Configuração da Página
st.set_page_config(page_title="Estimativa de Protótipos", layout="centered")
st.title("🏭 Estimativa de Prototipagem (POC)")
st.write("Insira as especificações da peça para gerar a estimativa de Prazo e Custo através do modelo XGBoost.")

# 2. Carregar os Modelos Treinados
@st.cache_resource # O cache evita que o modelo seja recarregado a cada clique
def carregar_modelos():
    modelo_p = joblib.load('modelo_prazo.pkl')
    modelo_c = joblib.load('modelo_custo.pkl')
    return modelo_p, modelo_c

try:
    xgb_p, xgb_c = carregar_modelos()
except Exception as e:
    st.error("Erro ao carregar os modelos. Certifique-se de que os arquivos .pkl estão no GitHub.")
    st.stop()

# 3. Lista exata de colunas que o modelo exige (Copiado do seu Jupyter Notebook)
COLUNAS_MODELO = [
    'QUANTITY', 'REWORK', 'MOCKUP', 
    'PRODUCT LINE _Design', 'PRODUCT LINE _Fabric Care', 'PRODUCT LINE _Food Preparation', 
    'PRODUCT LINE _Food Preservation', 'PRODUCT LINE _Ownership solutions', 'PRODUCT LINE _SDA', 'PRODUCT LINE _Water Care', 
    'Step_3D Printing', 'Step_CNC', 'Step_Bench', 'Step_Painting', 'Step_Assembly', 'Step_Outsourcing', 
    'MATERIAL_Acrilic', 'MATERIAL_Alumin', 'MATERIAL_Carbon steel', 'MATERIAL_EPS (Inform density in the description)', 
    'MATERIAL_Elastomer (Rubber)', 'MATERIAL_Glass', 'MATERIAL_High density Renshape (blue 1.2 g/cm³)', 
    'MATERIAL_Low density Renshape (yellow 0.3 g/cm³)', 'MATERIAL_MDF', 'MATERIAL_Medium density Renshape (pink 0.7 g/cm³)', 
    'MATERIAL_Nylon', 'MATERIAL_Other (inform material in description)', 'MATERIAL_PETG (additional in the description)', 
    'MATERIAL_PLA (additional in the description)', 'MATERIAL_Polyacetal', 'MATERIAL_Polycarbonate', 
    'MATERIAL_SLA (additional in description)', 'MATERIAL_SLS (additional in description)', 
    'MATERIAL_Stainless steel', 'MATERIAL_polypropylene'
]
# Adicionando as 131 colunas do BERT PCA dinamicamente
for i in range(1, 132):
    COLUNAS_MODELO.append(f'BERT_PCA_{i}')

# 4. Interface de Entrada de Dados (Formulário)
with st.form("form_estimativa"):
    col1, col2 = st.columns(2)
    
    with col1:
        quantidade = st.number_input("Quantidade", min_value=1, value=1, step=1)
        
        linha_produto = st.selectbox("Linha de Produto", [
            "Design", "Fabric Care", "Food Preparation", "Food Preservation", 
            "Ownership solutions", "SDA", "Water Care"
        ])
        
        material = st.selectbox("Material", [
            "Acrilic", "Alumin", "Carbon steel", "EPS (Inform density in the description)", 
            "Elastomer (Rubber)", "Glass", "High density Renshape (blue 1.2 g/cm³)", 
            "Low density Renshape (yellow 0.3 g/cm³)", "MDF", "Medium density Renshape (pink 0.7 g/cm³)", 
            "Nylon", "Other (inform material in description)", "PETG (additional in the description)", 
            "PLA (additional in the description)", "Polyacetal", "Polycarbonate", 
            "SLA (additional in description)", "SLS (additional in description)", 
            "Stainless steel", "polypropylene"
        ])

    with col2:
        rework = st.checkbox("É Retrabalho (Rework)?")
        mockup = st.checkbox("É Mockup?")
        
        etapas = st.multiselect("Etapas do Processo", [
            "3D Printing", "CNC", "Bench", "Painting", "Assembly", "Outsourcing"
        ])
        
    descricao = st.text_area("Descrição Técnica (Contexto adicional para a OS)", 
                             placeholder="Ex: Usinagem de topo de lavadora...")

    submit_button = st.form_submit_button(label="🚀 Calcular Estimativa")

# 5. Processamento e Previsão
if submit_button:
    # Passo A: Criar um dicionário base com todas as colunas zeradas
    dados_entrada = {col: 0 for col in COLUNAS_MODELO}
    
    # Passo B: Preencher com os dados do usuário
    dados_entrada['QUANTITY'] = quantidade
    dados_entrada['REWORK'] = 1 if rework else 0
    dados_entrada['MOCKUP'] = 1 if mockup else 0
    
    # Ativar as categorias corretas (One-Hot Encoding manual)
    chave_linha = f"PRODUCT LINE _{linha_produto}"
    if chave_linha in dados_entrada:
        dados_entrada[chave_linha] = 1
        
    chave_material = f"MATERIAL_{material}"
    if chave_material in dados_entrada:
        dados_entrada[chave_material] = 1
        
    for etapa in etapas:
        chave_etapa = f"Step_{etapa}"
        if chave_etapa in dados_entrada:
            dados_entrada[chave_etapa] = 1
            
    # As 131 colunas do BERT_PCA permanecem 0 (neutras) para otimização da interface web
            
    # Passo C: Converter para DataFrame com as colunas na ordem exata do treino
    df_input = pd.DataFrame([dados_entrada], columns=COLUNAS_MODELO)
    
    # Passo D: Fazer previsões e reverter o Log
    try:
        prazo_previsto = np.expm1(xgb_p.predict(df_input))[0]
        custo_previsto = np.expm1(xgb_c.predict(df_input))[0]
        
        # Limites de segurança estéticos (evitar prever 0.1 dias)
        prazo_previsto = max(1.0, prazo_previsto)
        custo_previsto = max(0.0, custo_previsto)

        st.success("✅ Previsão Concluída!")
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric(label="⏱️ Prazo Estimado", value=f"{prazo_previsto:.1f} dias")
        col_res2.metric(label="💰 Custo Estimado", value=f"R$ {custo_previsto:.2f}")
        
        st.caption("*Nota: O modelo processa as variáveis físicas instantaneamente. O processamento semântico completo via BERT requer execução em backend com aceleração por GPU.*")
        
    except Exception as e:
        st.error(f"Erro ao calcular: {e}")