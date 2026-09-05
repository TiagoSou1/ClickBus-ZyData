import pandas as pd
from datetime import datetime
from pathlib import Path

print("🚀 Início do script de segmentação")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "clickbus_tratado_final.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Leitura dos dados
df = pd.read_csv(DATA_FILE)
df['date_purchase'] = pd.to_datetime(df['date_purchase'], errors='coerce')

# 2. Data de referência para recência (data da última compra geral)
data_ref = df['date_purchase'].max()

# 3. Agregação por cliente (recência, frequência e valor monetário)
clientes_agg = df.groupby('fk_contact').agg(
    recencia = ('date_purchase', lambda x: (data_ref - x.max()).days),
    valor_total_gasto = ('gmv_success', 'sum'),
    valor_medio_por_compra = ('gmv_success', 'mean'),
)
clientes_agg = clientes_agg.reset_index()

# Calcular frequência (número de compras) como quantidade de linhas por cliente
frequencia = df.groupby('fk_contact').size().reset_index(name='frequencia_compras')

# Juntar frequência com os demais dados
clientes = pd.merge(clientes_agg, frequencia, on='fk_contact')

# 4. Cálculo dos scores RFM (5 quintis para cada dimensão)
clientes['R_score'] = pd.qcut(clientes['recencia'], 5, labels=[5,4,3,2,1]).astype(int)  # recência invertida
clientes['F_score'] = pd.qcut(clientes['frequencia_compras'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
clientes['M_score'] = pd.qcut(clientes['valor_total_gasto'], 5, labels=[1,2,3,4,5]).astype(int)

clientes['RFM_score'] = clientes['R_score'].astype(str) + clientes['F_score'].astype(str) + clientes['M_score'].astype(str)

# 5. Classificação inteligente
def classificar_segmento(r, f, m):
    if r == 5 and f == 5 and m >= 4:
        return 'Cliente VIP'
    elif f >= 4 and m <= 2:
        return 'Frequente Econômico'
    elif f <= 2 and m >= 4:
        return 'Gastador Ocasional'
    elif r <= 2:
        return 'Inativo'
    else:
        return 'Cliente Regular'

clientes['segmento_inteligente'] = clientes.apply(
    lambda row: classificar_segmento(row['R_score'], row['F_score'], row['M_score']),
    axis=1
)

# 6. Definição do tipo de viagem (com base nas colunas de origem e destino de ida e volta)
def definir_tipo_viagem(row):
    tem_ida = pd.notna(row['place_origin_departure']) and pd.notna(row['place_destination_departure'])
    tem_volta = pd.notna(row['place_origin_return']) and pd.notna(row['place_destination_return'])
    if tem_ida and tem_volta:
        return 'Ida e Volta'
    elif tem_ida:
        return 'Somente Ida'
    elif tem_volta:
        return 'Somente Volta'
    else:
        return 'Desconhecido'

# 7. Calcula o tipo de viagem para cada compra
df['tipo_viagem'] = df.apply(definir_tipo_viagem, axis=1)

# 8. Tipo de viagem preferido por cliente (moda)
tipo_viagem_preferido = df.groupby('fk_contact')['tipo_viagem'] \
                         .agg(lambda x: x.mode()[0] if not x.mode().empty else 'Desconhecido') \
                         .reset_index()
tipo_viagem_preferido.columns = ['fk_contact', 'tipo_viagem_preferido']

# 9. Merge com a tabela clientes
clientes = clientes.merge(tipo_viagem_preferido, on='fk_contact', how='left')

# 10. Padronizar a chave e exportar o resultado final
clientes = clientes.rename(columns={'fk_contact': 'cliente_id'})
clientes.to_csv(OUTPUT_DIR / 'clientes_segmentados.csv', index=False)

print("✅ Segmentação concluída com sucesso!")
