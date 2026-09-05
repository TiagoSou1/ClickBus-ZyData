import pandas as pd
import numpy as np
from datetime import datetime
from math import sqrt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, roc_auc_score, r2_score
import warnings
warnings.filterwarnings("ignore")

print("🟢 Script com arquitetura hierárquica iniciado...")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "clickbus_tratado_final.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------- CARREGAR E PREPARAR DADOS (Sem alterações) --------------------
try:
    df = pd.read_csv(DATA_FILE)
    print("✅ Arquivo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar o CSV: {e}")
    exit()

df['date_purchase'] = pd.to_datetime(df['date_purchase'])
df = df.sort_values(by=['fk_contact', 'date_purchase'])

# -------------------- FEATURE ENGINEERING (Sem alterações) --------------------
df['tipo_viagem'] = np.where(df['place_origin_return'].notna(), 'Ida e Volta', 'Somente Ida')
bins = [0, 50, 100, 200, 500, float('inf')]
labels = ['0-50', '51-100', '101-200', '201-500', '500+']
df['faixa_gmv'] = pd.cut(df['gmv_success'], bins=bins, labels=labels, right=False)
if df['faixa_gmv'].isnull().any():
    df['faixa_gmv'] = df['faixa_gmv'].cat.add_categories('Desconhecido').fillna('Desconhecido')

df['dias_para_proxima_compra'] = (df.groupby('fk_contact')['date_purchase'].shift(-1) - df['date_purchase']).dt.days
df['target_7_dias'] = (df['dias_para_proxima_compra'] <= 7).fillna(False).astype(int)

df['proximo_trecho'] = df.groupby('fk_contact')['place_origin_departure'].shift(-1).fillna('') + ' → ' + df.groupby('fk_contact')['place_destination_departure'].shift(-1).fillna('')
df['proximo_trecho'] = df['proximo_trecho'].replace(' → ', np.nan)

df['dia_da_semana'] = df['date_purchase'].dt.dayofweek
df['mes'] = df['date_purchase'].dt.month

top_origens = df['place_origin_departure'].value_counts().head(10).index
top_destinos = df['place_destination_departure'].value_counts().head(10).index
df['place_origin_departure'] = df['place_origin_departure'].apply(lambda x: x if x in top_origens else 'Outros')
df['place_destination_departure'] = df['place_destination_departure'].apply(lambda x: x if x in top_destinos else 'Outros')

df = df.dropna(subset=['dias_para_proxima_compra', 'proximo_trecho'])

# -------------------- PREPARAR FEATURES PARA ML (Sem alterações) --------------------
features = [
    'fk_contact', 'gmv_success', 'total_tickets_quantity_success', 'dia_da_semana',
    'mes', 'faixa_gmv', 'tipo_viagem', 'place_origin_departure', 'place_destination_departure'
]

df_ml = df[features + ['dias_para_proxima_compra', 'target_7_dias', 'proximo_trecho']].copy()
df_ml = pd.get_dummies(df_ml, columns=['faixa_gmv', 'tipo_viagem', 'place_origin_departure', 'place_destination_departure'], drop_first=True)

X = df_ml.drop(['fk_contact', 'dias_para_proxima_compra', 'target_7_dias', 'proximo_trecho'], axis=1)
y_bin = df_ml['target_7_dias']
y_reg = df_ml['dias_para_proxima_compra']
y_trecho = df_ml['proximo_trecho'] # Usando o df_ml para garantir alinhamento

top_trechos = y_trecho.value_counts().head(50).index
y_trecho = y_trecho.apply(lambda x: x if x in top_trechos else "Outros")

X = X.fillna(0)

# -------------------- SPLIT INICIAL: TREINO E TESTE --------------------
# Dividimos o conjunto completo uma única vez para ter um teste final "puro"
X_train, X_test, y_bin_train, y_bin_test, y_reg_train, y_reg_test, y_trecho_train, y_trecho_test = \
    train_test_split(X, y_bin, y_reg, y_trecho, test_size=0.2, random_state=42, stratify=y_bin)

# -------------------- TREINAMENTO DOS MODELOS HIERÁRQUICOS --------------------

# --- MODELO 1: O CLASSIFICADOR "GATEKEEPER" ---
print("🚦 Treinando Modelo 1: Classificador Gatekeeper (Compra em 7 dias?)...")
clf_gatekeeper = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_gatekeeper.fit(X_train, y_bin_train)

# --- MODELO 2: O REGRESSOR ESPECIALISTA DE CURTO PRAZO (<= 7 DIAS) ---
print("📈 Treinando Modelo 2: Regressor Especialista de Curto Prazo (dias = 0-7)...")
# Filtramos o treino APENAS para os casos onde a compra REAL foi em até 7 dias
X_train_short_term = X_train[y_bin_train == 1]
y_reg_train_short_term = y_reg_train[y_bin_train == 1]

reg_short_term = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
reg_short_term.fit(X_train_short_term, y_reg_train_short_term)

# --- MODELO 3: O REGRESSOR ESPECIALISTA DE LONGO PRAZO (> 7 DIAS) ---
print("📉 Treinando Modelo 3: Regressor Especialista de Longo Prazo (dias > 7)...")
# Filtramos o treino APENAS para os casos onde a compra REAL foi depois de 7 dias
X_train_long_term = X_train[y_bin_train == 0]
y_reg_train_long_term = y_reg_train[y_bin_train == 0]

reg_long_term = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=20)
reg_long_term.fit(X_train_long_term, y_reg_train_long_term)

# --- MODELO 4: O CLASSIFICADOR DE PRÓXIMO TRECHO ---
print("🚏 Treinando Modelo 4: Classificador de Próximo Trecho...")
trecho_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
trecho_clf.fit(X_train, y_trecho_train) # Treinado no conjunto completo de treino

# -------------------- PIPELINE DE PREDIÇÃO HIERÁRQUICA NO CONJUNTO DE TESTE --------------------
print("\n⚙️ Executando pipeline de predição no conjunto de teste...")

# 1. Usar o Gatekeeper para prever a classificação (0 ou 1) para todo o conjunto de teste
y_bin_pred = clf_gatekeeper.predict(X_test)
y_bin_proba = clf_gatekeeper.predict_proba(X_test)[:, 1]

# 2. Preparar um array vazio para as predições finais da regressão
y_reg_pred_final = pd.Series(index=X_test.index, dtype=float)

# 3. Separar o CONJUNTO DE TESTE com base nas PREDIÇÕES do Gatekeeper
X_test_short_term_idx = X_test[y_bin_pred == 1].index
X_test_long_term_idx = X_test[y_bin_pred == 0].index

# 4. Fazer predições com os modelos especialistas APENAS nos dados apropriados
if not X_test_short_term_idx.empty:
    preds_short = reg_short_term.predict(X_test.loc[X_test_short_term_idx])
    # Garantia extra: Limita a predição entre 0 e 7
    preds_short = np.clip(preds_short, 0, 7)
    y_reg_pred_final.loc[X_test_short_term_idx] = preds_short

if not X_test_long_term_idx.empty:
    preds_long = reg_long_term.predict(X_test.loc[X_test_long_term_idx])
    # Garantia extra: Limita a predição para ser no mínimo 8
    preds_long = np.clip(preds_long, 8, None)
    y_reg_pred_final.loc[X_test_long_term_idx] = preds_long

# 5. Fazer predição do próximo trecho
y_trecho_pred = trecho_clf.predict(X_test)


# -------------------- AVALIAÇÃO DOS RESULTADOS FINAIS --------------------
print("\n📊 Avaliação do Classificador Gatekeeper:")
print(classification_report(y_bin_test, y_bin_pred))
print(f"🎯 AUC-ROC: {roc_auc_score(y_bin_test, y_bin_proba):.4f}")

print("\n📈 Avaliação do Pipeline de Regressão Combinado:")
mae = mean_absolute_error(y_reg_test, y_reg_pred_final)
rmse = sqrt(mean_squared_error(y_reg_test, y_reg_pred_final))
r2 = r2_score(y_reg_test, y_reg_pred_final)
print(f"📉 MAE: {mae:.2f} dias")
print(f"📉 RMSE: {rmse:.2f} dias")
print(f"📈 R²: {r2:.2f}")


# -------------------- SALVAR PREDIÇÕES FINAIS E CONSISTENTES --------------------
print("\n💾 Salvando predições...")
predicoes = pd.DataFrame({
    'cliente_id': df.loc[X_test.index, 'fk_contact'].values,
    'ira_comprar_7_dias': y_bin_pred,
    'dias_ate_proxima_compra': y_reg_pred_final.astype(int),
    'proximo_trecho': y_trecho_pred
})

output_file = OUTPUT_DIR / 'predicoes_clickbus_hierarquico.csv'
predicoes.to_csv(output_file, index=False)
print(f"✅ Arquivo salvo em {output_file}")
