# ClickBus — Customer Intelligence & Marketing Analytics

Projeto acadêmico em grupo desenvolvido na FIAP para transformar dados de compras em segmentação de clientes, previsões de recompra, recomendação de trechos e regras de campanhas acionáveis.

O case conecta preparação de dados, análise RFM, machine learning e comunicação dos resultados em Power BI.

## Problema de negócio

Uma operação de viagens precisa entender quem são seus clientes, quando existe maior probabilidade de uma nova compra e qual comunicação pode ser mais relevante para cada perfil. O projeto organiza esse problema em três perguntas:

1. Quais clientes apresentam maior recência, frequência e valor?
2. O histórico disponível permite estimar recompra e próximo trecho?
3. Como transformar segmentos e predições em regras de campanha compreensíveis?

## Solução construída

- Normalização de cidades e pseudonimização dos identificadores de clientes.
- Segmentação RFM em cinco perfis operacionais.
- Classificador para recompra em até sete dias.
- Regressores especializados para estimar dias até a próxima compra.
- Classificador do próximo trecho entre as rotas mais frequentes.
- Regras de negócio que combinam segmento e predição em uma campanha sugerida.
- Dashboard de Power BI para exploração dos resultados.

## Fluxo analítico

```text
Dados tratados
      │
      ├── Segmentação RFM ───────────────┐
      │                                  │
      └── Modelos preditivos ────────────┼── Regras de campanha ── Power BI
                                         │
                                         └── Saídas CSV acionáveis
```

## Tecnologias

- Python, Pandas e NumPy
- scikit-learn
- Random Forest para classificação e regressão
- Análise RFM
- Power BI
- Arquivos CSV para entradas e saídas do pipeline

## Estrutura

```text
ClickBus-ZyData/
├── data/
│   ├── README.md
│   └── arquivos CSV locais não versionados
├── outputs/
│   └── resultados gerados localmente
├── src/
│   ├── anonymize_data.py
│   ├── segmentation.py
│   ├── model.py
│   └── campaigns.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Como executar

Pré-requisito: Python 3.10+.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Os dados completos do desafio não são redistribuídos neste repositório. Consulte `data/README.md`, confirme a autorização de uso da fonte e coloque a entrada local no diretório `data/` antes da execução.

Com o ambiente virtual ativado, execute a partir da raiz do repositório:

```bash
python src/anonymize_data.py
python src/segmentation.py
python src/model.py
python src/campaigns.py
```

Os três últimos scripts geram os arquivos analíticos em `outputs/`.

## Entregáveis

- Base de clientes segmentada por RFM.
- Predições de recompra, horizonte e próximo trecho.
- Tabela final com campanhas sugeridas.
- Dashboard para análise de perfis e resultados.

[Acessar o dashboard público no Power BI](https://app.powerbi.com/view?r=eyJrIjoiZDYwZTcwMDUtYTFiNy00OGRmLWI2MmMtNWZmOWZkZDJlOWE5IiwidCI6IjExZGJiZmUyLTg5YjgtNDU0OS1iZTEwLWNlYzM2NGU1OTU1MSIsImMiOjR9)

![Dashboard do projeto no Power BI](https://github.com/user-attachments/assets/30093a0d-4e8d-413e-aa4c-3af749842866)

## Limitações

- A validação atual usa uma separação aleatória estratificada por registro. Uma evolução importante é adotar validação temporal e impedir que registros do mesmo cliente apareçam nos dois conjuntos.
- As regras de campanha são heurísticas demonstrativas; não representam um experimento causal ou uma política pronta para produção.
- O segmento “em risco” é uma proxy baseada em RFM, não um modelo validado de churn.
- O projeto não documenta custos de erro, calibração de probabilidades ou monitoramento de drift.
- A disponibilidade e o uso dos dados continuam sujeitos às regras do desafio acadêmico e da fonte original.

## Próximas melhorias

- Implementar validação temporal e separação por cliente.
- Comparar o Random Forest com baselines mais simples.
- Adicionar métricas por classe e para o classificador de trechos.
- Transformar preparação e modelo em um pipeline testável do scikit-learn.
- Criar testes automatizados para schema, caminhos e regras de campanha.

## Autores

- Bruno de Souza Oliveira
- Daniel Gallo de Almeida Junior
- Ricardo Henrique Ramos Silva
- Rodrigo Silva Oshiro
- Tiago Sousa Leite

Projeto acadêmico desenvolvido na FIAP.
