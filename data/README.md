# Dados

Os dados completos do desafio acadêmico não são redistribuídos nesta versão pública do projeto. Os arquivos anteriormente presentes no repositório eram apenas ponteiros incompletos para objetos grandes e não permitiam reproduzir a análise.

## Entradas esperadas

Coloque em `data/` uma cópia cuja utilização e redistribuição estejam autorizadas:

- `clickbus_tratado 1.csv` — entrada do processo de pseudonimização.
- `clickbus_tratado_final.csv` — entrada dos scripts de segmentação e modelagem.

Colunas utilizadas pelo pipeline:

```text
fk_contact
date_purchase
gmv_success
total_tickets_quantity_success
place_origin_departure
place_destination_departure
place_origin_return
place_destination_return
```

## Uso responsável

- Não publique identificadores pessoais ou dados brutos adicionais.
- Confirme as condições do desafio e da fonte antes de usar ou redistribuir a base.
- A troca do identificador por um código sequencial é pseudonimização, não anonimização formal.
- Para demonstrações públicas, prefira uma amostra sintética que preserve apenas o schema necessário.

Os CSVs deste diretório são ignorados pelo Git para reduzir risco de publicação acidental.
