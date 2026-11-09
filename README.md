# 📦 Controle de Estoque de três produtos — Trabalho 1 · Pesquisa Operacional 2 (UFRJ)

**Alunos:** Bianca Sant'anna e Lucas Tesch · **Prof.** George Victor · EEI744

Programa em Python que, a partir da base **Online Retail (UCI)** traduzida para português
(vendas de 01/12/2010 a 09/12/2011):

1. faz o diagnóstico e a limpeza da base (duplicatas, serviços, ajustes, cancelamentos) e escolhe 3 itens;
2. divide os dados em treino (80% mais antigos) e teste e define as premissas de custo;
3. calcula uma política **⟨s,Q⟩** para um item e uma **⟨R,S⟩ conjunta** para dois itens, com nível de serviço de 99% por ciclo;
4. simula as duas políticas no período de teste (custo total e evolução do estoque);
5. **bônus:** previsão de demanda por suavização exponencial + dimensionamento de lotes por programação inteira mista (HiGHS), com proibição de falta e capacidade de armazenagem.

## Arquivos

| Arquivo | O que é |
|---|---|
| `controle_estoque.py` | Programa do trabalho (versão entregue). Gera tabelas e figuras em `resultados/`. |
| `app.py` | App Streamlit que apresenta cada etapa com gráficos interativos e o código correspondente. |
| `pipeline.py` | Executa as funções de `controle_estoque.py` na mesma ordem do `main()` e devolve os resultados ao app. |
| `dados/` | Onde fica a planilha da base (ver `dados/LEIA-ME.md`). |
| `tools/gerar_base_sintetica.py` | Gera uma base fictícia só para testar o app. |

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# programa original (gera a pasta resultados/)
python controle_estoque.py dados/Online_Retail_PT_BR_Completo.xlsx

# app de apresentação
streamlit run app.py
```

Na primeira execução com a base real o app leva alguns minutos (o abatimento de cancelamentos
percorre todas as linhas); depois o resultado fica em cache.

## Páginas do app

| Página | Etapa do manual | Funções mostradas |
|---|---|---|
| Início | visão geral, premissas | — |
| 1 · Dados e limpeza | tarefa 1 | `carregar`, `diagnosticar`, `preparar_base`, `abater_cancelamentos`, `preparar_periodos` |
| 2 · Treino/teste e seleção | tarefas 1 e 2 | `dias_de_operacao`, `matriz_diaria`, `selecionar_itens` |
| 3 · Demanda e outliers | tarefa 1 | `tratar_outliers` |
| 4 · Custos e parâmetros | tarefas 2 e 3 | `custos_item`, `lote_inteiro`, `desvio_acumulado`, `quantil_empirico`, `ciclo_inteiro` |
| 5 · Simulação no teste | tarefa 4 | `simular_sQ`, `simular_RS`, `ciclos_com_falta` |
| 6 · Bônus | tarefa 7 | `suav_exponencial`, `escolher_previsao`, `dimensionar_lotes` |
| 7 · Código completo | — | todas |

O botão **Modo apresentação** na barra lateral mostra em cada página quem da dupla apresenta aquela parte.

## Referências

CHEN, D. *Online Retail*. UCI Machine Learning Repository, 2015. DOI: 10.24432/C5BW33.
