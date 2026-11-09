Controle de Estoque de três produtos · Pesquisa Operacional (UFRJ)

Programa em Python que, a partir da base **Online Retail (UCI)** traduzida para português
(vendas de 01/12/2010 a 09/12/2011):

1. faz o diagnóstico e a limpeza da base (duplicatas, serviços, ajustes, cancelamentos) e escolhe 3 itens;
2. divide os dados em treino (80% mais antigos) e teste e define as premissas de custo;
3. calcula uma política **⟨s,Q⟩** para um item e uma **⟨R,S⟩ conjunta** para dois itens, com nível de serviço de 99% por ciclo;
4. simula as duas políticas no período de teste (custo total e evolução do estoque);
5. **bônus:** previsão de demanda por suavização exponencial + dimensionamento de lotes por programação inteira mista (HiGHS), com proibição de falta e capacidade de armazenagem.
