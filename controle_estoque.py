
"""
UFRJ- Pesquisa Operacional 2 - Prof. George Victor
Trabalho 1: Controle de Estoque
Alunos: Bianca Sant'anna / Lucas Tesch

Base: Online Retail (UCI), traduzida para PT-BR (vendas de 01/12/2010 a 09/12/2011).

Organizamos o programa na mesma sequência do trabalho:
  1) diagnóstico e pré-tratamento da base, escolha dos 3 itens
  2) divisão treino/teste e premissas de custo
  3) parâmetros <s,Q> (1 item) e <R,S> conjunto (2 itens) para 99%
  4) simulação no período de teste: custo total e gráfico do estoque
  7) bônus: previsão de demanda + dimensionamento de lotes por programação inteira

"""

import re
import sys
import unicodedata
import json
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import milp, LinearConstraint, Bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Valores que adotamos para o exercício
PASTA = Path(__file__).parent
ARQUIVO = Path(sys.argv[1]) if len(sys.argv) > 1 else PASTA / "Online_Retail_PT_BR_Completo.xlsx"
SAIDA = PASTA / "resultados"

#Aqui estamos guardando as configurações em variáveis
FRACAO_TREINO = 0.80
LEAD_TIME = 6                   
NIVEIS_SERVICO = [0.99]  
PTAX_GBP_BRL = 7.1059          
FRACAO_AQUISICAO = 0.60       
TAXA_ARMAZENAGEM = 0.25         
K_INDIVIDUAL = 20.0            
K_CONJUNTO = 20.0              
K_POR_ITEM = 2.5      
SEL_FREQ_MIN = 0.90             
SEL_MEDIA_MIN = 20.0            
SEL_MESES_MIN = 10              
SEL_CONC_MAX = 0.03            
SEL_TENDENCIA = (2 / 3, 1.5)    
LIMITE_IQR = 3.0                

#Aqui configuramos o bônus e os critérios para identificar serviços e analisar descrições
DIAS_POR_PERIODO = 6            
HORIZONTE = 20                  
CAPACIDADE = 2500               
VOLUME_UNITARIO = {"20725": 1.0, "22697": 2.0}   
NS_BONUS = 0.99                 
SERVICOS = {"POST", "DOT", "C2", "M", "D", "BANK CHARGES", "B", "AMAZONFEE", "CRUK", "S",
            "gift_0001_10", "gift_0001_20", "gift_0001_30", "gift_0001_40", "gift_0001_50"}

PADROES = {
    "Possível ajuste": r"\b(adjust\w*|ajust\w*|check\w*|count\w*|contagem|conferir|conferencia|inventario|correcao)\b",
    "Possível avaria": r"\b(damag\w*|broken|avaria\w*|danific\w*|quebrad\w*|molhad\w*|mofad\w*|amassad\w*|rachad\w*|defeito|improprio para venda|descartad\w*|destruid\w*)\b",
    "Possível perda": r"\b(missing|lost|falt\w*|perd\w*|sem estoque|nao encontrado|nao recebido)\b",
}


#Leitura, diagnóstico e limpeza
def carregar(arquivo):
    # Lemos os códigos como texto para não perder letras nem mudar sua identificação.
    df = pd.read_excel(arquivo, dtype={"Numero_Fatura": str, "Codigo_Produto": str})
    df["Numero_Fatura"] = df["Numero_Fatura"].str.strip().str.upper()
    df["Codigo_Produto"] = df["Codigo_Produto"].str.strip()
    return df


def classificar_descricao(descricao):
    if pd.isna(descricao) or not str(descricao).strip():
        return "Descrição vazia"
    # tira acentos para o regex pegar "avaria", "avária" etc.
    txt = "".join(c for c in unicodedata.normalize("NFKD", str(descricao).lower())
                  if not unicodedata.combining(c))
    cats = [nome for nome, p in PADROES.items() if re.search(p, txt)]
    return "; ".join(cats) or "Sem indício"


def diagnosticar(df):
    """Primeiro conferimos a base como ela veio, antes de excluir qualquer registro."""
    canc = df["Numero_Fatura"].str.startswith("C")
    diag = pd.DataFrame({
        "Indicador": ["Registros", "Duplicatas exatas", "Descrições ausentes", "Clientes ausentes",
                      "Linhas de cancelamento (C)", "Quantidade < 0 sem C (ajustes)",
                      "Preço = 0", "Preço < 0"],
        "Valor": [len(df), int(df.duplicated().sum()), int(df["Descricao_Produto"].isna().sum()),
                  int(df["ID_Cliente"].isna().sum()), int(canc.sum()),
                  int(((df["Quantidade"] < 0) & ~canc).sum()),
                  int((df["Preco_Unitario"] == 0).sum()), int((df["Preco_Unitario"] < 0).sum())],
    })
    neg = df[df["Quantidade"] < 0].copy()
    neg["Tipo"] = np.where(neg["Numero_Fatura"].str.startswith("C"), "Com C", "Sem C")
    neg["Indício"] = neg["Descricao_Produto"].map(classificar_descricao)
    auditoria = (neg.groupby(["Tipo", "Indício", "Descricao_Produto"], dropna=False)
                 .size().rename("Linhas").reset_index().sort_values("Linhas", ascending=False))
    return diag, auditoria


def chave_preco(valor):
    return Decimal(str(valor)).normalize()


def abater_cancelamentos(vendas, canc, corte=None):
    """Abate somente quando cliente+produto+preço+data têm uma única venda elegível.
    Associação inferida, não ligação documental com a fatura original. Não distribui
    quantidades entre vendas ambíguas. A versão do treino recebe só dados pré-corte.
    Retorna todas as vendas, inclusive saldo zero, para conservar a rastreabilidade.
    """
    #Guardamos quanto ainda pode ser abatido de cada venda.
    vendas = vendas.copy()
    restantes = vendas["Quantidade"].astype(float).to_dict()
    datas = vendas["Data_Fatura"].to_dict()
    grupos = {}
    for idx, row in vendas.iterrows():
        if pd.notna(row["ID_Cliente"]):
            k = (str(row["ID_Cliente"]), row["Codigo_Produto"], chave_preco(row["Preco_Unitario"]))
            grupos.setdefault(k, []).append(idx)
    registros = []
    for ci, c in canc.sort_values(["Data_Fatura", "Linha_Origem"], kind="stable").iterrows():
        pedido = float(-c["Quantidade"])
        reg = {"linha_cancelamento": int(c["Linha_Origem"]), "fatura_cancelamento": c["Numero_Fatura"],
               "cliente": c["ID_Cliente"], "produto": c["Codigo_Produto"],
               "data_cancelamento": c["Data_Fatura"], "preco_GBP": str(chave_preco(c["Preco_Unitario"])),
               "quantidade_cancelada": pedido, "quantidade_abatida": 0., "linha_venda": None,
               "fatura_venda": None, "data_venda": pd.NaT, "saldo_venda": np.nan,
               "candidatas": 0, "entre_periodos": False}
        if pd.isna(c["ID_Cliente"]):
            reg["status"] = "Cliente ausente"
        elif pedido <= 0:
            reg["status"] = "Quantidade de cancelamento inválida"
        else:
            k = (str(c["ID_Cliente"]), c["Codigo_Produto"], chave_preco(c["Preco_Unitario"]))
            candidatos = [i for i in grupos.get(k, []) if datas[i] <= c["Data_Fatura"] and restantes[i] > 0]
            reg["candidatas"] = len(candidatos)
            #Se houver mais de uma possibilidade, não escolhemos uma venda no chute.
            if not candidatos:
                reg["status"] = "Sem venda anterior elegível"
            elif len(candidatos) > 1:
                reg["status"] = "Múltiplas vendas candidatas"
            elif restantes[candidatos[0]] < pedido:
                reg["status"] = "Saldo insuficiente"
            else:
                i = candidatos[0]
                #Se vendeu 10 e cancelou 3, essa venda passa a contar com 7 unidades.
                restantes[i] -= pedido
                reg.update(status="Associação única inferida", quantidade_abatida=pedido,
                           linha_venda=int(vendas.loc[i, "Linha_Origem"]),
                           fatura_venda=vendas.loc[i, "Numero_Fatura"], data_venda=datas[i],
                           saldo_venda=restantes[i],
                           entre_periodos=bool(corte is not None and datas[i] < corte <= c["Data_Fatura"]))
        registros.append(reg)
    vendas["Quantidade_Original"] = vendas["Quantidade"]
    vendas["Quantidade"] = pd.Series(restantes, dtype=float)
    vendas["Quantidade_Abatida"] = vendas["Quantidade_Original"] - vendas["Quantidade"]
    audit = pd.DataFrame(registros, columns=["linha_cancelamento", "fatura_cancelamento", "cliente", "produto",
        "data_cancelamento", "preco_GBP", "quantidade_cancelada", "quantidade_abatida", "linha_venda",
        "fatura_venda", "data_venda", "saldo_venda", "candidatas", "entre_periodos", "status"])
    #Essas conferências evitam saldo negativo e desconto duplicado de unidades.
    assert (vendas["Quantidade"] >= 0).all()
    assert np.isclose(vendas["Quantidade_Abatida"].sum(), audit["quantidade_abatida"].sum())
    assert np.allclose(vendas["Quantidade_Original"], vendas["Quantidade"] + vendas["Quantidade_Abatida"])
    return vendas, audit


def preparar_base(df):
    log = [("Registros originais", len(df), 0)]
    def etapa(nome, manter):
        nonlocal df
        antes = len(df); df = df[manter].copy()
        log.append((nome, len(df), antes - len(df)))
    etapa("Duplicatas exatas (hipótese de repetição)", ~df.duplicated())
    df["Linha_Origem"] = df.index + 2
    etapa("Serviços e lançamentos (lista explícita)", ~df["Codigo_Produto"].isin(SERVICOS))
    etapa("Data e código válidos", df["Data_Fatura"].notna() & df["Codigo_Produto"].notna())
    canc = df["Numero_Fatura"].str.startswith("C", na=False)
    etapa("Quantidades não positivas sem C", canc | (df["Quantidade"] > 0))
    etapa("Preço unitário <= 0", df["Preco_Unitario"] > 0)
    canc = df["Numero_Fatura"].str.startswith("C", na=False)
    return df[~canc].copy(), df[canc].copy(), pd.DataFrame(log, columns=["Etapa", "Restantes", "Removidas"])


def preparar_periodos(vendas, canc, corte):
    #Aqui fica a correção mais importante: o treino só conhece o que aconteceu até o corte.
    tr, atr = abater_cancelamentos(vendas[vendas["Data_Fatura"] < corte],
                                  canc[canc["Data_Fatura"] < corte], corte)
    #No teste, fazemos uma conferência posterior dos cancelamentos.Se algum deles se refere ao treino, ele fica na auditoria, sem mudar o treino pronto.
    retrospectiva, audit = abater_cancelamentos(vendas, canc, corte)
    te = retrospectiva[retrospectiva["Data_Fatura"] >= corte]
    operacional = pd.concat([tr, te]).sort_index()
    operacional = operacional[operacional["Quantidade"] > 0].copy()
    return operacional, tr, te, audit, atr


#Séries diárias e escolha dos itens
def dias_de_operacao(df):
    #Hipótese: datas com alguma venda positiva na base representam dias de operação.
    #Ausência de transações não comprova fechamento. O calendário é fixado antes dos abatimentos.
    return pd.DatetimeIndex(sorted(df["Data_Fatura"].dt.normalize().unique()))


def matriz_diaria(df, dias):
    #Somamos as unidades por produto e por dia. Cada coluna vira uma data.
    #Se a loja teve vendas no dia, mas não desse produto, colocamos zero para ele.
    m = (df.groupby(["Codigo_Produto", df["Data_Fatura"].dt.normalize()])["Quantidade"].sum()
         .unstack(fill_value=0))
    return m.reindex(columns=dias, fill_value=0).astype(float)


def selecionar_itens(df, m_treino):
    n = m_treino.shape[1]
    fim_treino = m_treino.columns[-1] + pd.Timedelta(days=1)
    meses = (df[df["Data_Fatura"] < fim_treino].groupby("Codigo_Produto")["Data_Fatura"]
             .apply(lambda s: s.dt.to_period("M").nunique()))
    st = pd.DataFrame({
        "media": m_treino.mean(axis=1),
        "cv": m_treino.std(axis=1) / m_treino.mean(axis=1),
        "freq": (m_treino > 0).mean(axis=1),
        "meses": meses,
        "concentracao": m_treino.max(axis=1) / m_treino.sum(axis=1),
        "tendencia": m_treino.iloc[:, n // 2:].mean(axis=1) / m_treino.iloc[:, :n // 2].mean(axis=1),
    })
    st["descricao"] = df[df["Data_Fatura"] < fim_treino].groupby("Codigo_Produto")["Descricao_Produto"].agg(lambda s: s.mode()[0])
    ok = ((st["freq"] >= SEL_FREQ_MIN) & (st["media"] >= SEL_MEDIA_MIN)
          & (st["meses"] >= SEL_MESES_MIN) & (st["concentracao"] <= SEL_CONC_MAX)
          & st["tendencia"].between(*SEL_TENDENCIA))
    ranking = st[ok].sort_values("cv")
    if len(ranking) < 3:
        raise ValueError("Menos de três produtos elegíveis; reveja os critérios, sem usar o teste.")
    return ranking, ranking.index[0], list(ranking.index[1:3])


def tratar_outliers(serie, n_treino):
    """Limitamos os picos do treino em Q3 + 3*IQR, como escolha de tratamento.
    Isso não quer dizer que o pico seja um erro. O teste fica sem esse limite.
    Assim conseguimos ver como a política reage quando aparece uma venda grande.
    """
    tr = serie.iloc[:n_treino]
    q1, q3 = tr.quantile([0.25, 0.75])
    limite = q3 + LIMITE_IQR * (q3 - q1)
    tratada = pd.concat([tr.clip(upper=limite), serie.iloc[n_treino:]])
    return tratada, limite, int((tr > limite).sum())


#Parâmetros das políticas
def custos_item(df, item, dias_ano):
    #A base tem o preço de venda, não os custos da loja. Por isso usamos hipóteses.
    #Este df já deve conter somente as vendas do treinamento.
    p = df.loc[df["Codigo_Produto"] == item, "Preco_Unitario"].median()
    c = FRACAO_AQUISICAO * p
    return {"preco": p, "c": c, "h_ano": TAXA_ARMAZENAGEM * c,
            "h_dia": TAXA_ARMAZENAGEM * c / dias_ano, "falta": p - c}


def desvio_acumulado(serie_treino, janela):
    #Desvio da soma de 'janela' dias seguidos, medido direto no treino.
    #Preserva o efeito da dependência temporal, sem assumir que ela seja positiva.
    return serie_treino.rolling(janela).sum().dropna().std()


def quantil_empirico(serie_treino, janela, ns):
    #Somamos as vendas de cada janela e pegamos um valor que cubra 99% delas.
    #A opção higher escolhe um valor observado acima da posição do percentil.
    return serie_treino.rolling(janela).sum().dropna().quantile(ns, interpolation="higher")


def lote_inteiro(Q_real, mu, K, h_dia):
    #custo por dia K*mu/Q + h*Q/2, testado nos dois inteiros em volta do ótimo contínuo
    cands = {max(1, int(np.floor(Q_real))), max(1, int(np.ceil(Q_real)))}
    return min(cands, key=lambda q: K * mu / q + h_dia * q / 2)


def ciclo_inteiro(T_real, K_total, soma_h_mu):
    cands = {max(1, int(np.floor(T_real))), int(np.ceil(T_real))}
    return min(cands, key=lambda r: K_total / r + r * soma_h_mu / 2)

'''
#Simulação no período de teste
#Para as duas políticas, seguimos a mesma ordem de acontecimentos:
a cada dia: chega o que estava previsto, vende-se o possível e o resto é venda perdida; 
no fim do dia olha-se a posição de estoque e decide-se se pede; 
um pedido feito no fim do dia t fica disponível no dia t + L + 1, ou seja, a demanda de L dias inteiros acontece enquanto ele está a caminho;
começamos com um estoque calculado para representar um recebimento;
isso é uma condição inicial assumida, não um estoque observado na planilha;
'''

def ciclos_com_falta(perdas, chegadas):
    """Contamos só os ciclos que chegaram ao fim dentro do teste.
    O estoque inicial representa um recebimento em t=0. O trecho que sobrar
    depois do último recebimento fica separado, porque o teste acaba antes dele.
    """
    T = len(perdas)
    limites = [0] + sorted(set(d for d in chegadas if 0 < d <= T))
    completos = [perdas[a:b] for a, b in zip(limites[:-1], limites[1:])]
    dias_parciais = T - limites[-1]
    return {"ciclos_completos": len(completos),
            "ciclos_completos_com_falta": sum(bool(c.sum() > 0) for c in completos),
            "dias_ciclo_incompleto": dias_parciais,
            "perda_ciclo_incompleto": float(perdas[limites[-1]:].sum())}


def simular_sQ(demanda, s, Q, mu, cu):
    # s diz quando pedir; Q diz quantas unidades comprar em cada lote.
    T = len(demanda)
    estoque = np.zeros(T)
    perda = np.zeros(T)
    pedido = np.zeros(T)
    a_caminho = []                       
    I = round(s - mu * LEAD_TIME + Q)    
    inicial = I
    recebimentos = np.zeros(T)
    chegadas = []
    for t in range(T):
        recebimentos[t] = sum(q for d, q in a_caminho if d == t)
        I += recebimentos[t]
        a_caminho = [(d, q) for d, q in a_caminho if d != t]
        venda = min(I, demanda[t])
        perda[t] = demanda[t] - venda
        I -= venda
        # Não basta olhar o estoque físico: também contamos o que já foi pedido.
        posicao = I + sum(q for _, q in a_caminho)
        if posicao <= s:
            n_lotes = int(np.floor((s - posicao) / Q)) + 1
            pedido[t] = n_lotes * Q
            a_caminho.append((t + LEAD_TIME + 1, pedido[t]))
            chegadas.append(t + LEAD_TIME + 1)
        estoque[t] = I
    n_pedidos = int((pedido > 0).sum())
    custos = {
        "fixo": K_INDIVIDUAL * n_pedidos,
        "variavel": cu["c"] * pedido.sum(),
        "armazenagem": cu["h_dia"] * estoque.sum(),
        "falta": cu["falta"] * perda.sum(),
    }
    assert np.isclose(inicial + recebimentos.sum() - (np.sum(demanda) - perda.sum()), estoque[-1])
    return {"estoque": estoque, "perda": perda, "pedido": pedido, "recebimentos": recebimentos,
            "inicial": inicial, "final": I, "pendente": sum(q for _, q in a_caminho),
            "aquisicao_inicial": cu["c"] * inicial,
            "n_pedidos": n_pedidos, "chegadas": chegadas, "custos": custos}


def simular_RS(demandas, R, S, mus, cus):
    """A cada R dias, conferimos os dois produtos e completamos a posição até S.
    O custo fixo comum entra uma vez, e não uma vez para cada produto.
    """
    itens = list(demandas)
    T = len(demandas[itens[0]])
    estoque = {i: np.zeros(T) for i in itens}
    perda = {i: np.zeros(T) for i in itens}
    pedido = {i: np.zeros(T) for i in itens}
    a_caminho = {i: [] for i in itens}
    I = {i: round(S[i] - mus[i] * LEAD_TIME) for i in itens}    
    inicial = I.copy()
    recebimentos = {i: np.zeros(T) for i in itens}
    revisoes = set(range(R - LEAD_TIME - 1, T, R))
    fixo = 0.0
    chegadas = []
    for t in range(T):
        for i in itens:
            recebimentos[i][t] = sum(q for d, q in a_caminho[i] if d == t)
            I[i] += recebimentos[i][t]
            a_caminho[i] = [(d, q) for d, q in a_caminho[i] if d != t]
            venda = min(I[i], demandas[i][t])
            perda[i][t] = demandas[i][t] - venda
            I[i] -= venda
        if t in revisoes:
            pediu = 0
            for i in itens:
                posicao = I[i] + sum(q for _, q in a_caminho[i])
                q = max(0.0, S[i] - posicao)
                if q > 0:
                    pedido[i][t] = q
                    a_caminho[i].append((t + LEAD_TIME + 1, q))
                    pediu += 1
            if pediu:
                fixo += K_CONJUNTO + K_POR_ITEM * pediu
                chegadas.append(t + LEAD_TIME + 1)
        for i in itens:
            estoque[i][t] = I[i]
    custos = {
        "fixo": fixo,
        "variavel": sum(cus[i]["c"] * pedido[i].sum() for i in itens),
        "armazenagem": sum(cus[i]["h_dia"] * estoque[i].sum() for i in itens),
        "falta": sum(cus[i]["falta"] * perda[i].sum() for i in itens),
    }
    for i in itens:
        assert np.isclose(inicial[i] + recebimentos[i].sum() - (sum(demandas[i]) - perda[i].sum()), estoque[i][-1])
    return {"estoque": estoque, "perda": perda, "pedido": pedido, "recebimentos": recebimentos,
            "inicial": inicial, "final": I.copy(),
            "pendente": {i: sum(q for _, q in a_caminho[i]) for i in itens},
            "aquisicao_inicial": sum(cus[i]["c"] * inicial[i] for i in itens),
            "n_pedidos": len(chegadas), "chegadas": chegadas, "custos": custos}


#Bônus: previsão de demanda e dimensionamento de lotes
def em_periodos(serie, dias_por_periodo, do_fim=True):
    """Soma a série diária em blocos de 'dias_por_periodo' dias de operação.
    Para o treino os blocos são contados a partir do fim (os dias que sobram
    ficam no começo e são descartados); para o teste, a partir do início."""
    v = serie.values
    n = len(v) // dias_por_periodo
    v = v[len(v) - n * dias_por_periodo:] if do_fim else v[:n * dias_por_periodo]
    return v.reshape(n, dias_por_periodo).sum(axis=1)


def suav_exponencial(y, alpha, beta=None, phi=1.0, passos=1):
    #Devolve a previsão um passo à frente para cada ponto e a previsão para 'passos' à frente
    nivel, tend = y[0], (y[1] - y[0] if beta is not None else 0.0)
    ajuste = []
    for obs in y:
        ajuste.append(nivel + phi * tend)
        novo_nivel = alpha * obs + (1 - alpha) * (nivel + phi * tend)
        if beta is not None:
            tend = beta * (novo_nivel - nivel) + (1 - beta) * phi * tend
        nivel = novo_nivel
    futuro = [nivel + sum(phi ** k for k in range(1, h + 1)) * tend for h in range(1, passos + 1)]
    return np.array(ajuste), np.array(futuro)


def escolher_previsao(y, n_valid=8):
    """Separamos o fim do treino para comparar as previsões.
    Escolhemos o menor MAE: a média da diferença absoluta entre previsto e observado (o teste não participa dessa escolha).
    """
    base, valid = y[:-n_valid], y[-n_valid:]
    candidatos = []
    for a in np.arange(0.05, 0.55, 0.05):
        candidatos.append((f"SES a={a:.2f}", dict(alpha=a)))
        for b in (0.05, 0.1, 0.2):
            candidatos.append((f"Holt amortecido a={a:.2f} b={b:.2f}", dict(alpha=a, beta=b, phi=0.9)))
    melhor = None
    for nome, par in candidatos:
        _, prev = suav_exponencial(base, passos=n_valid, **par)
        mae = np.abs(prev - valid).mean()
        if melhor is None or mae < melhor[2]:
            melhor = (nome, par, mae)
    #média móvel de 8 períodos como referência
    mae_mm = np.abs(base[-8:].mean() - valid).mean()
    if mae_mm < melhor[2]:
        melhor = ("Média móvel 8", None, mae_mm)
    return melhor


def prever(y, modelo, passos):
    _, par, _ = modelo
    if par is None:
        ajuste = pd.Series(y).rolling(8).mean().shift(1).bfill().values
        return ajuste, np.full(passos, y[-8:].mean())
    return suav_exponencial(y, passos=passos, **par)


def dimensionar_lotes(D, est_min, est_inicial, c, h, volume, K0, k, cap):
    """Programação inteira mista para o ressuprimento conjunto, sem falta e com
    capacidade de armazenagem. D[i][t] é a demanda prevista do item i no período t.
    # Aqui a previsão já está pronta. Agora o modelo decide como abastecer.
    # x = unidades recebidas; I = estoque; z = incluir o item; y = pedido conjunto.
    """
    itens = list(D)
    n, T = len(itens), len(D[itens[0]])
    ix = lambda i, t: i * T + t
    iI = lambda i, t: n * T + i * T + t
    iz = lambda i, t: 2 * n * T + i * T + t
    iy = lambda t: 3 * n * T + t
    nv = 3 * n * T + T

    # Cada variável recebe o custo que ela acrescenta ao objetivo do modelo.
    custo = np.zeros(nv)
    for a, it in enumerate(itens):
        for t in range(T):
            custo[ix(a, t)] = c[it]
            custo[iI(a, t)] = h[it]
            custo[iz(a, t)] = k
    for t in range(T):
        custo[iy(t)] = K0

    linhas, lb, ub = [], [], []

    def nova(coefs, lo, hi):
        r = np.zeros(nv)
        for j, v in coefs:
            r[j] += v
        linhas.append(r); lb.append(lo); ub.append(hi)

    for a, it in enumerate(itens):
        for t in range(T):
            M = D[it][t:].sum() + est_min[it]
            coefs = [(iI(a, t), 1), (ix(a, t), -1)]
            rhs = -D[it][t]
            if t > 0:
                coefs.append((iI(a, t - 1), -1))
            else:
                rhs += est_inicial[it]
            nova(coefs, rhs, rhs)
            nova([(ix(a, t), 1), (iz(a, t), -M)], -np.inf, 0)
            nova([(iz(a, t), 1), (iy(t), -1)], -np.inf, 0)
    # Conferimos o espaço logo depois da chegada, quando o estoque está maior.
    for t in range(T):
        coefs, rhs = [], cap
        for a, it in enumerate(itens):
            coefs.append((ix(a, t), volume[it]))
            if t > 0:
                coefs.append((iI(a, t - 1), volume[it]))
            else:
                rhs -= volume[it] * est_inicial[it]
        nova(coefs, -np.inf, rhs)

    lo = np.zeros(nv)
    hi = np.full(nv, np.inf)
    if LEAD_TIME % DIAS_POR_PERIODO:
        raise ValueError("No bônus, o lead time deve corresponder a períodos completos.")
    atraso = LEAD_TIME // DIAS_POR_PERIODO

    #Não há pedidos em trânsito no instante inicial. O estoque inicial deve
    for a in range(n):
        for t in range(min(atraso, T)):
            hi[ix(a, t)] = 0.
    for a, it in enumerate(itens):
        for t in range(T):
            lo[iI(a, t)] = est_min[it]
    hi[2 * n * T:] = 1
    integral = np.zeros(nv)
    integral[2 * n * T:] = 1
    integral[:n * T] = 1      

#Buscamos os lotes de menor custo, conferimos as restrições e organizamos os resultados.
    res = milp(custo, constraints=LinearConstraint(np.array(linhas), lb, ub),
               bounds=Bounds(lo, hi), integrality=integral, options={"time_limit": 60, "mip_rel_gap": 1e-8})
    if res.x is None:
        raise RuntimeError(f"MILP sem solução: {res.message}")
    x = {it: np.round(res.x[[ix(a, t) for t in range(T)]]) + 0.0 for a, it in enumerate(itens)}
    I = {it: np.round(res.x[[iI(a, t) for t in range(T)]]) + 0.0 for a, it in enumerate(itens)}
    y = np.round(res.x[[iy(t) for t in range(T)]])
    # Verifica solução nas restrições antes de arredondar variáveis inteiras.
    Ax = np.array(linhas) @ res.x
    if not (np.all(Ax >= np.array(lb) - 1e-5) and np.all(Ax <= np.array(ub) + 1e-5)):
        raise RuntimeError("Solução MILP não respeita as restrições numéricas.")
    return {"x": x, "I": I, "y": y, "custo": res.fun, "status": res.message,
            "otimo": res.status == 0, "gap": float(getattr(res, "mip_gap", np.nan)),
            "atraso_periodos": atraso}


#Gráficos
def fig_demanda(series, n, itens, item_sQ, est, corte):
    fig, axs = plt.subplots(3, 1, figsize=(10, 4.3), sharex=True)
    for ax, it in zip(axs, itens):
        s = series[it]
        ax.plot(s.index[:n], s.values[:n], lw=0.7, color="#1f77b4")
        ax.plot(s.index[n - 1:], s.values[n - 1:], lw=0.7, color="#ff7f0e")
        ax.axvline(corte, color="0.3", ls="--", lw=0.8)
        pol = "<s,Q>" if it == item_sQ else "<R,S>"
        ax.set_title(f"{it} – {est.loc[it, 'descricao']} {pol}", fontsize=9, loc="left")
        ax.set_ylabel("un/dia", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(SAIDA / "fig1_demanda.png", dpi=200)
    plt.close(fig)


def fig_estoque(sims, datas, itens, item_sQ, par, est):
    #Evolução do estoque físico no teste, somente nível de serviço de 99%
    cores = {0.99: "#1f4e79"}
    fig, axs = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True)
    for ax, it in zip(axs, itens):
        for ns in NIVEIS_SERVICO:
            sim = sims[ns]
            if it == item_sQ:
                e, p = sim["sQ"]["estoque"], sim["sQ"]["perda"]
            else:
                e, p = sim["RS"]["estoque"][it], sim["RS"]["perda"][it]
            ax.step(datas, e, where="post", lw=1.2, color=cores[ns], label=f"estoque {ns:.0%}")
            dias_falta = datas[p > 0]
            if len(dias_falta):
                ax.plot(dias_falta, np.zeros(len(dias_falta)), "v", color="#c0392b", ms=4,
                        label="dia com venda perdida" if ns == NIVEIS_SERVICO[-1] else None)
        lin = par[(par["item"] == it) & (par["ns"] == 0.99)].iloc[0]
        nome = "s" if it == item_sQ else "S"
        ax.axhline(lin["s_ou_S"], color="0.5", ls=":", lw=0.9)
        topo = max(ax.get_ylim()[1], lin["s_ou_S"]) * 1.12
        ax.set_ylim(-0.04 * topo, topo)
        ax.text(datas[len(datas) // 2], lin["s_ou_S"], f"{nome} (99%) = {lin['s_ou_S']:,.0f}".replace(",", "."),
                fontsize=7, va="bottom", ha="center", color="0.35")
        pol = "<s,Q>" if it == item_sQ else "<R,S>"
        ax.set_title(f"{it} – {est.loc[it, 'descricao']} {pol}", fontsize=9, loc="left")
        ax.set_ylabel("unidades", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0].legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(SAIDA / "fig2_estoque_simulado.png", dpi=200)
    plt.close(fig)


def fig_bonus(hist, prev, real_teste, plano, itens, est):
    fig, axs = plt.subplots(len(itens) + 1, 1, figsize=(10, 6.4))
    for ax, it in zip(axs, itens):
        nh = len(hist[it])
        ax.plot(range(-nh + 1, 1), hist[it], color="0.45", lw=1, label="histórico (treino)")
        ax.plot(range(1, len(prev[it]) + 1), prev[it], color="#1f77b4", lw=1.4, label="previsão")
        ax.plot(range(1, len(real_teste[it]) + 1), real_teste[it], "o-", color="#ff7f0e", ms=3, lw=1,
                label="real (teste)")
        ax.axvline(0.5, color="0.3", ls="--", lw=0.8)
        ax.set_title(f"{it} – {est.loc[it, 'descricao']}: demanda por período (6 dias)", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    axs[0].legend(fontsize=7, ncol=3)
    ax = axs[-1]
    T = len(plano["y"])
    base = np.zeros(T)
    for it, cor in zip(itens, ["#9ecae1", "#3182bd"]):
        ocup = VOLUME_UNITARIO[it] * (np.r_[plano["I0"][it], plano["I"][it][:-1]] + plano["x"][it])
        ax.bar(range(1, T + 1), ocup, bottom=base, color=cor, label=f"{it} (após recebimento)")
        base += ocup
    ax.axhline(CAPACIDADE, color="#c0392b", ls="--", lw=1, label="capacidade")
    ax.set_title("Plano de lotes: ocupação do depósito por período (unid.-equivalentes)", fontsize=9, loc="left")
    ax.set_xlabel("período futuro", fontsize=8)
    ax.set_xticks(range(1, T + 1))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(SAIDA / "fig3_bonus_lotes.png", dpi=200)
    plt.close(fig)


#execução
def main():
    SAIDA.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 45)
    csv = dict(sep=";", encoding="utf-8-sig")

    print(f"Lendo {ARQUIVO.name} ...")
    bruto = carregar(ARQUIVO)
    diag, auditoria = diagnosticar(bruto)
    print("\n=== DIAGNÓSTICO DA BASE BRUTA ===\n" + diag.to_string(index=False))
    print("\nIndícios nas descrições de quantidades negativas:")
    print(auditoria.groupby(["Tipo", "Indício"])["Linhas"].sum().to_string())
    diag.to_csv(SAIDA / "diagnostico.csv", index=False, **csv)
    auditoria.to_csv(SAIDA / "auditoria_quantidades_negativas.csv", index=False, **csv)
    vendas, canc, log = preparar_base(bruto)
    dias = dias_de_operacao(vendas)
    n = int(FRACAO_TREINO * len(dias))
    corte = dias[n]
    df, treino_saldos, teste_saldos, audit, audit_tr = preparar_periodos(vendas, canc, corte)
    treino_df = treino_saldos[treino_saldos["Quantidade"] > 0]
    dias_ano = n * 365.25 / ((dias[n - 1] - dias[0]).days + 1)
    print("\n=== PRÉ-TRATAMENTO ===\n" + log.to_string(index=False))
    print("\nCancelamentos: associações inferidas; ambiguidades não são abatidas.")
    resumo_cancel = audit.groupby("status").agg(linhas=("quantidade_cancelada", "size"),
                    unidades_canceladas=("quantidade_cancelada", "sum"), unidades_abatidas=("quantidade_abatida", "sum"))
    print(resumo_cancel.to_string())
    print(f"Treino: {len(treino_df)} linhas positivas ajustadas; teste: {int((teste_saldos.Quantidade > 0).sum())}.")
    log.to_csv(SAIDA / "limpeza.csv", index=False, **csv)
    audit.to_csv(SAIDA / "auditoria_cancelamentos.csv", index=False, **csv)
    audit_tr.to_csv(SAIDA / "auditoria_cancelamentos_treino.csv", index=False, **csv)
    audit[audit["status"] != "Associação única inferida"].to_csv(SAIDA / "cancelamentos_nao_resolvidos.csv", index=False, **csv)
    resumo_cancel.to_csv(SAIDA / "resumo_cancelamentos.csv", **csv)
    vendas_exportar = pd.concat([treino_saldos.assign(fase="treino_congelado"), teste_saldos.assign(fase="teste_retrospectivo")])
    
    # O preço original fica em libra. A coluna extra mostra a conversão em reais.
    vendas_exportar["Preco_Unitario_BRL"] = vendas_exportar["Preco_Unitario"].map(
        lambda valor: float(Decimal(str(valor)) * Decimal(str(PTAX_GBP_BRL))))
    vendas_exportar.to_csv(SAIDA / "vendas_ajustadas_auditoria.csv", index=False, **csv)
    print(f"\n{len(dias)} datas com vendas (calendário assumido); {dias_ano:.1f} dias/ano estimados só no treino.")
    print(f"Treino: {n} dias até {dias[n - 1].date()} | Teste: {len(dias) - n} dias desde {corte.date()}")
    m = matriz_diaria(df, dias)
    ranking, item_sQ, itens_RS = selecionar_itens(treino_df, m.iloc[:, :n])
    print(f"\n=== SELEÇÃO (só treino): {len(ranking)} itens passaram nos filtros ===")
    print(ranking.round(3).to_string())
    ranking.round(4).to_csv(SAIDA / "ranking_selecao.csv", **csv)
    itens = [item_sQ] + itens_RS
    print(f"<s,Q>: {item_sQ} | <R,S>: {itens_RS}")

    # Depois de escolher os produtos, tratamos seus picos só no treinamento.
    series, est = {}, {}
    for it in itens:
        s, lim, nout = tratar_outliers(m.loc[it], n)
        series[it] = s
        tr, te = s.iloc[:n], s.iloc[n:]
        est[it] = {"descricao": ranking.loc[it, "descricao"], "media": tr.mean(), "dp": tr.std(),
                   "cv": tr.std() / tr.mean(), "media_teste": te.mean(),
                   "limite_outlier": lim, "dias_ajustados": nout}
    est = pd.DataFrame(est).T
    custos = {it: custos_item(treino_df, it, dias_ano) for it in itens}
    for it in itens:
        for k_ in ["preco", "c", "h_dia", "falta"]:
            est.loc[it, k_] = custos[it][k_]
            est.loc[it, k_ + "_BRL"] = custos[it][k_] * PTAX_GBP_BRL

    # Q e R equilibram custo de pedir e de armazenar
    # s e S acrescentam a proteção para a variação das vendas
    linhas = []
    mu, cu, tr = est.loc[item_sQ, "media"], custos[item_sQ], series[item_sQ].iloc[:n]
    Q = lote_inteiro(np.sqrt(2 * K_INDIVIDUAL * mu / cu["h_dia"]), mu, K_INDIVIDUAL, cu["h_dia"])
    sig_L = desvio_acumulado(tr, LEAD_TIME)
    for ns in NIVEIS_SERVICO:
        z = norm.ppf(ns)
        linhas.append({"item": item_sQ, "politica": "<s,Q>", "ns": ns, "z": z, "Q_ou_R": Q,
                       "sigma_protecao": sig_L, "ES": z * sig_L,
                       "s_ou_S": int(np.ceil(max(mu * LEAD_TIME + z * sig_L, quantil_empirico(tr, LEAD_TIME, ns)))),
                       "quantil_empirico": int(np.ceil(quantil_empirico(tr, LEAD_TIME, ns)))})

    K_total = K_CONJUNTO + K_POR_ITEM * len(itens_RS)
    soma_h_mu = sum(custos[i]["h_dia"] * est.loc[i, "media"] for i in itens_RS)
    T_real = np.sqrt(2 * K_total / soma_h_mu)
    R = ciclo_inteiro(T_real, K_total, soma_h_mu)
    print(f"\nCiclo conjunto ótimo T* = {T_real:.1f} dias -> R = {R} dias")
    for it in itens_RS:
        mu_i, tr_i = est.loc[it, "media"], series[it].iloc[:n]
        sig = desvio_acumulado(tr_i, R + LEAD_TIME)
        for ns in NIVEIS_SERVICO:
            z = norm.ppf(ns)
            linhas.append({"item": it, "politica": "<R,S>", "ns": ns, "z": z, "Q_ou_R": R,
                           "sigma_protecao": sig, "ES": z * sig,
                           "s_ou_S": int(np.ceil(max(mu_i * (R + LEAD_TIME) + z * sig, quantil_empirico(tr_i, R + LEAD_TIME, ns)))),
                           "quantil_empirico": int(np.ceil(quantil_empirico(tr_i, R + LEAD_TIME, ns)))})
    # Usamos o maior entre a aproximação normal e o percentil observado.
    # Isso cobre pelo menos 99% das janelas do treino, sem prometer o mesmo no teste.
    par = pd.DataFrame(linhas)
    for j, row in par.iterrows():
        janela = LEAD_TIME if row["politica"] == "<s,Q>" else R + LEAD_TIME
        acumulada = series[row["item"]].iloc[:n].rolling(janela).sum().dropna()
        par.loc[j, "cobertura_empirica_treino"] = (acumulada <= row["s_ou_S"]).mean()
        assert par.loc[j, "cobertura_empirica_treino"] >= row["ns"] - 1e-12

    print("\n=== ESTATÍSTICAS E CUSTOS (treino, após outliers) ===")
    print(est.drop(columns="descricao").astype(float).round(4).to_string())
    print("\n=== PARÂMETROS ===\n" + par.round(2).to_string(index=False))
    est.to_csv(SAIDA / "estatisticas_custos.csv", **csv)
    par.round(4).to_csv(SAIDA / "parametros.csv", index=False, **csv)
    pd.DataFrame(series).assign(fase=["treino"] * n + ["teste"] * (len(dias) - n)) \
        .to_csv(SAIDA / "demanda_diaria_tratada.csv", index_label="data", **csv)
    fig_demanda(series, n, itens, item_sQ, est, corte)

    #simulação no teste
    # Agora deixamos os parâmetros fixos e simulamos o que aconteceu no teste.
    datas_teste = dias[n:]
    dem = {it: series[it].iloc[n:].values for it in itens}
    sims, resumo, diario = {}, [], []
    for ns in NIVEIS_SERVICO:
        p = par[par["ns"] == ns].set_index("item")
        sq = simular_sQ(dem[item_sQ], p.loc[item_sQ, "s_ou_S"], Q, est.loc[item_sQ, "media"], custos[item_sQ])
        rs = simular_RS({i: dem[i] for i in itens_RS}, R, {i: p.loc[i, "s_ou_S"] for i in itens_RS},
                        {i: est.loc[i, "media"] for i in itens_RS}, {i: custos[i] for i in itens_RS})
        sims[ns] = {"sQ": sq, "RS": rs}

        for nome, sim, its in [("<s,Q>", sq, [item_sQ]), ("<R,S>", rs, itens_RS)]:
            # no <R,S> as perdas vêm por item; somadas dão a perda diária do sistema
            perda_dia = sim["perda"] if nome == "<s,Q>" else sum(sim["perda"][i] for i in its)
            perda_total = perda_dia.sum()
            dem_total = sum(dem[i].sum() for i in its)
            ciclos = ciclos_com_falta(perda_dia, sim["chegadas"])
            c_ = sim["custos"]
            resumo.append({"sistema": nome, "itens": "+".join(its), "ns_planejado": ns,
                           "pedidos": sim["n_pedidos"],
                           "custo_fixo": c_["fixo"], "custo_variavel": c_["variavel"],
                           "custo_armazenagem": c_["armazenagem"], "custo_falta": c_["falta"],
                           "custo_total": sum(c_.values()),
                           "custo_total_BRL": sum(c_.values()) * PTAX_GBP_BRL,
                           "demanda": dem_total, "vendas_perdidas": perda_total,
                           "fill_rate": 1 - perda_total / dem_total,
                           "maior_perda_em_um_dia": perda_dia.max(),
                           "ciclos": ciclos["ciclos_completos"],
                           "ciclos_sem_falta": ciclos["ciclos_completos"] - ciclos["ciclos_completos_com_falta"],
                           "dias_ciclo_incompleto": ciclos["dias_ciclo_incompleto"],
                           "perda_ciclo_incompleto": ciclos["perda_ciclo_incompleto"],
                           "aquisicao_estoque_inicial": sim["aquisicao_inicial"],
                           "total_com_aquisicao_inicial": sum(c_.values()) + sim["aquisicao_inicial"]})

        for t, d in enumerate(datas_teste):
            linha = {"data": d.date(), "ns": ns, f"{item_sQ}_demanda": dem[item_sQ][t],
                     f"{item_sQ}_estoque": sq["estoque"][t], f"{item_sQ}_perda": sq["perda"][t],
                     f"{item_sQ}_pedido": sq["pedido"][t], f"{item_sQ}_recebimento": sq["recebimentos"][t]}
            for i in itens_RS:
                linha.update({f"{i}_demanda": dem[i][t], f"{i}_estoque": rs["estoque"][i][t],
                              f"{i}_perda": rs["perda"][i][t], f"{i}_pedido": rs["pedido"][i][t], f"{i}_recebimento": rs["recebimentos"][i][t]})
            diario.append(linha)

    resumo = pd.DataFrame(resumo)
    print("\n=== SIMULAÇÃO NO TESTE (custos em £) ===")
    print(resumo.round(3).to_string(index=False))
    resumo.round(4).to_csv(SAIDA / "simulacao_resumo.csv", index=False, **csv)
    pd.DataFrame(diario).to_csv(SAIDA / "simulacao_diaria.csv", index=False, **csv)
    fig_estoque(sims, datas_teste, itens, item_sQ, par, est)
    estados = []
    for ns, si in sims.items():
        for i in itens:
            ss = si["sQ"] if i == item_sQ else si["RS"]
            estados.append({"item": i, "ns": ns,
                "inicial": ss["inicial"] if i == item_sQ else ss["inicial"][i],
                "final": ss["final"] if i == item_sQ else ss["final"][i],
                "em_transito_final": ss["pendente"] if i == item_sQ else ss["pendente"][i]})
    pd.DataFrame(estados).to_csv(SAIDA / "estados_estoque.csv", index=False, **csv)

    #previsão e dimensionamento de lotes 
    print("\n=== BÔNUS: PREVISÃO E DIMENSIONAMENTO DE LOTES ===")
    # No bônus, os mesmos produtos conjuntos passam a ter compras planejadas
    # pela previsão, em vez de seguir um intervalo fixo de reposição.
    hist, prev, real, est_min, modelos, prev_linhas = {}, {}, {}, {}, {}, []
    metricas_bonus = {}
    for it in itens_RS:
        y = em_periodos(series[it].iloc[:n], DIAS_POR_PERIODO, do_fim=True)
        modelo = escolher_previsao(y)
        ajuste, futuro = prever(y, modelo, HORIZONTE)
        sigma_erro = np.std(y[8:] - ajuste[8:], ddof=1)   # ignora o aquecimento do modelo
        hist[it], prev[it], modelos[it] = y, np.maximum(futuro, 0), modelo
        real[it] = em_periodos(series[it].iloc[n:], DIAS_POR_PERIODO, do_fim=False)
        est_min[it] = np.ceil(norm.ppf(NS_BONUS) * sigma_erro)
        erro_teste = np.abs(prev[it][:len(real[it])] - real[it]).mean()
        metricas_bonus[it] = {"mae_validacao": float(modelo[2]), "mae_teste": float(erro_teste),
                              "desvio_erro_ajuste": float(sigma_erro), "periodos_teste": len(real[it])}
        print(f"{it}: modelo {modelo[0]} (MAE validação {modelo[2]:.0f}) | erro 1 passo sd={sigma_erro:.0f} "
              f"| estoque mínimo {est_min[it]:.0f} | MAE no teste {erro_teste:.0f} un/período")
        for t in range(HORIZONTE):
            prev_linhas.append({"item": it, "periodo": t + 1, "previsao": prev[it][t],
                                "real_teste": real[it][t] if t < len(real[it]) else np.nan})

    D = {it: np.round(prev[it]) for it in itens_RS}

    # Volumes são premissas, não atributos medidos na base; outros itens recebem 1.
    for it in itens_RS:
        VOLUME_UNITARIO.setdefault(it, 1.0)
    c_b = {it: custos[it]["c"] for it in itens_RS}
    h_b = {it: custos[it]["h_dia"] * DIAS_POR_PERIODO for it in itens_RS}
    I0 = {it: est_min[it] + D[it][0] for it in itens_RS}   # começa cobrindo o 1º período
    plano = dimensionar_lotes(D, est_min, I0, c_b, h_b, VOLUME_UNITARIO, K_CONJUNTO, K_POR_ITEM, CAPACIDADE)
    plano["I0"] = I0
    sem_cap = dimensionar_lotes(D, est_min, I0, c_b, h_b, VOLUME_UNITARIO, K_CONJUNTO, K_POR_ITEM, 1e9)
    pico = lambda pl: max(sum(VOLUME_UNITARIO[i] * (np.r_[I0[i], pl["I"][i][:-1]][t] + pl["x"][i][t])
                              for i in itens_RS) for t in range(HORIZONTE))
    print(f"Plano com capacidade {CAPACIDADE}: {int(plano['y'].sum())} pedidos, custo £ {plano['custo']:.2f}, "
          f"pico {pico(plano):.0f} ({plano['status']})")
    print(f"Sem limite de capacidade:      {int(sem_cap['y'].sum())} pedidos, custo £ {sem_cap['custo']:.2f}, "
          f"pico {pico(sem_cap):.0f}")

    # Por último, conferimos se o plano previsto aguentaria as vendas reais.
    # Não ter falta na previsão não significa que nunca vai faltar na prática.
    for it in itens_RS:
        Iv, faltou = I0[it], 0.0
        for t in range(len(real[it])):
            Iv += plano["x"][it][t]
            venda = min(Iv, real[it][t])
            faltou += real[it][t] - venda
            Iv -= venda
        metricas_bonus[it]["falta_teste"] = float(faltou)
        print(f"  {it}: aplicando o plano à demanda real do teste ({len(real[it])} períodos), "
              f"faltariam {faltou:.0f} de {real[it].sum():.0f} unidades")

    plano_df = pd.DataFrame({"periodo": range(1, HORIZONTE + 1), "pedido_conjunto": plano["y"].astype(int)})
    for it in itens_RS:
        plano_df[f"{it}_previsao"] = D[it]
        plano_df[f"{it}_lote"] = plano["x"][it]
        plano_df[f"{it}_estoque_fim"] = np.round(plano["I"][it])
    print(plano_df.to_string(index=False))
    plano_df["periodo_emissao_dos_pedidos"] = plano_df["periodo"] - plano["atraso_periodos"]
    plano_df = plano_df.rename(columns={"pedido_conjunto": "recebimento_conjunto", **{f"{i}_lote": f"{i}_recebimento" for i in itens_RS}})
    plano_df.to_csv(SAIDA / "bonus_plano_lotes.csv", index=False, **csv)
    
    # x é recebimento; a emissão ocorre L dias operacionais antes, conhecida no plano.
    resumo_bonus = {"status": plano["status"], "otimo": plano["otimo"], "gap": plano["gap"],
        "custo": float(plano["custo"]), "pedidos": int(plano["y"].sum()), "pico": float(pico(plano)),
        "custo_sem_capacidade": float(sem_cap["custo"]), "pico_sem_capacidade": float(pico(sem_cap)),
        "pedidos_sem_capacidade": int(sem_cap["y"].sum()), "lead_time_periodos": plano["atraso_periodos"],
        "capacidade": CAPACIDADE, "volumes": {i: VOLUME_UNITARIO[i] for i in itens_RS},
        "estoque_inicial": {i: float(I0[i]) for i in itens_RS}, "modelos": {i: modelos[i][0] for i in itens_RS},
        "estoque_minimo": {i: float(est_min[i]) for i in itens_RS}, "horizonte": HORIZONTE,
        "metricas": metricas_bonus,
        "custo_exclui_aquisicao_inicial": True}
    (SAIDA / "bonus_resumo.json").write_text(json.dumps(resumo_bonus, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(prev_linhas).round(1).to_csv(SAIDA / "bonus_previsao.csv", index=False, **csv)
    fig_bonus(hist, prev, real, plano, itens_RS, est)

    meta = {"dias_treino": n, "dias_teste": len(dias) - n, "inicio_treino": str(dias[0].date()),
        "fim_treino": str(dias[n-1].date()), "inicio_teste": str(corte.date()), "fim_teste": str(dias[-1].date()),
        "item_sQ": item_sQ, "itens_RS": itens_RS, "nivel_servico": NIVEIS_SERVICO,
        "dias_ano_treino": dias_ano, "linhas_positivas_ajustadas": len(df),
        "unidades_brutas": float(vendas.Quantidade.sum()), "unidades_ajustadas": float(df.Quantidade.sum()),
        "cotacao_GBP_BRL": PTAX_GBP_BRL, "lead_time": LEAD_TIME,
        "cancelamentos_entre_periodos": int(audit.entre_periodos.sum())}
    (SAIDA / "metadados.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nArquivos gerados em {SAIDA}")


if __name__ == "__main__":
    main()
