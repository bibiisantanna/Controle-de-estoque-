"""
Pipeline usado pelo app Streamlit.

Reaproveita as funções de controle_estoque.py SEM alterá-las e repete a mesma
sequência do main(), mas em vez de imprimir e salvar arquivos devolve um
dicionário com todos os resultados intermediários (para os gráficos do app).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

import controle_estoque as ce


def janela_protecao(politica, R):
    return ce.LEAD_TIME if politica == "<s,Q>" else R + ce.LEAD_TIME


def parametros(series, n, est, item_sQ, itens_RS, custos, niveis):
    """Mesmo cálculo do main(): Q, R e os pontos s/S para cada nível de serviço."""
    linhas = []
    mu, cu, tr = est.loc[item_sQ, "media"], custos[item_sQ], series[item_sQ].iloc[:n]
    Q_real = np.sqrt(2 * ce.K_INDIVIDUAL * mu / cu["h_dia"])
    Q = ce.lote_inteiro(Q_real, mu, ce.K_INDIVIDUAL, cu["h_dia"])
    sig_L = ce.desvio_acumulado(tr, ce.LEAD_TIME)
    for ns in niveis:
        z = norm.ppf(ns)
        qe = ce.quantil_empirico(tr, ce.LEAD_TIME, ns)
        linhas.append({"item": item_sQ, "politica": "<s,Q>", "ns": ns, "z": z, "Q_ou_R": Q,
                       "media_dia": mu, "janela": ce.LEAD_TIME, "sigma_protecao": sig_L, "ES": z * sig_L,
                       "normal": mu * ce.LEAD_TIME + z * sig_L,
                       "s_ou_S": int(np.ceil(max(mu * ce.LEAD_TIME + z * sig_L, qe))),
                       "quantil_empirico": int(np.ceil(qe))})

    K_total = ce.K_CONJUNTO + ce.K_POR_ITEM * len(itens_RS)
    soma_h_mu = sum(custos[i]["h_dia"] * est.loc[i, "media"] for i in itens_RS)
    T_real = np.sqrt(2 * K_total / soma_h_mu)
    R = ce.ciclo_inteiro(T_real, K_total, soma_h_mu)
    for it in itens_RS:
        mu_i, tr_i = est.loc[it, "media"], series[it].iloc[:n]
        sig = ce.desvio_acumulado(tr_i, R + ce.LEAD_TIME)
        for ns in niveis:
            z = norm.ppf(ns)
            qe = ce.quantil_empirico(tr_i, R + ce.LEAD_TIME, ns)
            linhas.append({"item": it, "politica": "<R,S>", "ns": ns, "z": z, "Q_ou_R": R,
                           "media_dia": mu_i, "janela": R + ce.LEAD_TIME, "sigma_protecao": sig, "ES": z * sig,
                           "normal": mu_i * (R + ce.LEAD_TIME) + z * sig,
                           "s_ou_S": int(np.ceil(max(mu_i * (R + ce.LEAD_TIME) + z * sig, qe))),
                           "quantil_empirico": int(np.ceil(qe))})
    par = pd.DataFrame(linhas)
    for j, row in par.iterrows():
        acum = series[row["item"]].iloc[:n].rolling(int(row["janela"])).sum().dropna()
        par.loc[j, "cobertura_empirica_treino"] = (acum <= row["s_ou_S"]).mean()
    extras = {"Q_real": Q_real, "Q": Q, "T_real": T_real, "R": R, "K_total": K_total, "soma_h_mu": soma_h_mu}
    return par, extras


def simular(series, n, est, item_sQ, itens_RS, custos, par, Q, R, ns):
    """Roda as duas simulações do teste para um nível de serviço e monta o resumo."""
    dem = {it: series[it].iloc[n:].values for it in [item_sQ] + itens_RS}
    p = par[np.isclose(par["ns"], ns)].set_index("item")
    sq = ce.simular_sQ(dem[item_sQ], p.loc[item_sQ, "s_ou_S"], Q, est.loc[item_sQ, "media"], custos[item_sQ])
    rs = ce.simular_RS({i: dem[i] for i in itens_RS}, R, {i: p.loc[i, "s_ou_S"] for i in itens_RS},
                       {i: est.loc[i, "media"] for i in itens_RS}, {i: custos[i] for i in itens_RS})
    resumo = []
    for nome, sim, its in [("<s,Q>", sq, [item_sQ]), ("<R,S>", rs, itens_RS)]:
        perda_dia = sim["perda"] if nome == "<s,Q>" else sum(sim["perda"][i] for i in its)
        dem_total = sum(dem[i].sum() for i in its)
        ciclos = ce.ciclos_com_falta(perda_dia, sim["chegadas"])
        c_ = sim["custos"]
        resumo.append({"sistema": nome, "itens": "+".join(its), "ns_planejado": ns, "pedidos": sim["n_pedidos"],
                       "custo_fixo": c_["fixo"], "custo_variavel": c_["variavel"],
                       "custo_armazenagem": c_["armazenagem"], "custo_falta": c_["falta"],
                       "custo_total": sum(c_.values()), "custo_total_BRL": sum(c_.values()) * ce.PTAX_GBP_BRL,
                       "demanda": dem_total, "vendas_perdidas": perda_dia.sum(),
                       "fill_rate": 1 - perda_dia.sum() / dem_total, "maior_perda_em_um_dia": perda_dia.max(),
                       "ciclos": ciclos["ciclos_completos"],
                       "ciclos_sem_falta": ciclos["ciclos_completos"] - ciclos["ciclos_completos_com_falta"],
                       "aquisicao_estoque_inicial": sim["aquisicao_inicial"]})
    return {"sQ": sq, "RS": rs, "dem": dem, "resumo": pd.DataFrame(resumo)}


def rodar(arquivo):
    r = {}
    # 1) leitura, diagnóstico e limpeza
    bruto = ce.carregar(arquivo)
    r["diag"], r["auditoria"] = ce.diagnosticar(bruto)
    vendas, canc, r["log"] = ce.preparar_base(bruto)
    dias = ce.dias_de_operacao(vendas)
    n = int(ce.FRACAO_TREINO * len(dias))
    corte = dias[n]
    df, treino_saldos, teste_saldos, audit, audit_tr = ce.preparar_periodos(vendas, canc, corte)
    treino_df = treino_saldos[treino_saldos["Quantidade"] > 0]
    dias_ano = n * 365.25 / ((dias[n - 1] - dias[0]).days + 1)
    r["audit_canc"] = audit
    r["resumo_canc"] = audit.groupby("status").agg(
        linhas=("quantidade_cancelada", "size"), unidades_canceladas=("quantidade_cancelada", "sum"),
        unidades_abatidas=("quantidade_abatida", "sum")).reset_index()
    r["n_registros"] = len(bruto)
    r["unidades_brutas"] = float(vendas["Quantidade"].sum())
    r["unidades_ajustadas"] = float(df["Quantidade"].sum())
    r["n_canc_entre_periodos"] = int(audit["entre_periodos"].sum())

    # 2) calendário, matriz diária e seleção
    r.update(dias=dias, n=n, corte=corte, dias_ano=dias_ano)
    m = ce.matriz_diaria(df, dias)
    r["total_diario"] = m.sum(axis=0)
    ranking, item_sQ, itens_RS = ce.selecionar_itens(treino_df, m.iloc[:, :n])
    itens = [item_sQ] + itens_RS
    r.update(ranking=ranking, item_sQ=item_sQ, itens_RS=itens_RS, itens=itens)

    # estatísticas de todos os produtos do treino (para o funil de filtros)
    mt = m.iloc[:, :n]
    nn = mt.shape[1]
    fim_treino = mt.columns[-1] + pd.Timedelta(days=1)
    meses = (treino_df[treino_df["Data_Fatura"] < fim_treino].groupby("Codigo_Produto")["Data_Fatura"]
             .apply(lambda s: s.dt.to_period("M").nunique()))
    todos = pd.DataFrame({"media": mt.mean(axis=1), "cv": mt.std(axis=1) / mt.mean(axis=1),
                          "freq": (mt > 0).mean(axis=1), "meses": meses,
                          "concentracao": mt.max(axis=1) / mt.sum(axis=1),
                          "tendencia": mt.iloc[:, nn // 2:].mean(axis=1) / mt.iloc[:, :nn // 2].mean(axis=1)})
    r["stats_todos"] = todos.dropna(subset=["media"])

    # 3) outliers e estatísticas dos 3 itens
    series, brutas, est = {}, {}, {}
    for it in itens:
        s, lim, nout = ce.tratar_outliers(m.loc[it], n)
        series[it], brutas[it] = s, m.loc[it]
        tr, te = s.iloc[:n], s.iloc[n:]
        est[it] = {"descricao": ranking.loc[it, "descricao"], "media": tr.mean(), "dp": tr.std(),
                   "cv": tr.std() / tr.mean(), "media_teste": te.mean(),
                   "limite_outlier": lim, "dias_ajustados": nout}
    est = pd.DataFrame(est).T
    custos = {it: ce.custos_item(treino_df, it, dias_ano) for it in itens}
    for it in itens:
        for k_ in ["preco", "c", "h_ano", "h_dia", "falta"]:
            est.loc[it, k_] = custos[it][k_]
    for col in ["media", "dp", "cv", "media_teste", "limite_outlier", "dias_ajustados"]:
        est[col] = est[col].astype(float)
    r.update(series=series, brutas=brutas, est=est, custos=custos)

    # 4) parâmetros (99% do enunciado)
    par, extras = parametros(series, n, est, item_sQ, itens_RS, custos, ce.NIVEIS_SERVICO)
    r.update(par=par, **extras)

    # 5) simulação no teste
    ns = ce.NIVEIS_SERVICO[0]
    r["sim"] = simular(series, n, est, item_sQ, itens_RS, custos, par, extras["Q"], extras["R"], ns)
    r["datas_teste"] = dias[n:]

    # 6) bônus
    hist, prev, real, est_min, modelos, ajustes, metricas = {}, {}, {}, {}, {}, {}, {}
    for it in itens_RS:
        y = ce.em_periodos(series[it].iloc[:n], ce.DIAS_POR_PERIODO, do_fim=True)
        modelo = ce.escolher_previsao(y)
        ajuste, futuro = ce.prever(y, modelo, ce.HORIZONTE)
        sigma_erro = np.std(y[8:] - ajuste[8:], ddof=1)
        hist[it], prev[it], modelos[it], ajustes[it] = y, np.maximum(futuro, 0), modelo, ajuste
        real[it] = ce.em_periodos(series[it].iloc[n:], ce.DIAS_POR_PERIODO, do_fim=False)
        est_min[it] = np.ceil(norm.ppf(ce.NS_BONUS) * sigma_erro)
        metricas[it] = {"modelo": modelo[0], "mae_validacao": float(modelo[2]),
                        "mae_teste": float(np.abs(prev[it][:len(real[it])] - real[it]).mean()),
                        "sd_erro": float(sigma_erro), "estoque_minimo": float(est_min[it])}
    D = {it: np.round(prev[it]) for it in itens_RS}
    volume = dict(ce.VOLUME_UNITARIO)
    for it in itens_RS:
        volume.setdefault(it, 1.0)
    ce.VOLUME_UNITARIO.update(volume)
    c_b = {it: custos[it]["c"] for it in itens_RS}
    h_b = {it: custos[it]["h_dia"] * ce.DIAS_POR_PERIODO for it in itens_RS}
    I0 = {it: est_min[it] + D[it][0] for it in itens_RS}
    plano = ce.dimensionar_lotes(D, est_min, I0, c_b, h_b, volume, ce.K_CONJUNTO, ce.K_POR_ITEM, ce.CAPACIDADE)
    sem_cap = ce.dimensionar_lotes(D, est_min, I0, c_b, h_b, volume, ce.K_CONJUNTO, ce.K_POR_ITEM, 1e9)

    def ocupacao(pl):
        return {i: volume[i] * (np.r_[I0[i], pl["I"][i][:-1]] + pl["x"][i]) for i in itens_RS}

    for it in itens_RS:
        Iv, faltou = I0[it], 0.0
        for t in range(len(real[it])):
            Iv += plano["x"][it][t]
            venda = min(Iv, real[it][t])
            faltou += real[it][t] - venda
            Iv -= venda
        metricas[it]["falta_teste"] = faltou
        metricas[it]["demanda_teste"] = float(real[it].sum())
    r["bonus"] = {"hist": hist, "prev": prev, "real": real, "ajustes": ajustes, "est_min": est_min,
                  "metricas": metricas, "D": D, "I0": I0, "plano": plano, "sem_cap": sem_cap,
                  "volume": volume, "ocup": ocupacao(plano), "ocup_sem": ocupacao(sem_cap)}
    return r


if __name__ == "__main__":
    # Gera resultados_app.pkl para o app abrir sem recalcular:
    #   python pipeline.py Online_Retail_PT_BR_Completo.xlsx
    import pickle
    import sys
    from pathlib import Path
    arquivo = sys.argv[1] if len(sys.argv) > 1 else "Online_Retail_PT_BR_Completo.xlsx"
    saida = Path(__file__).parent / "resultados_app.pkl"
    with open(saida, "wb") as f:
        pickle.dump(rodar(arquivo), f)
    print(f"Resultados salvos em {saida}")
