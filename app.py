"""
App Streamlit do Trabalho 1 de Pesquisa Operacional 2 (UFRJ): controle de estoque.

Rodar:  streamlit run app.py
Cada página corresponde a uma etapa do controle_estoque.py e mostra o que a etapa
faz, os resultados em gráficos interativos e o trecho de código correspondente.
"""

import pickle
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm

import controle_estoque as ce
import pipeline

PASTA = Path(__file__).parent
st.set_page_config(page_title="Controle de Estoque · PO2", layout="wide")

AZUL, LARANJA, VERMELHO, VERDE, CINZA, AZUL_CLARO = "#1f4e79", "#e07b39", "#c0392b", "#2e8b57", "#8a8a8a", "#9ecae1"
GBP = ce.PTAX_GBP_BRL


# ----------------------------------------------------------------------------- utilidades
def fmt(v, casas=0):
    """Número no formato brasileiro: 1.234,5"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    s = f"{v:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v, casas=1):
    return fmt(100 * v, casas) + "%"


def libra(v):
    return f"£ {fmt(v, 2)}"


def real(v):
    return f"R$ {fmt(v, 2)}"


ARQUIVO_CODIGO = Path(__file__).parent / "controle_estoque.py"

def codigo_fonte():
    """Lê controle_estoque.py e guarda onde começa e termina cada função (sem usar inspect)."""
    try:
        texto = ARQUIVO_CODIGO.read_text(encoding="utf-8-sig")
    except Exception:
        return "", {}
    funcoes = {}
    try:
        for no in ast.parse(texto).body:
            if isinstance(no, ast.FunctionDef):
                funcoes[no.name] = (no.lineno, no.end_lineno)
    except SyntaxError:
        pass
    return texto, funcoes


def trecho(nome):
    texto, funcoes = codigo_fonte()
    if nome not in funcoes:
        return None, None
    a, b = funcoes[nome]
    return "\n".join(texto.splitlines()[a - 1:b]), a


def ver_codigo(*funcoes, titulo="Ver o código desta etapa"):
    with st.expander(titulo):
        for f in funcoes:
            codigo, _ = trecho(f.__name__)
            if codigo:
                st.code(codigo, language="python")


def metrica(col, rotulo, valor, nota=None, ajuda=None):
    """Métrica com uma nota explicativa embaixo (sem seta de variação)."""
    col.metric(rotulo, valor, help=ajuda)
    if nota:
        col.caption(nota)


def layout(fig, altura=380, legenda=True):
    fig.update_layout(height=altura, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0) if legenda else None,
                      showlegend=legenda, template="plotly_white", separators=",.")
    return fig


# ----------------------------------------------------------------------------- dados
NOME_BASE = "Online_Retail_PT_BR_Completo.xlsx"
CACHE = PASTA / "resultados_app.pkl"


@st.cache_resource(show_spinner=False)
def resultados():
    """Usa os resultados já calculados (resultados_app.pkl); se não der, roda o programa na planilha."""
    if CACHE.exists():
        try:
            with open(CACHE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    for caminho in [PASTA / NOME_BASE, PASTA / "dados" / NOME_BASE]:
        if caminho.exists():
            return pipeline.rodar(str(caminho))
    return None


# ----------------------------------------------------------------------------- páginas
def pagina_inicio(r):
    st.title("Controle de Estoque de três produtos")
    st.caption("UFRJ ·Pesquisa Operacional")
    st.markdown(
        "O programa `controle_estoque.py` usa a base **Online Retail (UCI)** traduzida para português "
        "(vendas de 01/12/2010 a 09/12/2011). Ele limpa os dados, escolhe três produtos, calcula uma política "
        "**⟨s,Q⟩** para um item e uma política **⟨R,S⟩ conjunta** para os outros dois, simula as duas no período "
        "de teste e, no bônus, planeja os lotes com **previsão de demanda + programação inteira**.")

    c = st.columns(6)
    c[0].metric("Registros na base", fmt(r["n_registros"]))
    c[1].metric("Dias de treino", r["n"])
    c[2].metric("Dias de teste", len(r["dias"]) - r["n"])
    c[3].metric("Item ⟨s,Q⟩", r["item_sQ"])
    c[4].metric("1º item ⟨R,S⟩", r["itens_RS"][0])
    c[5].metric("2º item ⟨R,S⟩", r["itens_RS"][1])

    st.subheader("Fluxo do programa")
    st.graphviz_chart("""
    digraph {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3f8", color="#1f4e79", fontname="Helvetica", fontsize=11];
      edge [color="#555555"];
      a [label="1 · Ler e diagnosticar\\ncarregar · diagnosticar"];
      b [label="Limpar e abater\\ncancelamentos\\npreparar_base · preparar_periodos"];
      c [label="2 · Treino 80% / teste 20%\\nsérie diária · seleção\\nselecionar_itens"];
      d [label="Outliers só no treino\\ntratar_outliers"];
      e [label="3 · Custos e parâmetros\\n⟨s,Q⟩ e ⟨R,S⟩ a 99%"];
      f [label="4 · Simulação diária\\nno teste\\nsimular_sQ · simular_RS"];
      g [label="7 · Bônus\\nprevisão + MILP\\ndimensionar_lotes", fillcolor="#fdf0e6", color="#e07b39"];
      a -> b -> c -> d -> e -> f; d -> g;
    }""", width="stretch")

    st.subheader("Premissas (constantes no topo do código)")
    prem = pd.DataFrame([
        ["Fração de treino", f"{ce.FRACAO_TREINO:.0%}", "FRACAO_TREINO"],
        ["Lead time", f"{ce.LEAD_TIME} dias de operação", "LEAD_TIME"],
        ["Nível de serviço (por ciclo)", ", ".join(f"{x:.0%}" for x in ce.NIVEIS_SERVICO), "NIVEIS_SERVICO"],
        ["Custo de aquisição", f"{ce.FRACAO_AQUISICAO:.0%} do preço", "FRACAO_AQUISICAO"],
        ["Armazenagem", f"{ce.TAXA_ARMAZENAGEM:.0%} a.a. sobre o custo", "TAXA_ARMAZENAGEM"],
        ["Custo fixo individual", libra(ce.K_INDIVIDUAL), "K_INDIVIDUAL"],
        ["Custo fixo conjunto", f"{libra(ce.K_CONJUNTO)} + {libra(ce.K_POR_ITEM)} por item", "K_CONJUNTO, K_POR_ITEM"],
        ["Falta", "venda perdida = preço − custo", "custos_item"],
        ["Câmbio", f"£ 1 = R$ {fmt(GBP, 4)} (PTAX)", "PTAX_GBP_BRL"],
    ], columns=["Premissa", "Valor", "Onde no código"])
    st.dataframe(prem, hide_index=True, width="stretch")


def pagina_dados(r):
    st.header("1 · Dados, diagnóstico e limpeza")
    st.markdown("Primeiro olhamos a base **como ela veio**, sem apagar nada. Só depois aplicamos os filtros, "
                "cada um registrado com quantas linhas removeu.")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.subheader("Diagnóstico da base bruta")
        d = r["diag"].copy()
        d["Valor"] = d["Valor"].map(fmt)
        st.dataframe(d, hide_index=True, width="stretch")
    with c2:
        st.subheader("Funil de limpeza")
        log = r["log"]
        fig = go.Figure(go.Funnel(y=log["Etapa"], x=log["Restantes"], textinfo="value",
                                  marker_color=[AZUL] + [AZUL_CLARO] * (len(log) - 1)))
        fig.update_traces(texttemplate="%{value:,.0f}")
        st.plotly_chart(layout(fig, 360, legenda=False), width="stretch")
        st.caption("Removidas em cada etapa: " + "; ".join(
            f"{e}: {fmt(v)}" for e, v in zip(log["Etapa"][1:], log["Removidas"][1:])))

    st.subheader("O que são as quantidades negativas?")
    aud = r["auditoria"].groupby(["Tipo", "Indício"])["Linhas"].sum().reset_index()
    fig = go.Figure()
    for tipo, cor in [("Com C", AZUL), ("Sem C", LARANJA)]:
        a = aud[aud["Tipo"] == tipo]
        fig.add_bar(y=a["Indício"], x=a["Linhas"], name=f"Fatura {tipo.lower()}", orientation="h", marker_color=cor)
    fig.update_layout(barmode="group", yaxis_title=None, xaxis_title="linhas")
    st.plotly_chart(layout(fig, 300), width="stretch")
    st.caption("Faturas com “C” são cancelamentos de venda. Negativas sem “C” têm descrições de ajuste, avaria "
               "ou perda (regex em `classificar_descricao`): são movimentos de estoque, não demanda.")
    ver_codigo(ce.carregar, ce.classificar_descricao, ce.diagnosticar, ce.preparar_base)

    st.divider()
    st.subheader("Cancelamentos: abatidos da venda original só quando não há dúvida")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        rc = r["resumo_canc"].sort_values("linhas", ascending=True)
        cores = [VERDE if s == "Associação única inferida" else CINZA for s in rc["status"]]
        fig = go.Figure(go.Bar(y=rc["status"], x=rc["linhas"], orientation="h", marker_color=cores,
                               text=[fmt(v) for v in rc["linhas"]], textposition="outside"))
        fig.update_layout(xaxis_title="linhas de cancelamento")
        st.plotly_chart(layout(fig, 300, legenda=False), width="stretch")
    with c2:
        st.markdown(
            "**Regra (`abater_cancelamentos`)**\n\n"
            "1. Procura vendas do **mesmo cliente, produto e preço**, com data até a do cancelamento e saldo.\n"
            "2. **Uma** candidata com saldo suficiente → abate (vendeu 10, cancelou 3 → conta 7).\n"
            "3. Zero, várias ou saldo insuficiente → não abate e registra o motivo.\n\n"
            "**Sem vazamento:** o treino só recebe cancelamentos anteriores ao corte (`preparar_periodos`).")
        ab = r["unidades_brutas"] - r["unidades_ajustadas"]
        metrica(st, "Unidades abatidas da demanda", fmt(ab), f"{ab / r['unidades_brutas']:.1%} do volume bruto")
        st.caption(f"Cancelamentos do teste que apontam para vendas do treino (só auditoria): "
                   f"{fmt(r['n_canc_entre_periodos'])}")
    with st.expander("Ver a auditoria dos cancelamentos (amostra)"):
        st.dataframe(r["audit_canc"].head(300), width="stretch")
    ver_codigo(ce.chave_preco, ce.abater_cancelamentos, ce.preparar_periodos)


def pagina_selecao(r):
    st.header("2 · Divisão treino/teste e seleção dos três itens")
    dias, n, corte = r["dias"], r["n"], r["corte"]
    st.markdown(f"A série usa **dias de operação**: datas em que a loja vendeu algo ({len(dias)} dias). "
                "Sábados e o recesso de fim de ano não viram zero. Os **80% mais antigos** são o treino.")
    tot = r["total_diario"]
    fig = go.Figure()
    fig.add_bar(x=tot.index[:n], y=tot.values[:n], name=f"Treino ({n} dias)", marker_color=AZUL)
    fig.add_bar(x=tot.index[n:], y=tot.values[n:], name=f"Teste ({len(dias) - n} dias)", marker_color=LARANJA)
    fig.add_vline(x=corte, line_dash="dash", line_color="#444")
    fig.update_layout(yaxis_title="unidades vendidas / dia (todos os produtos)", bargap=0)
    st.plotly_chart(layout(fig, 300), width="stretch")
    c = st.columns(3)
    c[0].metric("Fim do treino", dias[n - 1].strftime("%d/%m/%Y"))
    c[1].metric("Início do teste (corte)", corte.strftime("%d/%m/%Y"))
    c[2].metric("Dias de operação por ano (estimado no treino)", fmt(r["dias_ano"], 1),
                help="Converte a armazenagem anual em custo por dia de operação.")
    ver_codigo(ce.dias_de_operacao, ce.matriz_diaria)

    st.divider()
    st.subheader("Funil de filtros (calculado só no treino)")
    s = r["stats_todos"]
    filtros = [
        ("Todos os produtos", pd.Series(True, index=s.index)),
        (f"Vende em ≥ {ce.SEL_FREQ_MIN:.0%} dos dias", s["freq"] >= ce.SEL_FREQ_MIN),
        (f"Média ≥ {ce.SEL_MEDIA_MIN:.0f} un/dia", s["media"] >= ce.SEL_MEDIA_MIN),
        (f"Presente em ≥ {ce.SEL_MESES_MIN} meses", s["meses"] >= ce.SEL_MESES_MIN),
        (f"Maior dia ≤ {ce.SEL_CONC_MAX:.0%} do total", s["concentracao"] <= ce.SEL_CONC_MAX),
        ("Tendência entre 2/3 e 1,5", s["tendencia"].between(*ce.SEL_TENDENCIA)),
    ]
    acum, cont = pd.Series(True, index=s.index), []
    for nome, f in filtros:
        acum &= f.fillna(False)
        cont.append((nome, int(acum.sum())))
    c1, c2 = st.columns([1, 1.2])
    with c1:
        fig = go.Figure(go.Funnel(y=[a for a, _ in cont], x=[b for _, b in cont], textinfo="value",
                                  marker_color=[CINZA] * (len(cont) - 1) + [VERDE]))
        st.plotly_chart(layout(fig, 340, legenda=False), width="stretch")
    with c2:
        ok = acum
        fig = go.Figure()
        fig.add_scatter(x=s.loc[~ok, "media"], y=s.loc[~ok, "cv"], mode="markers", name="reprovados",
                        marker=dict(color="#cccccc", size=5), text=s.index[~ok],
                        hovertemplate="%{text}<br>média %{x:.1f}<br>CV %{y:.2f}<extra></extra>")
        fig.add_scatter(x=s.loc[ok, "media"], y=s.loc[ok, "cv"], mode="markers", name="aprovados",
                        marker=dict(color=AZUL_CLARO, size=9), text=s.index[ok],
                        hovertemplate="%{text}<br>média %{x:.1f}<br>CV %{y:.2f}<extra></extra>")
        for it, cor, pol in [(r["item_sQ"], AZUL, "⟨s,Q⟩")] + [(i, LARANJA, "⟨R,S⟩") for i in r["itens_RS"]]:
            fig.add_scatter(x=[s.loc[it, "media"]], y=[s.loc[it, "cv"]], mode="markers", name=f"{it} {pol}",
                            marker=dict(color=cor, size=14, line=dict(color="white", width=1.5)),
                            hovertemplate=f"{it} {pol}<br>média %{{x:.1f}}<br>CV %{{y:.2f}}<extra></extra>")
        fig.update_layout(xaxis_type="log", xaxis_title="média diária no treino (escala log)",
                          yaxis_title="CV (desvio / média)", yaxis_range=[0, min(6, s["cv"].quantile(0.98) * 1.1)])
        st.plotly_chart(layout(fig, 340), width="stretch")
    st.markdown("Entre os aprovados, ordenamos pelo **CV** (demanda mais estável primeiro): "
                "o **1º** recebe ⟨s,Q⟩ e o **2º e 3º** formam o pedido conjunto ⟨R,S⟩.")
    rk = r["ranking"].copy()
    rk.insert(0, "política", ["⟨s,Q⟩" if i == r["item_sQ"] else "⟨R,S⟩" if i in r["itens_RS"] else ""
                              for i in rk.index])
    st.dataframe(rk.style.format({"media": (lambda v: fmt(v, 1)), "cv": (lambda v: fmt(v, 3)), "freq": pct, "concentracao": (lambda v: pct(v, 2)),
                                  "tendencia": (lambda v: fmt(v, 2))}), width="stretch")
    ver_codigo(ce.selecionar_itens)


def pagina_demanda(r):
    st.header("3 · Demanda diária dos itens escolhidos e tratamento de outliers")
    st.markdown("No **treino**, dias acima de **Q3 + 3·IQR** são limitados a esse valor (o pico não é tratado "
                "como erro, só não deixamos que ele infle os parâmetros). No **teste**, a demanda fica **como "
                "aconteceu**, para ver a política reagir a picos reais.")
    n, est = r["n"], r["est"]
    for it in r["itens"]:
        pol = "⟨s,Q⟩" if it == r["item_sQ"] else "⟨R,S⟩"
        s, b = r["series"][it], r["brutas"][it]
        fig = go.Figure()
        cortados = b.iloc[:n][b.iloc[:n] > est.loc[it, "limite_outlier"]]
        fig.add_scatter(x=s.index[:n], y=s.values[:n], name="treino (tratado)", line=dict(color=AZUL, width=1))
        fig.add_scatter(x=s.index[n - 1:], y=s.values[n - 1:], name="teste (real)", line=dict(color=LARANJA, width=1))
        if len(cortados):
            fig.add_scatter(x=cortados.index, y=cortados.values, mode="markers", name="valor original do pico",
                            marker=dict(color=VERMELHO, symbol="x", size=8))
        fig.add_hline(y=est.loc[it, "limite_outlier"], line_dash="dot", line_color=CINZA,
                      annotation_text=f"limite Q3+3·IQR = {fmt(est.loc[it, 'limite_outlier'])}",
                      annotation_position="top left")
        fig.add_vline(x=r["corte"], line_dash="dash", line_color="#444")
        fig.update_layout(title=f"{it} – {est.loc[it, 'descricao']}  {pol}", yaxis_title="un/dia")
        st.plotly_chart(layout(fig, 280), width="stretch")
    tab = est[["descricao", "media", "dp", "cv", "media_teste", "limite_outlier", "dias_ajustados"]].copy()
    tab.columns = ["Produto", "Média treino", "Desvio treino", "CV", "Média teste", "Limite outlier", "Dias limitados"]
    st.dataframe(tab.style.format({"Média treino": (lambda v: fmt(v, 1)), "Desvio treino": (lambda v: fmt(v, 1)), "CV": (lambda v: fmt(v, 2)),
                                   "Média teste": (lambda v: fmt(v, 1)), "Limite outlier": fmt, "Dias limitados": fmt}),
                 width="stretch")
    ver_codigo(ce.tratar_outliers)


def curva_rolling(serie, janela):
    return serie.rolling(janela).sum().dropna()


def pagina_parametros(r):
    st.header("4 · Custos e parâmetros das políticas")
    est, par = r["est"], r["par"]
    st.subheader("Premissas de custo por item")
    st.markdown("A base só traz **preço de venda**. Então: preço p = mediana no treino; compra c = 60%·p; "
                "armazenagem h = 25% a.a. sobre c (dividido pelos dias de operação/ano); falta = p − c "
                "(margem da venda perdida).")
    cu = est[["descricao", "preco", "c", "h_ano", "h_dia", "falta"]].copy()
    cu["preço (R$)"] = cu["preco"] * GBP
    cu.columns = ["Produto", "Preço p (£)", "Compra c (£)", "h (£/un·ano)", "h (£/un·dia)", "Falta (£/un)", "Preço (R$)"]
    st.dataframe(cu.style.format({c: (lambda v: fmt(v, 4)) if "dia" in c else (lambda v: fmt(v, 2)) for c in cu.columns[1:]}),
                 width="stretch")
    ver_codigo(ce.custos_item)

    st.divider()
    it, Q = r["item_sQ"], r["Q"]
    mu, h = est.loc[it, "media"], est.loc[it, "h_dia"]
    linha = par[par["item"] == it].iloc[0]
    st.subheader(f"⟨s,Q⟩ — item {it} ({est.loc[it, 'descricao']})")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Q: lote econômico (EOQ)**")
        st.latex(r"Q^*=\sqrt{\frac{2K\mu}{h}}=\sqrt{\frac{2\cdot %s\cdot %s}{%s}}=%s\;\Rightarrow\;Q=%d"
                 % (fmt(ce.K_INDIVIDUAL), fmt(mu, 1), fmt(h, 5), fmt(r["Q_real"], 1), Q))
        qs = np.linspace(max(1, Q * 0.2), Q * 3, 300)
        fig = go.Figure()
        fig.add_scatter(x=qs, y=ce.K_INDIVIDUAL * mu / qs, name="pedir: Kμ/Q", line=dict(color=LARANJA))
        fig.add_scatter(x=qs, y=h * qs / 2, name="guardar: hQ/2", line=dict(color=AZUL_CLARO))
        fig.add_scatter(x=qs, y=ce.K_INDIVIDUAL * mu / qs + h * qs / 2, name="total", line=dict(color=AZUL, width=3))
        fig.add_vline(x=Q, line_dash="dash", annotation_text=f"Q = {Q}")
        fig.update_layout(xaxis_title="tamanho do lote Q (un)", yaxis_title="£ por dia")
        st.plotly_chart(layout(fig, 320), width="stretch")
    with c2:
        st.markdown(f"**s: cobre o lead time de {ce.LEAD_TIME} dias com 99%**")
        st.latex(r"s=\lceil\max(\mu L+z\,\sigma_L,\;q_{0{,}99})\rceil=\lceil\max(%s,\;%s)\rceil=%d"
                 % (fmt(linha["normal"], 1), fmt(linha["quantil_empirico"]), linha["s_ou_S"]))
        rol = curva_rolling(r["series"][it].iloc[:r["n"]], ce.LEAD_TIME)
        fig = histograma_janela(rol, linha, "s")
        st.plotly_chart(layout(fig, 320), width="stretch")
    st.caption("σ_L é o desvio das **somas móveis de L dias** do treino (não σ·√L): assim a dependência entre "
               "dias seguidos entra no cálculo. O quantil empírico não supõe normalidade; usamos o maior dos dois.")
    ver_codigo(ce.lote_inteiro, ce.desvio_acumulado, ce.quantil_empirico)

    st.divider()
    R = r["R"]
    st.subheader(f"⟨R,S⟩ conjunto — itens {' + '.join(r['itens_RS'])}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**R: ciclo comum ótimo**")
        st.latex(r"T^*=\sqrt{\frac{2(K_0+\sum k_i)}{\sum h_i\mu_i}}=\sqrt{\frac{2\cdot %s}{%s}}=%s\;\Rightarrow\;R=%d\text{ dias}"
                 % (fmt(r["K_total"], 1), fmt(r["soma_h_mu"], 4), fmt(r["T_real"], 1), R))
        ts = np.linspace(2, max(R * 3, 10), 300)
        fig = go.Figure()
        fig.add_scatter(x=ts, y=r["K_total"] / ts, name="pedir: K/T", line=dict(color=LARANJA))
        fig.add_scatter(x=ts, y=ts * r["soma_h_mu"] / 2, name="guardar: TΣhμ/2", line=dict(color=AZUL_CLARO))
        fig.add_scatter(x=ts, y=r["K_total"] / ts + ts * r["soma_h_mu"] / 2, name="total", line=dict(color=AZUL, width=3))
        fig.add_vline(x=R, line_dash="dash", annotation_text=f"R = {R}")
        fig.update_layout(xaxis_title="intervalo entre revisões T (dias)", yaxis_title="£ por dia")
        st.plotly_chart(layout(fig, 320), width="stretch")
    with c2:
        st.markdown(f"**S: cobre R + L = {R + ce.LEAD_TIME} dias com 99%**")
        alvo = st.radio("Item", r["itens_RS"], horizontal=True, label_visibility="collapsed")
        lin = par[par["item"] == alvo].iloc[0]
        st.latex(r"S=\lceil\max(\mu(R+L)+z\,\sigma_{R+L},\;q_{0{,}99})\rceil=\lceil\max(%s,\;%s)\rceil=%d"
                 % (fmt(lin["normal"], 1), fmt(lin["quantil_empirico"]), lin["s_ou_S"]))
        rol = curva_rolling(r["series"][alvo].iloc[:r["n"]], R + ce.LEAD_TIME)
        st.plotly_chart(layout(histograma_janela(rol, lin, "S"), 280), width="stretch")
    st.caption("Na revisão periódica, depois de um pedido só se pede de novo R dias depois, e ainda leva L "
               "para chegar: a proteção cobre R + L.")
    ver_codigo(ce.ciclo_inteiro)

    st.subheader("Tabela de parâmetros (99%)")
    t = par[["item", "politica", "Q_ou_R", "janela", "sigma_protecao", "ES", "normal", "quantil_empirico", "s_ou_S",
             "cobertura_empirica_treino"]].copy()
    t.columns = ["Item", "Política", "Q (un) ou R (dias)", "Janela (dias)", "σ proteção", "ES (z·σ)",
                 "μ·janela + z·σ", "Quantil 99%", "s ou S", "Cobertura no treino"]
    st.dataframe(t.style.format({"σ proteção": (lambda v: fmt(v, 1)), "ES (z·σ)": (lambda v: fmt(v, 1)), "μ·janela + z·σ": (lambda v: fmt(v, 1)),
                                 "Cobertura no treino": pct}), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Sensibilidade: e se o nível de serviço fosse outro?")
    st.markdown("Recalcula s e S e roda de novo a simulação do teste com as mesmas funções do código.")
    ns = st.select_slider("Nível de serviço por ciclo", options=[0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999],
                          value=0.95, format_func=lambda x: f"{x:.1%}")
    niveis = sorted({ns, 0.99})
    par2, ex2 = pipeline.parametros(r["series"], r["n"], est, r["item_sQ"], r["itens_RS"], r["custos"], niveis)
    linhas = []
    for x in niveis:
        sim = pipeline.simular(r["series"], r["n"], est, r["item_sQ"], r["itens_RS"], r["custos"], par2,
                               ex2["Q"], ex2["R"], x)
        for _, row in sim["resumo"].iterrows():
            pp = par2[np.isclose(par2["ns"], x) & par2["item"].isin(row["itens"].split("+"))]["s_ou_S"]
            linhas.append({"Nível": f"{x:.1%}", "Sistema": row["sistema"],
                           "s ou S": " / ".join(str(v) for v in pp), "Armazenagem (£)": row["custo_armazenagem"],
                           "Falta (£)": row["custo_falta"], "Total (£)": row["custo_total"],
                           "Fill rate": row["fill_rate"]})
    st.dataframe(pd.DataFrame(linhas).style.format({"Armazenagem (£)": (lambda v: fmt(v, 2)), "Falta (£)": (lambda v: fmt(v, 2)),
                                                    "Total (£)": (lambda v: fmt(v, 2)), "Fill rate": pct}),
                 hide_index=True, width="stretch")
    st.caption("Mais nível de serviço → s/S maiores → mais armazenagem e, em geral, menos falta. "
               "Compra e custo fixo quase não mudam: dependem do volume vendido.")


def histograma_janela(rol, linha, nome):
    fig = go.Figure()
    fig.add_histogram(x=rol, nbinsx=40, marker_color=AZUL_CLARO, name="somas móveis no treino")
    fig.add_vline(x=linha["media_dia"] * linha["janela"], line_color=CINZA, line_dash="dot",
                  annotation_text="μ·janela", annotation_position="top left")
    fig.add_vline(x=linha["quantil_empirico"], line_color=LARANJA, line_dash="dash",
                  annotation_text="quantil 99%", annotation_position="top")
    fig.add_vline(x=linha["s_ou_S"], line_color=AZUL, line_width=3,
                  annotation_text=f"{nome} = {linha['s_ou_S']}", annotation_position="top right")
    fig.update_layout(xaxis_title=f"demanda somada em {int(linha['janela'])} dias (un)", yaxis_title="nº de janelas",
                      showlegend=False)
    return fig


def pagina_simulacao(r):
    st.header("5 · Simulação no período de teste (99%)")
    st.markdown(f"Parâmetros **fixos**, demanda **real** do teste. A cada dia: **(1)** chega o que estava previsto, "
                f"**(2)** vende-se o possível (o resto é venda perdida), **(3)** no fim do dia olha-se a **posição** "
                f"(físico + em trânsito) e decide-se o pedido, **(4)** que chega em t + L + 1 = t + {ce.LEAD_TIME + 1}.")
    sim, datas, par = r["sim"], r["datas_teste"], r["par"]
    res = sim["resumo"]
    c = st.columns(len(res))
    for col, (_, row) in zip(c, res.iterrows()):
        with col:
            st.markdown(f"#### {row['sistema']} · {row['itens']}")
            a, b, cc = st.columns(3)
            metrica(a, "Custo total", libra(row["custo_total"]), real(row["custo_total_BRL"]))
            metrica(b, "Fill rate", pct(row['fill_rate']), f"{fmt(row['vendas_perdidas'])} un perdidas")
            metrica(cc, "Ciclos sem falta", f"{row['ciclos_sem_falta']} de {row['ciclos']}", f"{row['pedidos']} pedidos")

    st.subheader("Evolução do estoque")
    dia = st.slider("Dia do teste", 1, len(datas), len(datas))
    for it in r["itens"]:
        if it == r["item_sQ"]:
            s_ = sim["sQ"]
            e, p, ped, rec = s_["estoque"], s_["perda"], s_["pedido"], s_["recebimentos"]
            pol, nome = "⟨s,Q⟩", "s"
        else:
            s_ = sim["RS"]
            e, p, ped, rec = s_["estoque"][it], s_["perda"][it], s_["pedido"][it], s_["recebimentos"][it]
            pol, nome = "⟨R,S⟩", "S"
        d = sim["dem"][it]
        lim = par[par["item"] == it]["s_ou_S"].iloc[0]
        k = dia
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=datas[:k], y=d[:k], name="demanda do dia", marker_color="#e6e6e6", secondary_y=False)
        fig.add_scatter(x=datas[:k], y=e[:k], name="estoque físico (fim do dia)", line=dict(color=AZUL, width=2,
                        shape="hv"), secondary_y=False)
        m = ped[:k] > 0
        fig.add_scatter(x=datas[:k][m], y=e[:k][m], mode="markers", name="pedido emitido",
                        marker=dict(color=VERDE, symbol="triangle-up", size=11),
                        customdata=ped[:k][m], hovertemplate="pedido de %{customdata:.0f} un<extra></extra>")
        m = rec[:k] > 0
        fig.add_scatter(x=datas[:k][m], y=e[:k][m], mode="markers", name="recebimento",
                        marker=dict(color=AZUL, symbol="square", size=9),
                        customdata=rec[:k][m], hovertemplate="chegaram %{customdata:.0f} un<extra></extra>")
        m = p[:k] > 0
        fig.add_scatter(x=datas[:k][m], y=np.zeros(m.sum()), mode="markers", name="venda perdida",
                        marker=dict(color=VERMELHO, symbol="triangle-down", size=11),
                        customdata=p[:k][m], hovertemplate="perdeu %{customdata:.0f} un<extra></extra>")
        fig.add_hline(y=lim, line_dash="dot", line_color=CINZA, annotation_text=f"{nome} = {fmt(lim)}")
        fig.update_xaxes(range=[datas[0] - pd.Timedelta(days=1), datas[-1] + pd.Timedelta(days=1)])
        fig.update_yaxes(range=[-0.05 * max(e.max(), lim) * 1.15, max(e.max(), lim, d.max()) * 1.15], secondary_y=False)
        fig.update_layout(title=f"{it} – {r['est'].loc[it, 'descricao']}  {pol}", yaxis_title="unidades")
        st.plotly_chart(layout(fig, 300), width="stretch")

    with st.expander(f"O que aconteceu no dia {dia} ({datas[dia - 1].strftime('%d/%m/%Y')})", expanded=True):
        t = dia - 1
        linhas = []
        for it in r["itens"]:
            if it == r["item_sQ"]:
                s_ = sim["sQ"]; e, p, ped, rec = s_["estoque"], s_["perda"], s_["pedido"], s_["recebimentos"]
            else:
                s_ = sim["RS"]; e, p, ped, rec = (s_["estoque"][it], s_["perda"][it], s_["pedido"][it],
                                                  s_["recebimentos"][it])
            ini = s_["inicial"] if it == r["item_sQ"] else s_["inicial"][it]
            antes = e[t - 1] if t > 0 else ini
            linhas.append({"Item": it, "Estoque ao abrir": antes, "(1) Chegou": rec[t], "Disponível": antes + rec[t],
                           "Demanda": sim["dem"][it][t], "(2) Vendido": sim["dem"][it][t] - p[t],
                           "Venda perdida": p[t], "Estoque ao fechar": e[t], "(3-4) Pedido emitido": ped[t]})
        st.dataframe(pd.DataFrame(linhas).style.format({c: fmt for c in linhas[0] if c != "Item"}),
                     hide_index=True, width="stretch")

    st.subheader("Custos no teste")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = go.Figure()
        for col, nome, cor in [("custo_fixo", "fixo", CINZA), ("custo_variavel", "variável (compra)", AZUL_CLARO),
                               ("custo_armazenagem", "armazenagem", AZUL), ("custo_falta", "falta", VERMELHO)]:
            fig.add_bar(y=res["sistema"], x=res[col], name=nome, orientation="h", marker_color=cor)
        fig.update_layout(barmode="stack", xaxis_title="£ no período de teste")
        st.plotly_chart(layout(fig, 260), width="stretch")
    with c2:
        t = res[["sistema", "pedidos", "custo_fixo", "custo_variavel", "custo_armazenagem", "custo_falta",
                 "custo_total", "custo_total_BRL"]].copy()
        t.columns = ["Sistema", "Pedidos", "Fixo £", "Variável £", "Armaz. £", "Falta £", "Total £", "Total R$"]
        st.dataframe(t.style.format({c: (lambda v: fmt(v, 2)) for c in t.columns[2:]}), hide_index=True, width="stretch")
        st.caption("O custo variável domina e depende só do volume comprado. A compra do estoque inicial "
                   "(condição assumida) fica fora do total: " +
                   "; ".join(f"{row['sistema']}: {libra(row['aquisicao_estoque_inicial'])}" for _, row in res.iterrows()))
    ver_codigo(ce.simular_sQ, ce.simular_RS, ce.ciclos_com_falta)


def pagina_bonus(r):
    st.header("6 · Bônus: previsão de demanda e dimensionamento de lotes")
    b = r["bonus"]
    itens = r["itens_RS"]
    st.markdown(f"Para o par conjunto, a demanda é somada em **períodos de {ce.DIAS_POR_PERIODO} dias** "
                f"(= lead time, então o lead time vira 1 período). Escolhemos a previsão pelo menor **MAE** nos "
                f"últimos 8 períodos do treino e planejamos **{ce.HORIZONTE} períodos** com programação inteira.")

    st.subheader("1) Previsão")
    st.markdown("<style>[data-testid='stMetricValue'] div {font-size:1.25rem; white-space:normal;}</style>", unsafe_allow_html=True)
    cols = st.columns(len(itens))
    for col, it in zip(cols, itens):
        mt = b["metricas"][it]
        with col:
            y, aj, pv, re_ = b["hist"][it], b["ajustes"][it], b["prev"][it], b["real"][it]
            nh = len(y)
            fig = go.Figure()
            fig.add_scatter(x=list(range(-nh + 1, 1)), y=y, name="histórico (treino)", line=dict(color=CINZA))
            fig.add_scatter(x=list(range(-nh + 1, 1)), y=aj, name="ajuste 1 passo", line=dict(color=AZUL_CLARO, dash="dot"))
            fig.add_scatter(x=list(range(1, len(pv) + 1)), y=pv, name="previsão", line=dict(color=AZUL, width=3))
            fig.add_scatter(x=list(range(1, len(re_) + 1)), y=re_, name="real (teste)", mode="lines+markers",
                            line=dict(color=LARANJA))
            fig.add_vline(x=0.5, line_dash="dash")
            fig.add_vrect(x0=-7.5, x1=0.5, fillcolor="#f2f2f2", line_width=0, layer="below",
                          annotation_text="validação", annotation_position="bottom left")
            fig.update_layout(title=f"{it} – {r['est'].loc[it, 'descricao']}", xaxis_title="período (6 dias)",
                              yaxis_title="un / período")
            st.plotly_chart(layout(fig, 320), width="stretch")
            a, bb, c = st.columns(3)
            metrica(a, "Modelo", "Holt amort." if "Holt" in mt["modelo"] else mt["modelo"].split(" a=")[0],
                    mt["modelo"][mt["modelo"].find("a="):] if "a=" in mt["modelo"] else "")
            metrica(bb, "MAE valid. → teste", f"{fmt(mt['mae_validacao'])} → {fmt(mt['mae_teste'])}", "un / período")
            metrica(c, "Estoque mínimo", fmt(mt["estoque_minimo"]), f"z₀,₉₉ × σ do erro ({fmt(mt['sd_erro'])})")
    ver_codigo(ce.em_periodos, ce.suav_exponencial, ce.escolher_previsao, ce.prever)

    st.divider()
    st.subheader("2) Modelo de programação inteira mista (MILP)")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("**Variáveis** (item i, período t): x = unidades recebidas (inteiro) · I = estoque no fim · "
                    "z = item entra no pedido (0/1) · y = houve pedido conjunto (0/1)")
        st.latex(r"\min\;\sum_{i,t}\big(c_i x_{it}+h_i I_{it}+k\,z_{it}\big)+\sum_t K_0\,y_t")
        st.latex(r"\begin{aligned}"
                 r"&I_{it}=I_{i,t-1}+x_{it}-D_{it} && \text{balanço}\\"
                 r"&x_{it}\le M_{it}\,z_{it},\quad z_{it}\le y_t && \text{liga lote, item e pedido}\\"
                 r"&\textstyle\sum_i v_i\,(I_{i,t-1}+x_{it})\le \text{cap} && \text{depósito}\\"
                 r"&I_{it}\ge I^{\min}_i && \text{proíbe falta}\\"
                 r"&x_{i1}=0 && \text{lead time de 1 período}"
                 r"\end{aligned}")
    with c2:
        st.markdown(f"""
| Dado | Valor |
|---|---|
| Custo fixo conjunto K₀ | {libra(ce.K_CONJUNTO)} |
| Custo por item k | {libra(ce.K_POR_ITEM)} |
| Capacidade | {fmt(ce.CAPACIDADE)} unid.-equivalentes |
| Volume por unidade | {", ".join(f"{i}: {fmt(b['volume'][i], 1)}" for i in itens)} |
| Estoque inicial | {", ".join(f"{i}: {fmt(b['I0'][i])}" for i in itens)} (mínimo + 1º período) |
| Solver | HiGHS (`scipy.optimize.milp`) |
""")

    st.subheader("3) Plano ótimo")
    pl, sc = b["plano"], b["sem_cap"]
    pico = lambda oc: max(sum(oc[i][t] for i in itens) for t in range(ce.HORIZONTE))
    c = st.columns(3)
    metrica(c[0], "Pedidos conjuntos", int(pl["y"].sum()), f"{int(sc['y'].sum())} sem limite de capacidade")
    metrica(c[1], "Custo do plano", libra(pl["custo"]), f"{libra(sc['custo'])} sem limite")
    metrica(c[2], "Pico de ocupação", fmt(pico(b["ocup"])), f"{fmt(pico(b['ocup_sem']))} sem limite")

    ver_sem = st.toggle("Comparar com o plano sem limite de capacidade")
    oc = b["ocup_sem"] if ver_sem else b["ocup"]
    fig = go.Figure()
    for it, cor in zip(itens, [AZUL_CLARO, AZUL]):
        fig.add_bar(x=list(range(1, ce.HORIZONTE + 1)), y=oc[it], name=f"{it} (logo após o recebimento)", marker_color=cor)
    fig.add_hline(y=ce.CAPACIDADE, line_dash="dash", line_color=VERMELHO, annotation_text="capacidade")
    fig.update_layout(barmode="stack", xaxis_title="período futuro", yaxis_title="unid.-equivalentes",
                      title="Ocupação do depósito" + (" — SEM limite" if ver_sem else " — com limite de capacidade"))
    st.plotly_chart(layout(fig, 320), width="stretch")

    plano = pl if not ver_sem else sc
    df = pd.DataFrame({"Período": range(1, ce.HORIZONTE + 1), "Pedido conjunto": plano["y"].astype(int)})
    for it in itens:
        df[f"{it} previsão"] = b["D"][it]
        df[f"{it} recebe"] = plano["x"][it]
        df[f"{it} estoque fim"] = plano["I"][it]
    st.dataframe(df.style.format({c: fmt for c in df.columns[1:]})
                 .apply(lambda s: ["background-color:#e8f1fa" if v else "" for v in df["Pedido conjunto"]], axis=0),
                 hide_index=True, width="stretch")
    st.caption(f"x é recebimento; o pedido é emitido {pl['atraso_periodos']} período antes. Status do solver: {pl['status']}")

    st.subheader("4) O plano aguentaria a demanda real?")
    st.markdown(" · ".join(f"**{it}**: faltariam {fmt(b['metricas'][it]['falta_teste'])} de "
                           f"{fmt(b['metricas'][it]['demanda_teste'])} unidades ({len(b['real'][it])} períodos de teste)"
                           for it in itens))
    st.caption("Não ter falta na previsão não garante que nunca vai faltar na prática; o estoque mínimo é a proteção.")
    ver_codigo(ce.dimensionar_lotes)


def pagina_codigo(r):
    st.header("7 · Código completo")
    texto, funcoes = codigo_fonte()
    if not texto:
        st.warning("Não foi possível ler controle_estoque.py.")
        return
    ordem = sorted(funcoes, key=lambda n: funcoes[n][0])
    escolha = st.selectbox("Função", ["(arquivo inteiro)"] + ordem)
    if escolha == "(arquivo inteiro)":
        st.code(texto, language="python", line_numbers=True)
    else:
        codigo, inicio = trecho(escolha)
        st.caption(f"Linhas {inicio}–{funcoes[escolha][1]} de controle_estoque.py")
        st.code(codigo, language="python", line_numbers=True)


PAGINAS = {
    "Início": pagina_inicio,
    "1 · Dados e limpeza": pagina_dados,
    "2 · Treino/teste e seleção": pagina_selecao,
    "3 · Demanda e outliers": pagina_demanda,
    "4 · Custos e parâmetros": pagina_parametros,
    "5 · Simulação no teste": pagina_simulacao,
    "6 · Bônus: previsão e lotes": pagina_bonus,
    "7 · Código completo": pagina_codigo,
}


def main():
    st.sidebar.title("Controle de Estoque")
    st.sidebar.caption("Pesquisa Operacional  · UFRJ ")
    pagina = st.sidebar.radio("Etapas", list(PAGINAS), label_visibility="collapsed")
    with st.spinner("Carregando os resultados…"):
        r = resultados()
    if r is None:
        st.error(f"Não encontrei {NOME_BASE} nem resultados_app.pkl na pasta do projeto.")
        return
    PAGINAS[pagina](r)


if __name__ == "__main__":
    main()
