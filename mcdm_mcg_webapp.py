# -*- coding: utf-8 -*-
"""
MCDM Dashboard — Sistema de Apoio à Decisão Multicritério
Modelos de Decisão | MEGI ISEL 2025/2026

Atualizações:
- Entrada 100% Manual via interface (Tabelas Dinâmicas para Critérios e Matriz).
- Painel de Requisitos e Parâmetros otimizado.
- O AHP injeta os seus pesos globalmente.
- Correção do Método de Borda no Dashboard.
"""

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÃO DE PÁGINA
# =============================================================================
st.set_page_config(page_title="MCDM Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stMetric { background: rgba(120,120,120,0.06); padding: 0.6rem; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 MCDM Dashboard — Priorização Multicritério")
st.caption("Modelos de Decisão | Definição Manual de Requisitos e Matrizes")

# =============================================================================
# FUNÇÕES MATEMÁTICAS GERAIS E MCDM
# =============================================================================
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

def safe_call(fn, *args, **kwargs):
    try: return fn(*args, **kwargs), None
    except Exception as exc: return None, f"{type(exc).__name__}: {exc}"

def normalize_minmax(mat, types):
    mat = np.asarray(mat, dtype=float)
    out = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        rng = col.max() - col.min()
        if rng == 0: out[:, j] = 1.0; continue
        if types[j] == "max": out[:, j] = (col - col.min()) / rng
        else: out[:, j] = (col.max() - col) / rng
    return out

def normalize_vector(mat):
    mat = np.asarray(mat, dtype=float)
    denom = np.sqrt(np.sum(mat ** 2, axis=0))
    denom = np.where(denom == 0, 1.0, denom)
    return mat / denom

def normalize_sum(mat, types):
    mat = np.asarray(mat, dtype=float)
    out = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        if types[j] == "max":
            s = col.sum()
            out[:, j] = col / s if s != 0 else 1.0 / len(col)
        else:
            inv = 1.0 / np.where(col == 0, 1e-9, col)
            s = inv.sum()
            out[:, j] = inv / s if s != 0 else 1.0 / len(col)
    return out

def ranking_from_scores(scores, higher_is_better=True):
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores if higher_is_better else scores)
    rank = np.zeros(len(scores), dtype=int)
    rank[order] = np.arange(1, len(scores) + 1)
    return rank

# Modelos
def model_saw(mat, weights, types):
    norm = normalize_minmax(mat, types)
    scores = (norm * weights).sum(axis=1)
    return {"scores": scores, "ranking": ranking_from_scores(scores)}

def model_topsis(mat, weights, types):
    norm = normalize_vector(mat)
    weighted = norm * weights
    ideal = np.array([weighted[:, j].max() if types[j] == "max" else weighted[:, j].min() for j in range(mat.shape[1])])
    anti = np.array([weighted[:, j].min() if types[j] == "max" else weighted[:, j].max() for j in range(mat.shape[1])])
    d_plus = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted - anti) ** 2, axis=1))
    denom = np.where((d_plus + d_minus) == 0, 1e-9, d_plus + d_minus)
    ci = d_minus / denom
    return {"scores": ci, "ranking": ranking_from_scores(ci), "ideal": ideal, "anti_ideal": anti}

def preference(d, ftype="usual", p=None, q=None, sigma=None):
    if d <= 0: return 0.0
    if ftype == "usual": return 1.0
    if ftype == "linear": return float(min(d / p, 1.0)) if p else 0.0
    if ftype == "gaussian": return float(1.0 - np.exp(-(d ** 2) / (2 * (sigma or 1.0) ** 2)))
    return 0.0

def model_promethee(mat, weights, types, function="linear"):
    n_alt, n_crit = mat.shape
    pref = np.zeros((n_alt, n_alt))
    for j in range(n_crit):
        col = mat[:, j]
        rng = col.max() - col.min()
        p = rng * 0.5 if rng > 0 else 1.0
        sigma = rng * 0.3 if rng > 0 else 1.0
        for i in range(n_alt):
            for k in range(n_alt):
                if i == k: continue
                d = (col[i] - col[k]) if types[j] == "max" else (col[k] - col[i])
                pref[i, k] += weights[j] * preference(d, function, p=p, sigma=sigma)
    div = max(n_alt - 1, 1)
    phi_plus = pref.sum(axis=1) / div
    phi_minus = pref.sum(axis=0) / div
    phi_net = phi_plus - phi_minus
    return {"phi_plus": phi_plus, "phi_minus": phi_minus, "scores": phi_net, "ranking": ranking_from_scores(phi_net)}

def model_electre(mat, weights, types, c_thresh=0.6, d_thresh=0.4):
    n_alt, n_crit = mat.shape
    norm = normalize_minmax(mat, types)
    w_sum = weights.sum() if weights.sum() > 0 else 1.0
    concordance = np.zeros((n_alt, n_alt))
    discordance = np.zeros((n_alt, n_alt))
    grange = max(norm.max() - norm.min(), 1e-9)

    for i in range(n_alt):
        for k in range(n_alt):
            if i == k: continue
            concordance[i, k] = sum(weights[j] for j in range(n_crit) if norm[i, j] >= norm[k, j]) / w_sum
            diffs = [norm[k, j] - norm[i, j] for j in range(n_crit) if norm[k, j] > norm[i, j]]
            discordance[i, k] = (max(diffs) / grange) if diffs else 0.0

    outrank = (concordance >= c_thresh) & (discordance <= d_thresh)
    np.fill_diagonal(outrank, False)
    net_dominance = outrank.sum(axis=1) - outrank.sum(axis=0)
    return {"scores": net_dominance.astype(float), "ranking": ranking_from_scores(net_dominance.astype(float))}

def model_vikor(mat, weights, types, v=0.5):
    n_alt, n_crit = mat.shape
    f_best = np.array([mat[:, j].max() if types[j] == "max" else mat[:, j].min() for j in range(n_crit)])
    f_worst = np.array([mat[:, j].min() if types[j] == "max" else mat[:, j].max() for j in range(n_crit)])
    rng = np.where((f_best - f_worst) == 0, 1e-9, f_best - f_worst)
    S, R = np.zeros(n_alt), np.zeros(n_alt)
    for i in range(n_alt):
        terms = np.array([weights[j] * ((f_best[j] - mat[i, j]) if types[j] == "max" else (mat[i, j] - f_best[j])) / rng[j] for j in range(n_crit)])
        S[i], R[i] = terms.sum(), terms.max()
    s_rng, r_rng = max(S.max() - S.min(), 1e-9), max(R.max() - R.min(), 1e-9)
    Q = v * (S - S.min()) / s_rng + (1 - v) * (R - R.min()) / r_rng
    return {"scores": -Q, "ranking": ranking_from_scores(-Q)}

def model_copras(mat, weights, types):
    norm = normalize_sum(mat, types)
    weighted = norm * weights
    benefit = [j for j in range(mat.shape[1]) if types[j] == "max"]
    cost = [j for j in range(mat.shape[1]) if types[j] == "min"]
    S_plus = weighted[:, benefit].sum(axis=1) if benefit else np.zeros(mat.shape[0])
    S_minus = weighted[:, cost].sum(axis=1) if cost else np.zeros(mat.shape[0])
    if cost and S_minus.min() > 0:
        S_minus_safe = np.where(S_minus == 0, 1e-9, S_minus)
        sum_inv = (1 / S_minus_safe).sum()
        Q = S_plus + (S_minus.min() * sum_inv) / (S_minus_safe * sum_inv)
    else: Q = S_plus
    N = (Q / Q.max()) * 100 if Q.max() != 0 else Q
    return {"scores": N, "ranking": ranking_from_scores(N)}

def model_dematel(mat, weights, types):
    try:
        Z = np.abs(np.corrcoef(mat.T))
        np.fill_diagonal(Z, 0)
        s = max(Z.sum(axis=1).max(), Z.sum(axis=0).max())
        X = Z / s if s > 0 else Z
        T = X @ np.linalg.inv(np.eye(X.shape[0]) - X)
    except: T = np.eye(mat.shape[1])
    D = T.sum(axis=1)
    R = T.sum(axis=0)
    prominence = D + R
    adj = weights * prominence if prominence.sum() > 0 else weights
    adj = adj / adj.sum()
    norm = normalize_minmax(mat, types)
    scores = (norm * adj).sum(axis=1)
    return {"scores": scores, "ranking": ranking_from_scores(scores)}


# =============================================================================
# SIDEBAR: MODO DE ENTRADA E REQUISITOS (PARÂMETROS)
# =============================================================================
with st.sidebar:
    st.header("⚙️ Estrutura do Problema")
    input_method = st.radio("Método de Entrada dos Dados:", ["Entrada Manual", "Carregar Excel", "Dados de Demonstração"])
    
    st.divider()
    
    # SE MODO MANUAL
    if input_method == "Entrada Manual":
        st.subheader("📐 Dimensões")
        n_alts = st.number_input("Nº de Alternativas", min_value=2, max_value=50, value=4, step=1)
        n_crits = st.number_input("Nº de Critérios", min_value=2, max_value=20, value=4, step=1)
    elif input_method == "Carregar Excel":
        uploaded = st.file_uploader("Carregar Excel (.xlsx)", type=["xlsx", "xls"])
    
    st.divider()
    st.subheader("🎛️ Requisitos dos Modelos")
    st.caption("Ajuste os parâmetros abaixo para calibrar os modelos multicritério.")
    c_thresh = st.slider("ELECTRE: Limiar de Concordância (c)", 0.50, 0.95, 0.65, 0.01)
    d_thresh = st.slider("ELECTRE: Limiar de Discordância (d)", 0.05, 0.50, 0.35, 0.01)
    vikor_v = st.slider("VIKOR: Peso da estratégia (v)", 0.0, 1.0, 0.5, 0.05, help="1 = Máxima utilidade; 0 = Mínimo arrependimento")
    promethee_fn = st.selectbox("PROMETHEE: Função de Preferência", ["usual", "linear", "gaussian"], index=1)

# =============================================================================
# TAB 0 — PREENCHIMENTO E SETUP GERAL
# =============================================================================
TAB_LABELS = ["📋 Setup e Dados Base", "⚖️ Motores de Pesos", "🎯 TOPSIS", "🔗 ELECTRE", "📊 PROMETHEE", "⚖️ VIKOR", "🧮 COPRAS", "🌐 DEMATEL", "🏆 Dashboard Consolidado"]
tabs = st.tabs(TAB_LABELS)

# Variáveis globais de dados
alts, criteria, weights, types, mat = [], [], [], [], None
is_ready = False

with tabs[0]:
    if input_method == "Entrada Manual":
        st.header("🛠️ Configuração Manual do Problema")
        
        # 1. SETUP DE CRITÉRIOS
        st.subheader("1. Setup de Critérios")
        st.markdown("Defina o nome dos critérios, se são de **Benefício (max)** ou **Custo (min)** e o seu peso inicial.")
        
        # Criar dataframe por defeito para os critérios
        default_crits = pd.DataFrame({
            "Critério": [f"C{i+1}" for i in range(n_crits)],
            "Sentido": ["max"] * n_crits,
            "Peso": [1.0/n_crits] * n_crits
        })
        
        edited_crits = st.data_editor(
            default_crits, 
            column_config={
                "Critério": st.column_config.TextColumn("Nome do Critério", required=True),
                "Sentido": st.column_config.SelectboxColumn("Max / Min", options=["max", "min"], required=True),
                "Peso": st.column_config.NumberColumn("Peso (0 a 1)", min_value=0.0, step=0.05)
            },
            hide_index=True, use_container_width=True, key="crit_editor"
        )
        
        criteria = edited_crits["Critério"].tolist()
        types = edited_crits["Sentido"].tolist()
        weights_raw = np.array(edited_crits["Peso"].tolist())
        weights = weights_raw / weights_raw.sum() if weights_raw.sum() > 0 else np.ones(n_crits)/n_crits
        
        st.divider()
        
        # 2. PREENCHIMENTO DA MATRIZ
        st.subheader("2. Matriz de Decisão")
        st.markdown("Preencha as avaliações de cada alternativa para cada critério. As colunas são geradas automaticamente pelo passo anterior.")
        
        # Criar dataframe por defeito para a matriz
        default_matrix = pd.DataFrame(0.0, index=range(n_alts), columns=criteria)
        default_matrix.insert(0, "Alternativa", [f"Alt {i+1}" for i in range(n_alts)])
        
        edited_matrix = st.data_editor(
            default_matrix,
            column_config={"Alternativa": st.column_config.TextColumn("Alternativa", required=True)},
            hide_index=True, use_container_width=True, key="mat_editor"
        )
        
        alts = edited_matrix["Alternativa"].tolist()
        mat = edited_matrix[criteria].astype(float).values
        is_ready = True

    elif input_method == "Dados de Demonstração":
        st.header("📋 Dados de Demonstração (Caso MCG)")
        df = pd.DataFrame({
            "Alternativa": [f"A{i}" for i in range(1, 10)],
            "C1_VP": [250_000_000, 300_000, 900_000, 650_000, 5_000_000, 1_350_000, 10_500_000, 3_450_000, 15_000_000],
            "C2_PF": [0.25, 0.35, 0.50, 0.50, 0.40, 0.50, 0.40, 0.40, 0.60],
            "C3_EE": [24, 8, 8, 8, 24, 8, 16, 8, 24],
            "C4_FE": [4, 5, 3, 3, 4, 3, 3, 3, 4],
            "C5_UD": [180, 60, 60, 90, 30, 60, 180, 60, 300],
            "C6_RC": [4, 5, 5, 3, 3, 5, 4, 4, 3],
        })
        alts = df["Alternativa"].tolist()
        criteria = [c for c in df.columns if c != "Alternativa"]
        types = ["max", "max", "min", "max", "min", "max"]
        weights = np.array([0.4615, 0.1987, 0.0230, 0.0972, 0.0217, 0.1979])
        mat = df[criteria].astype(float).values
        
        st.write("**Critérios:**")
        st.dataframe(pd.DataFrame({"Critério": criteria, "Sentido": types, "Peso": weights}), hide_index=True)
        st.write("**Matriz:**")
        st.dataframe(df, hide_index=True)
        is_ready = True

    elif input_method == "Carregar Excel":
        st.header("📋 Dados do Excel")
        if uploaded:
            try:
                raw_df = pd.read_excel(pd.ExcelFile(uploaded), sheet_name="Dados")
                alts = raw_df.iloc[:, 0].astype(str).tolist()
                criteria = raw_df.select_dtypes(include=[np.number]).columns.tolist()
                mat = raw_df[criteria].astype(float).values
                types = ["max"] * len(criteria)
                weights = np.ones(len(criteria)) / len(criteria)
                st.success("Ficheiro carregado! Como usou Excel, assumiram-se pesos iguais e maximização. Use a Entrada Manual para maior controlo.")
                st.dataframe(raw_df, hide_index=True)
                is_ready = True
            except Exception as e:
                st.error(f"Erro a ler Excel: {e}")
        else:
            st.info("👈 Por favor carregue um ficheiro na barra lateral.")


# =============================================================================
# BLOQUEIO DE SEGURANÇA SE DADOS NÃO ESTIVEREM PRONTOS
# =============================================================================
if not is_ready:
    st.stop()


# =============================================================================
# TAB 1 — MOTORES DE PESOS (AHP INJECTA GLOBALMENTE)
# =============================================================================
with tabs[1]:
    st.header("⚖️ Motores de Geração de Pesos")
    st.markdown("Selecione o método AHP para gerar novos pesos baseados em comparações par-a-par. **Se gerar pesos aqui, eles irão sobrescrever os pesos definidos no Setup para todos os modelos seguintes.**")
    
    n_crit = len(criteria)
    
    st.subheader("🔺 Analytic Hierarchy Process (AHP)")
    st.markdown("**Passo 1: Matriz de Comparação Par-a-Par**. Edite a matriz abaixo (Escala de Saaty: 1 a 9).")
    
    if "ahp_matrix" not in st.session_state or st.session_state.ahp_matrix.shape != (n_crit, n_crit):
        st.session_state.ahp_matrix = np.ones((n_crit, n_crit))
        
    pw_df = pd.DataFrame(st.session_state.ahp_matrix, index=criteria, columns=criteria).round(4)
    edited_pw = st.data_editor(pw_df, use_container_width=True, key="ahp_manual")
    
    # Forçar reciprocidade
    E = edited_pw.values.astype(float).copy()
    for i in range(n_crit):
        for j in range(n_crit):
            if i == j: E[i, j] = 1.0
            elif i < j and E[i, j] != 0: E[j, i] = 1.0 / E[i, j]
    st.session_state.ahp_matrix = E
    
    col_sums = E.sum(axis=0)
    norm_E = E / col_sums
    approx_weights = norm_E.mean(axis=1)
    
    AW = E.dot(approx_weights)
    lambda_max = (AW / approx_weights).mean()
    CI = (lambda_max - n_crit) / (n_crit - 1) if n_crit > 1 else 0
    RI = RI_TABLE.get(n_crit, 1.59)
    CR = CI / RI if RI > 0 else 0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("λ_max", f"{lambda_max:.4f}")
    m2.metric("CI", f"{CI:.4f}")
    m3.metric("CR", f"{CR:.4f}", delta="Válido (<0.10)" if CR <= 0.10 else "Rever Valores", delta_color="normal" if CR <= 0.10 else "inverse")
    
    use_ahp = st.checkbox("🔥 **INJETAR ESTES PESOS AHP NO SISTEMA**", value=False)
    if use_ahp:
        weights = approx_weights  # OVERRIDE GLOBAL
        st.success("Pesos AHP ativados globalmente para TOPSIS, ELECTRE, etc!")
    
    st.write("Pesos Calculados:")
    st.dataframe(pd.DataFrame({"Critério": criteria, "Peso": approx_weights}).style.format({"Peso": "{:.4f}"}), hide_index=True)


# =============================================================================
# EXECUTAR TODOS OS MODELOS E ARMAZENAR RESULTADOS
# =============================================================================
all_results = {}

res_topsis, _ = safe_call(model_topsis, mat, weights, types)
if res_topsis: all_results["TOPSIS"] = res_topsis

res_electre, _ = safe_call(model_electre, mat, weights, types, c_thresh, d_thresh)
if res_electre: all_results["ELECTRE"] = res_electre

res_promethee, _ = safe_call(model_promethee, mat, weights, types, promethee_fn)
if res_promethee: all_results["PROMETHEE"] = res_promethee

res_vikor, _ = safe_call(model_vikor, mat, weights, types, vikor_v)
if res_vikor: all_results["VIKOR"] = res_vikor

res_copras, _ = safe_call(model_copras, mat, weights, types)
if res_copras: all_results["COPRAS"] = res_copras

res_dematel, _ = safe_call(model_dematel, mat, weights, types)
if res_dematel: all_results["DEMATEL"] = res_dematel

# =============================================================================
# TABS DE APRESENTAÇÃO DE RESULTADOS
# =============================================================================
def show_results(model_name, res_dict, sort_col, format_str="{:.4f}"):
    st.header(model_name)
    rdf = pd.DataFrame({"Alternativa": alts, "Score": res_dict["scores"], "Ranking": res_dict["ranking"]}).sort_values("Ranking")
    st.dataframe(rdf.style.format({"Score": format_str}), hide_index=True, use_container_width=True)

with tabs[2]: show_results("🎯 TOPSIS", all_results.get("TOPSIS", {}), "Score")
with tabs[3]: show_results("🔗 ELECTRE I", all_results.get("ELECTRE", {}), "Score")
with tabs[4]: show_results("📊 PROMETHEE II", all_results.get("PROMETHEE", {}), "Score")
with tabs[5]: show_results("⚖️ VIKOR", all_results.get("VIKOR", {}), "Score")
with tabs[6]: show_results("🧮 COPRAS", all_results.get("COPRAS", {}), "Score")
with tabs[7]: show_results("🌐 DEMATEL", all_results.get("DEMATEL", {}), "Score")

# =============================================================================
# TAB DASHBOARD CONSOLIDADO (Corrigido para Borda Invertido)
# =============================================================================
with tabs[8]:
    st.header("🏆 Dashboard Consolidado")
    if not all_results:
        st.warning("Sem resultados.")
    else:
        models_with_results = list(all_results.keys())
        rank_table = pd.DataFrame({"Alternativa": alts})
        for m in models_with_results: rank_table[m] = all_results[m]["ranking"]
        
        # Média de Posições (Quanto Menor, Melhor)
        rank_table["Posição Média"] = rank_table[models_with_results].mean(axis=1).round(2)
        
        # Correção aqui: higher_is_better=False, o valor mais baixo ganha
        rank_table["Ranking Final"] = ranking_from_scores(rank_table["Posição Média"].values, higher_is_better=False)
        rank_table = rank_table.sort_values("Ranking Final").reset_index(drop=True)

        st.subheader("Tabela de Rankings Consolidados (Método de Borda)")
        st.info("A agregação utiliza a **média das posições**. Um valor mais baixo (próximo de 1) indica lugares superiores no pódio consistentemente.")
        
        styled = rank_table.style.format({"Posição Média": "{:.2f}"}).background_gradient(subset=models_with_results + ["Posição Média", "Ranking Final"], cmap="RdYlGn_r")
        st.dataframe(styled, use_container_width=True, hide_index=True)

        top3_alts = rank_table.sort_values("Ranking Final").head(3)["Alternativa"].tolist()
        c1, c2, c3 = st.columns(3)
        c1.metric("🥇 1º lugar", top3_alts[0] if len(top3_alts) > 0 else "—")
        c2.metric("🥈 2º lugar", top3_alts[1] if len(top3_alts) > 1 else "—")
        c3.metric("🥉 3º lugar", top3_alts[2] if len(top3_alts) > 2 else "—")
