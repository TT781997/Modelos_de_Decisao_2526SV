# -*- coding: utf-8 -*-
"""
MCDM Dashboard — Sistema de Apoio à Decisão Multicritério
Modelos de Decisão | MEGI ISEL 2025/2026

Versão Merged Definitiva:
- Tabelas Dinâmicas Manuais c/ Preservação de Estado E Carregamento de Excel.
- Motores de Pesos Global (AHP, SWING, SMART, Entropia, CRITIC).
- Teoria, Passo-a-Passo Matemático e Análise de Sensibilidade Universal.
- Dashboard Consolidado (Borda), Gráficos Radar e Relatório Dinâmico Exportável.
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
st.set_page_config(page_title="MCDM Full Framework", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stMetric { background: rgba(120,120,120,0.06); padding: 0.6rem; border-radius: 8px; }
    .theory-box { background-color: rgba(30,144,255,0.05); border-left: 4px solid #1E90FF; padding: 1rem; border-radius: 4px; margin-bottom: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Framework Multicritério MCDM Consolidado")
st.caption("MEGI ISEL | Integração Completa: Matemática, Sensibilidade e Relatórios")

RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# =============================================================================
# INICIALIZAÇÃO DE ESTADO (MEMÓRIA DAS TABELAS DINÂMICAS)
# =============================================================================
if "manual_criteria" not in st.session_state:
    st.session_state.manual_criteria = pd.DataFrame({
        "Critério": ["C1", "C2", "C3"],
        "Sentido": ["max", "max", "min"],
        "Peso Inicial": [0.4, 0.4, 0.2]
    })

if "manual_matrix" not in st.session_state:
    st.session_state.manual_matrix = pd.DataFrame({
        "Alternativa": ["Alt 1", "Alt 2", "Alt 3"],
        "C1": [10.0, 20.0, 15.0],
        "C2": [15.0, 12.0, 18.0],
        "C3": [5.0, 8.0, 6.0]
    })

# =============================================================================
# FUNÇÕES DE CARREGAMENTO DE EXCEL
# =============================================================================
def load_excel(file):
    xls = pd.ExcelFile(file)
    if "Dados" not in xls.sheet_names:
        raise ValueError("A folha 'Dados' não foi encontrada no ficheiro.")
    df = pd.read_excel(xls, sheet_name="Dados")
    id_col = df.columns[0]
    df = df.dropna(subset=[id_col]).reset_index(drop=True)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    crits = [c for c in numeric if c != id_col]
    if not crits:
        raise ValueError("Nenhuma coluna numérica de critério foi detectada.")
    df[id_col] = df[id_col].astype(str)

    has_weights_sheet = "Pesos" in xls.sheet_names
    if has_weights_sheet:
        wdf = pd.read_excel(xls, sheet_name="Pesos", header=None)
        wvals_all = []
        for col in wdf.columns:
            wvals_all.extend(pd.to_numeric(wdf[col], errors="coerce").dropna().tolist())
        wvals = np.array(wvals_all, dtype=float)
        if len(wvals) >= len(crits):
            weights = wvals[:len(crits)]
        else:
            weights = np.ones(len(crits))
            has_weights_sheet = False
    else:
        weights = np.ones(len(crits))
    
    if weights.sum() <= 0 or np.any(weights < 0):
        weights = np.ones(len(crits))
        has_weights_sheet = False
    weights = weights / weights.sum()
    return df, weights, id_col, crits, has_weights_sheet

# =============================================================================
# OPERADORES E NORMALIZAÇÕES MATEMÁTICAS
# =============================================================================
def safe_call(fn, *args, **kwargs):
    try: return fn(*args, **kwargs), None
    except Exception as exc: return None, f"{type(exc).__name__}: {exc}"

def normalize_vector(mat):
    mat = np.asarray(mat, dtype=float)
    denom = np.sqrt(np.sum(mat ** 2, axis=0))
    denom = np.where(denom == 0, 1.0, denom)
    return mat / denom

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

def normalize_sum(mat, types):
    mat = np.asarray(mat, dtype=float)
    out = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        if types[j] == "max":
            s = col.sum(); out[:, j] = col / s if s != 0 else 1.0 / len(col)
        else:
            inv = 1.0 / np.where(col == 0, 1e-9, col)
            s = inv.sum(); out[:, j] = inv / s if s != 0 else 1.0 / len(col)
    return out

def ranking_from_scores(scores, higher_is_better=True):
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores if higher_is_better else scores)
    rank = np.zeros(len(scores), dtype=int)
    rank[order] = np.arange(1, len(scores) + 1)
    return rank

def calc_perturbed_weights(weights, target_idx, factor):
    w = np.array(weights, copy=True)
    w[target_idx] *= factor
    if w[target_idx] >= 1.0:
        w_new = np.zeros_like(w)
        w_new[target_idx] = 1.0
        return w_new
    rem_old = 1.0 - weights[target_idx]
    rem_new = 1.0 - w[target_idx]
    for k in range(len(w)):
        if k != target_idx:
            w[k] = weights[k] * (rem_new / rem_old) if rem_old > 0 else rem_new / (len(w)-1)
    return w

def render_sensitivity(model_func, mat, weights, types, alts, criteria, sens_pct, base_ranking, **kwargs):
    st.markdown("---")
    st.subheader(f"🔄 Análise de Sensibilidade Paralela (±{sens_pct}%)")
    df_plus = pd.DataFrame({"Alternativa": alts, "Base": base_ranking})
    df_minus = pd.DataFrame({"Alternativa": alts, "Base": base_ranking})
    for j, crit in enumerate(criteria):
        w_plus = calc_perturbed_weights(weights, j, 1 + sens_pct / 100.0)
        res_plus, err = safe_call(model_func, mat, w_plus, types, **kwargs)
        if not err and res_plus and "ranking" in res_plus: df_plus[f"+ {crit}"] = res_plus["ranking"]
        w_minus = calc_perturbed_weights(weights, j, 1 - sens_pct / 100.0)
        res_minus, err = safe_call(model_func, mat, w_minus, types, **kwargs)
        if not err and res_minus and "ranking" in res_minus: df_minus[f"- {crit}"] = res_minus["ranking"]
    def style_row(row):
        base = row['Base']
        return ['' if col in ['Alternativa', 'Base'] else ('color: #00B140; font-weight: bold;' if val < base else 'color: #D32F2F; font-weight: bold;' if val > base else 'color: gray;') for col, val in row.items()]
    c1, c2 = st.columns(2)
    with c1: st.write("**Aumento de Peso (+)**"); st.dataframe(df_plus.style.apply(style_row, axis=1), hide_index=True, use_container_width=True)
    with c2: st.write("**Redução de Peso (-)**"); st.dataframe(df_minus.style.apply(style_row, axis=1), hide_index=True, use_container_width=True)

# =============================================================================
# ALGORITMOS NATIVOS MCDM
# =============================================================================
def model_topsis(mat, weights, types):
    norm = normalize_vector(mat)
    weighted = norm * weights
    ideal = np.array([weighted[:, j].max() if types[j] == "max" else weighted[:, j].min() for j in range(mat.shape[1])])
    anti = np.array([weighted[:, j].min() if types[j] == "max" else weighted[:, j].max() for j in range(mat.shape[1])])
    d_plus = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted - anti) ** 2, axis=1))
    ci = d_minus / np.where((d_plus + d_minus) == 0, 1e-9, d_plus + d_minus)
    return {"normalized": norm, "weighted": weighted, "ideal": ideal, "anti_ideal": anti, "d_plus": d_plus, "d_minus": d_minus, "scores": ci, "ranking": ranking_from_scores(ci)}

def model_promethee(mat, weights, types, function="linear"):
    n_alt, n_crit = mat.shape
    pref = np.zeros((n_alt, n_alt))
    for j in range(n_crit):
        col = mat[:, j]
        rng = max(col.max() - col.min(), 1e-9)
        p = rng * 0.5
        for i in range(n_alt):
            for k in range(n_alt):
                if i == k: continue
                d = (col[i] - col[k]) if types[j] == "max" else (col[k] - col[i])
                if d > 0: pref[i, k] += weights[j] * (min(d / p, 1.0) if function == "linear" else 1.0)
    div = max(n_alt - 1, 1)
    phi_plus = pref.sum(axis=1) / div
    phi_minus = pref.sum(axis=0) / div
    phi_net = phi_plus - phi_minus
    return {"preference_matrix": pref, "phi_plus": phi_plus, "phi_minus": phi_minus, "scores": phi_net, "ranking": ranking_from_scores(phi_net)}

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
    return {"concordance": concordance, "discordance": discordance, "outrank": outrank, "scores": net_dominance.astype(float), "ranking": ranking_from_scores(net_dominance.astype(float))}

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
    return {"f_best": f_best, "f_worst": f_worst, "S": S, "R": R, "Q": Q, "scores": -Q, "ranking": ranking_from_scores(-Q)}

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
    return {"S_plus": S_plus, "S_minus": S_minus, "Q": Q, "scores": N, "ranking": ranking_from_scores(N)}

def model_maut(mat, weights, types):
    norm = normalize_minmax(mat, types)
    U = (norm * weights).sum(axis=1)
    return {"utility_matrix": norm, "scores": U, "ranking": ranking_from_scores(U)}

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
    relation = D - R
    adj = weights * prominence if prominence.sum() > 0 else weights
    adj = adj / adj.sum()
    norm = normalize_minmax(mat, types)
    scores = (norm * adj).sum(axis=1)
    return {"T": T, "prominence": prominence, "relation": relation, "scores": scores, "ranking": ranking_from_scores(scores)}

def model_fuzzy_topsis(mat, weights, types, spread=0.10):
    n_alt, n_crit = mat.shape
    l = mat * (1 - spread); m = mat.copy(); u = mat * (1 + spread)
    L, M, U = np.zeros_like(l), np.zeros_like(m), np.zeros_like(u)
    for j in range(n_crit):
        if types[j] == "max":
            u_max = max(u[:, j].max(), 1e-9)
            L[:, j] = l[:, j] / u_max; M[:, j] = m[:, j] / u_max; U[:, j] = u[:, j] / u_max
        else:
            l_min = max(l[:, j].min(), 1e-9)
            L[:, j] = l_min / np.where(u[:, j] == 0, 1e-9, u[:, j])
            M[:, j] = l_min / np.where(m[:, j] == 0, 1e-9, m[:, j])
            U[:, j] = l_min / np.where(l[:, j] == 0, 1e-9, l[:, j])
    Lw, Mw, Uw = L * weights, M * weights, U * weights
    d_plus = np.zeros(n_alt); d_minus = np.zeros(n_alt)
    for i in range(n_alt):
        for j in range(n_crit):
            d_plus[i] += np.sqrt(((Lw[i, j] - weights[j])**2 + (Mw[i, j] - weights[j])**2 + (Uw[i, j] - weights[j])**2)/3.0) if types[j] == "max" else np.sqrt((Lw[i, j]**2 + Mw[i, j]**2 + Uw[i, j]**2)/3.0)
            d_minus[i] += np.sqrt((Lw[i, j]**2 + Mw[i, j]**2 + Uw[i, j]**2)/3.0) if types[j] == "max" else np.sqrt(((Lw[i, j] - weights[j])**2 + (Mw[i, j] - weights[j])**2 + (Uw[i, j] - weights[j])**2)/3.0)
    cc = d_minus / np.where((d_plus + d_minus) == 0, 1e-9, d_plus + d_minus)
    return {"Lw": Lw, "Mw": Mw, "Uw": Uw, "d_plus": d_plus, "d_minus": d_minus, "scores": cc, "ranking": ranking_from_scores(cc)}

# =============================================================================
# BARRA LATERAL CONTROLO DE PARÂMETROS
# =============================================================================
with st.sidebar:
    st.header("⚙️ Painel de Operações")
    input_method = st.radio("Origem dos Dados estruturados:", ["Entrada Manual Direta", "Carregar Excel", "Dados de Demonstração MCG"])
    
    uploaded_file = None
    if input_method == "Carregar Excel":
        uploaded_file = st.file_uploader("Carregar Excel (.xlsx)", type=["xlsx", "xls"], help="Folha 'Dados' obrigatória. Folha 'Pesos' opcional.")
        
    st.divider()
    st.subheader("🎛️ Requisitos e Modelos")
    c_thresh = st.slider("ELECTRE: Limiar Concordância (c)", 0.50, 0.95, 0.65, 0.01)
    d_thresh = st.slider("ELECTRE: Limiar Discordância (d)", 0.05, 0.50, 0.35, 0.01)
    vikor_v = st.slider("VIKOR: Coeficiente de Consenso (v)", 0.0, 1.0, 0.5, 0.05)
    promethee_fn = st.selectbox("PROMETHEE: Tipo de Preferência", ["usual", "linear"])
    sens_pct = st.slider("Variação de Sensibilidade (±%)", 5, 50, 20, 5)

# =============================================================================
# TAB LABELS CONFIGURATION
# =============================================================================
TAB_LABELS = [
    "📋 Dados", "⚖️ Motores de Pesos", "🎯 TOPSIS", "🔗 ELECTRE", 
    "📊 PROMETHEE", "⚖️ VIKOR", "📐 MAUT", "🧮 COPRAS", "🌐 DEMATEL", 
    "🌫️ Fuzzy TOPSIS", "🏆 Dashboard", "📄 Relatório"
]
tabs = st.tabs(TAB_LABELS)

# =============================================================================
# TAB 0 — GESTÃO DAS MATRIZES BASE E EXCEL
# =============================================================================
alts, criteria, types, weights_setup, mat, data_df = [], [], [], [], None, None

with tabs[0]:
    if input_method == "Dados de Demonstração MCG":
        alts = [f"A{i}" for i in range(1, 10)]
        criteria = ["C1_VP", "C2_PF", "C3_EE", "C4_FE", "C5_UD", "C6_RC"]
        types = ["max", "max", "min", "max", "min", "max"]
        weights_setup = np.array([0.4615, 0.1987, 0.0230, 0.0972, 0.0217, 0.1979])
        mat = np.array([
            [250000000, 0.25, 24, 4, 180, 4], [300000, 0.35, 8, 5, 60, 5],
            [900000, 0.50, 8, 3, 60, 5], [650000, 0.50, 8, 3, 90, 3],
            [5000000, 0.40, 24, 4, 30, 3], [1350000, 0.50, 8, 3, 60, 5],
            [10500000, 0.40, 16, 3, 180, 4], [3450000, 0.40, 8, 3, 60, 4],
            [15000000, 0.60, 24, 4, 300, 3]
        ], dtype=float)
        data_df = pd.DataFrame(mat, columns=criteria)
        data_df.insert(0, "Alternativa", alts)
        st.header("📋 Dados de Demonstração Carregados (Caso MCG)")
        st.dataframe(data_df, hide_index=True, use_container_width=True)

    elif input_method == "Carregar Excel":
        st.header("📋 Dados Carregados via Excel")
        if uploaded_file is not None:
            res, err = safe_call(load_excel, uploaded_file)
            if err:
                st.error(f"❌ Erro ao ler Excel: {err}")
                st.stop()
            else:
                data_df_raw, loaded_weights, id_col, loaded_criteria, has_w = res
                st.dataframe(data_df_raw, hide_index=True, use_container_width=True)
                
                # Editor rápido para definir se é Max ou Min para o Excel importado
                st.subheader("🎯 Configurar Sentido dos Critérios do Excel")
                type_defaults = ["max"] * len(loaded_criteria)
                config_df = pd.DataFrame({
                    "Critério": loaded_criteria,
                    "Sentido": type_defaults,
                    "Peso Inicial": loaded_weights
                })
                edited_cfg = st.data_editor(config_df, hide_index=True, use_container_width=True)
                
                alts = data_df_raw[id_col].tolist()
                criteria = loaded_criteria
                types = edited_cfg["Sentido"].tolist()
                
                w_raw = np.array(edited_cfg["Peso Inicial"].tolist(), dtype=float)
                weights_setup = w_raw / w_raw.sum() if w_raw.sum() > 0 else np.ones(len(criteria))/len(criteria)
                mat = data_df_raw[criteria].astype(float).values
                data_df = data_df_raw.copy()
        else:
            st.info("👈 Por favor, faça o upload de um ficheiro Excel na barra lateral.")
            st.stop()

    else:
        st.header("🛠️ Configuração de Matrizes do Decisor (Manual)")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("1. Setup dos Critérios e Sentidos")
            edited_crit_df = st.data_editor(st.session_state.manual_criteria, column_config={"Critério": st.column_config.TextColumn("ID Critério", required=True), "Sentido": st.column_config.SelectboxColumn("Direção (max/min)", options=["max", "min"], required=True), "Peso Inicial": st.column_config.NumberColumn("Peso Base", min_value=0.0, step=0.05)}, num_rows="dynamic", hide_index=True, use_container_width=True)
            st.session_state.manual_criteria = edited_crit_df
        active_crits = edited_crit_df["Critério"].dropna().tolist()
        if len(active_crits) < 2:
            st.warning("⚠️ Forneça pelo menos 2 critérios operacionais.")
            st.stop()
        
        current_mat_df = st.session_state.manual_matrix
        cols_to_keep = ["Alternativa"] + [c for c in active_crits if c in current_mat_df.columns]
        updated_mat_df = current_mat_df[cols_to_keep].copy()
        for c in active_crits:
            if c not in updated_mat_df.columns: updated_mat_df[c] = 0.0
            
        mat_cols_cfg = {"Alternativa": st.column_config.TextColumn("Nome da Alternativa", required=True)}
        for c in active_crits: mat_cols_cfg[c] = st.column_config.NumberColumn(c, default=0.0)
        
        with c2:
            st.subheader("2. Matriz de Avaliação Bruta")
            edited_matrix_df = st.data_editor(updated_mat_df, column_config=mat_cols_cfg, num_rows="dynamic", hide_index=True, use_container_width=True)
            st.session_state.manual_matrix = edited_matrix_df
            
        alts = edited_matrix_df["Alternativa"].dropna().tolist()
        criteria = active_crits
        types = edited_crit_df["Sentido"].tolist()
        w_raw = np.array(edited_crit_df["Peso Inicial"].tolist(), dtype=float)
        weights_setup = w_raw / w_raw.sum() if w_raw.sum() > 0 else np.ones(len(criteria))/len(criteria)
        mat = edited_matrix_df[criteria].astype(float).values
        data_df = edited_matrix_df.copy()
        if len(alts) < 2: st.info("💡 Introduza pelo menos duas alternativas robustas."); st.stop()

# =============================================================================
# TAB 1 — MOTORES DE PESOS (INTEGRAÇÃO COMPLETA DAS FORMULAÇÕES)
# =============================================================================
with tabs[1]:
    st.header("⚖️ Motores de Geração de Pesos")
    weight_method = st.radio("Selecione a Metodologia Avançada:", ["AHP", "SWING", "SMART", "Entropia de Shannon", "CRITIC"], horizontal=True)
    n_crit = len(criteria)
    generated_weights = weights_setup.copy()
    
    if weight_method == "AHP":
        st.markdown("<div class='theory-box'><b>AHP (Analytic Hierarchy Process):</b> Baseado em julgamentos par-a-par subjectivos recorrendo à Escala de Saaty (1-9). Requer a verificação do Rácio de Consistência (CR) que deve ser inferior a 0.10.</div>", unsafe_allow_html=True)
        if "ahp_matrix_data" not in st.session_state or st.session_state.ahp_matrix_data.shape != (n_crit, n_crit):
            st.session_state.ahp_matrix_data = np.ones((n_crit, n_crit))
        ahp_df = pd.DataFrame(st.session_state.ahp_matrix_data, index=criteria, columns=criteria)
        edited_ahp = st.data_editor(ahp_df, use_container_width=True)
        A = edited_ahp.values.astype(float).copy()
        for i in range(n_crit):
            for j in range(n_crit):
                if i == j: A[i, j] = 1.0
                elif i < j and A[i, j] != 0: A[j, i] = 1.0 / A[i, j]
        st.session_state.ahp_matrix_data = A
        col_sums = A.sum(axis=0)
        norm_A = A / np.where(col_sums == 0, 1, col_sums)
        generated_weights = norm_A.mean(axis=1)
        lambdamax = (A.dot(generated_weights) / np.where(generated_weights == 0, 1, generated_weights)).mean()
        ci = (lambdamax - n_crit) / (n_crit - 1) if n_crit > 1 else 0
        ri = RI_TABLE.get(n_crit, 1.59)
        cr = ci / ri if ri > 0 else 0
        st.latex(r"CI = \frac{\lambda_{\max}-n}{n-1}, \quad CR = \frac{CI}{RI}")
        m1, m2, m3 = st.columns(3)
        m1.metric("λ_max", f"{lambdamax:.4f}")
        m2.metric("CI", f"{ci:.4f}")
        m3.metric("CR", f"{cr:.4f}", delta="Consistente (<0.10)" if cr <= 0.10 else "Inconsistente", delta_color="normal" if cr <= 0.10 else "inverse")

    elif weight_method == "SWING":
        st.markdown("<div class='theory-box'><b>SWING Weighting:</b> Método subjetivo baseado na variação do pior cenário para o melhor. Atribui-se 100 pontos ao critério de maior impacto e pontuações relativas aos restantes.</div>", unsafe_allow_html=True)
        if "swing_pts" not in st.session_state or len(st.session_state.swing_pts) != n_crit:
            st.session_state.swing_pts = [100.0] + [50.0]*(n_crit-1)
        pts_df = pd.DataFrame({"Critério": criteria, "Pontos (0-100)": st.session_state.swing_pts})
        edited_pts = st.data_editor(pts_df, hide_index=True, use_container_width=True)
        p = edited_pts["Pontos (0-100)"].values.astype(float)
        st.session_state.swing_pts = p
        generated_weights = p / p.sum() if p.sum() > 0 else np.ones(n_crit)/n_crit
        st.latex(r"w_j = \frac{p_j}{\sum_{k=1}^n p_k}")

    elif weight_method == "SMART":
        st.markdown("<div class='theory-box'><b>SMART:</b> Atribuição direta de pontuações absolutas (0-100) refletindo a importância nativa do critério, procedendo-se de seguida à normalização linear estável.</div>", unsafe_allow_html=True)
        if "smart_pts" not in st.session_state or len(st.session_state.smart_pts) != n_crit:
            st.session_state.smart_pts = [50.0]*n_crit
        pts_df = pd.DataFrame({"Critério": criteria, "Pontos (0-100)": st.session_state.smart_pts})
        edited_pts = st.data_editor(pts_df, hide_index=True, use_container_width=True)
        p = edited_pts["Pontos (0-100)"].values.astype(float)
        st.session_state.smart_pts = p
        generated_weights = p / p.sum() if p.sum() > 0 else np.ones(n_crit)/n_crit

    elif weight_method == "Entropia de Shannon":
        st.markdown("<div class='theory-box'><b>Entropia de Shannon:</b> Método puramente quantitativo que extrai os pesos com base na dispersão interna dos dados da matriz. Maior dispersão = maior peso.</div>", unsafe_allow_html=True)
        norm_e = np.zeros_like(mat)
        for j in range(n_crit):
            if types[j] == "max":
                s = mat[:, j].sum(); norm_e[:, j] = mat[:, j] / s if s > 0 else 1.0/len(alts)
            else:
                inv = 1.0 / np.where(mat[:, j] == 0, 1e-9, mat[:, j])
                s = inv.sum(); norm_e[:, j] = inv / s if s > 0 else 1.0/len(alts)
        k = 1.0 / np.log(len(alts)) if len(alts) > 1 else 1.0
        norm_safe = np.where(norm_e == 0, 1e-9, norm_e)
        ej = -k * np.sum(norm_safe * np.log(norm_safe), axis=0)
        dj = 1.0 - ej
        st.latex(r"E_j = -k \sum_{i=1}^m p_{ij} \ln p_{ij}, \quad d_j = 1 - E_j")
        st.dataframe(pd.DataFrame({"Critério": criteria, "Entropia (E)": ej, "Divergência (d)": dj}).round(4), hide_index=True, use_container_width=True)
        generated_weights = dj / dj.sum() if dj.sum() > 0 else np.ones(n_crit)/n_crit

    elif weight_method == "CRITIC":
        st.markdown("<div class='theory-box'><b>CRITIC:</b> Método objetivo que quantifica a importância combinando o desvio padrão interno de cada critério com a correlação linear intercritérios (conflito).</div>", unsafe_allow_html=True)
        norm_c = normalize_minmax(mat, types)
        std = np.std(norm_c, axis=0)
        corr = np.corrcoef(norm_c.T)
        corr = np.nan_to_num(corr, nan=0.0)
        conflict = std * np.sum(1 - corr, axis=1)
        st.latex(r"C_j = \sigma_j \sum_{k=1}^n (1 - r_{jk}), \quad w_j = \frac{C_j}{\sum C_j}")
        st.dataframe(pd.DataFrame({"Critério": criteria, "Conflito (C)": conflict, "Desvio Padrão (σ)": std}).round(4), hide_index=True, use_container_width=True)
        generated_weights = conflict / conflict.sum() if conflict.sum() > 0 else np.ones(n_crit)/n_crit

    activate_weights = st.toggle(f"🔥 INJETAR PESOS '{weight_method.upper()}' EM TODOS OS MODELOS", value=False)
    final_weights = generated_weights if activate_weights else weights_setup
    st.subheader("Pesos Finais Ativos no Momento:")
    st.dataframe(pd.DataFrame({"Critério": criteria, "Peso Absoluto": final_weights}).style.format({"Peso Absoluto": "{:.4f}"}), hide_index=True, use_container_width=True)

# =============================================================================
# MAPEAMENTO DINÂMICO E EXECUÇÃO GERAL DOS MODELOS
# =============================================================================
all_results = {}
res_topsis, _ = safe_call(model_topsis, mat, final_weights, types)
if res_topsis: all_results["TOPSIS"] = res_topsis
res_electre, _ = safe_call(model_electre, mat, final_weights, types, c_thresh, d_thresh)
if res_electre: all_results["ELECTRE"] = res_electre
res_promethee, _ = safe_call(model_promethee, mat, final_weights, types, promethee_fn)
if res_promethee: all_results["PROMETHEE"] = res_promethee
res_vikor, _ = safe_call(model_vikor, mat, final_weights, types, vikor_v)
if res_vikor: all_results["VIKOR"] = res_vikor
res_maut, _ = safe_call(model_maut, mat, final_weights, types)
if res_maut: all_results["MAUT"] = res_maut
res_copras, _ = safe_call(model_copras, mat, final_weights, types)
if res_copras: all_results["COPRAS"] = res_copras
res_dematel, _ = safe_call(model_dematel, mat, final_weights, types)
if res_dematel: all_results["DEMATEL"] = res_dematel
res_fuzzy, _ = safe_call(model_fuzzy_topsis, mat, final_weights, types, spread=0.10)
if res_fuzzy: all_results["Fuzzy TOPSIS"] = res_fuzzy

# =============================================================================
# DETALHAMENTO DE CADA ABA INDIVIDUAL (TEORIA, PASSOS, SENSIBILIDADE)
# =============================================================================

with tabs[2]:
    st.header("🎯 TOPSIS — Técnica de Proximidade ao Ideal")
    st.markdown("<div class='theory-box'><b>Resumo Teórico:</b> Método compensatório que seleciona a alternativa com menor distância geométrica à Solução Ideal Positiva ($A^+$) e maior distância à Solução Ideal Negativa ($A^-$).<br>Fórmula de Normalização Vectorial: $r_{ij} = \frac{x_{ij}}{\sqrt{\sum x_{ij}^2}}$</div>", unsafe_allow_html=True)
    if "TOPSIS" in all_results:
        r = all_results["TOPSIS"]
        st.write("**Passo 1 & 2: Matriz Normalizada Vectorial**")
        st.dataframe(pd.DataFrame(r["normalized"], index=alts, columns=criteria).round(4), use_container_width=True)
        st.write("**Passo 3: Matriz Ponderada** ($v_{ij} = w_j \cdot r_{ij}$)")
        st.dataframe(pd.DataFrame(r["weighted"], index=alts, columns=criteria).round(4), use_container_width=True)
        st.write("**Passo 4: Vetores de Referência ($A^+$ e $A^-$)**")
        st.dataframe(pd.DataFrame({"Critério": criteria, "A+ (Ideal)": r["ideal"], "A- (Anti-Ideal)": r["anti_ideal"]}).round(4), hide_index=True, use_container_width=True)
        st.write("**Passo 5 & 6: Distâncias e Coeficiente de Proximidade ($CC_i$)**")
        st.latex(r"CC_i = \frac{D_i^-}{D_i^+ + D_i^-}")
        rdf = pd.DataFrame({"Alternativa": alts, "D+": r["d_plus"], "D-": r["d_minus"], "Score (CCi)": r["scores"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"D+": "{:.4f}", "D-": "{:.4f}", "Score (CCi)": "{:.4f}"}), hide_index=True, use_container_width=True)
        render_sensitivity(model_topsis, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"])

with tabs[3]:
    st.header("🔗 ELECTRE I — Relações de Sobreclassificação")
    st.markdown("<div class='theory-box'><b>Resumo Teórico:</b> Abordagem estruturada não compensatória fundamentada nos conceitos de Concordância (maioria ponderada apoia a decisão) e Discordância (nenhum critério opõe veto forte).</div>", unsafe_allow_html=True)
    if "ELECTRE" in all_results:
        r = all_results["ELECTRE"]
        st.write("**Passo 1: Matriz de Concordância (C)**")
        st.dataframe(pd.DataFrame(r["concordance"], index=alts, columns=alts).round(4), use_container_width=True)
        st.write("**Passo 2: Matriz de Discordância (D)**")
        st.dataframe(pd.DataFrame(r["discordance"], index=alts, columns=alts).round(4), use_container_width=True)
        st.write("**Passo 3: Relação de Sobreclassificação Booleana**")
        st.dataframe(pd.DataFrame(r["outrank"].astype(int), index=alts, columns=alts), use_container_width=True)
        rdf = pd.DataFrame({"Alternativa": alts, "Score Dominância": r["scores"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf, hide_index=True, use_container_width=True)
        render_sensitivity(model_electre, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"], c_thresh=c_thresh, d_thresh=d_thresh)

with tabs[4]:
    st.header("📊 PROMETHEE II — Ordenação Completa por Par")
    st.markdown("<div class='theory-box'><b>Resumo Teórico:</b> Método não-compensatório baseado em comparações par-a-par de todas as alternativas, calculando fluxos líquidos de superação sobre o espaço de decisão.<br>Índice Global: $\pi(a,b) = \sum w_j \cdot P_j(a,b)$</div>", unsafe_allow_html=True)
    if "PROMETHEE" in all_results:
        r = all_results["PROMETHEE"]
        st.write("**Passo 3 & 4: Matriz de Preferência Agregada** $\pi(a,b)$")
        st.dataframe(pd.DataFrame(r["preference_matrix"], index=alts, columns=alts).round(4), use_container_width=True)
        st.write("**Passo 5 & 6: Fluxos de Preferência Líquidos ($\Phi$)**")
        st.latex(r"\phi(a) = \phi^+(a) - \phi^-(a)")
        rdf = pd.DataFrame({"Alternativa": alts, "Φ+ (Saída)": r["phi_plus"], "Φ- (Entrada)": r["phi_minus"], "Fluxo Líquido (Φ)": r["scores"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Φ+ (Saída)": "{:.4f}", "Φ- (Entrada)": "{:.4f}", "Fluxo Líquido (Φ)": "{:.4f}"}), hide_index=True, use_container_width=True)
        render_sensitivity(model_promethee, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"], function=promethee_fn)

with tabs[5]:
    st.header("⚖️ VIKOR — Solução de Compromisso Estável")
    st.markdown("<div class='theory-box'><b>Resumo Teórico:</b> Desenvolvido para lidar com critérios altamente conflituosos, otimizando simultaneamente a utilidade da maioria ($S_j$) e minimizando o arrependimento do oponente ($R_j$). O menor índice $Q$ determina o pódio.</div>", unsafe_allow_html=True)
    if "VIKOR" in all_results:
        r = all_results["VIKOR"]
        st.write("**Passo 1: Valores Máximos e Mínimos das Fronteiras**")
        st.dataframe(pd.DataFrame({"Critério": criteria, "Melhor (f*)": r["f_best"], "Pior (f-)": r["f_worst"]}).round(4), hide_index=True, use_container_width=True)
        st.write("**Passo 2 & 3: Índices de Desvio Intermédios e Compromisso (Q)**")
        st.latex(r"Q_j = v \frac{S_j - S^*}{S^- - S^*} + (1-v) \frac{R_j - R^*}{R^- - R^*}")
        rdf = pd.DataFrame({"Alternativa": alts, "Utilidade (S)": r["S"], "Arrependimento (R)": r["R"], "Índice de Compromisso (Q)": r["Q"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Utilidade (S)": "{:.4f}", "Arrependimento (R)": "{:.4f}", "Índice de Compromisso (Q)": "{:.4f}"}), hide_index=True, use_container_width=True)
        render_sensitivity(model_vikor, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"], v=vikor_v)

with tabs[6]:
    st.header("📐 MAUT — Multi-Attribute Utility Theory")
    st.markdown("<div class='theory-box'><b>Resumo Teórico:</b> Transforma desempenhos reais numa escala de utilidade comum $[0,1]$ através de funções (ex.: lineares) e agrega-os de forma aditiva limpa.</div>", unsafe_allow_html=True)
    if "MAUT" in all_results:
        r = all_results["MAUT"]
        st.write("**Passo 4: Matriz de Utilidades Parciais Normalizadas**")
        st.dataframe(pd.DataFrame(r["utility_matrix"], index=alts, columns=criteria).round(4), use_container_width=True)
        st.write("**Passo 5: Utilidade Global Composta e Ordenação**")
        st.latex(r"U_i = \sum w_j \cdot U_j(x_{ij})")
        rdf = pd.DataFrame({"Alternativa": alts, "Utilidade Composta": r["scores"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Utilidade Composta": "{:.4f}"}), hide_index=True, use_container_width=True)
        render_sensitivity(model_maut, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"])

with tabs[7]:
    st.header("🧮 COPRAS — Complex Proportional Assessment")
    st.markdown("<div class='theory-box'><b>Resumo Teórico:</b> Avalia as alternativas assumindo a proporcionalidade direta dos índices de benefício ($S^+$) e inversa das componentes de custo ($S^-$).</div>", unsafe_allow_html=True)
    if "COPRAS" in all_results:
        r = all_results["COPRAS"]
        st.write("**Passo 1 & 2: Índices Parciais de Custos/Benefícios e Grau de Utilidade (N)**")
        st.latex(r"N_i = \frac{Q_i}{\max Q_k} \times 100\%")
        rdf = pd.DataFrame({"Alternativa": alts, "S+ (Benefício)": r["S_plus"], "S- (Custo)": r["S_minus"], "Significância (Q)": r["Q"], "Grau Utilidade (N %)": r["scores"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"S+ (Benefício)": "{:.4f}", "S- (Custo)": "{:.4f}", "Significância (Q)": "{:.4f}", "Grau Utilidade (N %)": "{:.2f}"}), hide_index=True, use_container_width=True)
        render_sensitivity(model_copras, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"])

with tabs[8]:
    st.header("🌐 DEMATEL — Diagrama Causa-Efeito Estrutural")
    st.markdown("<div class='theory-box'><b>Resumo Teórico:</b> Focado no mapeamento das influências diretas e indiretas entre fatores, gerando os eixos de Prominência ($D+R$) e Relação ($D-R$).</div>", unsafe_allow_html=True)
    if "DEMATEL" in all_results:
        r = all_results["DEMATEL"]
        st.write("**Passo 3: Matriz de Influência Total (T)**")
        st.dataframe(pd.DataFrame(r["T"], index=criteria, columns=criteria).round(4), use_container_width=True)
        st.write("**Passo 4 & 5: Prioridades Ajustadas e Rankings das Alternativas**")
        rdf = pd.DataFrame({"Alternativa": alts, "Score": r["scores"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Score": "{:.4f}"}), hide_index=True, use_container_width=True)
        render_sensitivity(model_dematel, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"])

with tabs[9]:
    st.header("🌫️ Fuzzy TOPSIS (Chen, 2000)")
    st.markdown("<div class='theory-box'><b>Resumo Teórico:</b> Substitui os valores crisp por Números Fuzzy Triangulares (TFN) $(l, m, u)$ para capturar a imprecisão linguística inerente dos decisores.<br>Cálculo de Distância Fuzzy Vertex: $d(\tilde{a},\tilde{b}) = \sqrt{\frac{1}{3}[(l_a-l_b)^2+(m_a-m_b)^2+(u_a-u_b)^2]}$</div>", unsafe_allow_html=True)
    if "Fuzzy TOPSIS" in all_results:
        r = all_results["Fuzzy TOPSIS"]
        st.write("**Passo 3: Matriz Ponderada Normalizada Fuzzy (Amostra do Centro m)**")
        st.dataframe(pd.DataFrame(r["Mw"], index=alts, columns=criteria).round(4), use_container_width=True)
        st.write("**Passo 4 & 5: Distâncias aos FPIS/FNIS Fuzzy e Proximidade**")
        rdf = pd.DataFrame({"Alternativa": alts, "d+ (FPIS)": r["d_plus"], "d- (FNIS)": r["d_minus"], "Score (CCi Fuzzy)": r["scores"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"d+ (FPIS)": "{:.4f}", "d- (FNIS)": "{:.4f}", "Score (CCi Fuzzy)": "{:.4f}"}), hide_index=True, use_container_width=True)
        render_sensitivity(model_fuzzy_topsis, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"])

# =============================================================================
# TAB 10 — DASHBOARD CONSOLIDADO (BORDA + RADAR + EXPORT EXCEL)
# =============================================================================
rank_table_global = pd.DataFrame()
score_table_global = pd.DataFrame()

with tabs[10]:
    st.header("🏆 Matriz de Consenso e Dashboard Final")
    models_executed = list(all_results.keys())
    if not models_executed:
        st.error("Execute ou introduza dados válidos.")
    else:
        consolidated_df = pd.DataFrame({"Alternativa": alts})
        for m in models_executed: consolidated_df[m] = all_results[m]["ranking"]
        consolidated_df["Posição Média"] = consolidated_df[models_executed].mean(axis=1).round(2)
        
        # AGRAVAÇÃO INVERTIDA DO MÉTODO DE BORDA (MENOR MÉDIA GANHA O PODIO)
        consolidated_df["Ranking Consolidado"] = ranking_from_scores(consolidated_df["Posição Média"].values, higher_is_better=False)
        consolidated_df = consolidated_df.sort_values("Ranking Consolidado").reset_index(drop=True)
        
        rank_table_global = consolidated_df.copy() # para exportação e relatório

        score_table = pd.DataFrame({"Alternativa": alts})
        for m in models_executed: score_table[m] = all_results[m]["scores"]
        score_table_global = score_table.copy()
        
        st.info("💡 **Concluido via Método de Borda:** A ordenação é baseada na menor posição média obtida entre todos os modelos matemáticos.")
        styled_matrix = consolidated_df.style.format({"Posição Média": "{:.2f}"}).background_gradient(subset=models_executed + ["Posição Média", "Ranking Consolidado"], cmap="RdYlGn_r")
        st.dataframe(styled_matrix, use_container_width=True, hide_index=True)
        
        podium = consolidated_df["Alternativa"].tolist()
        p1, p2, p3 = st.columns(3)
        p1.metric("🥇 Vencedor Absoluto", podium[0] if len(podium) > 0 else "—")
        p2.metric("🥈 2º Classificado", podium[1] if len(podium) > 1 else "—")
        p3.metric("🥉 3º Classificado", podium[2] if len(podium) > 2 else "—")

        # Heatmap
        st.subheader("Heatmap de Posições por Modelo")
        heat_df = consolidated_df.set_index("Alternativa")[models_executed]
        fig_heat = px.imshow(heat_df.values, labels=dict(x="Modelo", y="Alternativa", color="Ranking"), x=models_executed, y=heat_df.index, color_continuous_scale="RdYlGn_r", aspect="auto", text_auto=True)
        fig_heat.update_layout(height=420)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Gráfico Radar para Top-3
        st.subheader("Perfil Multicritério — Top-3 (Radar Normalizado)")
        try:
            norm_radar = normalize_minmax(mat, types)
            fig_radar = go.Figure()
            colors = ["#e63946", "#f4a261", "#2a9d8f"]
            top3_alts = podium[:3]
            for k, alt_name in enumerate(top3_alts):
                idx = alts.index(alt_name)
                vals = list(norm_radar[idx]) + [norm_radar[idx, 0]]
                axes = criteria + [criteria[0]]
                fig_radar.add_trace(go.Scatterpolar(r=vals, theta=axes, fill="toself", name=alt_name, line=dict(color=colors[k % len(colors)], width=2), opacity=0.65))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=460, title="Top-3 — Perfil normalizado por critério")
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception as e:
            st.warning("Não foi possível gerar o Gráfico Radar.")

        # Exportação Excel
        st.subheader("📥 Exportação")
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                data_df.to_excel(writer, sheet_name="Dados", index=False)
                pd.DataFrame({"Critério": criteria, "Peso Final": final_weights, "Sentido": types}).to_excel(writer, sheet_name="Pesos_e_Tipos", index=False)
                consolidated_df.to_excel(writer, sheet_name="Rankings", index=False)
                score_table_global.to_excel(writer, sheet_name="Scores", index=False)
            st.download_button("Descarregar Excel com Todos os Resultados", data=buffer.getvalue(), file_name="mcdm_resultados_completos.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as exc:
            st.error(f"Erro na exportação: {exc}")

# =============================================================================
# TAB 11 — RELATÓRIO DINÂMICO
# =============================================================================
with tabs[11]:
    st.header("📄 Relatório de Análise Multicritério")

    if not all_results:
        st.warning("⚠️ Nenhum modelo foi executado com sucesso. Visita as tabs anteriores primeiro.")
    else:
        n_alt = len(alts)
        n_crit = len(criteria)
        n_max = types.count("max")
        n_min = types.count("min")
        models_with_results = list(all_results.keys())

        top3_alts_rel = rank_table_global["Alternativa"].head(3).tolist()
        top1 = top3_alts_rel[0] if len(top3_alts_rel) > 0 else "—"

        if top1 != "—":
            top1_row = rank_table_global[rank_table_global["Alternativa"] == top1].iloc[0]
            top1_in_top3 = sum(1 for m in models_with_results if top1_row[m] <= 3)
            conv_pct = (top1_in_top3 / len(models_with_results)) * 100
        else:
            top1_in_top3 = 0
            conv_pct = 0

        stats = data_df[criteria].astype(float).describe().T

        st.markdown(
            f"**Data:** {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}  \n"
            f"**Modelos aplicados com sucesso:** {len(models_with_results)} / 8  \n"
            f"**Alternativas avaliadas:** {n_alt}  \n"
            f"**Critérios:** {n_crit} ({n_max} benefício · {n_min} custo)"
        )

        report_lines = []
        report_lines.append(f"# Relatório de Análise Multicritério\n")
        report_lines.append(f"**Gerado em:** {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}\n")

        # 1. Sumário executivo
        report_lines.append("## 1. Sumário executivo\n")
        report_lines.append(f"Foi realizada uma análise de decisão multicritério sobre **{n_alt} alternativas** avaliadas segundo **{n_crit} critérios** ({n_max} de benefício e {n_min} de custo). Foram aplicados **{len(models_with_results)} modelos** MCDM ({', '.join(models_with_results)}), agregados por método de Borda invertido (média de posições).\n")
        report_lines.append(f"**Alternativa recomendada:** `{top1}`, com posição média de `{rank_table_global['Posição Média'].iloc[0]:.2f}`. Aparece no Top-3 em **{top1_in_top3} de {len(models_with_results)}** modelos ({conv_pct:.0f}% de convergência inter-modelo).\n")
        
        if len(top3_alts_rel) >= 3:
            report_lines.append(f"**Top-3 agregado:** 1º `{top3_alts_rel[0]}` · 2º `{top3_alts_rel[1]}` · 3º `{top3_alts_rel[2]}`\n")

        # 2. Contexto e dados
        report_lines.append("## 2. Contexto e dados de entrada\n")
        report_lines.append("### 2.1 Critérios, Pesos Finais e Sentidos\n")
        report_lines.append("| Critério | Peso Final | Sentido |")
        report_lines.append("|----------|------|---------|")
        for c, w, t in zip(criteria, final_weights, types): report_lines.append(f"| {c} | {w:.4f} | {t} |")
        report_lines.append("\n### 2.2 Estatísticas descritivas dos critérios\n")
        report_lines.append("| Critério | Mín | Mediana | Máx | Média | Desvio padrão |")
        report_lines.append("|----------|-----|---------|-----|-------|---------------|")
        for c in criteria:
            s = stats.loc[c]
            report_lines.append(f"| {c} | {s['min']:.4g} | {s['50%']:.4g} | {s['max']:.4g} | {s['mean']:.4g} | {s['std']:.4g} |")
        report_lines.append("")

        # 3. Metodologia
        report_lines.append("## 3. Metodologia\n")
        report_lines.append("Foram aplicados os seguintes modelos suportados pela framework matemática interativa:\n")
        for m in models_with_results: report_lines.append(f"- **{m}**")
        report_lines.append("\nOs rankings foram agregados por **média de posições** (Borda invertido) — a alternativa com menor média é a recomendação final.\n")

        # 4. Resultados por modelo
        report_lines.append("## 4. Resultados por modelo (Top-3)\n")
        report_lines.append("| Modelo | 1º | 2º | 3º |")
        report_lines.append("|--------|-----|-----|-----|")
        for m in models_with_results:
            rank_m = all_results[m]["ranking"]
            top_indices = sorted(range(len(rank_m)), key=lambda i: rank_m[i])[:3]
            top_names = [alts[i] for i in top_indices]
            row = f"| {m} |"
            for i in range(3): row += f" {top_names[i] if i < len(top_names) else '—'} |"
            report_lines.append(row)
        report_lines.append("")

        # 5. Ranking consolidado
        report_lines.append("## 5. Ranking consolidado (Borda invertido)\n")
        report_lines.append("| Posição | Alternativa | Posição média |")
        report_lines.append("|---------|-------------|---------------|")
        for i, row in rank_table_global.head(min(10, n_alt)).iterrows():
            report_lines.append(f"| {i+1} | {row['Alternativa']} | {row['Posição Média']:.2f} |")
        report_lines.append("")

        # 6. Recomendação Final
        report_lines.append("## 6. Recomendação e Robustez\n")
        if conv_pct >= 60:
            verd = f"A análise apresenta **alta convergência** ({conv_pct:.0f}%) em torno da alternativa `{top1}`. **Recomenda-se a sua selecção** com elevado grau de confiança."
        elif conv_pct >= 40:
            verd = f"A análise apresenta **convergência moderada** ({conv_pct:.0f}%) em torno da alternativa `{top1}`. Recomenda-se selecção, complementada pela análise de sensibilidade gráfica constante nas abas do dashboard."
        else:
            verd = f"A análise apresenta **baixa convergência** ({conv_pct:.0f}%). Recomenda-se reavaliação dos pesos e da influência das análises de sensibilidade."
        report_lines.append(verd + "\n")

        report_md = "\n".join(report_lines)

        st.subheader(f"🥇 Alternativa recomendada: {top1}")
        st.metric("Convergência inter-modelo (Top-1 no Top-3)", f"{conv_pct:.0f}%", f"{top1_in_top3} / {len(models_with_results)} modelos")

        st.markdown("---")
        st.markdown(report_md)

        st.markdown("---")
        st.subheader("📥 Descarregar relatório")
        c1, c2 = st.columns(2)
        with c1: st.download_button("Descarregar como Markdown (.md)", data=report_md.encode("utf-8"), file_name=f"relatorio_mcdm_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md", mime="text/markdown")
        with c2: st.download_button("Descarregar como texto (.txt)", data=report_md.encode("utf-8"), file_name=f"relatorio_mcdm_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt", mime="text/plain")
