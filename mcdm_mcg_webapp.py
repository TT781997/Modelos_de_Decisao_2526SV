# -*- coding: utf-8 -*-
"""
MCDM Dashboard — Sistema de Apoio à Decisão Multicritério
Modelos de Decisão | MEGI ISEL 2025/2026

Atualizações Definitivas:
- Restauro COMPLETO do Passo-a-Passo Matemático em todas as Tabs.
- Restauro da Análise de Sensibilidade Paralela.
- Proteção total de estado (Tabelas Dinâmicas que não apagam dados).
- Motor Global de Pesos (AHP, SWING, SMART, Entropia, CRITIC).
"""

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st

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
st.caption("Modelos de Decisão | Matemática Passo-a-Passo e Sensibilidade")

RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# =============================================================================
# INICIALIZAÇÃO DO ESTADO DE SESSÃO
# =============================================================================
if "manual_criteria" not in st.session_state:
    st.session_state.manual_criteria = pd.DataFrame({
        "Critério": ["C1", "C2", "C3"],
        "Sentido": ["max", "max", "min"],
        "Peso Inicial": [0.4, 0.4, 0.2]
    })

if "manual_matrix" not in st.session_state:
    st.session_state.manual_matrix = pd.DataFrame({
        "Alternativa": ["Alt 1", "Alt 2"],
        "C1": [10.0, 20.0],
        "C2": [15.0, 12.0],
        "C3": [5.0, 8.0]
    })

# =============================================================================
# FUNÇÕES MATEMÁTICAS GERAIS E SENSIBILIDADE
# =============================================================================
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
    st.subheader(f"🔄 Análise de Sensibilidade (±{sens_pct}%)")
    st.markdown("Mostra como o ranking final se altera se aumentarmos/reduzirmos isoladamente o peso de cada critério. **Verde** = subiu no ranking; **Vermelho** = desceu.")
    
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
        styles = []
        for col, val in row.items():
            if col in ['Alternativa', 'Base']: styles.append('')
            elif val < base: styles.append('color: #00B140; font-weight: bold;') 
            elif val > base: styles.append('color: #D32F2F; font-weight: bold;') 
            else: styles.append('color: gray;')
        return styles

    c1, c2 = st.columns(2)
    with c1: 
        st.write(f"**+{sens_pct}% no Peso**")
        st.dataframe(df_plus.style.apply(style_row, axis=1), hide_index=True, use_container_width=True)
    with c2: 
        st.write(f"**-{sens_pct}% no Peso**")
        st.dataframe(df_minus.style.apply(style_row, axis=1), hide_index=True, use_container_width=True)


# =============================================================================
# ALGORITMOS DE DECISÃO MULTICRITÉRIO (COM RETURNS DETALHADOS)
# =============================================================================
def model_topsis(mat, weights, types):
    norm = normalize_vector(mat)
    weighted = norm * weights
    ideal = np.array([weighted[:, j].max() if types[j] == "max" else weighted[:, j].min() for j in range(mat.shape[1])])
    anti = np.array([weighted[:, j].min() if types[j] == "max" else weighted[:, j].max() for j in range(mat.shape[1])])
    d_plus = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted - anti) ** 2, axis=1))
    ci = d_minus / np.where((d_plus + d_minus) == 0, 1e-9, d_plus + d_minus)
    return {"normalized": norm, "weighted": weighted, "ideal": ideal, "anti_ideal": anti, 
            "d_plus": d_plus, "d_minus": d_minus, "scores": ci, "ranking": ranking_from_scores(ci)}

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
    return {"concordance": concordance, "discordance": discordance, "outrank": outrank,
            "scores": net_dominance.astype(float), "ranking": ranking_from_scores(net_dominance.astype(float))}

def model_promethee(mat, weights, types, function="linear"):
    n_alt, n_crit = mat.shape
    pref = np.zeros((n_alt, n_alt))
    for j in range(n_crit):
        col = mat[:, j]
        rng = col.max() - col.min()
        p = rng * 0.5 if rng > 0 else 1.0
        for i in range(n_alt):
            for k in range(n_alt):
                if i == k: continue
                d = (col[i] - col[k]) if types[j] == "max" else (col[k] - col[i])
                if d > 0: pref[i, k] += weights[j] * (min(d / p, 1.0) if function == "linear" else 1.0)
    div = max(n_alt - 1, 1)
    phi_plus = pref.sum(axis=1) / div
    phi_minus = pref.sum(axis=0) / div
    phi_net = phi_plus - phi_minus
    return {"preference_matrix": pref, "phi_plus": phi_plus, "phi_minus": phi_minus, 
            "scores": phi_net, "ranking": ranking_from_scores(phi_net)}

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
    return {"S": S, "R": R, "Q": Q, "scores": -Q, "ranking": ranking_from_scores(-Q)}

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
    return {"T": T, "prominence": prominence, "relation": relation, "adj_weights": adj, "scores": scores, "ranking": ranking_from_scores(scores)}


# =============================================================================
# BARRA LATERAL: PARÂMETROS E REQUISITOS DOS MODELOS
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configurações Globais")
    input_method = st.radio("Origem dos Dados:", ["Entrada Manual Direta", "Dados de Demonstração MCG"])
    
    st.divider()
    st.subheader("🎛️ Requisitos Técnicos")
    c_thresh = st.slider("ELECTRE: Limiar Concordância (c)", 0.50, 0.95, 0.65, 0.01)
    d_thresh = st.slider("ELECTRE: Limiar Discordância (d)", 0.05, 0.50, 0.35, 0.01)
    vikor_v = st.slider("VIKOR: Estratégia Consenso (v)", 0.0, 1.0, 0.5, 0.05)
    promethee_fn = st.selectbox("PROMETHEE: Critério", ["usual", "linear"], index=1)
    sens_pct = st.slider("Variação Análise Sensibilidade (±%)", 5, 50, 20, 5)

# =============================================================================
# CONFIGURAÇÃO DE ABAS
# =============================================================================
TAB_LABELS = [
    "📋 Dados", "⚖️ Motores de Pesos", "🎯 TOPSIS", "🔗 ELECTRE", 
    "📊 PROMETHEE", "⚖️ VIKOR", "🧮 COPRAS", "🌐 DEMATEL", "🏆 Dashboard"
]
tabs = st.tabs(TAB_LABELS)

# =============================================================================
# TAB 0 — SETUP DA GRELHA
# =============================================================================
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
        st.header("📋 Dados de Demonstração Carregados")
        df_display = pd.DataFrame(mat, index=alts, columns=criteria)
        df_display.insert(0, "Alternativa", alts)
        st.dataframe(df_display, hide_index=True, use_container_width=True)
    else:
        st.header("🛠️ Configuração da Grelha de Decisão")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("1. Definição de Critérios")
            edited_crit_df = st.data_editor(
                st.session_state.manual_criteria,
                column_config={
                    "Critério": st.column_config.TextColumn("ID Critério", required=True),
                    "Sentido": st.column_config.SelectboxColumn("Tipo", options=["max", "min"], required=True),
                    "Peso Inicial": st.column_config.NumberColumn("Peso Base", min_value=0.0, step=0.05)
                },
                num_rows="dynamic", hide_index=True, use_container_width=True
            )
            st.session_state.manual_criteria = edited_crit_df
            
        active_crits = edited_crit_df["Critério"].dropna().tolist()
        if len(active_crits) < 2:
            st.warning("⚠️ Adicione pelo menos 2 critérios.")
            st.stop()
            
        current_mat_df = st.session_state.manual_matrix
        cols_to_keep = ["Alternativa"] + [c for c in active_crits if c in current_mat_df.columns]
        updated_mat_df = current_mat_df[cols_to_keep].copy()
        for c in active_crits:
            if c not in updated_mat_df.columns:
                updated_mat_df[c] = 0.0
                
        mat_cols_cfg = {"Alternativa": st.column_config.TextColumn("Nome da Alternativa", required=True)}
        for c in active_crits:
            mat_cols_cfg[c] = st.column_config.NumberColumn(c, default=0.0)
            
        with c2:
            st.subheader("2. Avaliação das Alternativas")
            edited_matrix_df = st.data_editor(
                updated_mat_df, column_config=mat_cols_cfg,
                num_rows="dynamic", hide_index=True, use_container_width=True
            )
            st.session_state.manual_matrix = edited_matrix_df

        alts = edited_matrix_df["Alternativa"].dropna().tolist()
        criteria = active_crits
        types = edited_crit_df["Sentido"].tolist()
        w_raw = np.array(edited_crit_df["Peso Inicial"].tolist(), dtype=float)
        weights_setup = w_raw / w_raw.sum() if w_raw.sum() > 0 else np.ones(len(criteria))/len(criteria)
        mat = edited_matrix_df[criteria].astype(float).values

        if len(alts) < 2:
            st.info("💡 Adicione pelo menos 2 alternativas.")
            st.stop()

# =============================================================================
# TAB 1 — MOTORES DE PESOS (ESCOLHA E INJEÇÃO GLOBAL)
# =============================================================================
with tabs[1]:
    st.header("⚖️ Motores de Geração de Pesos")
    st.markdown("Selecione um método e visualize o passo-a-passo. Ative a *switch* para forçar os outros modelos a usar estes pesos.")
    
    weight_method = st.radio("Metodologia de Extração de Pesos:", ["AHP", "SWING", "SMART", "Entropia de Shannon", "CRITIC"], horizontal=True)
    n_crit = len(criteria)
    generated_weights = weights_setup.copy()
    
    if weight_method == "AHP":
        st.subheader("🔺 Passo-a-Passo: AHP")
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
        m1, m2, m3 = st.columns(3)
        m1.metric("λ_max", f"{lambdamax:.4f}")
        m2.metric("CI", f"{ci:.4f}")
        m3.metric("CR", f"{cr:.4f}", delta="Válido (<0.10)" if cr <= 0.10 else "Inconsistente (>0.10)", delta_color="normal" if cr <= 0.10 else "inverse")

    elif weight_method == "SWING":
        st.subheader("🎯 Passo-a-Passo: SWING")
        if "swing_pts" not in st.session_state or len(st.session_state.swing_pts) != n_crit:
            st.session_state.swing_pts = [100.0] + [50.0]*(n_crit-1)
        pts_df = pd.DataFrame({"Critério": criteria, "Pontos (0-100)": st.session_state.swing_pts})
        edited_pts = st.data_editor(pts_df, hide_index=True, use_container_width=True)
        p = edited_pts["Pontos (0-100)"].values.astype(float)
        st.session_state.swing_pts = p
        generated_weights = p / p.sum() if p.sum() > 0 else np.ones(n_crit)/n_crit

    elif weight_method == "SMART":
        st.subheader("📊 Passo-a-Passo: SMART")
        if "smart_pts" not in st.session_state or len(st.session_state.smart_pts) != n_crit:
            st.session_state.smart_pts = [50.0]*n_crit
        pts_df = pd.DataFrame({"Critério": criteria, "Pontos (0-100)": st.session_state.smart_pts})
        edited_pts = st.data_editor(pts_df, hide_index=True, use_container_width=True)
        p = edited_pts["Pontos (0-100)"].values.astype(float)
        st.session_state.smart_pts = p
        generated_weights = p / p.sum() if p.sum() > 0 else np.ones(n_crit)/n_crit

    elif weight_method == "Entropia de Shannon":
        st.subheader("📐 Passo-a-Passo: Entropia")
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
        st.dataframe(pd.DataFrame({"Critério": criteria, "Entropia (E)": ej, "Divergência (d)": dj}).round(4), hide_index=True)
        generated_weights = dj / dj.sum() if dj.sum() > 0 else np.ones(n_crit)/n_crit

    elif weight_method == "CRITIC":
        st.subheader("📈 Passo-a-Passo: CRITIC")
        norm_c = normalize_minmax(mat, types)
        std = np.std(norm_c, axis=0)
        corr = np.corrcoef(norm_c.T)
        corr = np.nan_to_num(corr, nan=0.0)
        conflict = std * np.sum(1 - corr, axis=1)
        st.dataframe(pd.DataFrame({"Critério": criteria, "Conflito (C)": conflict, "Desvio Padrão": std}).round(4), hide_index=True)
        generated_weights = conflict / conflict.sum() if conflict.sum() > 0 else np.ones(n_crit)/n_crit

    activate_weights = st.toggle(f"🔥 INJETAR PESOS '{weight_method.upper()}' EM TODOS OS MODELOS", value=False)
    
    # OVERRIDE GLOBAL
    final_weights = generated_weights if activate_weights else weights_setup
    
    if activate_weights:
        st.success(f"✅ Pesos do método {weight_method} estão a alimentar as restantes Tabs!")
    else:
        st.info("⚠️ Atualmente os modelos estão a usar os 'Pesos Iniciais' da Tab de Dados.")
        
    st.dataframe(pd.DataFrame({"Critério": criteria, "Peso Final Usado": final_weights}).style.format({"Peso Final Usado": "{:.4f}"}), hide_index=True)


# =============================================================================
# PREPARAÇÃO E EXECUÇÃO SEGURA DOS MODELOS
# =============================================================================
all_results = {}

res_topsis, err_top = safe_call(model_topsis, mat, final_weights, types)
if not err_top: all_results["TOPSIS"] = res_topsis

res_electre, err_elec = safe_call(model_electre, mat, final_weights, types, c_thresh, d_thresh)
if not err_elec: all_results["ELECTRE"] = res_electre

res_prom, err_prom = safe_call(model_promethee, mat, final_weights, types, promethee_fn)
if not err_prom: all_results["PROMETHEE"] = res_prom

res_vikor, err_vik = safe_call(model_vikor, mat, final_weights, types, vikor_v)
if not err_vik: all_results["VIKOR"] = res_vikor

res_cop, err_cop = safe_call(model_copras, mat, final_weights, types)
if not err_cop: all_results["COPRAS"] = res_cop

res_dem, err_dem = safe_call(model_dematel, mat, final_weights, types)
if not err_dem: all_results["DEMATEL"] = res_dem


# =============================================================================
# RENDERIZAÇÃO DETALHADA DAS TABS DE CADA MODELO (PASSO-A-PASSO E SENSIBILIDADE)
# =============================================================================

# TAB 2: TOPSIS
with tabs[2]:
    st.header("🎯 TOPSIS")
    if "TOPSIS" in all_results:
        r = all_results["TOPSIS"]
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Matriz Normalizada**")
            st.dataframe(pd.DataFrame(r["normalized"], index=alts, columns=criteria).round(4))
        with c2:
            st.write("**Matriz Ponderada**")
            st.dataframe(pd.DataFrame(r["weighted"], index=alts, columns=criteria).round(4))
            
        st.write("**Soluções Ideal e Anti-Ideal**")
        st.dataframe(pd.DataFrame({"Critério": criteria, "A+ (Ideal)": r["ideal"], "A- (Anti-Ideal)": r["anti_ideal"]}).round(4), hide_index=True)
        
        st.subheader("Ranking Final")
        rdf = pd.DataFrame({"Alternativa": alts, "D+": r["d_plus"], "D-": r["d_minus"], "Score (Ci)": r["scores"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"D+": "{:.4f}", "D-": "{:.4f}", "Score (Ci)": "{:.4f}"}), hide_index=True, use_container_width=True)
        
        render_sensitivity(model_topsis, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"])
    else: st.error("Erro ao calcular TOPSIS. Verifique os dados.")

# TAB 3: ELECTRE
with tabs[3]:
    st.header("🔗 ELECTRE I")
    if "ELECTRE" in all_results:
        r = all_results["ELECTRE"]
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Matriz de Concordância (C)**")
            st.dataframe(pd.DataFrame(r["concordance"], index=alts, columns=alts).round(3))
        with c2:
            st.write("**Matriz de Discordância (D)**")
            st.dataframe(pd.DataFrame(r["discordance"], index=alts, columns=alts).round(3))
            
        st.write("**Matriz de Sobreclassificação (Booleana)**")
        st.dataframe(pd.DataFrame(r["outrank"], index=alts, columns=alts))
        
        st.subheader("Ranking Final")
        rdf = pd.DataFrame({"Alternativa": alts, "Score Dominância": r["scores"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Score Dominância": "{:.2f}"}), hide_index=True, use_container_width=True)
        
        render_sensitivity(model_electre, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"], c_thresh=c_thresh, d_thresh=d_thresh)
    else: st.error("Erro ao calcular ELECTRE.")

# TAB 4: PROMETHEE
with tabs[4]:
    st.header("📊 PROMETHEE II")
    if "PROMETHEE" in all_results:
        r = all_results["PROMETHEE"]
        st.write("**Matriz de Preferência Agregada**")
        st.dataframe(pd.DataFrame(r["preference_matrix"], index=alts, columns=alts).round(4))
        
        st.subheader("Ranking Final")
        rdf = pd.DataFrame({"Alternativa": alts, "Fluxo Positivo (Φ+)": r["phi_plus"], "Fluxo Negativo (Φ-)": r["phi_minus"], "Fluxo Líquido (Φ)": r["scores"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Fluxo Positivo (Φ+)": "{:.4f}", "Fluxo Negativo (Φ-)": "{:.4f}", "Fluxo Líquido (Φ)": "{:.4f}"}), hide_index=True, use_container_width=True)
        
        render_sensitivity(model_promethee, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"], function=promethee_fn)
    else: st.error("Erro ao calcular PROMETHEE.")

# TAB 5: VIKOR
with tabs[5]:
    st.header("⚖️ VIKOR")
    if "VIKOR" in all_results:
        r = all_results["VIKOR"]
        st.subheader("Ranking Final (S, R e Índice Q)")
        st.info("Nota: No método VIKOR o menor Q representa a melhor solução. A coluna Score converte isso internamente.")
        rdf = pd.DataFrame({"Alternativa": alts, "Utilidade (S)": r["S"], "Arrependimento (R)": r["R"], "Índice (Q)": r["Q"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Utilidade (S)": "{:.4f}", "Arrependimento (R)": "{:.4f}", "Índice (Q)": "{:.4f}"}), hide_index=True, use_container_width=True)
        
        render_sensitivity(model_vikor, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"], v=vikor_v)
    else: st.error("Erro ao calcular VIKOR.")

# TAB 6: COPRAS
with tabs[6]:
    st.header("🧮 COPRAS")
    if "COPRAS" in all_results:
        r = all_results["COPRAS"]
        st.subheader("Ranking Final")
        rdf = pd.DataFrame({"Alternativa": alts, "Benefícios (S+)": r["S_plus"], "Custos (S-)": r["S_minus"], "Peso Relativo (Q)": r["Q"], "Grau Utilidade % (N)": r["scores"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Benefícios (S+)": "{:.4f}", "Custos (S-)": "{:.4f}", "Peso Relativo (Q)": "{:.4f}", "Grau Utilidade % (N)": "{:.2f}"}), hide_index=True, use_container_width=True)
        
        render_sensitivity(model_copras, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"])
    else: st.error("Erro ao calcular COPRAS.")

# TAB 7: DEMATEL
with tabs[7]:
    st.header("🌐 DEMATEL")
    if "DEMATEL" in all_results:
        r = all_results["DEMATEL"]
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write("**Matriz de Relação Total (T)**")
            st.dataframe(pd.DataFrame(r["T"], index=criteria, columns=criteria).round(4))
        with c2:
            st.write("**Prominência e Relação**")
            st.dataframe(pd.DataFrame({"Critério": criteria, "D+R": r["prominence"], "D-R": r["relation"]}).round(4), hide_index=True)
            
        st.subheader("Ranking Final (após ajuste de pesos pelo D+R)")
        rdf = pd.DataFrame({"Alternativa": alts, "Score": r["scores"], "Ranking": r["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Score": "{:.4f}"}), hide_index=True, use_container_width=True)
        
        render_sensitivity(model_dematel, mat, final_weights, types, alts, criteria, sens_pct, r["ranking"])
    else: st.error("Erro ao calcular DEMATEL.")

# =============================================================================
# TAB 8 — DASHBOARD CONSOLIDADO (BORDA)
# =============================================================================
with tabs[8]:
    st.header("🏆 Matriz de Consenso e Dashboard Final")
    models_executed = list(all_results.keys())
    
    if not models_executed:
        st.error("Nenhum modelo produziu resultados válidos.")
    else:
        consolidated_df = pd.DataFrame({"Alternativa": alts})
        for m in models_executed: consolidated_df[m] = all_results[m]["ranking"]
            
        consolidated_df["Posição Média"] = consolidated_df[models_executed].mean(axis=1).round(2)
        # Borda invertido: A menor média fica em 1º
        consolidated_df["Ranking Consolidado"] = ranking_from_scores(consolidated_df["Posição Média"].values, higher_is_better=False)
        consolidated_df = consolidated_df.sort_values("Ranking Consolidado").reset_index(drop=True)
        
        st.info("💡 **Método de Borda:** A ordenação baseia-se na menor posição média. Valores menores (próximos de 1) são melhores.")
        styled_matrix = consolidated_df.style.format({"Posição Média": "{:.2f}"}).background_gradient(subset=models_executed + ["Posição Média", "Ranking Consolidado"], cmap="RdYlGn_r")
        st.dataframe(styled_matrix, use_container_width=True, hide_index=True)
        
        podium = consolidated_df["Alternativa"].tolist()
        p1, p2, p3 = st.columns(3)
        p1.metric("🥇 Vencedor Consensual", podium[0] if len(podium) > 0 else "—")
        p2.metric("🥈 2º Classificado", podium[1] if len(podium) > 1 else "—")
        p3.metric("🥉 3º Classificado", podium[2] if len(podium) > 2 else "—")
