# -*- coding: utf-8 -*-
"""
MCDM Dashboard — Sistema de Apoio à Decisão Multicritério
Modelos de Decisão | MEGI ISEL 2025/2026 | Caso de Estudo MCG

Aplicação Streamlit single-file com 15 abas:
Visão Geral, AHP, ANP, TOPSIS, ELECTRE, PROMETHEE, VIKOR, MAUT,
COPRAS, DEMATEL, Fuzzy AHP, Fuzzy TOPSIS, Fuzzy ANP, Dashboard, Relatório.

Execução: streamlit run app.py
Requisitos: streamlit pandas numpy scipy openpyxl plotly
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
st.set_page_config(
    page_title="MCDM Dashboard | MCG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { letter-spacing: -0.02em; }
    .stMetric { background: rgba(120,120,120,0.06); padding: 0.6rem; border-radius: 8px; }
    div[data-testid="stExpander"] details { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 MCDM Dashboard — Priorização Multicritério")
st.caption("Modelos de Decisão | MEGI ISEL 2025/2026 | Caso de Estudo MCG")

# =============================================================================
# UTILITIES E FUNÇÕES BASE
# =============================================================================
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
            7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.54,
            13: 1.56, 14: 1.57, 15: 1.59}

def safe_call(fn, *args, **kwargs):
    """Executa uma função e devolve (resultado, erro_str). Nunca rebenta o Streamlit."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"

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
        if rng == 0:
            out[:, j] = 1.0
            continue
        if types[j] == "max":
            out[:, j] = (col - col.min()) / rng
        else:
            out[:, j] = (col.max() - col) / rng
    return out

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

def calc_perturbed_weights(weights, target_idx, factor):
    """Calcula novos pesos variando um alvo e renormalizando os restantes."""
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
    """Módulo transversal para renderizar a Sensibilidade em Paralelo em qualquer método."""
    st.markdown("---")
    st.subheader(f"🔄 Análise de Sensibilidade Paralela (±{sens_pct}%)")
    st.markdown(
        "Variação isolada do peso de **cada critério**, com **renormalização automática** dos restantes. "
        "A cor **verde** indica melhoria no ranking e **vermelho** indica perda de posição face ao cenário Base."
    )

    df_plus = pd.DataFrame({"Alternativa": alts, "Base": base_ranking})
    df_minus = pd.DataFrame({"Alternativa": alts, "Base": base_ranking})

    for j, crit in enumerate(criteria):
        # +% Variação
        w_plus = calc_perturbed_weights(weights, j, 1 + sens_pct / 100.0)
        res_plus, err = safe_call(model_func, mat, w_plus, types, **kwargs)
        if not err and "ranking" in res_plus: df_plus[f"+ {crit}"] = res_plus["ranking"]

        # -% Variação
        w_minus = calc_perturbed_weights(weights, j, 1 - sens_pct / 100.0)
        res_minus, err = safe_call(model_func, mat, w_minus, types, **kwargs)
        if not err and "ranking" in res_minus: df_minus[f"- {crit}"] = res_minus["ranking"]

    def style_row(row):
        base = row['Base']
        styles = []
        for col, val in row.items():
            if col in ['Alternativa', 'Base']: styles.append('')
            elif val < base: styles.append('color: #00B140; font-weight: bold;') # Melhor = verde
            elif val > base: styles.append('color: #D32F2F; font-weight: bold;') # Pior = vermelho
            else: styles.append('color: gray;')
        return styles

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Aumento de +{sens_pct}% no Peso**")
        st.dataframe(df_plus.style.apply(style_row, axis=1), use_container_width=True, hide_index=True)
    with c2:
        st.write(f"**Redução de -{sens_pct}% no Peso**")
        st.dataframe(df_minus.style.apply(style_row, axis=1), use_container_width=True, hide_index=True)

# =============================================================================
# MODELOS MCDM
# =============================================================================
def model_saw(mat, weights, types):
    """Proxy para cálculo de scores do AHP."""
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
    return {"normalized": norm, "weighted": weighted, "ideal": ideal, "anti_ideal": anti,
            "d_plus": d_plus, "d_minus": d_minus, "scores": ci, "ranking": ranking_from_scores(ci)}

def preference(d, ftype="usual", p=None, q=None, sigma=None):
    if d <= 0: return 0.0
    if ftype == "usual": return 1.0
    if ftype == "u_shape": return 1.0 if d > (q or 0) else 0.0
    if ftype in ("v_shape", "linear"): return float(min(d / p, 1.0)) if p else 0.0
    if ftype == "level": return 0.0 if d <= (q or 0) else 0.5 if d <= (p or 1) else 1.0
    if ftype == "linear_indiff": return 0.0 if d <= (q or 0) else (d - q) / (p - q) if p > q else 1.0
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
    return {"preference_matrix": pref, "phi_plus": phi_plus, "phi_minus": phi_minus,
            "scores": phi_net, "ranking": ranking_from_scores(phi_net)}

def model_electre(mat, weights, types, c_thresh=0.6, d_thresh=0.4):
    n_alt, n_crit = mat.shape
    norm = normalize_minmax(mat, types)
    w_sum = weights.sum() if weights.sum() > 0 else 1.0
    concordance = np.zeros((n_alt, n_alt))
    discordance = np.zeros((n_alt, n_alt))
    global_range = max(norm.max() - norm.min(), 1e-9)

    for i in range(n_alt):
        for k in range(n_alt):
            if i == k: continue
            c = sum(weights[j] for j in range(n_crit) if norm[i, j] >= norm[k, j])
            concordance[i, k] = c / w_sum
            diffs = [norm[k, j] - norm[i, j] for j in range(n_crit) if norm[k, j] > norm[i, j]]
            discordance[i, k] = (max(diffs) / global_range) if diffs else 0.0

    outrank = (concordance >= c_thresh) & (discordance <= d_thresh)
    np.fill_diagonal(outrank, False)

    kernel = []
    for i in range(n_alt):
        dominated = any(outrank[k, i] and not outrank[i, k] for k in range(n_alt) if k != i)
        if not dominated: kernel.append(i)

    net_dominance = outrank.sum(axis=1) - outrank.sum(axis=0)
    return {"concordance": concordance, "discordance": discordance, "outrank": outrank,
            "kernel": kernel, "scores": net_dominance.astype(float), "ranking": ranking_from_scores(net_dominance.astype(float))}

def model_ahp(pairwise):
    A = np.asarray(pairwise, dtype=float)
    n = A.shape[0]
    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmax(eigvals.real))
    lam_max = float(eigvals[idx].real)
    w = np.abs(eigvecs[:, idx].real)
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    CI = (lam_max - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.59)
    CR = CI / RI if RI > 0 else 0.0
    return {"weights": w, "lambda_max": lam_max, "CI": CI, "CR": CR, "consistent": CR < 0.10}

def model_vikor(mat, weights, types, v=0.5):
    n_alt, n_crit = mat.shape
    f_best = np.array([mat[:, j].max() if types[j] == "max" else mat[:, j].min() for j in range(n_crit)])
    f_worst = np.array([mat[:, j].min() if types[j] == "max" else mat[:, j].max() for j in range(n_crit)])
    rng = np.where((f_best - f_worst) == 0, 1e-9, f_best - f_worst)
    S = np.zeros(n_alt)
    R = np.zeros(n_alt)
    for i in range(n_alt):
        terms = np.zeros(n_crit)
        for j in range(n_crit):
            d = (f_best[j] - mat[i, j]) if types[j] == "max" else (mat[i, j] - f_best[j])
            terms[j] = weights[j] * d / rng[j]
        S[i] = terms.sum()
        R[i] = terms.max()
    s_b, s_w = S.min(), S.max()
    r_b, r_w = R.min(), R.max()
    s_rng = (s_w - s_b) if s_w != s_b else 1e-9
    r_rng = (r_w - r_b) if r_w != r_b else 1e-9
    Q = v * (S - s_b) / s_rng + (1 - v) * (R - r_b) / r_rng
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
    else:
        Q = S_plus
    N = (Q / Q.max()) * 100 if Q.max() != 0 else Q
    return {"S_plus": S_plus, "S_minus": S_minus, "Q": Q, "N": N, "scores": N, "ranking": ranking_from_scores(N)}

def model_maut(mat, weights, types):
    norm = normalize_minmax(mat, types)
    U = (norm * weights).sum(axis=1)
    return {"utility_matrix": norm, "scores": U, "ranking": ranking_from_scores(U)}

def _influence_supermatrix(mat):
    try:
        corr = np.corrcoef(mat.T)
        corr = np.nan_to_num(np.abs(corr), nan=0.0)
        col_sums = corr.sum(axis=0)
        col_sums = np.where(col_sums == 0, 1, col_sums)
        return corr / col_sums
    except Exception:
        n = mat.shape[1]
        return np.eye(n)

def _limit_matrix(M, iters=60, tol=1e-9):
    prev = M.copy()
    for _ in range(iters):
        nxt = prev @ M
        if np.allclose(prev, nxt, atol=tol): return nxt
        prev = nxt
    return prev

def model_anp(mat, weights, types):
    M = _influence_supermatrix(mat)
    L = _limit_matrix(M)
    adj = L @ weights
    adj = adj / adj.sum() if adj.sum() > 0 else weights
    norm = normalize_minmax(mat, types)
    scores = (norm * adj).sum(axis=1)
    return {"adjusted_weights": adj, "limit_matrix": L, "scores": scores, "ranking": ranking_from_scores(scores)}

def model_dematel(mat, weights, types):
    try:
        Z = np.abs(np.corrcoef(mat.T))
        Z = np.nan_to_num(Z)
        np.fill_diagonal(Z, 0)
        s = max(Z.sum(axis=1).max(), Z.sum(axis=0).max())
        X = Z / s if s > 0 else Z
        n = X.shape[0]
        T = X @ np.linalg.inv(np.eye(n) - X)
    except Exception:
        T = np.eye(mat.shape[1])
    D = T.sum(axis=1)
    R = T.sum(axis=0)
    prominence = D + R
    relation = D - R
    if prominence.sum() > 0:
        adj = weights * prominence
        adj = adj / adj.sum()
    else:
        adj = weights
    norm = normalize_minmax(mat, types)
    scores = (norm * adj).sum(axis=1)
    return {"T": T, "D": D, "R": R, "prominence": prominence, "relation": relation,
            "adjusted_weights": adj, "scores": scores, "ranking": ranking_from_scores(scores)}

def model_fuzzy_ahp(weights):
    fuzzy = np.array([(w * 0.8, w, w * 1.2) for w in weights])
    crisp = fuzzy.mean(axis=1)
    crisp = crisp / crisp.sum() if crisp.sum() > 0 else weights
    return {"fuzzy_weights": fuzzy, "crisp_weights": crisp}

def model_fuzzy_topsis(mat, weights, types, spread=0.10):
    l = mat * (1 - spread); m = mat.copy(); u = mat * (1 + spread)
    n_alt, n_crit = mat.shape
    L = np.zeros_like(l); M = np.zeros_like(m); U = np.zeros_like(u)
    for j in range(n_crit):
        if types[j] == "max":
            denom = max(u[:, j].max(), 1e-9)
            L[:, j] = l[:, j] / denom; M[:, j] = m[:, j] / denom; U[:, j] = u[:, j] / denom
        else:
            num = max(l[:, j].min(), 1e-9)
            L[:, j] = num / np.where(u[:, j] == 0, 1e-9, u[:, j])
            M[:, j] = num / np.where(m[:, j] == 0, 1e-9, m[:, j])
            U[:, j] = num / np.where(l[:, j] == 0, 1e-9, l[:, j])
    Lw, Mw, Uw = L * weights, M * weights, U * weights
    fpis = np.array([(Uw[:, j].max(), Uw[:, j].max(), Uw[:, j].max()) for j in range(n_crit)])
    fnis = np.array([(Lw[:, j].min(), Lw[:, j].min(), Lw[:, j].min()) for j in range(n_crit)])
    def vd(al, am, au, bl, bm, bu): return np.sqrt(((al - bl) ** 2 + (am - bm) ** 2 + (au - bu) ** 2) / 3.0)
    d_plus = np.zeros(n_alt); d_minus = np.zeros(n_alt)
    for i in range(n_alt):
        for j in range(n_crit):
            d_plus[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], fpis[j, 0], fpis[j, 1], fpis[j, 2])
            d_minus[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], fnis[j, 0], fnis[j, 1], fnis[j, 2])
    denom = np.where((d_plus + d_minus) == 0, 1e-9, d_plus + d_minus)
    cc = d_minus / denom
    return {"d_plus": d_plus, "d_minus": d_minus, "scores": cc, "ranking": ranking_from_scores(cc)}

def model_fuzzy_anp(mat, weights, types):
    fahp = model_fuzzy_ahp(weights)
    fw = fahp["crisp_weights"]
    M = _influence_supermatrix(mat)
    L = _limit_matrix(M, iters=40)
    adj = L @ fw
    adj = adj / adj.sum() if adj.sum() > 0 else fw
    norm = normalize_minmax(mat, types)
    scores = (norm * adj).sum(axis=1)
    return {"crisp_fuzzy_weights": fw, "adjusted_weights": adj, "scores": scores, "ranking": ranking_from_scores(scores)}


# =============================================================================
# CARREGAMENTO DE DADOS E CONFIGURAÇÃO NA SIDEBAR
# =============================================================================
def build_demo_data():
    df = pd.DataFrame({
        "Alternativa": [f"A{i}" for i in range(1, 10)],
        "C1_VP": [250_000_000, 300_000, 900_000, 650_000, 5_000_000, 1_350_000, 10_500_000, 3_450_000, 15_000_000],
        "C2_PF": [0.25, 0.35, 0.50, 0.50, 0.40, 0.50, 0.40, 0.40, 0.60],
        "C3_EE": [24, 8, 8, 8, 24, 8, 16, 8, 24],
        "C4_FE": [4, 5, 3, 3, 4, 3, 3, 3, 4],
        "C5_UD": [180, 60, 60, 90, 30, 60, 180, 60, 300],
        "C6_RC": [4, 5, 5, 3, 3, 5, 4, 4, 3],
    })
    weights = [0.4615, 0.1987, 0.0230, 0.0972, 0.0217, 0.1979]
    return df, weights

with st.sidebar:
    st.header("⚙️ Configuração")
    uploaded = st.file_uploader("Carregar Excel (.xlsx)", type=["xlsx", "xls"])
    use_demo = st.checkbox("Usar dados de demonstração MCG", value=False, key="use_demo_data")
    
    st.divider()
    st.subheader("🎛️ Parâmetros de modelos")
    c_thresh = st.slider("ELECTRE — limiar de concordância (c)", 0.50, 0.95, 0.65, 0.01)
    d_thresh = st.slider("ELECTRE — limiar de discordância (d)", 0.05, 0.50, 0.35, 0.01)
    promethee_fn = st.selectbox("PROMETHEE — função de preferência", ["usual", "linear", "gaussian"], index=1)
    vikor_v = st.slider("VIKOR — peso da estratégia v", 0.0, 1.0, 0.5, 0.05)
    sens_pct = st.slider("Variação para Análise de Sensibilidade (±%)", 5, 50, 20, 5)

if "loaded" not in st.session_state: st.session_state.loaded = False

if use_demo:
    raw_df, raw_weights = build_demo_data()
    id_col = "Alternativa"
    all_criteria = [c for c in raw_df.columns if c != id_col]
    st.session_state.loaded = True
elif uploaded is not None:
    try:
        xls = pd.ExcelFile(uploaded)
        raw_df = pd.read_excel(xls, sheet_name="Dados")
        id_col = raw_df.columns[0]
        raw_df[id_col] = raw_df[id_col].astype(str)
        all_criteria = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if "Pesos" in xls.sheet_names:
            wdf = pd.read_excel(xls, sheet_name="Pesos", header=None)
            raw_weights = pd.to_numeric(wdf.iloc[0], errors="coerce").dropna().tolist()[:len(all_criteria)]
        else:
            raw_weights = [1.0/len(all_criteria)] * len(all_criteria)
            st.sidebar.warning("⚠️ Folha `Pesos` ausente. Pesos uniformes aplicados.")
        st.session_state.loaded = True
    except Exception as e:
        st.sidebar.error(f"Erro no carregamento: {e}")
        st.session_state.loaded = False

if st.session_state.loaded:
    with st.sidebar:
        st.divider()
        st.subheader("🎯 Adicionar/Remover Critérios")
        active_criteria = st.multiselect("Critérios Ativos", all_criteria, default=all_criteria)
        
        if len(active_criteria) < 2:
            st.error("Selecione pelo menos 2 critérios.")
            st.stop()

        indices = [all_criteria.index(c) for c in active_criteria]
        filtered_weights = [raw_weights[i] if i < len(raw_weights) else 1.0/len(active_criteria) for i in indices]
        
        def _is_cost_criterion(name: str) -> bool:
            low = name.lower()
            for sub in ("custo", "cost", "esforço", "esforco", "prazo", "tempo", "delay", "risk", "risco", "urgenc"):
                if sub in low: return True
            tokens = low.replace("-", "_").split("_")
            return any(t in {"ee", "ud", "urg", "dias"} for t in tokens)
        
        type_defaults = ["min" if _is_cost_criterion(c) else "max" for c in active_criteria]
        config_df = pd.DataFrame({"Critério": active_criteria, "Sentido": type_defaults, "Peso": filtered_weights})
        
        st.caption("Edite os pesos e sentidos na tabela. Os pesos são normalizados na hora.")
        edited_cfg = st.data_editor(
            config_df,
            column_config={
                "Critério": st.column_config.TextColumn("Critério", disabled=True),
                "Sentido": st.column_config.SelectboxColumn("Sentido", options=["max", "min"], required=True),
                "Peso": st.column_config.NumberColumn("Peso", min_value=0.0, step=0.01),
            },
            hide_index=True, use_container_width=True
        )
        
        types = edited_cfg["Sentido"].tolist()
        weights = np.array(edited_cfg["Peso"].tolist())
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(active_criteria))/len(active_criteria)
        criteria = active_criteria

# =============================================================================
# HELPERS DE RENDERIZAÇÃO
# =============================================================================
def render_ranking_chart(alts, scores, title, label="Score"):
    df = pd.DataFrame({"Alternativa": alts, label: scores}).sort_values(label, ascending=False)
    fig = px.bar(df, x="Alternativa", y=label, title=title, text_auto=".3f", color=label, color_continuous_scale="Tealgrn")
    fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig

if not st.session_state.loaded:
    st.info("👈 Carregue um ficheiro Excel na sidebar (ou active o modo demo) para começar.")
    st.stop()

# =============================================================================
# DEFINIR TABS
# =============================================================================
TAB_LABELS = [
    "📋 Visão Geral", "🔺 AHP", "🕸️ ANP", "🎯 TOPSIS", "🔗 ELECTRE",
    "📊 PROMETHEE", "⚖️ VIKOR", "📐 MAUT", "🧮 COPRAS", "🌐 DEMATEL",
    "🌫️ Fuzzy AHP", "🌫️ Fuzzy TOPSIS", "🌫️ Fuzzy ANP", "🏆 Dashboard", "📄 Relatório"
]
tabs = st.tabs(TAB_LABELS)
all_results = {}

# Extração da Base para edição
base_df = raw_df[[id_col] + active_criteria]

# =============================================================================
# TAB 0 — VISÃO GERAL (Alternativas Dinâmicas)
# =============================================================================
with tabs[0]:
    st.header("📋 Visão Geral e Edição Dinâmica")
    st.markdown("Pode **adicionar ou remover alternativas** diretamente na tabela abaixo adicionando/apagando linhas. A matriz será recalculada em tempo real.")
    
    edited_data_df = st.data_editor(base_df, num_rows="dynamic", use_container_width=True, hide_index=True)
    data_df = edited_data_df.dropna(subset=[id_col])
    alts = data_df[id_col].tolist()
    mat = data_df[criteria].astype(float).values

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Pesos e Sentidos (Renormalizados)")
        st.dataframe(pd.DataFrame({"Critério": criteria, "Peso": [f"{w:.4f}" for w in weights], "Sentido": types}), use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Estatísticas Descritivas")
        st.dataframe(data_df[criteria].describe().T, use_container_width=True)

    st.subheader("Heatmap normalizado (min-max)")
    try:
        norm = normalize_minmax(mat, types)
        fig = px.imshow(norm, labels=dict(x="Critério", y="Alternativa", color="Valor"), x=criteria, y=alts, color_continuous_scale="RdYlGn", aspect="auto", text_auto=".2f")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Erro no heatmap: {exc}")


# =============================================================================
# TAB 1 — AHP
# =============================================================================
with tabs[1]:
    st.header("🔺 AHP — Analytic Hierarchy Process (Saaty, 1980)")
    st.markdown("O vetor de pesos é o autovector principal associado ao maior autovalor $\\lambda_{max}$:")
    st.latex(r"A = [a_{ij}],\quad a_{ji} = 1/a_{ij},\quad a_{ii} = 1 \implies A\,w = \lambda_{max}\,w")
    
    n = len(criteria)
    init = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and weights[j] != 0: init[i, j] = weights[i] / weights[j]
    pw_df = pd.DataFrame(init, index=criteria, columns=criteria).round(4)
    st.markdown("Pode **ajustar os julgamentos par-a-par** diretamente na tabela:")
    edited_pw = st.data_editor(pw_df, use_container_width=True, key="ahp_pw")
    
    E = edited_pw.values.astype(float).copy()
    for i in range(n):
        for j in range(n):
            if i == j: E[i, j] = 1.0
            elif i < j and E[i, j] != 0: E[j, i] = 1.0 / E[i, j]

    res, err = safe_call(model_ahp, E)
    if err: st.error(f"Erro AHP: {err}")
    else:
        st.subheader("Passo 1: Verificação de Consistência")
        st.latex(r"CI = \frac{\lambda_{max} - n}{n - 1},\quad CR = \frac{CI}{RI(n)}")
        m1, m2, m3 = st.columns(3)
        m1.metric("λ_max", f"{res['lambda_max']:.4f}")
        m2.metric("CI", f"{res['CI']:.4f}")
        m3.metric("CR", f"{res['CR']:.4f}", delta="Consistente" if res['consistent'] else "Inconsistente", delta_color="normal" if res['consistent'] else "inverse")

        if not res['consistent']:
            w_ahp = res['weights']
            worst_pair, worst_diff = None, 0
            for i in range(n):
                for j in range(i + 1, n):
                    if w_ahp[j] != 0:
                        expected, observed = w_ahp[i] / w_ahp[j], E[i, j]
                        if expected > 0:
                            diff = abs(np.log(observed / expected))
                            if diff > worst_diff: worst_diff, worst_pair = diff, (i, j, observed, expected)
            msg = f"⚠️ **Matriz inconsistente: CR = {res['CR']:.4f} > 0.10.** Julgamentos devem ser revistos."
            if worst_pair is not None:
                msg += f"\nPar mais crítico: **{criteria[worst_pair[0]]} vs {criteria[worst_pair[1]]}** → inseriu `{worst_pair[2]:.2f}`, ideal seria `{worst_pair[3]:.2f}`."
            st.warning(msg)
        else:
            st.success(f"✅ Matriz consistente (CR < 0.10).")

        st.subheader("Passo 2: Vector de Pesos AHP e Ranking Additivo")
        comp_df = pd.DataFrame({"Critério": criteria, "Peso AHP": res["weights"]})
        st.dataframe(comp_df.style.format({"Peso AHP": "{:.4f}"}), use_container_width=True, hide_index=True)
        
        score_res = model_saw(mat, res["weights"], types)
        rdf = pd.DataFrame({"Alternativa": alts, "Score AHP": score_res["scores"], "Ranking": score_res["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Score AHP": "{:.4f}"}), use_container_width=True, hide_index=True)
        all_results["AHP"] = {"scores": score_res["scores"], "ranking": score_res["ranking"]}
        
        render_sensitivity(model_saw, mat, res["weights"], types, alts, criteria, sens_pct, score_res["ranking"])


# =============================================================================
# TAB 2 — ANP
# =============================================================================
with tabs[2]:
    st.header("🕸️ ANP — Analytic Network Process (Saaty, 1996)")
    st.markdown("Captura **dependências entre critérios**. A solução de rede estável é extraída da matriz limite.")
    st.latex(r"W^\infty = \lim_{k \to \infty} W^k, \quad W_{ij} = \frac{|\rho(C_i, C_j)|}{\sum_k |\rho(C_k, C_j)|}")
    
    res, err = safe_call(model_anp, mat, weights, types)
    if err: st.error(f"Erro ANP: {err}")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Passo 1: Matriz limite")
            st.dataframe(pd.DataFrame(res["limit_matrix"].round(4), index=criteria, columns=criteria), use_container_width=True)
        with c2:
            st.subheader("Passo 2: Pesos ajustados (ANP)")
            st.dataframe(pd.DataFrame({"Critério": criteria, "Peso original": weights, "Peso ANP": res["adjusted_weights"]}).round(4), use_container_width=True, hide_index=True)

        st.subheader("Passo 3: Ranking Final")
        rdf = pd.DataFrame({"Alternativa": alts, "Score ANP": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Score ANP": "{:.4f}"}), use_container_width=True, hide_index=True)
        
        all_results["ANP"] = {"scores": res["scores"], "ranking": res["ranking"]}
        render_sensitivity(model_anp, mat, weights, types, alts, criteria, sens_pct, res["ranking"])


# =============================================================================
# TAB 3 — TOPSIS
# =============================================================================
with tabs[3]:
    st.header("🎯 TOPSIS (Hwang & Yoon, 1981)")
    res, err = safe_call(model_topsis, mat, weights, types)
    if err: st.error(f"Erro TOPSIS: {err}")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Passo 1: Normalização Euclidiana")
            st.latex(r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^m x_{kj}^2}}")
            st.dataframe(pd.DataFrame(res["normalized"], index=alts, columns=criteria).round(4), use_container_width=True)
        with c2:
            st.subheader("Passo 2: Matriz ponderada")
            st.latex(r"v_{ij} = w_j \cdot r_{ij}")
            st.dataframe(pd.DataFrame(res["weighted"], index=alts, columns=criteria).round(4), use_container_width=True)

        st.subheader("Passo 3: Ideal ($A^+$), Anti-Ideal ($A^-$)")
        st.dataframe(pd.DataFrame({"Critério": criteria, "A+": res["ideal"], "A−": res["anti_ideal"]}).round(4), use_container_width=True, hide_index=True)

        st.subheader("Passo 4: Distâncias e Coeficiente ($C_i^*$)")
        st.latex(r"D_i^+ = \sqrt{\sum_{j=1}^n (v_{ij} - v_j^+)^2}, \quad D_i^- = \sqrt{\sum_{j=1}^n (v_{ij} - v_j^-)^2}, \quad C_i^* = \frac{D_i^-}{D_i^+ + D_i^-}")
        rdf = pd.DataFrame({"Alternativa": alts, "D+": res["d_plus"], "D−": res["d_minus"], "Ci*": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"D+": "{:.4f}", "D−": "{:.4f}", "Ci*": "{:.4f}"}), use_container_width=True, hide_index=True)
        
        all_results["TOPSIS"] = {"scores": res["scores"], "ranking": res["ranking"]}
        render_sensitivity(model_topsis, mat, weights, types, alts, criteria, sens_pct, res["ranking"])


# =============================================================================
# TAB 4 — ELECTRE
# =============================================================================
with tabs[4]:
    st.header("🔗 ELECTRE I — Relações de Sobreclassificação (Roy, 1968)")
    st.markdown(f"**Limiares correntes**: Concordância $c$ = {c_thresh:.2f} | Discordância $d$ = {d_thresh:.2f} (Ajustáveis na Sidebar)")
    
    res, err = safe_call(model_electre, mat, weights, types, c_thresh, d_thresh)
    if err: st.error(f"Erro ELECTRE: {err}")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Passo 1: Matriz de Concordância (C)")
            st.latex(r"C(a, b) = \frac{1}{\sum w_j} \sum_{j : a_j \succeq b_j} w_j")
            st.dataframe(pd.DataFrame(res["concordance"], index=alts, columns=alts).round(3), use_container_width=True)
        with c2:
            st.subheader("Passo 2: Matriz de Discordância (D)")
            st.latex(r"D(a, b) = \frac{\max_j \{r_{bj} - r_{aj} : r_{bj} > r_{aj}\}}{\max_{j,k,l} |r_{kj} - r_{lj}|}")
            st.dataframe(pd.DataFrame(res["discordance"], index=alts, columns=alts).round(3), use_container_width=True)

        st.subheader("Passo 3: Matriz de Sobreclassificação ($S$)")
        st.latex(r"a\,S\,b \iff C(a,b) \ge c \;\land\; D(a,b) \le d")
        outrank_df = pd.DataFrame(res["outrank"].astype(int), index=alts, columns=alts)
        st.dataframe(outrank_df, use_container_width=True)

        st.subheader("Passo 4: Kernel (Candidatas robustas)")
        kernel_alts = [alts[i] for i in res["kernel"]]
        st.success(f"**Kernel:** {', '.join(kernel_alts) if kernel_alts else '∅ (vazio)'}")
        
        rdf = pd.DataFrame({"Alternativa": alts, "No kernel": ["✅" if i in res["kernel"] else "—" for i in range(len(alts))], "Score Dominância": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf, use_container_width=True, hide_index=True)
        
        # Mapa de Calor de Sensibilidade do Kernel Original (Mantido pela excelência visual)
        try:
            c_grid = np.round(np.arange(max(c_thresh - 0.10, 0.5), min(c_thresh + 0.11, 1.0), 0.05), 2)
            d_grid = np.round(np.arange(max(d_thresh - 0.10, 0.0), min(d_thresh + 0.11, 1.0), 0.05), 2)
            heat = np.zeros((len(d_grid), len(c_grid)))
            for ii, dv in enumerate(d_grid):
                for jj, cv in enumerate(c_grid):
                    r2, e2 = safe_call(model_electre, mat, weights, types, cv, dv)
                    heat[ii, jj] = len(r2["kernel"]) if e2 is None else np.nan
            fig = px.imshow(heat, labels=dict(x="c", y="d", color="|Kernel|"), x=[f"{v:.2f}" for v in c_grid], y=[f"{v:.2f}" for v in d_grid], color_continuous_scale="Viridis", text_auto=True, aspect="auto", title="Sensibilidade Cruzada Limiares (Tamanho do Kernel)")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception: pass

        all_results["ELECTRE"] = {"scores": res["scores"], "ranking": res["ranking"]}
        render_sensitivity(model_electre, mat, weights, types, alts, criteria, sens_pct, res["ranking"], c_thresh=c_thresh, d_thresh=d_thresh)


# =============================================================================
# TAB 5 — PROMETHEE
# =============================================================================
with tabs[5]:
    st.header("📊 PROMETHEE II — Fluxos Líquidos (Brans, 1985)")
    st.markdown(f"**Função de preferência activa:** `{promethee_fn}`")
    
    res, err = safe_call(model_promethee, mat, weights, types, promethee_fn)
    if err: st.error(f"Erro PROMETHEE: {err}")
    else:
        st.subheader("Passo 1: Preferência Agregada $\pi(a,b)$")
        st.latex(r"\pi(a, b) = \sum_{j=1}^n w_j \cdot P_j(d_{ab})")
        st.dataframe(pd.DataFrame(res["preference_matrix"], index=alts, columns=alts).round(4), use_container_width=True)

        st.subheader("Passo 2: Fluxos Positivo ($\phi^+$), Negativo ($\phi^-$) e Líquido ($\phi$)")
        st.latex(r"\phi^+(a) = \frac{1}{m-1} \sum \pi(a, b), \quad \phi^-(a) = \frac{1}{m-1} \sum \pi(b, a), \quad \phi(a) = \phi^+(a) - \phi^-(a)")
        rdf = pd.DataFrame({"Alternativa": alts, "φ+": res["phi_plus"], "φ−": res["phi_minus"], "φ líq": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"φ+": "{:.4f}", "φ−": "{:.4f}", "φ líq": "{:.4f}"}), use_container_width=True, hide_index=True)
        
        all_results["PROMETHEE"] = {"scores": res["scores"], "ranking": res["ranking"]}
        render_sensitivity(model_promethee, mat, weights, types, alts, criteria, sens_pct, res["ranking"], function=promethee_fn)


# =============================================================================
# TAB 6 — VIKOR
# =============================================================================
with tabs[6]:
    st.header("⚖️ VIKOR — Compromisso (Opricovic, 1998)")
    st.markdown(f"**Peso da estratégia $v$ = {vikor_v:.2f}** (1 = utilidade grupo máxima; 0 = arrependimento individual mínimo)")
    
    res, err = safe_call(model_vikor, mat, weights, types, vikor_v)
    if err: st.error(f"Erro VIKOR: {err}")
    else:
        st.subheader("Passo 1: Utilidade ($S_i$) e Arrependimento ($R_i$)")
        st.latex(r"S_i = \sum_{j=1}^n w_j \frac{f_j^* - f_{ij}}{f_j^* - f_j^-}, \quad R_i = \max_j \left[ w_j \frac{f_j^* - f_{ij}}{f_j^* - f_j^-} \right]")
        
        st.subheader("Passo 2: Índice VIKOR ($Q_i$) e Ranking")
        st.latex(r"Q_i = v \cdot \frac{S_i - S^*}{S^- - S^*} + (1-v) \cdot \frac{R_i - R^*}{R^- - R^*}")
        rdf = pd.DataFrame({"Alternativa": alts, "S": res["S"], "R": res["R"], "Q": res["Q"], "Ranking (Menor = Melhor)": res["ranking"]}).sort_values("Ranking (Menor = Melhor)")
        st.dataframe(rdf.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"}), use_container_width=True, hide_index=True)
        
        all_results["VIKOR"] = {"scores": res["scores"], "ranking": res["ranking"]}
        render_sensitivity(model_vikor, mat, weights, types, alts, criteria, sens_pct, res["ranking"], v=vikor_v)


# =============================================================================
# TAB 7 — MAUT
# =============================================================================
with tabs[7]:
    st.header("📐 MAUT — Multi-Attribute Utility Theory (Keeney & Raiffa)")
    res, err = safe_call(model_maut, mat, weights, types)
    if err: st.error(f"Erro MAUT: {err}")
    else:
        st.subheader("Passo 1: Utilidades Parciais Min-Max")
        st.latex(r"u_j(x_{ij}) = \frac{x_{ij} - \min}{\max - \min}")
        st.dataframe(pd.DataFrame(res["utility_matrix"], index=alts, columns=criteria).round(4), use_container_width=True)

        st.subheader("Passo 2: Utilidade Global ($U_i$) e Ranking")
        st.latex(r"U_i = \sum_{j=1}^n w_j \cdot u_j(x_{ij})")
        rdf = pd.DataFrame({"Alternativa": alts, "Utilidade U": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Utilidade U": "{:.4f}"}), use_container_width=True, hide_index=True)
        
        all_results["MAUT"] = {"scores": res["scores"], "ranking": res["ranking"]}
        render_sensitivity(model_maut, mat, weights, types, alts, criteria, sens_pct, res["ranking"])


# =============================================================================
# TAB 8 — COPRAS
# =============================================================================
with tabs[8]:
    st.header("🧮 COPRAS — Complex Proportional Assessment")
    res, err = safe_call(model_copras, mat, weights, types)
    if err: st.error(f"Erro COPRAS: {err}")
    else:
        st.subheader("Passo 1: Avaliação de Benefícios ($S^+$) e Custos ($S^-$)")
        st.latex(r"S_i^+ = \sum_{j \in J^+} w_j \cdot \bar{x}_{ij},\quad S_i^- = \sum_{j \in J^-} w_j \cdot \bar{x}_{ij}")
        
        st.subheader("Passo 2: Importância Relativa ($Q_i$) e Grau Utilidade ($N_i$)")
        st.latex(r"Q_i = S_i^+ + \frac{\min S_k^- \cdot \sum (1/S_k^-)}{S_i^- \cdot \sum (1/S_k^-)}, \quad N_i = \frac{Q_i}{\max Q_k} \times 100\%")
        
        rdf = pd.DataFrame({"Alternativa": alts, "S+": res["S_plus"], "S−": res["S_minus"], "Q": res["Q"], "N (%)": res["N"], "Ranking": res["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"S+": "{:.4f}", "S−": "{:.4f}", "Q": "{:.4f}", "N (%)": "{:.2f}"}), use_container_width=True, hide_index=True)
        
        all_results["COPRAS"] = {"scores": res["N"], "ranking": res["ranking"]}
        render_sensitivity(model_copras, mat, weights, types, alts, criteria, sens_pct, res["ranking"])


# =============================================================================
# TAB 9 — DEMATEL
# =============================================================================
with tabs[9]:
    st.header("🌐 DEMATEL (Gabus & Fontela, 1972)")
    res, err = safe_call(model_dematel, mat, weights, types)
    if err: st.error(f"Erro DEMATEL: {err}")
    else:
        st.subheader("Passo 1: Matriz de Relação Total ($T$)")
        st.latex(r"T = X \cdot (I - X)^{-1}")
        c1, c2 = st.columns(2)
        with c1: st.dataframe(pd.DataFrame(res["T"].round(4), index=criteria, columns=criteria), use_container_width=True)
        with c2:
            st.markdown("**Prominência ($D+R$):** Força Global. **Relação ($D-R$):** Causa se $>0$, Efeito se $<0$.")
            st.dataframe(pd.DataFrame({"Critério": criteria, "D+R": res["prominence"], "D-R": res["relation"]}).round(4), use_container_width=True, hide_index=True)

        st.subheader("Passo 2: Ranking Final (Pesos Modulados por Prominência)")
        rdf = pd.DataFrame({"Alternativa": alts, "Score": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Score": "{:.4f}"}), use_container_width=True, hide_index=True)
        
        all_results["DEMATEL"] = {"scores": res["scores"], "ranking": res["ranking"]}
        render_sensitivity(model_dematel, mat, weights, types, alts, criteria, sens_pct, res["ranking"])


# =============================================================================
# TAB 10 — FUZZY AHP
# =============================================================================
with tabs[10]:
    st.header("🌫️ Fuzzy AHP (Chang, 1996)")
    res, err = safe_call(model_fuzzy_ahp, weights)
    if err: st.error(f"Erro: {err}")
    else:
        st.subheader("Passo 1: Números Triangulares e Defuzzificação")
        st.latex(r"\tilde{a} = (l, m, u),\quad w^{\text{crisp}} = \frac{l + m + u}{3}")
        fdf = pd.DataFrame(res["fuzzy_weights"], index=criteria, columns=["l", "m", "u"])
        fdf["crisp"] = res["crisp_weights"]
        st.dataframe(fdf.round(4), use_container_width=True)

        st.subheader("Passo 2: Ranking Aditivo")
        score_res = model_saw(mat, res["crisp_weights"], types)
        rdf = pd.DataFrame({"Alternativa": alts, "Score": score_res["scores"], "Ranking": score_res["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Score": "{:.4f}"}), use_container_width=True, hide_index=True)
        
        all_results["Fuzzy AHP"] = {"scores": score_res["scores"], "ranking": score_res["ranking"]}
        render_sensitivity(model_saw, mat, res["crisp_weights"], types, alts, criteria, sens_pct, score_res["ranking"])


# =============================================================================
# TAB 11 — FUZZY TOPSIS
# =============================================================================
with tabs[11]:
    st.header("🌫️ Fuzzy TOPSIS (Chen, 2000)")
    spread = st.slider("Spread fuzzy (%)", 5, 30, 10, 5) / 100.0
    res, err = safe_call(model_fuzzy_topsis, mat, weights, types, spread)
    if err: st.error(f"Erro: {err}")
    else:
        st.subheader("Passo 1: Distâncias ao FPIS e FNIS (Método do Vértice)")
        st.latex(r"d(\tilde{a}, \tilde{b}) = \sqrt{\frac{1}{3}\left[(a_l - b_l)^2 + (a_m - b_m)^2 + (a_u - b_u)^2\right]}")
        
        st.subheader("Passo 2: Proximidade Coeficiente ($CC_i$)")
        st.latex(r"CC_i = \frac{d_i^-}{d_i^+ + d_i^-}")
        rdf = pd.DataFrame({"Alternativa": alts, "d+": res["d_plus"], "d−": res["d_minus"], "CC": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"d+": "{:.4f}", "d−": "{:.4f}", "CC": "{:.4f}"}), use_container_width=True, hide_index=True)
        
        all_results["Fuzzy TOPSIS"] = {"scores": res["scores"], "ranking": res["ranking"]}
        render_sensitivity(model_fuzzy_topsis, mat, weights, types, alts, criteria, sens_pct, res["ranking"], spread=spread)


# =============================================================================
# TAB 12 — FUZZY ANP
# =============================================================================
with tabs[12]:
    st.header("🌫️ Fuzzy ANP")
    res, err = safe_call(model_fuzzy_anp, mat, weights, types)
    if err: st.error(f"Erro: {err}")
    else:
        st.subheader("Passo 1: Pesos Fuzzy Cruzados com Supermatriz")
        st.dataframe(pd.DataFrame({"Critério": criteria, "F-AHP": res["crisp_fuzzy_weights"], "Fuzzy ANP Ajustado": res["adjusted_weights"]}).round(4), use_container_width=True, hide_index=True)

        st.subheader("Passo 2: Ranking F-ANP")
        rdf = pd.DataFrame({"Alternativa": alts, "Score": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking")
        st.dataframe(rdf.style.format({"Score": "{:.4f}"}), use_container_width=True, hide_index=True)
        
        all_results["Fuzzy ANP"] = {"scores": res["scores"], "ranking": res["ranking"]}
        render_sensitivity(model_fuzzy_anp, mat, weights, types, alts, criteria, sens_pct, res["ranking"])


# =============================================================================
# TAB 13 — DASHBOARD CONSOLIDADO
# =============================================================================
with tabs[13]:
    st.header("🏆 Dashboard Consolidado")
    if not all_results:
        st.warning("⚠️ Nenhum modelo executado. Visite as tabs anteriores.")
    else:
        models_with_results = list(all_results.keys())
        rank_table = pd.DataFrame({"Alternativa": alts})
        for m in models_with_results: rank_table[m] = all_results[m]["ranking"]
        rank_table["Posição Média"] = rank_table[models_with_results].mean(axis=1).round(2)
        rank_table["Ranking Final"] = ranking_from_scores(-rank_table["Posição Média"].values)
        rank_table = rank_table.sort_values("Ranking Final").reset_index(drop=True)

        st.subheader("Tabela de Rankings Consolidados (Método Borda)")
        styled = rank_table.style.format({"Posição Média": "{:.2f}"}).background_gradient(subset=models_with_results, cmap="RdYlGn_r").background_gradient(subset=["Posição Média", "Ranking Final"], cmap="RdYlGn_r")
        st.dataframe(styled, use_container_width=True, hide_index=True)

        top3_alts = rank_table.sort_values("Ranking Final").head(3)["Alternativa"].tolist()
        c1, c2, c3 = st.columns(3)
        c1.metric("🥇 1º lugar", top3_alts[0] if len(top3_alts) > 0 else "—")
        c2.metric("🥈 2º lugar", top3_alts[1] if len(top3_alts) > 1 else "—")
        c3.metric("🥉 3º lugar", top3_alts[2] if len(top3_alts) > 2 else "—")


# =============================================================================
# TAB 14 — RELATÓRIO
# =============================================================================
with tabs[14]:
    st.header("📄 Relatório de Análise Multicritério")
    if not all_results: st.warning("⚠️ Execução Pendente.")
    else:
        n_alt = len(alts)
        n_crit = len(criteria)
        models_with_results = list(all_results.keys())
        
        rank_table_rel = pd.DataFrame({"Alternativa": alts})
        for m in models_with_results: rank_table_rel[m] = all_results[m]["ranking"]
        rank_table_rel["Posição Média"] = rank_table_rel[models_with_results].mean(axis=1)
        rank_table_rel = rank_table_rel.sort_values("Posição Média").reset_index(drop=True)
        top1 = rank_table_rel["Alternativa"].iloc[0]

        top1_in_top3 = sum(1 for m in models_with_results if rank_table_rel.iloc[0][m] <= 3)
        conv_pct = (top1_in_top3 / len(models_with_results)) * 100

        report_md = (
            f"# Relatório de Análise MCDM\n**Gerado:** {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}\n\n"
            f"**Alternativa Recomendada:** `{top1}`\n\n"
            f"**Convergência:** {conv_pct:.0f}% (Top-3 em {top1_in_top3} de {len(models_with_results)} modelos).\n\n"
            f"**Modelos Avaliados:** {', '.join(models_with_results)}"
        )
        
        st.markdown("---")
        st.markdown(report_md)
        st.download_button("Descarregar Markdown (.md)", data=report_md.encode("utf-8"), file_name="relatorio.md", mime="text/markdown")
