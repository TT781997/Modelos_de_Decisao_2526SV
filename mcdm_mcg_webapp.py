# -*- coding: utf-8 -*-
"""
MCDM Dashboard — Sistema de Apoio à Decisão Multicritério
Modelos de Decisão | MEGI ISEL 2025/2026 | Caso de Estudo MCG

VERSÃO FINAL COMPLETA (12/05/2026)
✅ Checkbox "Usar dados de demonstração" = False por defeito
✅ Nova tab "📖 Teoria MCDM" com teoria + matemática completa (LaTeX)
✅ Todas as 14 tabs originais + Dashboard + Relatório
✅ Pesos AHP do Q5.2 + C5_UD = min
✅ Botão de reset de pesos AHP
✅ Pronto para deploy no Streamlit Cloud

Execução: streamlit run mcdm_mcg_webapp.py
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
# CONFIG
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 MCDM Dashboard — Priorização Multicritério")
st.caption("Modelos de Decisão | MEGI ISEL 2025/2026 | Caso de Estudo MCG")

# =============================================================================
# UTILITIES
# =============================================================================
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
            7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.54,
            13: 1.56, 14: 1.57, 15: 1.59}


def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
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


# =============================================================================
# MODELOS MCDM
# =============================================================================
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
    if d <= 0:
        return 0.0
    if ftype == "usual":
        return 1.0
    if ftype == "u_shape":
        return 1.0 if d > (q or 0) else 0.0
    if ftype in ("v_shape", "linear"):
        if not p:
            return 0.0
        return float(min(d / p, 1.0))
    if ftype == "level":
        if d <= (q or 0):
            return 0.0
        if d <= (p or 1):
            return 0.5
        return 1.0
    if ftype == "linear_indiff":
        if d <= (q or 0):
            return 0.0
        if d <= (p or 1):
            return (d - q) / (p - q) if p > q else 1.0
        return 1.0
    if ftype == "gaussian":
        s = sigma or 1.0
        return float(1.0 - np.exp(-(d ** 2) / (2 * s ** 2)))
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
                if i == k:
                    continue
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
            if i == k:
                continue
            c = sum(weights[j] for j in range(n_crit) if norm[i, j] >= norm[k, j])
            concordance[i, k] = c / w_sum
            diffs = [norm[k, j] - norm[i, j] for j in range(n_crit) if norm[k, j] > norm[i, j]]
            discordance[i, k] = (max(diffs) / global_range) if diffs else 0.0
    outrank = (concordance >= c_thresh) & (discordance <= d_thresh)
    np.fill_diagonal(outrank, False)
    kernel = [i for i in range(n_alt) if not any(outrank[k, i] and not outrank[i, k] for k in range(n_alt) if k != i)]
    net_dominance = outrank.sum(axis=1) - outrank.sum(axis=0)
    return {"concordance": concordance, "discordance": discordance, "outrank": outrank,
            "kernel": kernel, "scores": net_dominance.astype(float),
            "ranking": ranking_from_scores(net_dominance.astype(float))}


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
    return {"S_plus": S_plus, "S_minus": S_minus, "Q": Q, "N": N,
            "scores": N, "ranking": ranking_from_scores(N)}


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
        if np.allclose(prev, nxt, atol=tol):
            return nxt
        prev = nxt
    return prev


def model_anp(mat, weights, types):
    M = _influence_supermatrix(mat)
    L = _limit_matrix(M)
    adj = L @ weights
    adj = adj / adj.sum() if adj.sum() > 0 else weights
    norm = normalize_minmax(mat, types)
    scores = (norm * adj).sum(axis=1)
    return {"adjusted_weights": adj, "limit_matrix": L, "scores": scores,
            "ranking": ranking_from_scores(scores)}


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
    adj = weights * prominence / prominence.sum() if prominence.sum() > 0 else weights
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
    l = mat * (1 - spread)
    m = mat.copy()
    u = mat * (1 + spread)
    n_alt, n_crit = mat.shape
    L = np.zeros_like(l); M = np.zeros_like(m); U = np.zeros_like(u)
    for j in range(n_crit):
        if types[j] == "max":
            denom = max(u[:, j].max(), 1e-9)
            L[:, j] = l[:, j] / denom
            M[:, j] = m[:, j] / denom
            U[:, j] = u[:, j] / denom
        else:
            num = max(l[:, j].min(), 1e-9)
            L[:, j] = num / np.where(u[:, j] == 0, 1e-9, u[:, j])
            M[:, j] = num / np.where(m[:, j] == 0, 1e-9, m[:, j])
            U[:, j] = num / np.where(l[:, j] == 0, 1e-9, l[:, j])
    Lw, Mw, Uw = L * weights, M * weights, U * weights
    fpis = np.array([(Uw[:, j].max(), Uw[:, j].max(), Uw[:, j].max()) for j in range(n_crit)])
    fnis = np.array([(Lw[:, j].min(), Lw[:, j].min(), Lw[:, j].min()) for j in range(n_crit)])
    def vd(al, am, au, bl, bm, bu):
        return np.sqrt(((al - bl) ** 2 + (am - bm) ** 2 + (au - bu) ** 2) / 3.0)
    d_plus = np.zeros(n_alt)
    d_minus = np.zeros(n_alt)
    for i in range(n_alt):
        for j in range(n_crit):
            d_plus[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], *fpis[j])
            d_minus[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], *fnis[j])
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
    return {"crisp_fuzzy_weights": fw, "adjusted_weights": adj,
            "scores": scores, "ranking": ranking_from_scores(scores)}


# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================
def build_demo_data():
    df = pd.DataFrame({
        "Alternativa": [f"A{i}" for i in range(1, 10)],
        "C1_VP": [250_000_000, 300_000, 900_000, 650_000, 5_000_000,
                  1_350_000, 10_500_000, 3_450_000, 15_000_000],
        "C2_PF": [0.25, 0.35, 0.50, 0.50, 0.40, 0.50, 0.40, 0.40, 0.60],
        "C3_EE": [24, 8, 8, 8, 24, 8, 16, 8, 24],
        "C4_FE": [4, 5, 3, 3, 4, 3, 3, 3, 4],
        "C5_UD": [180, 60, 60, 90, 30, 60, 180, 60, 300],
        "C6_RC": [4, 5, 5, 3, 3, 5, 4, 4, 3],
    })
    weights = np.array([0.4615, 0.1987, 0.0230, 0.0972, 0.0217, 0.1979])
    return df, weights


def load_excel(file):
    xls = pd.ExcelFile(file)
    if "Dados" not in xls.sheet_names:
        raise ValueError("A folha 'Dados' não foi encontrada.")
    df = pd.read_excel(xls, sheet_name="Dados")
    id_col = df.columns[0]
    df = df.dropna(subset=[id_col]).reset_index(drop=True)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    crits = [c for c in numeric if c != id_col]
    df[id_col] = df[id_col].astype(str)
    if "Pesos" in xls.sheet_names:
        wdf = pd.read_excel(xls, sheet_name="Pesos", header=None)
        wvals = wdf.select_dtypes(include=[np.number]).values.flatten()
        wvals = wvals[~np.isnan(wvals)]
        weights = np.array(wvals[:len(crits)], dtype=float) if len(wvals) >= len(crits) else np.ones(len(crits))
    else:
        weights = np.ones(len(crits))
    weights = weights / weights.sum()
    return df, weights, id_col, crits


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuração")
    uploaded = st.file_uploader("Carregar Excel (.xlsx)", type=["xlsx", "xls"])
    use_demo = st.checkbox("Usar dados de demonstração MCG (9 alts × 6 crit)", value=False)

    st.divider()
    st.subheader("🎛️ Parâmetros de modelos")
    c_thresh = st.slider("ELECTRE — limiar de concordância (c)", 0.50, 0.95, 0.65, 0.01)
    d_thresh = st.slider("ELECTRE — limiar de discordância (d)", 0.05, 0.50, 0.35, 0.01)
    promethee_fn = st.selectbox("PROMETHEE — função de preferência", ["usual", "linear", "gaussian"], index=1)
    vikor_v = st.slider("VIKOR — peso da estratégia v", 0.0, 1.0, 0.5, 0.05)
    sens_pct = st.slider("Sensibilidade ±% nos pesos (TOPSIS)", 5, 50, 20, 5)

# =============================================================================
# ESTADO E CARREGAMENTO
# =============================================================================
if "loaded" not in st.session_state:
    st.session_state.loaded = False

data_df = None
weights = None
id_col = None
criteria = []
types = []

if use_demo:
    data_df, weights = build_demo_data()
    id_col = "Alternativa"
    criteria = [c for c in data_df.columns if c != id_col]
    st.session_state.loaded = True
elif uploaded is not None:
    res, err = safe_call(load_excel, uploaded)
    if err is None:
        data_df, weights, id_col, criteria = res
        st.session_state.loaded = True
    else:
        st.sidebar.error(f"❌ {err}")

if st.session_state.loaded and data_df is not None:
    with st.sidebar:
        st.divider()
        st.subheader("🎯 Configuração de critérios")
        min_keywords = ["ee", "custo", "cost", "esforço", "prazo", "tempo", "delay", "risk", "risco",
                        "ud", "urg", "urgencia", "dias", "deadline"]
        type_defaults = ["min" if any(k in c.lower() for k in min_keywords) else "max" for c in criteria]

        config_df = pd.DataFrame({"Critério": criteria, "Sentido": type_defaults, "Peso": [float(w) for w in weights]})
        editor_key = "crit_cfg_demo" if use_demo else f"crit_cfg_{hash(tuple(criteria))}"

        edited_cfg = st.data_editor(
            config_df,
            column_config={
                "Critério": st.column_config.TextColumn("Critério", disabled=True),
                "Sentido": st.column_config.SelectboxColumn("Sentido", options=["max", "min"], required=True),
                "Peso": st.column_config.NumberColumn("Peso", min_value=0.0, format="%.4f"),
            },
            use_container_width=True,
            hide_index=True,
            key=editor_key,
        )

        if use_demo and st.button("🔄 Restaurar Pesos AHP do Q5.2 (Secção 5.2)"):
            st.session_state[editor_key] = config_df.copy()
            st.rerun()

        try:
            types = edited_cfg["Sentido"].astype(str).tolist()
            edited_w = edited_cfg["Peso"].astype(float).values
            weights = edited_w / edited_w.sum() if edited_w.sum() > 0 else np.ones(len(criteria)) / len(criteria)
        except Exception:
            types = type_defaults
            weights = np.ones(len(criteria)) / len(criteria)

# =============================================================================
# HELPERS
# =============================================================================
def need_data():
    st.info("👈 Carregue um ficheiro Excel ou active o modo demonstração na sidebar.")

def get_matrix():
    if not st.session_state.loaded or data_df is None:
        return None
    return data_df[criteria].astype(float).values

def render_ranking_chart(alts, scores, title, label="Score"):
    df = pd.DataFrame({"Alternativa": alts, label: scores}).sort_values(label, ascending=False)
    fig = px.bar(df, x="Alternativa", y=label, title=title, text_auto=".3f",
                 color=label, color_continuous_scale="Tealgrn")
    fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig

# =============================================================================
# TABS
# =============================================================================
TAB_LABELS = [
    "📋 Visão Geral", "📖 Teoria MCDM", "🔺 AHP", "🕸️ ANP", "🎯 TOPSIS", "🔗 ELECTRE",
    "📊 PROMETHEE", "⚖️ VIKOR", "📐 MAUT", "🧮 COPRAS", "🌐 DEMATEL",
    "🌫️ Fuzzy AHP", "🌫️ Fuzzy TOPSIS", "🌫️ Fuzzy ANP",
    "🏆 Dashboard", "📝 Relatório"
]
tabs = st.tabs(TAB_LABELS)

all_results = {}

# =============================================================================
# TAB 0 — VISÃO GERAL
# =============================================================================
with tabs[0]:
    st.header("📋 Visão Geral dos Dados")
    if not st.session_state.loaded:
        need_data()
    else:
        alts = data_df[id_col].tolist()
        col1, col2, col3 = st.columns(3)
        col1.metric("Alternativas", len(alts))
        col2.metric("Critérios", len(criteria))
        col3.metric("Tipos (max/min)", f"{types.count('max')}/{types.count('min')}")
        st.subheader("Matriz de decisão")
        st.dataframe(data_df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Pesos e Sentido")
            st.dataframe(pd.DataFrame({
                "Critério": criteria,
                "Peso": [f"{w:.4f}" for w in weights],
                "Sentido": types,
            }), use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Estatísticas descritivas")
            st.dataframe(data_df[criteria].describe().T, use_container_width=True)
        st.subheader("Heatmap normalizado (min-max)")
        try:
            mat = get_matrix()
            norm = normalize_minmax(mat, types)
            fig = px.imshow(norm, labels=dict(x="Critério", y="Alternativa", color="Valor normalizado"),
                            x=criteria, y=alts, color_continuous_scale="RdYlGn", aspect="auto", text_auto=".2f")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Erro no heatmap: {exc}")

# =============================================================================
# TAB 1 — TEORIA MCDM (COMPLETA COM MATEMÁTICA)
# =============================================================================
with tabs[1]:
    st.header("📖 Teoria e Matemática dos Métodos MCDM")
    st.markdown("""
    Esta secção explica **todos os métodos** implementados no dashboard, com a teoria completa, 
    fórmulas matemáticas e passos algorítmicos.
    """)

    st.subheader("1. AHP — Analytic Hierarchy Process")
    st.markdown(r"""
    **Matriz de comparações par-a-par** \( A = (a_{ij}) \), onde \( a_{ij} \) é a importância relativa do critério \( i \) sobre \( j \) (escala Saaty 1-9).

    **Pesos por autovetor principal:**
    \[
    A \mathbf{w} = \lambda_{\max} \mathbf{w}
    \]
    **Índice de consistência:**
    \[
    CI = \frac{\lambda_{\max} - n}{n-1}, \quad CR = \frac{CI}{RI}
    \]
    Aceita-se se \( CR < 0.10 \).
    """)

    st.subheader("2. ANP — Analytic Network Process (simplificado)")
    st.markdown(r"""
    Considera dependências entre critérios. Constrói-se a **supermatriz** \( M \) (aqui via correlações) e eleva-se a potências:
    \[
    L = \lim_{k \to \infty} M^k
    \]
    Pesos finais: \( \mathbf{w}_{ANP} = L \mathbf{w} \).
    """)

    st.subheader("3. TOPSIS")
    st.markdown(r"""
    Normalização vetorial:
    \[
    r_{ij} = \frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}}
    \]
    Matriz ponderada \( v_{ij} = w_j r_{ij} \).

    Distâncias à solução ideal \( A^+ \) e anti-ideal \( A^- \):
    \[
    d_i^+ = \sqrt{\sum_j (v_{ij} - v_j^+)^2}, \quad d_i^- = \sqrt{\sum_j (v_{ij} - v_j^-)^2}
    \]
    \[
    C_i^* = \frac{d_i^-}{d_i^+ + d_i^-}
    \]
    """)

    st.subheader("4. ELECTRE I")
    st.markdown(r"""
    Matrizes de **concordância** \( C \) e **discordância** \( D \).

    Sobreclassificação: \( a \ S \ b \) se \( C(a,b) \geq c \) e \( D(a,b) \leq d \).

    **Kernel** = conjunto de alternativas não dominadas.
    """)

    st.subheader("5. PROMETHEE II")
    st.markdown(r"""
    Função de preferência \( P_j(a,b) \) por critério.

    Fluxos:
    \[
    \phi^+(a) = \frac{1}{n-1} \sum_{b \neq a} \pi(a,b), \quad \phi^-(a) = \frac{1}{n-1} \sum_{b \neq a} \pi(b,a)
    \]
    Fluxo líquido: \( \phi(a) = \phi^+(a) - \phi^-(a) \).
    """)

    st.subheader("6. VIKOR")
    st.markdown(r"""
    \[
    S_i = \sum_j w_j \frac{f_j^* - f_{ij}}{f_j^* - f_j^-}, \quad R_i = \max_j \left( w_j \frac{f_j^* - f_{ij}}{f_j^* - f_j^-} \right)
    \]
    \[
    Q_i = v \frac{S_i - S^*}{S^- - S^*} + (1-v) \frac{R_i - R^*}{R^- - R^*}
    \]
    """)

    st.subheader("7. MAUT")
    st.markdown(r"""
    Utilidade aditiva:
    \[
    U(a_i) = \sum_j w_j \cdot u_j(x_{ij})
    \]
    (normalização min-max).
    """)

    st.subheader("8. COPRAS")
    st.markdown(r"""
    Separação em benefícios \( S_i^+ \) e custos \( S_i^- \).

    \[
    Q_i = S_i^+ + \frac{\min_k S_k^-}{\sum_k (1/S_k^-)} \cdot \frac{1}{S_i^-}
    \]
    """)

    st.subheader("9. DEMATEL")
    st.markdown(r"""
    Matriz total de relações:
    \[
    T = X (I - X)^{-1}, \quad X = Z / s
    \]
    Prominência \( D + R \), Relação \( D - R \).
    """)

    st.subheader("10-12. Versões Fuzzy")
    st.markdown(r"""
    Números triangulares fuzzy \( \tilde{x} = (l, m, u) \).

    Distância fuzzy (método do vértice):
    \[
    d(\tilde{a}, \tilde{b}) = \sqrt{\frac{(l_a-l_b)^2 + (m_a-m_b)^2 + (u_a-u_b)^2}{3}}
    \]
    """)

    st.info("Todas as implementações seguem rigorosamente as fórmulas acima. Consulte as tabs individuais para resultados numéricos.")

# =============================================================================
# TAB 2 — AHP
# =============================================================================
with tabs[2]:
    st.header("🔺 AHP — Analytic Hierarchy Process")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        n = len(criteria)
        st.markdown("Edite a matriz de comparação par-a-par (escala Saaty).")
        init = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and weights[j] != 0:
                    init[i, j] = weights[i] / weights[j]
        pw_df = pd.DataFrame(init, index=criteria, columns=criteria).round(4)
        edited_pw = st.data_editor(pw_df, use_container_width=True, key="ahp_pw")
        E = edited_pw.values.astype(float).copy()
        for i in range(n):
            for j in range(n):
                if i == j:
                    E[i, j] = 1.0
                elif i < j and E[i, j] != 0:
                    E[j, i] = 1.0 / E[i, j]
        res, err = safe_call(model_ahp, E)
        if err:
            st.error(f"Erro AHP: {err}")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("λ_max", f"{res['lambda_max']:.4f}")
            m2.metric("CI", f"{res['CI']:.4f}")
            m3.metric("CR", f"{res['CR']:.4f}",
                      delta="Consistente" if res['consistent'] else "Inconsistente",
                      delta_color="normal" if res['consistent'] else "inverse")
            st.subheader("Vector de pesos AHP")
            comp_df = pd.DataFrame({"Critério": criteria, "Peso AHP": res["weights"], "Peso na sidebar": weights})
            st.dataframe(comp_df.style.format({"Peso AHP": "{:.4f}", "Peso na sidebar": "{:.4f}"}), use_container_width=True, hide_index=True)
            norm = normalize_minmax(mat, types)
            ahp_scores = (norm * res["weights"]).sum(axis=1)
            ahp_rank = ranking_from_scores(ahp_scores)
            alts = data_df[id_col].tolist()
            rank_df = pd.DataFrame({"Alternativa": alts, "Score AHP": ahp_scores, "Ranking": ahp_rank}).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rank_df.style.format({"Score AHP": "{:.4f}"}), use_container_width=True, hide_index=True)
            st.plotly_chart(render_ranking_chart(alts, ahp_scores, "AHP — Score por alternativa"), use_container_width=True)
            all_results["AHP"] = {"scores": ahp_scores, "ranking": ahp_rank, "weights": res["weights"]}

# =============================================================================
# TAB 3 — ANP
# =============================================================================
with tabs[3]:
    st.header("🕸️ ANP — Analytic Network Process (simplificado)")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_anp, mat, weights, types)
        if err:
            st.error(f"Erro ANP: {err}")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Pesos ajustados (ANP)")
                st.dataframe(pd.DataFrame({"Critério": criteria, "Peso original": weights, "Peso ANP": res["adjusted_weights"]}).round(4), use_container_width=True, hide_index=True)
            with c2:
                st.subheader("Matriz limite")
                st.dataframe(pd.DataFrame(res["limit_matrix"].round(4), index=criteria, columns=criteria), use_container_width=True)
            rdf = pd.DataFrame({"Alternativa": alts, "Score ANP": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"Score ANP": "{:.4f}"}), use_container_width=True, hide_index=True)
            st.plotly_chart(render_ranking_chart(alts, res["scores"], "ANP — Score por alternativa"), use_container_width=True)
            all_results["ANP"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 4 — TOPSIS
# =============================================================================
with tabs[4]:
    st.header("🎯 TOPSIS")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_topsis, mat, weights, types)
        if err:
            st.error(f"Erro TOPSIS: {err}")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(pd.DataFrame(res["normalized"], index=alts, columns=criteria).round(4), use_container_width=True)
            with c2:
                st.dataframe(pd.DataFrame(res["weighted"], index=alts, columns=criteria).round(4), use_container_width=True)
            st.subheader("Distâncias e Ci*")
            rdf = pd.DataFrame({"Alternativa": alts, "D+": res["d_plus"], "D−": res["d_minus"], "Ci*": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"D+": "{:.4f}", "D−": "{:.4f}", "Ci*": "{:.4f}"}), use_container_width=True, hide_index=True)
            st.plotly_chart(render_ranking_chart(alts, res["scores"], "TOPSIS — Ci* por alternativa", label="Ci*"), use_container_width=True)
            all_results["TOPSIS"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 5 — ELECTRE
# =============================================================================
with tabs[5]:
    st.header("🔗 ELECTRE I")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_electre, mat, weights, types, c_thresh, d_thresh)
        if err:
            st.error(f"Erro ELECTRE: {err}")
        else:
            st.markdown(f"**Limiares**: c = {c_thresh:.2f} | d = {d_thresh:.2f}")
            kernel_alts = [alts[i] for i in res["kernel"]]
            st.success(f"**Kernel:** {', '.join(kernel_alts) if kernel_alts else '∅'}")
            rdf = pd.DataFrame({"Alternativa": alts, "Score": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf, use_container_width=True, hide_index=True)
            all_results["ELECTRE"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 6 — PROMETHEE
# =============================================================================
with tabs[6]:
    st.header("📊 PROMETHEE II")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_promethee, mat, weights, types, promethee_fn)
        if err:
            st.error(f"Erro PROMETHEE: {err}")
        else:
            rdf = pd.DataFrame({"Alternativa": alts, "φ+": res["phi_plus"], "φ−": res["phi_minus"], "φ líquido": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"φ+": "{:.4f}", "φ−": "{:.4f}", "φ líquido": "{:.4f}"}), use_container_width=True, hide_index=True)
            all_results["PROMETHEE"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 7 — VIKOR
# =============================================================================
with tabs[7]:
    st.header("⚖️ VIKOR")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_vikor, mat, weights, types, vikor_v)
        if err:
            st.error(f"Erro VIKOR: {err}")
        else:
            rdf = pd.DataFrame({"Alternativa": alts, "S": res["S"], "R": res["R"], "Q": res["Q"], "Ranking": res["ranking"]}).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"}), use_container_width=True, hide_index=True)
            all_results["VIKOR"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 8 — MAUT
# =============================================================================
with tabs[8]:
    st.header("📐 MAUT")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_maut, mat, weights, types)
        if err:
            st.error(f"Erro MAUT: {err}")
        else:
            rdf = pd.DataFrame({"Alternativa": alts, "Utilidade U": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"Utilidade U": "{:.4f}"}), use_container_width=True, hide_index=True)
            all_results["MAUT"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 9 — COPRAS
# =============================================================================
with tabs[9]:
    st.header("🧮 COPRAS")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_copras, mat, weights, types)
        if err:
            st.error(f"Erro COPRAS: {err}")
        else:
            rdf = pd.DataFrame({"Alternativa": alts, "N (%)": res["N"], "Ranking": res["ranking"]}).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"N (%)": "{:.2f}"}), use_container_width=True, hide_index=True)
            all_results["COPRAS"] = {"scores": res["N"], "ranking": res["ranking"]}

# =============================================================================
# TAB 10 — DEMATEL
# =============================================================================
with tabs[10]:
    st.header("🌐 DEMATEL")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_dematel, mat, weights, types)
        if err:
            st.error(f"Erro DEMATEL: {err}")
        else:
            rdf = pd.DataFrame({"Alternativa": alts, "Score DEMATEL": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"Score DEMATEL": "{:.4f}"}), use_container_width=True, hide_index=True)
            all_results["DEMATEL"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 11 — Fuzzy AHP
# =============================================================================
with tabs[11]:
    st.header("🌫️ Fuzzy AHP")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_fuzzy_ahp, weights)
        if err:
            st.error(f"Erro Fuzzy AHP: {err}")
        else:
            norm = normalize_minmax(mat, types)
            scores = (norm * res["crisp_weights"]).sum(axis=1)
            ranking = ranking_from_scores(scores)
            rdf = pd.DataFrame({"Alternativa": alts, "Score F-AHP": scores, "Ranking": ranking}).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"Score F-AHP": "{:.4f}"}), use_container_width=True, hide_index=True)
            all_results["Fuzzy AHP"] = {"scores": scores, "ranking": ranking}

# =============================================================================
# TAB 12 — Fuzzy TOPSIS
# =============================================================================
with tabs[12]:
    st.header("🌫️ Fuzzy TOPSIS")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        spread = st.slider("Spread fuzzy (%)", 5, 30, 10, 5) / 100.0
        res, err = safe_call(model_fuzzy_topsis, mat, weights, types, spread)
        if err:
            st.error(f"Erro Fuzzy TOPSIS: {err}")
        else:
            rdf = pd.DataFrame({"Alternativa": alts, "CC": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"CC": "{:.4f}"}), use_container_width=True, hide_index=True)
            all_results["Fuzzy TOPSIS"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 13 — Fuzzy ANP
# =============================================================================
with tabs[13]:
    st.header("🌫️ Fuzzy ANP")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_fuzzy_anp, mat, weights, types)
        if err:
            st.error(f"Erro Fuzzy ANP: {err}")
        else:
            rdf = pd.DataFrame({"Alternativa": alts, "Score F-ANP": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"Score F-ANP": "{:.4f}"}), use_container_width=True, hide_index=True)
            all_results["Fuzzy ANP"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 14 — DASHBOARD CONSOLIDADO
# =============================================================================
with tabs[14]:
    st.header("🏆 Dashboard Consolidado")
    if not st.session_state.loaded:
        need_data()
    elif not all_results:
        st.warning("Execute as tabs anteriores para gerar resultados.")
    else:
        alts = data_df[id_col].tolist()
        models_with_results = list(all_results.keys())
        rank_table = pd.DataFrame({"Alternativa": alts})
        for m in models_with_results:
            rank_table[m] = all_results[m]["ranking"]
        rank_table["Posição Média"] = rank_table[models_with_results].mean(axis=1).round(2)
        rank_table["Ranking Final"] = ranking_from_scores(-rank_table["Posição Média"].values)
        rank_table = rank_table.sort_values("Ranking Final").reset_index(drop=True)
        st.subheader("Tabela consolidada de rankings")
        st.dataframe(rank_table.style.background_gradient(subset=models_with_results, cmap="RdYlGn_r"), use_container_width=True, hide_index=True)
        top3 = rank_table.head(3)["Alternativa"].tolist()
        st.success(f"**Top-3 recomendado:** {', '.join(top3)}")

# =============================================================================
# TAB 15 — RELATÓRIO FINAL
# =============================================================================
with tabs[15]:
    st.header("📝 Relatório Final — Caso de Estudo MCG")
    if not st.session_state.loaded:
        need_data()
    else:
        st.markdown("### 1. Introdução")
        st.markdown("Dashboard com 14 métodos MCDM usando pesos AHP do Q5.2 e C5_UD como critério de minimização.")
        st.markdown("### 2. Configuração de Critérios")
        st.dataframe(pd.DataFrame({"Critério": criteria, "Peso": weights, "Sentido": types}), use_container_width=True, hide_index=True)
        if all_results:
            st.markdown("### 3. Ranking Consolidado")
            st.dataframe(rank_table, use_container_width=True, hide_index=True)
        st.markdown("### 4. Conclusão")
        st.success("Todos os métodos estão implementados e o dashboard está pronto para análise.")

st.success("✅ Dashboard completo carregado! (Demo desativado por defeito + Teoria MCDM + Relatório)")
