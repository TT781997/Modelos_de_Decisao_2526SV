# -*- coding: utf-8 -*-
"""
MCDM Dashboard — Sistema de Apoio à Decisão Multicritério
Modelos de Decisão | MEGI ISEL 2025/2026 | Caso de Estudo MCG

Aplicação Streamlit single-file com 14 abas:
Visão Geral, AHP, ANP, TOPSIS, ELECTRE, PROMETHEE, VIKOR, MAUT,
COPRAS, DEMATEL, Fuzzy AHP, Fuzzy TOPSIS, Fuzzy ANP, Dashboard Consolidado.

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
    div[data-testid="stExpander"] details { border-radius: 8px; }
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
    """Executa uma função e devolve (resultado, erro_str). Nunca rebenta o Streamlit."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def normalize_vector(mat):
    """Normalização vectorial (Euclidiana) por critério — usada no TOPSIS."""
    mat = np.asarray(mat, dtype=float)
    denom = np.sqrt(np.sum(mat ** 2, axis=0))
    denom = np.where(denom == 0, 1.0, denom)
    return mat / denom


def normalize_minmax(mat, types):
    """Normalização min-max com inversão para critérios de minimização."""
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
    """Normalização por soma com inversão para minimização (1/x)."""
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
    """Devolve posições no ranking (1 = melhor)."""
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
    ideal = np.array([
        weighted[:, j].max() if types[j] == "max" else weighted[:, j].min()
        for j in range(mat.shape[1])
    ])
    anti = np.array([
        weighted[:, j].min() if types[j] == "max" else weighted[:, j].max()
        for j in range(mat.shape[1])
    ])
    d_plus = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted - anti) ** 2, axis=1))
    denom = np.where((d_plus + d_minus) == 0, 1e-9, d_plus + d_minus)
    ci = d_minus / denom
    return {
        "normalized": norm,
        "weighted": weighted,
        "ideal": ideal,
        "anti_ideal": anti,
        "d_plus": d_plus,
        "d_minus": d_minus,
        "scores": ci,
        "ranking": ranking_from_scores(ci),
    }


def preference(d, ftype="usual", p=None, q=None, sigma=None):
    """Funções de preferência PROMETHEE."""
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
    return {
        "preference_matrix": pref,
        "phi_plus": phi_plus,
        "phi_minus": phi_minus,
        "scores": phi_net,
        "ranking": ranking_from_scores(phi_net),
    }


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

    kernel = []
    for i in range(n_alt):
        dominated = any(
            outrank[k, i] and not outrank[i, k]
            for k in range(n_alt) if k != i
        )
        if not dominated:
            kernel.append(i)

    net_dominance = outrank.sum(axis=1) - outrank.sum(axis=0)
    return {
        "concordance": concordance,
        "discordance": discordance,
        "outrank": outrank,
        "kernel": kernel,
        "scores": net_dominance.astype(float),
        "ranking": ranking_from_scores(net_dominance.astype(float)),
    }


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
    """MAUT — utilidade linear aditiva (forma simples e robusta)."""
    norm = normalize_minmax(mat, types)
    U = (norm * weights).sum(axis=1)
    return {"utility_matrix": norm, "scores": U, "ranking": ranking_from_scores(U)}


def _influence_supermatrix(mat):
    """Constrói matriz de influência inter-critério a partir de correlações."""
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
    """Eleva a supermatriz a potências sucessivas até convergir."""
    prev = M.copy()
    for _ in range(iters):
        nxt = prev @ M
        if np.allclose(prev, nxt, atol=tol):
            return nxt
        prev = nxt
    return prev


def model_anp(mat, weights, types):
    """ANP simplificado — usa supermatriz de influência inter-critério (proxy via correlação)."""
    M = _influence_supermatrix(mat)
    L = _limit_matrix(M)
    adj = L @ weights
    adj = adj / adj.sum() if adj.sum() > 0 else weights
    norm = normalize_minmax(mat, types)
    scores = (norm * adj).sum(axis=1)
    return {"adjusted_weights": adj, "limit_matrix": L, "scores": scores,
            "ranking": ranking_from_scores(scores)}


def model_dematel(mat, weights, types):
    """DEMATEL — modela influências entre critérios; usa prominência como modulador dos pesos."""
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
            "adjusted_weights": adj, "scores": scores,
            "ranking": ranking_from_scores(scores)}


def model_fuzzy_ahp(weights):
    """Fuzzy AHP — gera números triangulares a partir dos pesos crisp (spread ±20%) e defuzzifica."""
    fuzzy = np.array([(w * 0.8, w, w * 1.2) for w in weights])
    crisp = fuzzy.mean(axis=1)
    crisp = crisp / crisp.sum() if crisp.sum() > 0 else weights
    return {"fuzzy_weights": fuzzy, "crisp_weights": crisp}


def model_fuzzy_topsis(mat, weights, types, spread=0.10):
    """Fuzzy TOPSIS — números triangulares (val·(1−s), val, val·(1+s)) com método do vértice."""
    l = mat * (1 - spread)
    m = mat.copy()
    u = mat * (1 + spread)
    n_alt, n_crit = mat.shape

    # Normalização fuzzy
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

    # Solução fuzzy ideal positiva (FPIS) e negativa (FNIS)
    fpis = np.array([(Uw[:, j].max(), Uw[:, j].max(), Uw[:, j].max()) for j in range(n_crit)])
    fnis = np.array([(Lw[:, j].min(), Lw[:, j].min(), Lw[:, j].min()) for j in range(n_crit)])

    def vd(al, am, au, bl, bm, bu):
        return np.sqrt(((al - bl) ** 2 + (am - bm) ** 2 + (au - bu) ** 2) / 3.0)

    d_plus = np.zeros(n_alt)
    d_minus = np.zeros(n_alt)
    for i in range(n_alt):
        for j in range(n_crit):
            d_plus[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], fpis[j, 0], fpis[j, 1], fpis[j, 2])
            d_minus[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], fnis[j, 0], fnis[j, 1], fnis[j, 2])
    denom = np.where((d_plus + d_minus) == 0, 1e-9, d_plus + d_minus)
    cc = d_minus / denom
    return {"d_plus": d_plus, "d_minus": d_minus, "scores": cc,
            "ranking": ranking_from_scores(cc)}


def model_fuzzy_anp(mat, weights, types):
    """Fuzzy ANP — combina Fuzzy AHP (pesos com spread) com ajuste por supermatriz."""
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
# SIDEBAR — Upload e parâmetros
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuração")

    uploaded = st.file_uploader(
        "Carregar Excel (.xlsx)",
        type=["xlsx", "xls"],
        help="Folha 'Dados' obrigatória. Folha 'Pesos' opcional."
    )

    with st.expander("📌 Formato esperado", expanded=False):
        st.markdown(
            "**Folha `Dados`** — 1ª coluna = identificador da alternativa, restantes colunas = critérios numéricos.\n\n"
            "**Folha `Pesos`** — vector de pesos na mesma ordem dos critérios (linha ou coluna). "
            "Se ausente, é aplicada ponderação uniforme."
        )

    # Importante: key + value=False garantem que ao abrir o URL a checkbox arranca SEMPRE desligada
    use_demo = st.checkbox(
        "Usar dados de demonstração MCG (9 alts × 6 crit)",
        value=False,
        key="use_demo_data",
        help="Activa apenas para testar a app sem carregar Excel próprio."
    )

    st.divider()
    st.subheader("🎛️ Parâmetros de modelos")
    c_thresh = st.slider("ELECTRE — limiar de concordância (c)", 0.50, 0.95, 0.65, 0.01)
    d_thresh = st.slider("ELECTRE — limiar de discordância (d)", 0.05, 0.50, 0.35, 0.01)
    promethee_fn = st.selectbox(
        "PROMETHEE — função de preferência",
        ["usual", "linear", "gaussian"],
        index=1,
    )
    vikor_v = st.slider("VIKOR — peso da estratégia v", 0.0, 1.0, 0.5, 0.05)
    sens_pct = st.slider("Sensibilidade ±% nos pesos (TOPSIS)", 5, 50, 20, 5)


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
        # Lê tudo como texto/raw e converte forçadamente para numérico
        wdf = pd.read_excel(xls, sheet_name="Pesos", header=None)
        # Converte cada coluna a numérica (não-numérico → NaN), achata e descarta NaN
        wvals_all = []
        for col in wdf.columns:
            wvals_all.extend(pd.to_numeric(wdf[col], errors="coerce").dropna().tolist())
        wvals = np.array(wvals_all, dtype=float)
        if len(wvals) >= len(crits):
            weights = wvals[:len(crits)]
        else:
            weights = np.ones(len(crits))
            has_weights_sheet = False  # tamanho inválido = trata como ausente
    else:
        weights = np.ones(len(crits))
    # Normalização e validação
    if weights.sum() <= 0 or np.any(weights < 0):
        weights = np.ones(len(crits))
        has_weights_sheet = False
    weights = weights / weights.sum()
    return df, weights, id_col, crits, has_weights_sheet


# Estado
if "loaded" not in st.session_state:
    st.session_state.loaded = False

data_df = None
weights = None
id_col = None
criteria = []
err_load = None
has_weights_sheet = False  # True só quando o Excel carregado trazia folha 'Pesos' válida

if use_demo:
    data_df, weights = build_demo_data()
    id_col = "Alternativa"
    criteria = [c for c in data_df.columns if c != id_col]
    has_weights_sheet = True  # demo já vem com pesos AHP do caso MCG
    st.session_state.loaded = True
elif uploaded is not None:
    res, err_load = safe_call(load_excel, uploaded)
    if err_load is None:
        data_df, weights, id_col, criteria, has_weights_sheet = res
        st.session_state.loaded = True
        if not has_weights_sheet:
            st.sidebar.warning(
                "⚠️ **Folha `Pesos` não encontrada** — a usar pesos uniformes "
                f"({1/len(criteria):.4f} cada). Para usar pesos AHP, acrescenta a "
                "folha `Pesos` ao Excel ou edita manualmente na tabela abaixo."
            )
    else:
        st.sidebar.error(f"❌ {err_load}")

# Editor de tipos e pesos — UI única, compacta e escalável para qualquer nº de critérios
types = []
if st.session_state.loaded and data_df is not None:
    with st.sidebar:
        st.divider()
        st.subheader("🎯 Configuração de critérios")
        st.caption(
            f"**{len(criteria)} critérios** detectados. Edite o sentido (max/min) e o peso "
            "directamente na tabela. Os pesos são normalizados automaticamente."
        )

        # Heurística de defaults — critérios cujo nome sugere custo/minimização
        def _is_cost_criterion(name: str) -> bool:
            low = name.lower()
            # Substrings (palavras longas — sem risco de falso positivo)
            for sub in ("custo", "cost", "esforço", "esforco", "prazo", "tempo",
                        "delay", "risk", "risco", "urgenc", "urgênc"):
                if sub in low:
                    return True
            # Tokens curtos — testa apenas como segmento (entre _ ou - ou início/fim)
            tokens = low.replace("-", "_").split("_")
            return any(t in {"ee", "ud", "urg", "dias"} for t in tokens)

        type_defaults = ["min" if _is_cost_criterion(c) else "max" for c in criteria]

        # Chave dependente dos critérios — força reset quando se troca de ficheiro
        editor_key = f"crit_cfg_{hash(tuple(criteria))}"

        config_df = pd.DataFrame({
            "Critério": criteria,
            "Sentido": type_defaults,
            "Peso": [float(w) for w in weights],
        })

        # Calcula altura adaptativa: limite máximo para não ocupar a sidebar toda
        n = len(criteria)
        row_h = 35
        editor_height = min(35 + row_h * n, 500)  # cap a ~13 linhas visíveis, restantes via scroll

        edited_cfg = st.data_editor(
            config_df,
            column_config={
                "Critério": st.column_config.TextColumn("Critério", disabled=True, width="medium"),
                "Sentido": st.column_config.SelectboxColumn(
                    "Sentido", options=["max", "min"], required=True, width="small",
                    help="max = benefício (quanto maior melhor); min = custo (quanto menor melhor)",
                ),
                "Peso": st.column_config.NumberColumn(
                    "Peso", min_value=0.0, max_value=1.0, step=0.001, format="%.4f", width="small",
                    help="Pesos relativos — serão normalizados para somar 1.0",
                ),
            },
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            height=editor_height,
            key=editor_key,
        )

        # Extrair valores actualizados
        try:
            types = edited_cfg["Sentido"].astype(str).tolist()
            edited_w = edited_cfg["Peso"].astype(float).values
            if edited_w.sum() > 0:
                weights = edited_w / edited_w.sum()
            else:
                weights = np.ones(len(criteria)) / len(criteria)
                st.warning("⚠️ Soma de pesos = 0. A usar pesos uniformes.")
        except Exception as exc:
            st.error(f"Erro na configuração: {exc}")
            types = type_defaults
            weights = np.ones(len(criteria)) / len(criteria)

        # Resumo + acções rápidas
        st.caption(
            f"📊 max: **{types.count('max')}** · min: **{types.count('min')}** · "
            f"Σ pesos (antes da norm.): **{float(edited_cfg['Peso'].sum()):.4f}**"
        )


# =============================================================================
# HELPERS DE RENDERIZAÇÃO
# =============================================================================
def need_data():
    st.info("👈 Carregue um ficheiro Excel na sidebar (ou active o modo demo) para começar.")


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
# DEFINIR TABS
# =============================================================================
TAB_LABELS = [
    "📋 Visão Geral", "🔺 AHP", "🕸️ ANP", "🎯 TOPSIS", "🔗 ELECTRE",
    "📊 PROMETHEE", "⚖️ VIKOR", "📐 MAUT", "🧮 COPRAS", "🌐 DEMATEL",
    "🌫️ Fuzzy AHP", "🌫️ Fuzzy TOPSIS", "🌫️ Fuzzy ANP", "🏆 Dashboard",
    "📚 Teoria & Matemática", "📄 Relatório",
]
tabs = st.tabs(TAB_LABELS)

# Estrutura para o dashboard final
all_results = {}

# =============================================================================
# TAB 1 — VISÃO GERAL
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
            fig = px.imshow(
                norm,
                labels=dict(x="Critério", y="Alternativa", color="Valor normalizado"),
                x=criteria, y=alts, color_continuous_scale="RdYlGn", aspect="auto",
                text_auto=".2f",
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Erro no heatmap: {exc}")


# =============================================================================
# TAB 2 — AHP
# =============================================================================
with tabs[1]:
    st.header("🔺 AHP — Analytic Hierarchy Process")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        n = len(criteria)
        st.markdown(
            "Edite a matriz de comparação par-a-par (escala Saaty). O triângulo inferior "
            "é actualizado automaticamente como recíproco. Os pesos são recalculados via método do autovector."
        )

        # Matriz inicial a partir dos pesos correntes
        init = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and weights[j] != 0:
                    init[i, j] = weights[i] / weights[j]

        pw_df = pd.DataFrame(init, index=criteria, columns=criteria).round(4)
        edited_pw = st.data_editor(pw_df, use_container_width=True, key="ahp_pw")

        # Forçar reciprocidade
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

            # Aviso de inconsistência + identificação do par mais problemático
            if not res['consistent']:
                # Procura o par (i,j) com maior discrepância entre julgamento e razão de pesos
                w_ahp = res['weights']
                worst_pair = None
                worst_diff = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if w_ahp[j] != 0:
                            expected = w_ahp[i] / w_ahp[j]
                            observed = E[i, j]
                            if expected > 0:
                                diff = abs(np.log(observed / expected))
                                if diff > worst_diff:
                                    worst_diff = diff
                                    worst_pair = (i, j, observed, expected)
                msg = (
                    f"⚠️ **Matriz inconsistente: CR = {res['CR']:.4f} > 0.10.** "
                    "Segundo Saaty, julgamentos com CR > 0.10 devem ser revistos."
                )
                if worst_pair is not None:
                    i, j, obs, exp = worst_pair
                    msg += (
                        f"\n\nPar mais inconsistente: **{criteria[i]} vs {criteria[j]}** "
                        f"→ atribuído `{obs:.2f}`, sugerido `{exp:.2f}` "
                        f"(racio dos pesos calculados). Considera ajustar este julgamento "
                        "para baixar o CR; depois reavalia."
                    )
                st.warning(msg)
            else:
                st.success(f"✅ Matriz consistente (CR = {res['CR']:.4f} < 0.10).")

            st.subheader("Vector de pesos AHP")
            comp_df = pd.DataFrame({
                "Critério": criteria,
                "Peso AHP": res["weights"],
                "Peso na sidebar": weights,
            })
            st.dataframe(comp_df.style.format({"Peso AHP": "{:.4f}", "Peso na sidebar": "{:.4f}"}),
                         use_container_width=True, hide_index=True)

            # Ranking AHP (utilidade aditiva com pesos AHP)
            st.subheader("Ranking AHP (aplicado às alternativas)")
            norm = normalize_minmax(mat, types)
            ahp_scores = (norm * res["weights"]).sum(axis=1)
            ahp_rank = ranking_from_scores(ahp_scores)
            alts = data_df[id_col].tolist()
            rank_df = pd.DataFrame({
                "Alternativa": alts,
                "Score AHP": ahp_scores,
                "Ranking": ahp_rank,
            }).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rank_df.style.format({"Score AHP": "{:.4f}"}),
                         use_container_width=True, hide_index=True)

            st.plotly_chart(render_ranking_chart(alts, ahp_scores, "AHP — Score por alternativa"),
                            use_container_width=True)

            all_results["AHP"] = {"scores": ahp_scores, "ranking": ahp_rank, "weights": res["weights"]}


# =============================================================================
# TAB 3 — ANP
# =============================================================================
with tabs[2]:
    st.header("🕸️ ANP — Analytic Network Process (simplificado)")
    if not st.session_state.loaded:
        need_data()
    else:
        st.caption(
            "Aproximação: a influência inter-critério é estimada via correlações entre os "
            "valores observados nas alternativas. A supermatriz é elevada a potências até convergir, "
            "ajustando os pesos AHP para reflectir dependências."
        )
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_anp, mat, weights, types)
        if err:
            st.error(f"Erro ANP: {err}")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Pesos ajustados (ANP)")
                st.dataframe(pd.DataFrame({
                    "Critério": criteria,
                    "Peso original": weights,
                    "Peso ANP": res["adjusted_weights"],
                }).round(4), use_container_width=True, hide_index=True)
            with c2:
                st.subheader("Matriz limite (supermatriz convergida)")
                st.dataframe(pd.DataFrame(res["limit_matrix"].round(4),
                                          index=criteria, columns=criteria),
                             use_container_width=True)

            st.subheader("Ranking ANP")
            rdf = pd.DataFrame({
                "Alternativa": alts,
                "Score ANP": res["scores"],
                "Ranking": res["ranking"],
            }).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"Score ANP": "{:.4f}"}),
                         use_container_width=True, hide_index=True)
            st.plotly_chart(render_ranking_chart(alts, res["scores"], "ANP — Score por alternativa"),
                            use_container_width=True)

            all_results["ANP"] = {"scores": res["scores"], "ranking": res["ranking"]}


# =============================================================================
# TAB 4 — TOPSIS
# =============================================================================
with tabs[3]:
    st.header("🎯 TOPSIS — Technique for Order Preference by Similarity to Ideal Solution")
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
                st.subheader("Matriz normalizada (vectorial)")
                st.dataframe(pd.DataFrame(res["normalized"], index=alts, columns=criteria).round(4),
                             use_container_width=True)
            with c2:
                st.subheader("Matriz ponderada")
                st.dataframe(pd.DataFrame(res["weighted"], index=alts, columns=criteria).round(4),
                             use_container_width=True)

            st.subheader("Soluções ideal (A+) e anti-ideal (A−)")
            st.dataframe(pd.DataFrame({"Critério": criteria,
                                       "A+": res["ideal"], "A−": res["anti_ideal"]}).round(4),
                         use_container_width=True, hide_index=True)

            st.subheader("Distâncias e Coeficiente de Proximidade Ci*")
            rdf = pd.DataFrame({
                "Alternativa": alts,
                "D+": res["d_plus"],
                "D−": res["d_minus"],
                "Ci*": res["scores"],
                "Ranking": res["ranking"],
            }).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"D+": "{:.4f}", "D−": "{:.4f}", "Ci*": "{:.4f}"}),
                         use_container_width=True, hide_index=True)

            st.plotly_chart(render_ranking_chart(alts, res["scores"], "TOPSIS — Ci* por alternativa",
                                                 label="Ci*"), use_container_width=True)

            # Análise de sensibilidade ±sens_pct% nos pesos
            st.subheader(f"Análise de sensibilidade (±{sens_pct}% nos pesos)")
            try:
                ci_low = np.full(len(alts), np.nan)
                ci_high = np.full(len(alts), np.nan)
                ci_min = np.copy(res["scores"]); ci_max = np.copy(res["scores"])
                for j in range(len(criteria)):
                    for delta in [-sens_pct / 100.0, sens_pct / 100.0]:
                        w_perturbed = weights.copy()
                        w_perturbed[j] = max(weights[j] * (1 + delta), 0)
                        if w_perturbed.sum() > 0:
                            w_perturbed = w_perturbed / w_perturbed.sum()
                        r2, e2 = safe_call(model_topsis, mat, w_perturbed, types)
                        if e2 is None:
                            ci_min = np.minimum(ci_min, r2["scores"])
                            ci_max = np.maximum(ci_max, r2["scores"])

                sens_df = pd.DataFrame({
                    "Alternativa": alts,
                    "Ci* mínimo": ci_min,
                    "Ci* base": res["scores"],
                    "Ci* máximo": ci_max,
                })
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=sens_df["Alternativa"], y=sens_df["Ci* base"],
                    name="Ci* base",
                    error_y=dict(
                        type="data",
                        array=sens_df["Ci* máximo"] - sens_df["Ci* base"],
                        arrayminus=sens_df["Ci* base"] - sens_df["Ci* mínimo"],
                    ),
                    marker_color="#2E86AB",
                ))
                fig.update_layout(title=f"Sensibilidade TOPSIS — variação ±{sens_pct}% nos pesos",
                                  height=400, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(sens_df.round(4), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Erro na sensibilidade: {exc}")

            all_results["TOPSIS"] = {"scores": res["scores"], "ranking": res["ranking"]}


# =============================================================================
# TAB 5 — ELECTRE
# =============================================================================
with tabs[4]:
    st.header("🔗 ELECTRE I — relação de sobreclassificação")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_electre, mat, weights, types, c_thresh, d_thresh)
        if err:
            st.error(f"Erro ELECTRE: {err}")
        else:
            st.markdown(f"**Limiares correntes**: c = {c_thresh:.2f} | d = {d_thresh:.2f} (ajustáveis na sidebar)")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Matriz de Concordância (C)")
                st.dataframe(pd.DataFrame(res["concordance"], index=alts, columns=alts).round(3),
                             use_container_width=True)
            with c2:
                st.subheader("Matriz de Discordância (D)")
                st.dataframe(pd.DataFrame(res["discordance"], index=alts, columns=alts).round(3),
                             use_container_width=True)

            st.subheader("Matriz de Sobreclassificação (S)")
            outrank_df = pd.DataFrame(res["outrank"].astype(int), index=alts, columns=alts)
            st.dataframe(outrank_df, use_container_width=True)

            st.subheader("Kernel (conjunto de escolha)")
            kernel_alts = [alts[i] for i in res["kernel"]]
            st.success(f"**Kernel:** {', '.join(kernel_alts) if kernel_alts else '∅ (vazio)'}")

            st.subheader("Ranking por dominância líquida")
            rdf = pd.DataFrame({
                "Alternativa": alts,
                "No kernel": ["✅" if i in res["kernel"] else "—" for i in range(len(alts))],
                "Score (dominância líquida)": res["scores"],
                "Ranking": res["ranking"],
            }).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf, use_container_width=True, hide_index=True)

            # Mapa de sensibilidade c × d
            st.subheader("Mapa de sensibilidade — kernel vs (c, d)")
            try:
                c_grid = np.round(np.arange(max(c_thresh - 0.10, 0.5), min(c_thresh + 0.11, 1.0), 0.05), 2)
                d_grid = np.round(np.arange(max(d_thresh - 0.10, 0.0), min(d_thresh + 0.11, 1.0), 0.05), 2)
                heat = np.zeros((len(d_grid), len(c_grid)))
                for ii, dv in enumerate(d_grid):
                    for jj, cv in enumerate(c_grid):
                        r2, e2 = safe_call(model_electre, mat, weights, types, cv, dv)
                        heat[ii, jj] = len(r2["kernel"]) if e2 is None else np.nan
                fig = px.imshow(heat,
                                labels=dict(x="c", y="d", color="|Kernel|"),
                                x=[f"{v:.2f}" for v in c_grid],
                                y=[f"{v:.2f}" for v in d_grid],
                                color_continuous_scale="Viridis",
                                text_auto=True, aspect="auto")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Erro no mapa: {exc}")

            all_results["ELECTRE"] = {"scores": res["scores"], "ranking": res["ranking"]}


# =============================================================================
# TAB 6 — PROMETHEE
# =============================================================================
with tabs[5]:
    st.header("📊 PROMETHEE II — fluxos líquidos e ranking")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        st.markdown(f"**Função de preferência activa:** `{promethee_fn}` (ajustável na sidebar)")

        res, err = safe_call(model_promethee, mat, weights, types, promethee_fn)
        if err:
            st.error(f"Erro PROMETHEE: {err}")
        else:
            st.subheader("Matriz de preferência agregada π(a,b)")
            st.dataframe(pd.DataFrame(res["preference_matrix"], index=alts, columns=alts).round(4),
                         use_container_width=True)

            st.subheader("Fluxos φ+, φ−, φ líquido")
            rdf = pd.DataFrame({
                "Alternativa": alts,
                "φ+": res["phi_plus"],
                "φ−": res["phi_minus"],
                "φ líquido": res["scores"],
                "Ranking": res["ranking"],
            }).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"φ+": "{:.4f}", "φ−": "{:.4f}", "φ líquido": "{:.4f}"}),
                         use_container_width=True, hide_index=True)

            # Gráfico de fluxos
            fig = go.Figure()
            fig.add_trace(go.Bar(name="φ+", x=alts, y=res["phi_plus"], marker_color="#2A9D8F"))
            fig.add_trace(go.Bar(name="φ−", x=alts, y=-res["phi_minus"], marker_color="#E76F51"))
            fig.add_trace(go.Scatter(name="φ líquido", x=alts, y=res["scores"],
                                     mode="markers+lines", marker=dict(size=10, color="#264653")))
            fig.update_layout(barmode="relative", title="PROMETHEE II — fluxos por alternativa",
                              height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Comparação entre funções de preferência
            st.subheader("Comparação entre funções de preferência")
            try:
                comp = pd.DataFrame({"Alternativa": alts})
                for fn in ["usual", "linear", "gaussian"]:
                    r2, _ = safe_call(model_promethee, mat, weights, types, fn)
                    if r2 is not None:
                        comp[f"φ ({fn})"] = r2["scores"]
                st.dataframe(comp.round(4), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Erro comparativo: {exc}")

            all_results["PROMETHEE"] = {"scores": res["scores"], "ranking": res["ranking"]}


# =============================================================================
# TAB 7 — VIKOR
# =============================================================================
with tabs[6]:
    st.header("⚖️ VIKOR — compromisso entre utilidade e arrependimento")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_vikor, mat, weights, types, vikor_v)
        if err:
            st.error(f"Erro VIKOR: {err}")
        else:
            st.markdown(f"**Peso da estratégia v = {vikor_v:.2f}** (1 = utilidade pura; 0 = arrependimento puro)")
            rdf = pd.DataFrame({
                "Alternativa": alts,
                "S (utilidade)": res["S"],
                "R (arrependimento)": res["R"],
                "Q (índice VIKOR)": res["Q"],
                "Ranking (Q menor = melhor)": res["ranking"],
            }).sort_values("Ranking (Q menor = melhor)").reset_index(drop=True)
            st.dataframe(rdf.style.format({"S (utilidade)": "{:.4f}", "R (arrependimento)": "{:.4f}",
                                           "Q (índice VIKOR)": "{:.4f}"}),
                         use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(name="S", x=alts, y=res["S"], marker_color="#3a86ff"))
            fig.add_trace(go.Bar(name="R", x=alts, y=res["R"], marker_color="#fb5607"))
            fig.add_trace(go.Scatter(name="Q", x=alts, y=res["Q"], mode="markers+lines",
                                     marker=dict(size=10, color="#264653")))
            fig.update_layout(barmode="group", title="VIKOR — S, R, Q por alternativa",
                              height=400, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

            all_results["VIKOR"] = {"scores": res["scores"], "ranking": res["ranking"]}


# =============================================================================
# TAB 8 — MAUT
# =============================================================================
with tabs[7]:
    st.header("📐 MAUT — Multi-Attribute Utility Theory (linear aditiva)")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_maut, mat, weights, types)
        if err:
            st.error(f"Erro MAUT: {err}")
        else:
            st.subheader("Matriz de utilidades parciais (normalização min-max)")
            st.dataframe(pd.DataFrame(res["utility_matrix"], index=alts, columns=criteria).round(4),
                         use_container_width=True)

            st.subheader("Utilidade global e ranking")
            rdf = pd.DataFrame({
                "Alternativa": alts,
                "Utilidade U": res["scores"],
                "Ranking": res["ranking"],
            }).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"Utilidade U": "{:.4f}"}),
                         use_container_width=True, hide_index=True)
            st.plotly_chart(render_ranking_chart(alts, res["scores"], "MAUT — Utilidade U",
                                                 label="Utilidade U"), use_container_width=True)

            all_results["MAUT"] = {"scores": res["scores"], "ranking": res["ranking"]}


# =============================================================================
# TAB 9 — COPRAS
# =============================================================================
with tabs[8]:
    st.header("🧮 COPRAS — Complex Proportional Assessment")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        res, err = safe_call(model_copras, mat, weights, types)
        if err:
            st.error(f"Erro COPRAS: {err}")
        else:
            rdf = pd.DataFrame({
                "Alternativa": alts,
                "S+ (benefícios)": res["S_plus"],
                "S− (custos)": res["S_minus"],
                "Q (importância relativa)": res["Q"],
                "N (utilidade %)": res["N"],
                "Ranking": res["ranking"],
            }).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"S+ (benefícios)": "{:.4f}", "S− (custos)": "{:.4f}",
                                           "Q (importância relativa)": "{:.4f}",
                                           "N (utilidade %)": "{:.2f}"}),
                         use_container_width=True, hide_index=True)
            st.plotly_chart(render_ranking_chart(alts, res["N"], "COPRAS — Utilidade N (%)",
                                                 label="N (%)"), use_container_width=True)

            all_results["COPRAS"] = {"scores": res["N"], "ranking": res["ranking"]}


# =============================================================================
# TAB 10 — DEMATEL
# =============================================================================
with tabs[9]:
    st.header("🌐 DEMATEL — Decision Making Trial and Evaluation Laboratory")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        st.caption(
            "Aproximação: matriz de relação directa estimada via correlações entre critérios. "
            "A prominência (D+R) modula os pesos para o ranking final."
        )
        res, err = safe_call(model_dematel, mat, weights, types)
        if err:
            st.error(f"Erro DEMATEL: {err}")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Matriz total de relações T")
                st.dataframe(pd.DataFrame(res["T"].round(4), index=criteria, columns=criteria),
                             use_container_width=True)
            with c2:
                st.subheader("Prominência (D+R) e relação (D−R)")
                st.dataframe(pd.DataFrame({
                    "Critério": criteria,
                    "D": res["D"], "R": res["R"],
                    "Prominência D+R": res["prominence"],
                    "Relação D−R": res["relation"],
                }).round(4), use_container_width=True, hide_index=True)

            # Diagrama de influência: D+R (x) vs D-R (y)
            try:
                fig = px.scatter(
                    x=res["prominence"], y=res["relation"], text=criteria,
                    labels={"x": "Prominência (D+R)", "y": "Relação (D−R)"},
                    title="DEMATEL — diagrama causa-efeito"
                )
                fig.update_traces(textposition="top center", marker=dict(size=14, color="#6a4c93"))
                fig.add_hline(y=0, line_dash="dash", line_color="grey")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

            st.subheader("Ranking DEMATEL (pesos ajustados pela prominência)")
            rdf = pd.DataFrame({
                "Alternativa": alts,
                "Score DEMATEL": res["scores"],
                "Ranking": res["ranking"],
            }).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"Score DEMATEL": "{:.4f}"}),
                         use_container_width=True, hide_index=True)

            all_results["DEMATEL"] = {"scores": res["scores"], "ranking": res["ranking"]}


# =============================================================================
# TAB 11 — FUZZY AHP
# =============================================================================
with tabs[10]:
    st.header("🌫️ Fuzzy AHP")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        st.caption(
            "Os pesos são tratados como números triangulares fuzzy (l, m, u) com spread ±20%, "
            "depois defuzzificados pelo método do centro de área."
        )
        res, err = safe_call(model_fuzzy_ahp, weights)
        if err:
            st.error(f"Erro Fuzzy AHP: {err}")
        else:
            fdf = pd.DataFrame(res["fuzzy_weights"], index=criteria, columns=["l", "m", "u"])
            fdf["crisp (defuzz)"] = res["crisp_weights"]
            st.subheader("Pesos fuzzy triangulares e versão crisp")
            st.dataframe(fdf.round(4), use_container_width=True)

            # Ranking usando os pesos defuzzificados
            norm = normalize_minmax(mat, types)
            scores = (norm * res["crisp_weights"]).sum(axis=1)
            ranking = ranking_from_scores(scores)
            rdf = pd.DataFrame({
                "Alternativa": alts, "Score F-AHP": scores, "Ranking": ranking
            }).sort_values("Ranking").reset_index(drop=True)
            st.subheader("Ranking Fuzzy AHP")
            st.dataframe(rdf.style.format({"Score F-AHP": "{:.4f}"}),
                         use_container_width=True, hide_index=True)
            st.plotly_chart(render_ranking_chart(alts, scores, "Fuzzy AHP — Score"),
                            use_container_width=True)

            all_results["Fuzzy AHP"] = {"scores": scores, "ranking": ranking}


# =============================================================================
# TAB 12 — FUZZY TOPSIS
# =============================================================================
with tabs[11]:
    st.header("🌫️ Fuzzy TOPSIS")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        st.caption(
            "Valores tratados como números triangulares (val·(1−s), val, val·(1+s)); distância pelo método do vértice."
        )
        spread = st.slider("Spread fuzzy (%)", 5, 30, 10, 5) / 100.0
        res, err = safe_call(model_fuzzy_topsis, mat, weights, types, spread)
        if err:
            st.error(f"Erro Fuzzy TOPSIS: {err}")
        else:
            rdf = pd.DataFrame({
                "Alternativa": alts,
                "d+ (FPIS)": res["d_plus"],
                "d− (FNIS)": res["d_minus"],
                "CC (proximidade)": res["scores"],
                "Ranking": res["ranking"],
            }).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"d+ (FPIS)": "{:.4f}", "d− (FNIS)": "{:.4f}",
                                           "CC (proximidade)": "{:.4f}"}),
                         use_container_width=True, hide_index=True)
            st.plotly_chart(render_ranking_chart(alts, res["scores"], "Fuzzy TOPSIS — CC",
                                                 label="CC"), use_container_width=True)

            all_results["Fuzzy TOPSIS"] = {"scores": res["scores"], "ranking": res["ranking"]}


# =============================================================================
# TAB 13 — FUZZY ANP
# =============================================================================
with tabs[12]:
    st.header("🌫️ Fuzzy ANP")
    if not st.session_state.loaded:
        need_data()
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        st.caption(
            "Combina pesos fuzzy (do Fuzzy AHP) com ajuste por supermatriz de influência inter-critério (ANP simplificado)."
        )
        res, err = safe_call(model_fuzzy_anp, mat, weights, types)
        if err:
            st.error(f"Erro Fuzzy ANP: {err}")
        else:
            st.subheader("Pesos finais (fuzzy + ANP)")
            st.dataframe(pd.DataFrame({
                "Critério": criteria,
                "Peso fuzzy (defuzz)": res["crisp_fuzzy_weights"],
                "Peso Fuzzy ANP": res["adjusted_weights"],
            }).round(4), use_container_width=True, hide_index=True)

            rdf = pd.DataFrame({
                "Alternativa": alts,
                "Score F-ANP": res["scores"],
                "Ranking": res["ranking"],
            }).sort_values("Ranking").reset_index(drop=True)
            st.dataframe(rdf.style.format({"Score F-ANP": "{:.4f}"}),
                         use_container_width=True, hide_index=True)
            st.plotly_chart(render_ranking_chart(alts, res["scores"], "Fuzzy ANP — Score"),
                            use_container_width=True)

            all_results["Fuzzy ANP"] = {"scores": res["scores"], "ranking": res["ranking"]}


# =============================================================================
# TAB 14 — DASHBOARD CONSOLIDADO
# =============================================================================
with tabs[13]:
    st.header("🏆 Dashboard Consolidado")
    if not st.session_state.loaded:
        need_data()
    elif not all_results:
        st.warning("⚠️ Nenhum modelo foi executado com sucesso. Volte às tabs anteriores e verifique os erros.")
    else:
        alts = data_df[id_col].tolist()
        n_alt = len(alts)

        # Tabela consolidada de rankings
        models_with_results = list(all_results.keys())
        rank_table = pd.DataFrame({"Alternativa": alts})
        score_table = pd.DataFrame({"Alternativa": alts})
        for m in models_with_results:
            rank_table[m] = all_results[m]["ranking"]
            score_table[m] = all_results[m]["scores"]

        # Ranking agregado (média de posições — método de Borda invertido)
        rank_table["Posição Média"] = rank_table[models_with_results].mean(axis=1).round(2)
        rank_table["Ranking Final"] = ranking_from_scores(-rank_table["Posição Média"].values)
        rank_table = rank_table.sort_values("Ranking Final").reset_index(drop=True)

        st.subheader("Tabela consolidada de rankings (1 = melhor)")
        styled = rank_table.style.format({"Posição Média": "{:.2f}"})\
            .background_gradient(subset=models_with_results, cmap="RdYlGn_r")\
            .background_gradient(subset=["Posição Média", "Ranking Final"], cmap="RdYlGn_r")
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Convergência: nº de modelos que colocam cada alternativa no Top-3
        top3_count = (rank_table[models_with_results] <= 3).sum(axis=1)
        rank_table["Top-3 em N modelos"] = top3_count

        st.subheader("📌 Painel de recomendação")
        top3 = rank_table.sort_values("Ranking Final").head(3)
        top3_alts = top3["Alternativa"].tolist()
        convergence = (rank_table[rank_table["Alternativa"].isin(top3_alts)]
                       [models_with_results] <= 3).sum().sum()
        max_conv = 3 * len(models_with_results)
        conv_pct = (convergence / max_conv) * 100 if max_conv else 0

        rec_col1, rec_col2, rec_col3 = st.columns(3)
        rec_col1.metric("🥇 1º lugar", top3_alts[0] if len(top3_alts) > 0 else "—")
        rec_col2.metric("🥈 2º lugar", top3_alts[1] if len(top3_alts) > 1 else "—")
        rec_col3.metric("🥉 3º lugar", top3_alts[2] if len(top3_alts) > 2 else "—")

        st.info(
            f"**Convergência:** {convergence}/{max_conv} ({conv_pct:.0f}%) — "
            f"i.e., dos 3 × {len(models_with_results)} = {max_conv} possíveis posições Top-3, "
            f"{convergence} foram atribuídas às alternativas {', '.join(top3_alts)}.\n\n"
            f"**Top-3 recomendado:** {', '.join(top3_alts)} | "
            f"**Modelos avaliados:** {', '.join(models_with_results)}"
        )

        # Gráfico de calor de rankings
        st.subheader("Heatmap de posições por modelo")
        try:
            heat_df = rank_table.set_index("Alternativa")[models_with_results]
            fig = px.imshow(heat_df.values,
                            labels=dict(x="Modelo", y="Alternativa", color="Ranking"),
                            x=models_with_results, y=heat_df.index,
                            color_continuous_scale="RdYlGn_r",
                            aspect="auto", text_auto=True)
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Erro no heatmap: {exc}")

        # Gráfico radar para Top-3
        st.subheader("Perfil multicritério — Top-3 (radar normalizado)")
        try:
            mat = get_matrix()
            norm = normalize_minmax(mat, types)
            fig = go.Figure()
            colors = ["#e63946", "#f4a261", "#2a9d8f"]
            for k, alt_name in enumerate(top3_alts):
                idx = alts.index(alt_name)
                vals = list(norm[idx]) + [norm[idx, 0]]
                axes = criteria + [criteria[0]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=axes, fill="toself", name=alt_name,
                    line=dict(color=colors[k % len(colors)], width=2),
                    opacity=0.65,
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, height=460, margin=dict(l=10, r=10, t=30, b=10),
                title="Top-3 — perfil normalizado por critério",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Erro no radar: {exc}")

        # Exportação Excel
        st.subheader("📥 Exportação")
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                data_df.to_excel(writer, sheet_name="Dados", index=False)
                pd.DataFrame({"Critério": criteria, "Peso": weights, "Sentido": types})\
                    .to_excel(writer, sheet_name="Pesos_e_Tipos", index=False)
                rank_table.to_excel(writer, sheet_name="Rankings", index=False)
                score_table.to_excel(writer, sheet_name="Scores", index=False)
            st.download_button(
                "Descarregar Excel com todos os resultados",
                data=buffer.getvalue(),
                file_name="mcdm_resultados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as exc:
            st.error(f"Erro na exportação: {exc}")

# =============================================================================
# TAB 15 — TEORIA & MATEMÁTICA
# =============================================================================
with tabs[14]:
    st.header("📚 Teoria & Matemática dos Modelos MCDM")
    st.markdown(
        "Esta secção descreve a fundamentação matemática de **cada um dos 13 modelos** "
        "implementados nas tabs anteriores. Pode ser consultada em qualquer momento, "
        "independentemente dos dados carregados."
    )

    st.markdown("---")
    st.subheader("Introdução: o problema MCDM")
    st.markdown(
        "Um problema de **Decisão Multicritério** (MCDM, *Multi-Criteria Decision Making*) "
        "envolve a escolha, ordenação ou classificação de **alternativas** "
        "$A_1, A_2, \\dots, A_m$ avaliadas segundo **critérios** $C_1, C_2, \\dots, C_n$, "
        "frequentemente em conflito (e.g., maximizar receita vs. minimizar custo). "
        "Cada critério tem um **peso** $w_j$ (com $\\sum w_j = 1$) que reflecte a sua importância "
        "relativa, e um **sentido** (max = benefício, min = custo)."
    )
    st.latex(r"""
    \text{Matriz de decisão:}\quad
    X = \begin{bmatrix}
    x_{11} & x_{12} & \cdots & x_{1n} \\
    x_{21} & x_{22} & \cdots & x_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    x_{m1} & x_{m2} & \cdots & x_{mn}
    \end{bmatrix},\quad
    w = (w_1, w_2, \dots, w_n)
    """)

    # ============ AHP ============
    st.markdown("---")
    st.subheader("1. AHP — Analytic Hierarchy Process (Saaty, 1980)")
    st.markdown(
        "Determina pesos $w_j$ a partir de uma **matriz de comparação par-a-par** $A$ "
        "onde $a_{ij}$ representa quantas vezes o critério $i$ é mais importante que $j$, "
        "usando a escala de Saaty (1=igual, 3=moderadamente, 5=fortemente, 7=muito fortemente, 9=extremamente)."
    )
    st.latex(r"A = [a_{ij}],\quad a_{ji} = 1/a_{ij},\quad a_{ii} = 1")
    st.markdown("O vector de pesos é o **autovector principal** associado ao maior autovalor $\\lambda_{max}$:")
    st.latex(r"A\,w = \lambda_{max}\,w")
    st.markdown("**Verificação de consistência:**")
    st.latex(r"CI = \frac{\lambda_{max} - n}{n - 1},\quad CR = \frac{CI}{RI(n)}")
    st.markdown(
        "Onde $RI(n)$ é o **Random Index** (índice de consistência aleatório) tabelado. "
        "Saaty considera a matriz consistente se **CR < 0.10**; caso contrário, recomenda revisão dos julgamentos."
    )

    # ============ ANP ============
    st.markdown("---")
    st.subheader("2. ANP — Analytic Network Process (Saaty, 1996)")
    st.markdown(
        "Generalização do AHP que captura **dependências entre critérios** (rede em vez de hierarquia). "
        "Constrói-se uma **supermatriz** $W$ que combina os pesos dos elementos com as influências cruzadas. "
        "A solução estável é a **matriz limite**:"
    )
    st.latex(r"W^\infty = \lim_{k \to \infty} W^k")
    st.markdown(
        "Nesta implementação, na ausência de uma matriz de influência elicitada do decisor, "
        "estimamos a influência inter-critério via **correlações** entre os valores observados:"
    )
    st.latex(r"W_{ij} = \frac{|\rho(C_i, C_j)|}{\sum_k |\rho(C_k, C_j)|}")
    st.markdown(
        "Os pesos AHP são então modulados pela matriz limite para obter pesos finais "
        "que reflectem a estrutura de dependências detectada nos dados."
    )

    # ============ TOPSIS ============
    st.markdown("---")
    st.subheader("3. TOPSIS — Technique for Order Preference by Similarity to Ideal Solution (Hwang & Yoon, 1981)")
    st.markdown("**Passo 1.** Normalização vectorial (Euclidiana) por critério:")
    st.latex(r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^m x_{kj}^2}}")
    st.markdown("**Passo 2.** Ponderação:")
    st.latex(r"v_{ij} = w_j \cdot r_{ij}")
    st.markdown("**Passo 3.** Solução ideal $A^+$ e anti-ideal $A^-$:")
    st.latex(r"""
    A^+ = \{v_j^+\} = \begin{cases} \max_i v_{ij} & \text{se } j \in J_{\text{max}} \\ \min_i v_{ij} & \text{se } j \in J_{\text{min}} \end{cases}
    """)
    st.latex(r"""
    A^- = \{v_j^-\} = \begin{cases} \min_i v_{ij} & \text{se } j \in J_{\text{max}} \\ \max_i v_{ij} & \text{se } j \in J_{\text{min}} \end{cases}
    """)
    st.markdown("**Passo 4.** Distâncias Euclidianas:")
    st.latex(r"D_i^+ = \sqrt{\sum_{j=1}^n (v_{ij} - v_j^+)^2},\quad D_i^- = \sqrt{\sum_{j=1}^n (v_{ij} - v_j^-)^2}")
    st.markdown("**Passo 5.** Coeficiente de proximidade ao ideal (quanto maior, melhor):")
    st.latex(r"C_i^* = \frac{D_i^-}{D_i^+ + D_i^-},\quad 0 \le C_i^* \le 1")

    # ============ ELECTRE I ============
    st.markdown("---")
    st.subheader("4. ELECTRE I — ELimination Et Choix Traduisant la REalité (Roy, 1968)")
    st.markdown(
        "Constrói **relações de sobreclassificação** binárias: dado um par $(a, b)$, "
        "$a$ sobreclassifica $b$ ($a \\,S\\, b$) se a evidência a favor é suficiente e contra é fraca."
    )
    st.markdown("**Índice de concordância:**")
    st.latex(r"C(a, b) = \frac{1}{\sum_j w_j} \sum_{j \in J(a,b)} w_j,\quad J(a,b) = \{j : a_j \succeq b_j\}")
    st.markdown("**Índice de discordância:**")
    st.latex(r"D(a, b) = \frac{\max_j \{r_{bj} - r_{aj} : r_{bj} > r_{aj}\}}{\max_{j,k,l} |r_{kj} - r_{lj}|}")
    st.markdown(
        "Define-se a relação de sobreclassificação com limiares $c$ (concordância) e $d$ (discordância):"
    )
    st.latex(r"a\,S\,b \iff C(a,b) \ge c \;\land\; D(a,b) \le d")
    st.markdown(
        "O **kernel** é o subconjunto de alternativas que não são sobreclassificadas por nenhuma "
        "fora do kernel — o conjunto recomendado de candidatas robustas."
    )

    # ============ PROMETHEE II ============
    st.markdown("---")
    st.subheader("5. PROMETHEE II — Preference Ranking Organisation Method (Brans, 1985)")
    st.markdown("Para cada par $(a, b)$ e critério $j$, define-se a **função de preferência** $P_j(d)$ "
                "onde $d = x_{aj} - x_{bj}$ (com sinal invertido para critérios de custo):")
    st.latex(r"""
    P_j^{\text{usual}}(d) = \begin{cases} 0 & d \le 0 \\ 1 & d > 0 \end{cases}
    """)
    st.latex(r"""
    P_j^{\text{linear}}(d) = \begin{cases} 0 & d \le 0 \\ d/p & 0 < d < p \\ 1 & d \ge p \end{cases}
    """)
    st.latex(r"""
    P_j^{\text{gaussian}}(d) = \begin{cases} 0 & d \le 0 \\ 1 - e^{-d^2/(2\sigma^2)} & d > 0 \end{cases}
    """)
    st.markdown("**Grau de preferência agregado:**")
    st.latex(r"\pi(a, b) = \sum_{j=1}^n w_j \cdot P_j(d_{ab})")
    st.markdown("**Fluxos de preferência:**")
    st.latex(r"""
    \phi^+(a) = \frac{1}{m-1} \sum_{b \ne a} \pi(a, b),\quad
    \phi^-(a) = \frac{1}{m-1} \sum_{b \ne a} \pi(b, a)
    """)
    st.markdown("**Fluxo líquido (ranking PROMETHEE II):**")
    st.latex(r"\phi(a) = \phi^+(a) - \phi^-(a)")

    # ============ VIKOR ============
    st.markdown("---")
    st.subheader("6. VIKOR — VIseKriterijumska Optimizacija I Kompromisno Resenje (Opricovic, 1998)")
    st.markdown("Procura a **solução de compromisso** entre máxima utilidade do grupo e mínimo arrependimento individual.")
    st.markdown("Sejam $f_j^* = \\max_i f_{ij}$ (ideal) e $f_j^- = \\min_i f_{ij}$ (anti-ideal) por critério.")
    st.latex(r"S_i = \sum_{j=1}^n w_j \frac{f_j^* - f_{ij}}{f_j^* - f_j^-}")
    st.latex(r"R_i = \max_j \left[ w_j \frac{f_j^* - f_{ij}}{f_j^* - f_j^-} \right]")
    st.latex(r"Q_i = v \cdot \frac{S_i - S^*}{S^- - S^*} + (1-v) \cdot \frac{R_i - R^*}{R^- - R^*}")
    st.markdown(
        "$S_i$ representa **utilidade de grupo** (distância à ideal), $R_i$ representa **arrependimento individual** "
        "(pior critério para a alternativa $i$). O parâmetro $v \\in [0,1]$ regula a estratégia: $v=1$ privilegia utilidade, $v=0$ privilegia equidade. A melhor alternativa é a de **menor $Q_i$**."
    )

    # ============ MAUT ============
    st.markdown("---")
    st.subheader("7. MAUT — Multi-Attribute Utility Theory (Keeney & Raiffa, 1976)")
    st.markdown(
        "Forma simples de utilidade linear aditiva. Cada valor é convertido numa **utilidade parcial** "
        "$u_j(x_{ij}) \\in [0, 1]$ via normalização min-max (com inversão para custos):"
    )
    st.latex(r"""
    u_j(x_{ij}) = \begin{cases}
    \dfrac{x_{ij} - \min_k x_{kj}}{\max_k x_{kj} - \min_k x_{kj}} & \text{(benefício)} \\[6pt]
    \dfrac{\max_k x_{kj} - x_{ij}}{\max_k x_{kj} - \min_k x_{kj}} & \text{(custo)}
    \end{cases}
    """)
    st.markdown("**Utilidade global** (ranking por valor decrescente):")
    st.latex(r"U_i = \sum_{j=1}^n w_j \cdot u_j(x_{ij})")

    # ============ COPRAS ============
    st.markdown("---")
    st.subheader("8. COPRAS — COmplex PRoportional ASsessment (Zavadskas & Kaklauskas, 1996)")
    st.markdown("Separa critérios em **benefícios** ($J^+$) e **custos** ($J^-$) e calcula utilidades parciais ponderadas:")
    st.latex(r"""
    S_i^+ = \sum_{j \in J^+} w_j \cdot \bar{x}_{ij},\quad
    S_i^- = \sum_{j \in J^-} w_j \cdot \bar{x}_{ij}
    """)
    st.markdown("**Importância relativa:**")
    st.latex(r"""
    Q_i = S_i^+ + \frac{\min_k S_k^- \cdot \sum_{k=1}^m (1/S_k^-)}{S_i^- \cdot \sum_{k=1}^m (1/S_k^-)}
    """)
    st.markdown("**Grau de utilidade** (normalizado a 100):")
    st.latex(r"N_i = \frac{Q_i}{\max_k Q_k} \times 100\,\%")

    # ============ DEMATEL ============
    st.markdown("---")
    st.subheader("9. DEMATEL — Decision Making Trial and Evaluation Laboratory (Gabus & Fontela, 1972)")
    st.markdown(
        "Modela **influências causais entre critérios**. Partindo de uma matriz de relação directa $Z$ "
        "(aqui estimada por correlações, na ausência de elicitação directa), normaliza-se:"
    )
    st.latex(r"X = \frac{Z}{\max\left(\max_i \sum_j z_{ij},\; \max_j \sum_i z_{ij}\right)}")
    st.markdown("A **matriz de relação total** é obtida por:")
    st.latex(r"T = X \cdot (I - X)^{-1}")
    st.markdown("Definem-se as somas $D_i = \\sum_j t_{ij}$ (influência exercida) e $R_i = \\sum_j t_{ji}$ (influência recebida).")
    st.latex(r"""
    \begin{aligned}
    D + R & \quad \text{(prominência — importância global do critério)} \\
    D - R & \quad \text{(relação — causal se positivo, efeito se negativo)}
    \end{aligned}
    """)

    # ============ FUZZY AHP ============
    st.markdown("---")
    st.subheader("10. Fuzzy AHP (Chang, 1996)")
    st.markdown(
        "Estende o AHP usando **números triangulares fuzzy** (TFN) para capturar incerteza nos julgamentos. "
        "Um TFN é representado por um tuplo $(l, m, u)$ — limite inferior, valor central, limite superior."
    )
    st.latex(r"\tilde{a} = (l, m, u),\quad l \le m \le u")
    st.markdown("Para defuzzificar (obter um peso crisp), usa-se o **método do centro de área**:")
    st.latex(r"w^{\text{crisp}} = \frac{l + m + u}{3}")
    st.markdown(
        "Nesta implementação, os pesos crisp do AHP são expandidos em TFNs com spread $\\pm 20\\%$ "
        "como aproximação inicial; em aplicações reais, o decisor especifica directamente os TFNs."
    )

    # ============ FUZZY TOPSIS ============
    st.markdown("---")
    st.subheader("11. Fuzzy TOPSIS (Chen, 2000)")
    st.markdown("Aplica o TOPSIS clássico a uma matriz de decisão fuzzy. Cada elemento é um TFN $\\tilde{x}_{ij} = (l_{ij}, m_{ij}, u_{ij})$.")
    st.markdown("A **distância entre TFNs** (método do vértice):")
    st.latex(r"""
    d(\tilde{a}, \tilde{b}) = \sqrt{\frac{1}{3}\left[(a_l - b_l)^2 + (a_m - b_m)^2 + (a_u - b_u)^2\right]}
    """)
    st.markdown(
        "Calculam-se a **Fuzzy Positive Ideal Solution** (FPIS) e a **Fuzzy Negative Ideal Solution** (FNIS) "
        "e o coeficiente de proximidade fuzzy:"
    )
    st.latex(r"CC_i = \frac{d_i^-}{d_i^+ + d_i^-}")

    # ============ FUZZY ANP ============
    st.markdown("---")
    st.subheader("12. Fuzzy ANP (Mikhailov, 2003)")
    st.markdown(
        "Combina os pesos fuzzy do Fuzzy AHP com a estrutura de supermatriz do ANP. "
        "Permite modelar simultaneamente **incerteza** (via TFN) e **dependências entre critérios** "
        "(via supermatriz de influência). Os pesos defuzzificados são propagados pela matriz limite "
        "para obter pesos finais que reflectem ambos os aspectos."
    )

    # ============ Agregação ============
    st.markdown("---")
    st.subheader("13. Ranking Consolidado — Método de Borda invertido")
    st.markdown(
        "Para agregar os rankings dos vários modelos, usa-se a **média de posições** "
        "(uma variante do método de Borda): a alternativa com menor média de ranking "
        "é a recomendada agregadamente."
    )
    st.latex(r"\bar{r}_i = \frac{1}{K} \sum_{k=1}^K r_{ik}")
    st.markdown(
        "Onde $r_{ik}$ é a posição da alternativa $i$ no modelo $k$, e $K$ é o número total de modelos. "
        "Calcula-se também a **convergência Top-3**: percentagem dos modelos que colocam cada "
        "alternativa nas três primeiras posições — indicador de robustez."
    )

    st.markdown("---")
    st.subheader("Referências bibliográficas")
    st.markdown(
        "- Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.\n"
        "- Saaty, T. L. (1996). *Decision Making with Dependence and Feedback: The Analytic Network Process*. RWS.\n"
        "- Hwang, C.-L., Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer.\n"
        "- Roy, B. (1968). Classement et choix en présence de points de vue multiples (méthode ELECTRE). *RIRO* 8.\n"
        "- Brans, J. P., Vincke, P. (1985). A Preference Ranking Organisation Method. *Management Science* 31(6).\n"
        "- Opricovic, S., Tzeng, G.-H. (2004). Compromise solution by MCDM methods. *EJOR* 156(2).\n"
        "- Keeney, R. L., Raiffa, H. (1976). *Decisions with Multiple Objectives*. Wiley.\n"
        "- Zavadskas, E. K., Kaklauskas, A. (1996). *Multiple Criteria Evaluation of Buildings*. Vilnius Tech.\n"
        "- Gabus, A., Fontela, E. (1972). *World Problems, an Invitation to Further Thought*. Battelle.\n"
        "- Chang, D.-Y. (1996). Applications of the extent analysis method on fuzzy AHP. *EJOR* 95(3).\n"
        "- Chen, C.-T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy environment. *Fuzzy Sets and Systems* 114(1).\n"
        "- Mikhailov, L. (2003). Deriving priorities from fuzzy pairwise comparison judgments. *Fuzzy Sets and Systems* 134(3)."
    )


# =============================================================================
# TAB 16 — RELATÓRIO (DINÂMICO)
# =============================================================================
with tabs[15]:
    st.header("📄 Relatório de Análise Multicritério")

    if not st.session_state.loaded:
        need_data()
    elif not all_results:
        st.warning("⚠️ Nenhum modelo foi executado com sucesso. Visita as tabs anteriores primeiro.")
    else:
        mat = get_matrix()
        alts = data_df[id_col].tolist()
        n_alt = len(alts)
        n_crit = len(criteria)
        n_max = types.count("max")
        n_min = types.count("min")

        # ===== Sumário executivo =====
        models_with_results = list(all_results.keys())

        # Calcula ranking consolidado (Borda invertido)
        rank_table_rel = pd.DataFrame({"Alternativa": alts})
        for m in models_with_results:
            rank_table_rel[m] = all_results[m]["ranking"]
        rank_table_rel["Posição Média"] = rank_table_rel[models_with_results].mean(axis=1)
        rank_table_rel = rank_table_rel.sort_values("Posição Média").reset_index(drop=True)

        top3_alts_rel = rank_table_rel["Alternativa"].head(3).tolist()
        top1 = top3_alts_rel[0] if len(top3_alts_rel) > 0 else "—"

        # Convergência: quantos modelos têm a top-1 no top-3?
        if top1 != "—":
            top1_row = rank_table_rel[rank_table_rel["Alternativa"] == top1].iloc[0]
            top1_in_top3 = sum(1 for m in models_with_results if top1_row[m] <= 3)
            conv_pct = (top1_in_top3 / len(models_with_results)) * 100
        else:
            top1_in_top3 = 0
            conv_pct = 0

        # Estatísticas dos critérios (para relatório)
        stats = data_df[criteria].describe().T

        st.markdown(
            f"**Data:** {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}  \n"
            f"**Modelos aplicados com sucesso:** {len(models_with_results)} / 13  \n"
            f"**Alternativas avaliadas:** {n_alt}  \n"
            f"**Critérios:** {n_crit} ({n_max} benefício · {n_min} custo)"
        )

        # ===== Construção do relatório em markdown (downloadable) =====
        report_lines = []
        report_lines.append(f"# Relatório de Análise Multicritério\n")
        report_lines.append(f"**Gerado em:** {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}\n")

        # 1. Sumário executivo
        report_lines.append("## 1. Sumário executivo\n")
        report_lines.append(
            f"Foi realizada uma análise de decisão multicritério sobre **{n_alt} alternativas** "
            f"avaliadas segundo **{n_crit} critérios** ({n_max} de benefício e {n_min} de custo). "
            f"Foram aplicados **{len(models_with_results)} modelos** MCDM "
            f"({', '.join(models_with_results)}), agregados por método de Borda invertido "
            f"(média de posições).\n"
        )
        report_lines.append(
            f"**Alternativa recomendada:** `{top1}`, com posição média de "
            f"`{rank_table_rel['Posição Média'].iloc[0]:.2f}`. "
            f"Aparece no Top-3 em **{top1_in_top3} de {len(models_with_results)}** modelos "
            f"({conv_pct:.0f}% de convergência inter-modelo).\n"
        )
        if len(top3_alts_rel) >= 3:
            report_lines.append(
                f"**Top-3 agregado:** 1º `{top3_alts_rel[0]}` · "
                f"2º `{top3_alts_rel[1]}` · 3º `{top3_alts_rel[2]}`\n"
            )

        # 2. Contexto e dados
        report_lines.append("## 2. Contexto e dados de entrada\n")
        report_lines.append("### 2.1 Critérios, pesos e sentidos\n")
        report_lines.append("| Critério | Peso | Sentido |")
        report_lines.append("|----------|------|---------|")
        for c, w, t in zip(criteria, weights, types):
            report_lines.append(f"| {c} | {w:.4f} | {t} |")
        report_lines.append("")

        report_lines.append("### 2.2 Estatísticas descritivas dos critérios\n")
        report_lines.append("| Critério | Mín | Mediana | Máx | Média | Desvio padrão |")
        report_lines.append("|----------|-----|---------|-----|-------|---------------|")
        for c in criteria:
            s = stats.loc[c]
            report_lines.append(
                f"| {c} | {s['min']:.4g} | {s['50%']:.4g} | {s['max']:.4g} | "
                f"{s['mean']:.4g} | {s['std']:.4g} |"
            )
        report_lines.append("")

        # 3. Metodologia
        report_lines.append("## 3. Metodologia\n")
        report_lines.append(
            "Foram aplicados os seguintes modelos (ver Tab `Teoria & Matemática` para fundamentação):\n"
        )
        for m in models_with_results:
            report_lines.append(f"- **{m}**")
        report_lines.append("")
        report_lines.append(
            "Para cada modelo foram registados o **score** e o **ranking** por alternativa. "
            "Os rankings foram agregados por **média de posições** (Borda invertido) — "
            "a alternativa com menor média é a recomendação final.\n"
        )

        # 4. Resultados por modelo
        report_lines.append("## 4. Resultados por modelo\n")
        report_lines.append("Top-3 alternativas em cada modelo:\n")
        report_lines.append("| Modelo | 1º | 2º | 3º |")
        report_lines.append("|--------|-----|-----|-----|")
        for m in models_with_results:
            rank_m = all_results[m]["ranking"]
            top_indices = sorted(range(len(rank_m)), key=lambda i: rank_m[i])[:3]
            top_names = [alts[i] for i in top_indices]
            row = f"| {m} |"
            for i in range(3):
                row += f" {top_names[i] if i < len(top_names) else '—'} |"
            report_lines.append(row)
        report_lines.append("")

        # 5. Ranking consolidado
        report_lines.append("## 5. Ranking consolidado (Borda invertido)\n")
        report_lines.append("| Posição | Alternativa | Posição média |")
        report_lines.append("|---------|-------------|---------------|")
        for i, row in rank_table_rel.head(min(10, n_alt)).iterrows():
            report_lines.append(
                f"| {i+1} | {row['Alternativa']} | {row['Posição Média']:.2f} |"
            )
        report_lines.append("")

        # 6. Convergência e robustez
        report_lines.append("## 6. Análise de convergência e robustez\n")
        report_lines.append(
            f"A alternativa Top-1 (`{top1}`) aparece no Top-3 de **{top1_in_top3}/{len(models_with_results)}** "
            f"modelos avaliados, indicando uma robustez de **{conv_pct:.0f}%**.\n"
        )
        # Convergência por alternativa Top-3
        report_lines.append("Convergência Top-3 por alternativa (nº de modelos que colocam no Top-3):\n")
        report_lines.append("| Alternativa | Modelos com Top-3 | % |")
        report_lines.append("|-------------|-------------------|---|")
        for alt in alts:
            row = rank_table_rel[rank_table_rel["Alternativa"] == alt].iloc[0]
            n_top3 = sum(1 for m in models_with_results if row[m] <= 3)
            if n_top3 > 0:
                report_lines.append(
                    f"| {alt} | {n_top3}/{len(models_with_results)} | "
                    f"{(n_top3/len(models_with_results))*100:.0f}% |"
                )
        report_lines.append("")

        # 7. Recomendação
        report_lines.append("## 7. Recomendação\n")
        if conv_pct >= 60:
            verd = (
                f"A análise apresenta **alta convergência** ({conv_pct:.0f}%) "
                f"em torno da alternativa `{top1}`. **Recomenda-se a sua selecção** com "
                f"elevado grau de confiança."
            )
        elif conv_pct >= 40:
            verd = (
                f"A análise apresenta **convergência moderada** ({conv_pct:.0f}%) "
                f"em torno da alternativa `{top1}`. Recomenda-se selecção, mas com **análise "
                f"complementar de sensibilidade** aos pesos."
            )
        else:
            verd = (
                f"A análise apresenta **baixa convergência** ({conv_pct:.0f}%). "
                f"Recomenda-se reavaliação dos pesos e/ou alargamento do conjunto de alternativas "
                f"antes de decidir."
            )
        report_lines.append(verd + "\n")

        # 8. Limitações
        report_lines.append("## 8. Limitações e observações\n")
        if "AHP" in all_results and "weights" in all_results["AHP"]:
            ahp_w = all_results["AHP"]["weights"]
            max_diff = max(abs(ahp_w - weights))
            if max_diff > 0.05:
                report_lines.append(
                    f"- Os pesos AHP calculados na tab AHP diferem dos pesos correntes em até "
                    f"`{max_diff:.4f}`. Considera aplicar os pesos AHP para resultados mais coerentes "
                    f"com o método.\n"
                )
        report_lines.append(
            "- ANP, DEMATEL e variantes Fuzzy ANP usam **proxy data-driven** "
            "(correlações entre critérios) para estimar dependências. Numa aplicação "
            "rigorosa, esta matriz seria elicitada directamente do decisor.\n"
            "- Fuzzy AHP/TOPSIS usam **spread fixo** (±10%-20%) como aproximação dos TFN. "
            "Em aplicações reais o decisor define o spread por julgamento.\n"
            "- Todas as escalas ordinais (e.g., 1-5) são tratadas como contínuas para fins de normalização.\n"
        )

        # Junta tudo
        report_md = "\n".join(report_lines)

        # ===== Apresentação na tab =====
        st.subheader(f"🥇 Alternativa recomendada: {top1}")
        st.metric(
            "Convergência inter-modelo (Top-1 no Top-3)",
            f"{conv_pct:.0f}%",
            f"{top1_in_top3} / {len(models_with_results)} modelos"
        )

        st.markdown("---")
        st.markdown(report_md)

        # Download como .md e .txt
        st.markdown("---")
        st.subheader("📥 Descarregar relatório")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Descarregar como Markdown (.md)",
                data=report_md.encode("utf-8"),
                file_name=f"relatorio_mcdm_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
            )
        with c2:
            st.download_button(
                "Descarregar como texto (.txt)",
                data=report_md.encode("utf-8"),
                file_name=f"relatorio_mcdm_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
            )
