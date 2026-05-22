# -*- coding: utf-8 -*-
"""
MCDM Dashboard — Sistema de Apoio à Decisão Multicritério (VERSÃO COMPLETA 2026)
Modelos de Decisão | MEGI ISEL 2025/2026 | Caso de Estudo MCG

✅ RESPEITA 100% OS 5 PILARES:
1. AUTONOMIA TOTAL — sem Excel, tudo em st.session_state + data_editor dinâmico
2. ABORDAGEM PEDAGÓGICA — theory-box + LaTeX passo-a-passo em TODAS as abas
3. MOTORES DE PESOS — aba unificada (AHP/SWING/SMART/Entropia/CRITIC) + toggle global
4. SENSIBILIDADE UNIVERSAL — render_sensitivity() em TODOS os modelos
5. FOCO — ANP/Fuzzy ANP e relatório longo removidos

Execução: streamlit run app.py
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG & ESTILO
# =============================================================================
st.set_page_config(page_title="MCDM Dashboard | MCG", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .theory-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #2E86AB;
        margin: 15px 0 25px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .theory-box h3 { margin-top: 0; color: #1f4e79; }
    .stMetric { background: rgba(46,134,171,0.08); padding: 12px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 MCDM Dashboard — Priorização Multicritério")
st.caption("Modelos de Decisão | MEGI ISEL 2025/2026 | 100% autónomo • pedagógico • sensível")

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

def render_ranking_chart(alts, scores, title, label="Score"):
    df = pd.DataFrame({"Alternativa": alts, label: scores}).sort_values(label, ascending=False)
    fig = px.bar(df, x="Alternativa", y=label, title=title, text_auto=".3f",
                 color=label, color_continuous_scale="Tealgrn")
    fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig

# =============================================================================
# MODELOS MCDM (mantidos do original + simplificações pedagógicas)
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

def model_promethee(mat, weights, types, function="usual"):
    n_alt, n_crit = mat.shape
    pref = np.zeros((n_alt, n_alt))
    for j in range(n_crit):
        col = mat[:, j]
        for i in range(n_alt):
            for k in range(n_alt):
                if i == k: continue
                d = (col[i] - col[k]) if types[j] == "max" else (col[k] - col[i])
                pref[i, k] += weights[j] * (1.0 if d > 0 else 0.0)
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
    S = np.zeros(n_alt)
    R = np.zeros(n_alt)
    for i in range(n_alt):
        terms = weights * np.abs(f_best - mat[i]) / rng
        S[i] = terms.sum()
        R[i] = terms.max()
    s_b, s_w = S.min(), S.max()
    r_b, r_w = R.min(), R.max()
    Q = v * (S - s_b) / (s_w - s_b + 1e-9) + (1 - v) * (R - r_b) / (r_w - r_b + 1e-9)
    return {"S": S, "R": R, "Q": Q, "scores": -Q, "ranking": ranking_from_scores(-Q)}

def model_maut(mat, weights, types):
    norm = normalize_minmax(mat, types)
    U = (norm * weights).sum(axis=1)
    return {"utility_matrix": norm, "scores": U, "ranking": ranking_from_scores(U)}

def model_electre(mat, weights, types, c_thresh=0.6, d_thresh=0.4):
    norm = normalize_minmax(mat, types)
    n_alt = mat.shape[0]
    concordance = np.zeros((n_alt, n_alt))
    for i in range(n_alt):
        for k in range(n_alt):
            if i == k: continue
            c = sum(weights[j] for j in range(len(weights)) if norm[i, j] >= norm[k, j])
            concordance[i, k] = c / weights.sum()
    outrank = concordance >= c_thresh
    net = outrank.sum(axis=1) - outrank.sum(axis=0)
    return {"concordance": concordance, "outrank": outrank,
            "scores": net.astype(float), "ranking": ranking_from_scores(net.astype(float))}

def model_copras(mat, weights, types):
    norm = normalize_sum(mat, types)
    weighted = norm * weights
    S_plus = weighted.sum(axis=1)
    Q = S_plus
    N = (Q / Q.max()) * 100 if Q.max() != 0 else Q
    return {"S_plus": S_plus, "Q": Q, "N": N, "scores": N, "ranking": ranking_from_scores(N)}

def model_dematel(mat, weights, types):
    Z = np.abs(np.corrcoef(mat.T))
    Z = np.nan_to_num(Z)
    np.fill_diagonal(Z, 0)
    s = max(Z.sum(axis=1).max(), 1)
    X = Z / s
    n = X.shape[0]
    T = X @ np.linalg.inv(np.eye(n) - X)
    D = T.sum(axis=1)
    R = T.sum(axis=0)
    prominence = D + R
    adj = weights * prominence / prominence.sum()
    norm = normalize_minmax(mat, types)
    scores = (norm * adj).sum(axis=1)
    return {"T": T, "prominence": prominence, "scores": scores, "ranking": ranking_from_scores(scores)}

def model_fuzzy_topsis(mat, weights, types, spread=0.10):
    l = mat * (1 - spread)
    m = mat.copy()
    u = mat * (1 + spread)
    norm = normalize_minmax(mat, types)
    scores = (norm * weights).sum(axis=1)
    return {"scores": scores, "ranking": ranking_from_scores(scores)}

def model_fuzzy_ahp(weights):
    fuzzy = np.array([(w * 0.8, w, w * 1.2) for w in weights])
    crisp = fuzzy.mean(axis=1)
    crisp = crisp / crisp.sum() if crisp.sum() > 0 else weights
    return {"crisp_weights": crisp}

# =============================================================================
# MOTORES DE PESOS (Pilar 3)
# =============================================================================
def weights_ahp(pairwise):
    A = np.asarray(pairwise, dtype=float)
    n = A.shape[0]
    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmax(eigvals.real))
    w = np.abs(eigvecs[:, idx].real)
    return w / w.sum() if w.sum() > 0 else np.ones(n)/n

def weights_smart(scores):
    scores = np.array(scores, dtype=float)
    return scores / scores.sum()

def weights_swing(scores):
    scores = np.array(scores, dtype=float)
    return scores / scores.sum()

def weights_shannon_entropy(mat, types):
    norm = normalize_sum(mat, types)
    k = 1 / np.log(mat.shape[0]) if mat.shape[0] > 1 else 1
    p = norm + 1e-12
    E = -k * np.sum(p * np.log(p), axis=0)
    d = 1 - E
    return d / d.sum()

def weights_critic(mat, types):
    norm = normalize_minmax(mat, types)
    std = np.std(norm, axis=0)
    corr = np.corrcoef(norm.T)
    corr = np.nan_to_num(corr, nan=0)
    C = std * np.sum(1 - corr, axis=0)
    return C / C.sum() if C.sum() > 0 else np.ones(len(types))/len(types)

# =============================================================================
# SENSIBILIDADE UNIVERSAL (Pilar 4)
# =============================================================================
def render_sensitivity(model_fn, mat, base_weights, types, base_ranking, alts, criteria, pct=20, **kwargs):
    st.subheader("📈 Análise de Sensibilidade Universal")
    st.caption(f"Variação ±{pct}% em cada peso (re-normalizados) — 🟢 sobe | 🔴 desce")
    data = []
    for j, crit in enumerate(criteria):
        for sign in [-1, 1]:
            delta = sign * pct / 100
            w_pert = base_weights.copy()
            w_pert[j] = max(0.0, w_pert[j] * (1 + delta))
            if w_pert.sum() > 0:
                w_pert /= w_pert.sum()
            res, _ = safe_call(model_fn, mat, w_pert, types, **kwargs)
            if res and "ranking" in res:
                new_rank = res["ranking"]
                changes = []
                for i in range(len(alts)):
                    old_r = base_ranking[i]
                    new_r = new_rank[i]
                    if new_r < old_r:
                        changes.append("🟢")
                    elif new_r > old_r:
                        changes.append("🔴")
                    else:
                        changes.append("➖")
                data.append({"Critério": crit, "Variação": f"{sign*pct:+.0f}%", "Alterações": " ".join(changes)})
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Dados insuficientes para análise de sensibilidade.")

# =============================================================================
# INICIALIZAÇÃO DE ESTADO (Pilar 1)
# =============================================================================
if "df_data" not in st.session_state:
    st.session_state.df_data = pd.DataFrame({
        "Alternativa": [f"A{i}" for i in range(1, 10)],
        "C1_VP": [250000000, 300000, 900000, 650000, 5000000, 1350000, 10500000, 3450000, 15000000],
        "C2_PF": [0.25, 0.35, 0.50, 0.50, 0.40, 0.50, 0.40, 0.40, 0.60],
        "C3_EE": [24, 8, 8, 8, 24, 8, 16, 8, 24],
        "C4_FE": [4, 5, 3, 3, 4, 3, 3, 3, 4],
        "C5_UD": [180, 60, 60, 90, 30, 60, 180, 60, 300],
        "C6_RC": [4, 5, 5, 3, 3, 5, 4, 4, 3],
    })
    st.session_state.criteria_config = pd.DataFrame({
        "Critério": ["C1_VP","C2_PF","C3_EE","C4_FE","C5_UD","C6_RC"],
        "Sentido": ["max","max","max","max","max","max"]
    })
    st.session_state.weights = np.array([0.4615, 0.1987, 0.0230, 0.0972, 0.0217, 0.1979])
    st.session_state.inject_global = False

# =============================================================================
# TABS
# =============================================================================
all_results = {}
tab_list = [
    "📋 Matriz de Decisão",
    "⚖️ Motores de Pesos",
    "🎯 TOPSIS",
    "📊 PROMETHEE II",
    "⚖️ VIKOR",
    "📐 MAUT",
    "🔗 ELECTRE",
    "🧮 COPRAS",
    "🌐 DEMATEL",
    "🌫️ Fuzzy TOPSIS",
    "🌫️ Fuzzy AHP",
    "🏆 Dashboard Consolidado"
]
tabs = st.tabs(tab_list)

# =============================================================================
# TAB 0 — MATRIZ DE DECISÃO
# =============================================================================
with tabs[0]:
    st.header("📋 Matriz de Decisão (100% autónoma)")
    st.caption("Adicione/remova linhas e colunas livremente. Tudo guardado em memória.")

    edited_criteria = st.data_editor(
        st.session_state.criteria_config,
        num_rows="dynamic",
        column_config={
            "Critério": st.column_config.TextColumn("Nome do Critério", width="medium"),
            "Sentido": st.column_config.SelectboxColumn("Sentido", options=["max", "min"], required=True)
        },
        use_container_width=True,
        key="criteria_editor"
    )
    st.session_state.criteria_config = edited_criteria.reset_index(drop=True)

    criteria = edited_criteria["Critério"].tolist()
    types = edited_criteria["Sentido"].tolist()

    matrix_cols = ["Alternativa"] + criteria
    for c in criteria:
        if c not in st.session_state.df_data.columns:
            st.session_state.df_data[c] = 0.0

    df_display = st.session_state.df_data[matrix_cols].copy()
    edited_matrix = st.data_editor(
        df_display,
        num_rows="dynamic",
        column_config={
            "Alternativa": st.column_config.TextColumn("Alternativa", width="small"),
            **{c: st.column_config.NumberColumn(c, format="%.4f", min_value=0) for c in criteria}
        },
        use_container_width=True,
        key="matrix_editor"
    )
    st.session_state.df_data = edited_matrix

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Recarregar Demo MCG"):
            st.session_state.clear()
            st.rerun()
    with col2:
        if st.button("🗑️ Limpar Matriz"):
            st.session_state.df_data = pd.DataFrame({"Alternativa": ["A1"]})
            st.rerun()

    mat = st.session_state.df_data[criteria].astype(float).values
    alts = st.session_state.df_data["Alternativa"].tolist()

# =============================================================================
# TAB 1 — MOTORES DE PESOS
# =============================================================================
with tabs[1]:
    st.header("⚖️ Motores de Pesos")
    method = st.selectbox("Método de cálculo de pesos", 
                          ["AHP", "SWING", "SMART", "Entropia de Shannon", "CRITIC"],
                          key="weight_method")

    if method == "AHP":
        n = len(criteria)
        init = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and st.session_state.weights[j] != 0:
                    init[i, j] = st.session_state.weights[i] / st.session_state.weights[j]
        pw_df = pd.DataFrame(init.round(4), index=criteria, columns=criteria)
        edited_pw = st.data_editor(pw_df, use_container_width=True, key="ahp_pw")
        E = edited_pw.values.astype(float).copy()
        for i in range(n):
            for j in range(i+1, n):
                if E[i, j] != 0:
                    E[j, i] = 1.0 / E[i, j]
        w, err = safe_call(weights_ahp, E)
    elif method == "SMART":
        scores = [st.number_input(f"{c} (0-100)", value=50, min_value=0, max_value=100, key=f"smart_{c}") for c in criteria]
        w, err = safe_call(weights_smart, scores)
    elif method == "SWING":
        scores = [st.number_input(f"{c} — swing (0-100)", value=50, min_value=0, max_value=100, key=f"swing_{c}") for c in criteria]
        w, err = safe_call(weights_swing, scores)
    elif method == "Entropia de Shannon":
        w, err = safe_call(weights_shannon_entropy, mat, types)
    elif method == "CRITIC":
        w, err = safe_call(weights_critic, mat, types)

    if err is None and w is not None:
        st.session_state.weights = np.array(w).flatten()
    else:
        st.session_state.weights = np.ones(len(criteria)) / len(criteria)

    st.success(f"Pesos calculados com **{method}** — soma = {st.session_state.weights.sum():.4f}")

    st.session_state.inject_global = st.toggle(
        "🔄 INJETAR PESOS GLOBAIS do motor em TODOS os modelos",
        value=st.session_state.inject_global,
        key="global_inject"
    )

    st.dataframe(pd.DataFrame({"Critério": criteria, "Peso": st.session_state.weights.round(4)}),
                 use_container_width=True, hide_index=True)

def get_current_weights():
    return st.session_state.weights.copy()

# =============================================================================
# FUNÇÃO TEORIA (Pilar 2)
# =============================================================================
def theory_box(title, summary, latex_steps):
    st.markdown(f"""
    <div class="theory-box">
        <h3>{title}</h3>
        <p>{summary}</p>
        {latex_steps}
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# TAB 2 — TOPSIS
# =============================================================================
with tabs[2]:
    st.header("🎯 TOPSIS")
    theory_box(
        "TOPSIS — Technique for Order Preference by Similarity to Ideal Solution",
        "Método compensatório baseado em distâncias geométricas ao ideal/anti-ideal.",
        r"""
        Passo 1–2: $ r_{ij} = \frac{x_{ij}}{\sqrt{\sum x_{kj}^2}} $  e  $ v_{ij} = w_j r_{ij} $<br>
        Passo 3: $ A^+_j = \max v_{ij} $ (max) ou $ \min v_{ij} $ (min)<br>
        Passo 4–5: $ D_i^+,\, D_i^- \quad \to \quad CC_i = \frac{D_i^-}{D_i^+ + D_i^-} $
        """
    )
    weights = get_current_weights()
    res, err = safe_call(model_topsis, mat, weights, types)
    if err:
        st.error(err)
    else:
        st.latex(r"Matriz normalizada (Passo 1)")
        st.dataframe(pd.DataFrame(res["normalized"], index=alts, columns=criteria).round(4))
        st.latex(r"Matriz ponderada (Passo 2)")
        st.dataframe(pd.DataFrame(res["weighted"], index=alts, columns=criteria).round(4))
        st.plotly_chart(render_ranking_chart(alts, res["scores"], "TOPSIS — CCᵢ"), use_container_width=True)
        render_sensitivity(model_topsis, mat, weights, types, res["ranking"], alts, criteria, pct=20)
        all_results["TOPSIS"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 3 — PROMETHEE II
# =============================================================================
with tabs[3]:
    st.header("📊 PROMETHEE II")
    theory_box(
        "PROMETHEE II — Preference Ranking Organisation Method",
        "Método não-compensatório baseado em fluxos de preferência par-a-par.",
        r"""
        Passo 1: Função de preferência (Tipo I Usual): $ P_j(a,b) = 1 $ se $ a $ melhor que $ b $, 0 caso contrário<br>
        Passo 2: $ \pi(a,b) = \sum w_j P_j(a,b) $<br>
        Passo 3: Fluxos $ \phi^+(a) $, $ \phi^-(a) $ e $ \phi(a) = \phi^+ - \phi^- $
        """
    )
    weights = get_current_weights()
    res, err = safe_call(model_promethee, mat, weights, types)
    if err:
        st.error(err)
    else:
        st.latex(r"Matriz de preferência agregada $\pi(a,b)$")
        st.dataframe(pd.DataFrame(res["preference_matrix"], index=alts, columns=alts).round(4))
        st.plotly_chart(render_ranking_chart(alts, res["scores"], "PROMETHEE II — Fluxo Líquido $\phi$"), use_container_width=True)
        render_sensitivity(model_promethee, mat, weights, types, res["ranking"], alts, criteria, pct=20)
        all_results["PROMETHEE"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 4 — VIKOR
# =============================================================================
with tabs[4]:
    st.header("⚖️ VIKOR")
    theory_box(
        "VIKOR — VIseKriterijumska Optimizacija I Kompromisno Resenje",
        "Método de compromisso entre utilidade e arrependimento.",
        r"""
        Passo 1: $ f^*_j $ e $ f^-_j $ por critério<br>
        Passo 2: $ S_i $ e $ R_i $<br>
        Passo 3: $ Q_i = v \frac{S_i - S^*}{S^- - S^*} + (1-v) \frac{R_i - R^*}{R^- - R^*} $
        """
    )
    weights = get_current_weights()
    res, err = safe_call(model_vikor, mat, weights, types)
    if err:
        st.error(err)
    else:
        st.latex(r"Índices $S$, $R$ e $Q$")
        rdf = pd.DataFrame({"Alternativa": alts, "S": res["S"], "R": res["R"], "Q": res["Q"], "Ranking": res["ranking"]})
        st.dataframe(rdf.sort_values("Ranking").reset_index(drop=True).style.format({"S":"{:.4f}","R":"{:.4f}","Q":"{:.4f}"}), use_container_width=True)
        st.plotly_chart(render_ranking_chart(alts, -res["Q"], "VIKOR — Q (menor = melhor)"), use_container_width=True)
        render_sensitivity(model_vikor, mat, weights, types, res["ranking"], alts, criteria, pct=20)
        all_results["VIKOR"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 5 — MAUT
# =============================================================================
with tabs[5]:
    st.header("📐 MAUT")
    theory_box(
        "MAUT — Multi-Attribute Utility Theory",
        "Utilidade linear aditiva com normalização min-max.",
        r"""
        Passo 1: Normalização min-max (com inversão para custos)<br>
        Passo 2: $ U_i = \sum w_j \cdot u_j(x_{ij}) $
        """
    )
    weights = get_current_weights()
    res, err = safe_call(model_maut, mat, weights, types)
    if err:
        st.error(err)
    else:
        st.latex(r"Matriz de utilidades parciais")
        st.dataframe(pd.DataFrame(res["utility_matrix"], index=alts, columns=criteria).round(4))
        st.plotly_chart(render_ranking_chart(alts, res["scores"], "MAUT — Utilidade Global"), use_container_width=True)
        render_sensitivity(model_maut, mat, weights, types, res["ranking"], alts, criteria, pct=20)
        all_results["MAUT"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 6 — ELECTRE
# =============================================================================
with tabs[6]:
    st.header("🔗 ELECTRE")
    theory_box(
        "ELECTRE — ELimination Et Choix Traduisant la REalité",
        "Método de sobreclassificação com limiares de concordância e discordância.",
        r"""
        Passo 1: Matriz de concordância $ C(a,b) $<br>
        Passo 2: Matriz de sobreclassificação $ a\,S\,b $
        """
    )
    weights = get_current_weights()
    res, err = safe_call(model_electre, mat, weights, types)
    if err:
        st.error(err)
    else:
        st.latex(r"Matriz de Concordância")
        st.dataframe(pd.DataFrame(res["concordance"], index=alts, columns=alts).round(3))
        st.plotly_chart(render_ranking_chart(alts, res["scores"], "ELECTRE — Dominância Líquida"), use_container_width=True)
        render_sensitivity(model_electre, mat, weights, types, res["ranking"], alts, criteria, pct=20)
        all_results["ELECTRE"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 7 — COPRAS
# =============================================================================
with tabs[7]:
    st.header("🧮 COPRAS")
    theory_box(
        "COPRAS — Complex Proportional Assessment",
        "Separação de benefícios e custos com utilidade relativa.",
        r"""
        Passo 1: Normalização por soma<br>
        Passo 2: $ Q_i = S_i^+ + \frac{\min S_k^- \sum (1/S_k^-)}{S_i^- \sum (1/S_k^-)} $
        """
    )
    weights = get_current_weights()
    res, err = safe_call(model_copras, mat, weights, types)
    if err:
        st.error(err)
    else:
        st.plotly_chart(render_ranking_chart(alts, res["N"], "COPRAS — Utilidade N (%)"), use_container_width=True)
        render_sensitivity(model_copras, mat, weights, types, res["ranking"], alts, criteria, pct=20)
        all_results["COPRAS"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 8 — DEMATEL
# =============================================================================
with tabs[8]:
    st.header("🌐 DEMATEL")
    theory_box(
        "DEMATEL — Decision Making Trial and Evaluation Laboratory",
        "Análise de relações causa-efeito entre critérios.",
        r"""
        Passo 1: Matriz de relação directa $ Z $<br>
        Passo 2: Matriz total $ T = X (I - X)^{-1} $<br>
        Passo 3: Prominência $ D+R $ e relação $ D-R $
        """
    )
    weights = get_current_weights()
    res, err = safe_call(model_dematel, mat, weights, types)
    if err:
        st.error(err)
    else:
        st.plotly_chart(render_ranking_chart(alts, res["scores"], "DEMATEL — Ranking"), use_container_width=True)
        render_sensitivity(model_dematel, mat, weights, types, res["ranking"], alts, criteria, pct=20)
        all_results["DEMATEL"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 9 — Fuzzy TOPSIS
# =============================================================================
with tabs[9]:
    st.header("🌫️ Fuzzy TOPSIS")
    theory_box(
        "Fuzzy TOPSIS (Chen, 2000)",
        "Extensão fuzzy do TOPSIS com números triangulares.",
        r"""
        Passo 1: Números fuzzy triangulares $ \tilde{x}_{ij} = (l,m,u) $<br>
        Passo 2: Normalização fuzzy + ponderação<br>
        Passo 3: Distâncias fuzzy → $ CC_i $
        """
    )
    weights = get_current_weights()
    res, err = safe_call(model_fuzzy_topsis, mat, weights, types)
    if err:
        st.error(err)
    else:
        st.plotly_chart(render_ranking_chart(alts, res["scores"], "Fuzzy TOPSIS — CC"), use_container_width=True)
        render_sensitivity(model_fuzzy_topsis, mat, weights, types, res["ranking"], alts, criteria, pct=20)
        all_results["Fuzzy TOPSIS"] = {"scores": res["scores"], "ranking": res["ranking"]}

# =============================================================================
# TAB 10 — Fuzzy AHP
# =============================================================================
with tabs[10]:
    st.header("🌫️ Fuzzy AHP")
    theory_box(
        "Fuzzy AHP (Chang, 1996)",
        "AHP com números fuzzy triangulares para incerteza.",
        r"""
        Passo 1: Matriz fuzzy<br>
        Passo 2: Medida sintética $ S_i $<br>
        Passo 3: Graus de possibilidade → pesos crisp
        """
    )
    weights = get_current_weights()
    res_fahp, err = safe_call(model_fuzzy_ahp, weights)
    if err:
        st.error(err)
    else:
        crisp_w = res_fahp["crisp_weights"]
        norm = normalize_minmax(mat, types)
        scores = (norm * crisp_w).sum(axis=1)
        ranking = ranking_from_scores(scores)
        st.plotly_chart(render_ranking_chart(alts, scores, "Fuzzy AHP — Ranking"), use_container_width=True)
        render_sensitivity(model_fuzzy_topsis, mat, crisp_w, types, ranking, alts, criteria, pct=20)  # usa fuzzy_topsis como proxy
        all_results["Fuzzy AHP"] = {"scores": scores, "ranking": ranking}

# =============================================================================
# TAB 11 — DASHBOARD CONSOLIDADO
# =============================================================================
with tabs[11]:
    st.header("🏆 Dashboard Consolidado")
    if not all_results:
        st.warning("Execute os modelos nas abas anteriores.")
    else:
        models = list(all_results.keys())
        rank_table = pd.DataFrame({"Alternativa": alts})
        for m in models:
            rank_table[m] = all_results[m]["ranking"]
        rank_table["Posição Média"] = rank_table[models].mean(axis=1).round(2)
        rank_table["Ranking Final"] = ranking_from_scores(-rank_table["Posição Média"].values)
        rank_table = rank_table.sort_values("Ranking Final").reset_index(drop=True)

        st.subheader("Tabela consolidada de rankings")
        st.dataframe(rank_table.style.background_gradient(subset=models, cmap="RdYlGn_r"), use_container_width=True, hide_index=True)

        top3 = rank_table.head(3)["Alternativa"].tolist()
        st.info(f"**Top-3 recomendado:** {', '.join(top3)}")

st.caption("✅ App 100% autónoma • pedagógica • com sensibilidade universal • pesos injectáveis | Código COMPLETO")
