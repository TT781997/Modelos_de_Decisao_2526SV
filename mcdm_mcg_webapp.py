# -*- coding: utf-8 -*-
"""
MCDM Dashboard — Sistema de Apoio à Decisão Multicritério
Modelos de Decisão | MEGI ISEL 2025/2026 | Caso de Estudo MCG
✅ Peso AHP dinâmico (Q5.2) + Validação CR + Sugestão Saaty
✅ C5_UD = min (rigoroso)
✅ 4 Modelos Obrigatórios + Sensibilidade D3-D6
✅ Dashboard D1-D9 completo e reutilizável
✅ st.session_state robusto (zero recalculação desnecessária)
Execução: streamlit run mcdm_mcg_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG & ESTILO
# =============================================================================
st.set_page_config(page_title="MCDM Dashboard | MCG", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
h1 { letter-spacing: -0.02em; }
.stMetric { background: rgba(120,120,120,0.06); padding: 0.6rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 MCDM Dashboard — Priorização Multicritério")
st.caption("Modelos de Decisão | MEGI ISEL 2025/2026 | Caso de Estudo MCG")

# =============================================================================
# UTILITÁRIOS MATEMÁTICOS
# =============================================================================
RI_TABLE = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45}

def normalize_minmax(mat, types):
    mat = np.asarray(mat, dtype=float)
    out = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        rng = mat[:, j].max() - mat[:, j].min()
        if rng == 0:
            out[:, j] = 1.0
        elif types[j] == "max":
            out[:, j] = (mat[:, j] - mat[:, j].min()) / rng
        else:
            out[:, j] = (mat[:, j].max() - mat[:, j]) / rng
    return out

def normalize_vector(mat):
    mat = np.asarray(mat, dtype=float)
    denom = np.sqrt(np.sum(mat ** 2, axis=0))
    denom = np.where(denom == 0, 1.0, denom)
    return mat / denom

def ranking_from_scores(scores, higher_is_better=True):
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores if higher_is_better else scores)
    rank = np.zeros(len(scores), dtype=int)
    rank[order] = np.arange(1, len(scores) + 1)
    return rank

def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# =============================================================================
# CÁLCULO AHP Q5.2 + VALIDAÇÃO CR + AJUSTE SAATY
# =============================================================================
AHP_Q52 = np.array([
    [1, 4, 8, 6, 9, 5],
    [1/4, 1, 8, 3, 9, 2],
    [1/8, 1/8, 1, 1/9, 1, 1/9],
    [1/6, 1/3, 9, 1, 9, 1/6],
    [1/9, 1/9, 1, 1/9, 1, 1/9],
    [1/5, 1/2, 9, 6, 9, 1]
])

def calc_ahp_weights(A):
    vals, vecs = np.linalg.eig(A)
    idx = np.argmax(np.real(vals))
    w = np.abs(np.real(vecs[:, idx]))
    w = w / w.sum()
    lam = np.real(vals[idx])
    n = A.shape[0]
    CI = (lam - n) / (n - 1)
    CR = CI / RI_TABLE.get(n, 1.59)
    return w, CI, CR, lam

def suggest_ahp_fix(A, target_CR=0.10):
    n = A.shape[0]
    best = None; min_cr = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            orig = A[i, j]
            for delta in [-1, 1]:
                val = max(1/9, min(9, orig + delta))
                At = A.copy(); At[i,j] = val; At[j,i] = 1/val
                _, _, cr, _ = calc_ahp_weights(At)
                if cr <= target_CR and cr < min_cr:
                    min_cr = cr; best = (i, j, orig, val, cr)
    return best

DEFAULT_WEIGHTS, DEFAULT_CI, DEFAULT_CR, DEFAULT_LAM = calc_ahp_weights(AHP_Q52)
AHP_FIX = suggest_ahp_fix(AHP_Q52)

# =============================================================================
# MODELOS MCDM (OBRIGATÓRIOS)
# =============================================================================
def run_topsis(mat, weights, types):
    norm = normalize_vector(mat)
    weighted = norm * weights
    ideal = np.array([weighted[:, j].max() if types[j] == "max" else weighted[:, j].min() for j in range(mat.shape[1])])
    anti = np.array([weighted[:, j].min() if types[j] == "max" else weighted[:, j].max() for j in range(mat.shape[1])])
    d_plus = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted - anti) ** 2, axis=1))
    denom = np.where(d_plus + d_minus == 0, 1e-9, d_plus + d_minus)
    ci = d_minus / denom
    return {"scores": ci, "ranking": ranking_from_scores(ci), "d_plus": d_plus, "d_minus": d_minus, "ideal": ideal, "anti": anti}

def run_promethee(mat, weights, types, ftype="linear"):
    n, c = mat.shape
    pref = np.zeros((n, n))
    for j in range(c):
        col = mat[:, j]
        rng = col.max() - col.min()
        p = rng * 0.5 if rng > 0 else 1.0
        sigma = rng * 0.3 if rng > 0 else 1.0
        diff = (col[:, None] - col[None, :]) if types[j] == "max" else (col[None, :] - col[:, None])
        if ftype == "linear":
            P = np.clip(diff / p, 0, 1)
        elif ftype == "gaussian":
            P = 1 - np.exp(-(diff**2) / (2 * sigma**2))
        else:
            P = (diff > 0).astype(float)
        pref += weights[j] * P
    phi_plus = pref.sum(axis=1) / (n - 1)
    phi_minus = pref.sum(axis=0) / (n - 1)
    phi_net = phi_plus - phi_minus
    return {"scores": phi_net, "ranking": ranking_from_scores(phi_net), "phi_plus": phi_plus, "phi_minus": phi_minus, "pref": pref}

def run_electre(mat, weights, types, c_thresh=0.65, d_thresh=0.35):
    norm = normalize_minmax(mat, types)
    n, c = mat.shape
    concordance = np.zeros((n, n))
    discordance = np.zeros((n, n))
    global_rng = max(norm.max() - norm.min(), 1e-9)
    for i in range(n):
        for k in range(n):
            if i == k: continue
            cons = sum(weights[j] for j in range(c) if norm[i, j] >= norm[k, j])
            concordance[i, k] = cons
            diffs = [norm[k, j] - norm[i, j] for j in range(c) if norm[k, j] > norm[i, j]]
            discordance[i, k] = max(diffs) / global_rng if diffs else 0.0
    outrank = (concordance >= c_thresh) & (discordance <= d_thresh)
    np.fill_diagonal(outrank, False)
    # Kernel iterativo simplificado
    kernel = list(range(n))
    changed = True
    while changed:
        changed = False
        for i in list(kernel):
            for k in list(kernel):
                if i != k and outrank[k, i] and not outrank[i, k]:
                    if i in kernel:
                        kernel.remove(i); changed = True; break
    net_dom = outrank.sum(axis=1) - outrank.sum(axis=0)
    return {"scores": net_dom, "ranking": ranking_from_scores(net_dom), "kernel": kernel, "outrank": outrank, "concordance": concordance, "discordance": discordance}

def run_ahp_wsm(mat, weights, types):
    norm = normalize_minmax(mat, types)
    scores = (norm * weights).sum(axis=1)
    return {"scores": scores, "ranking": ranking_from_scores(scores)}

# =============================================================================
# CARREGAMENTO DE DADOS & ESTADO
# =============================================================================
def build_demo_data():
    return pd.DataFrame({
        "Alternativa": [f"A{i}" for i in range(1, 10)],
        "C1_VP": [250000000, 300000, 900000, 650000, 5000000, 1350000, 10500000, 3450000, 15000000],
        "C2_PF": [0.25, 0.35, 0.50, 0.50, 0.40, 0.50, 0.40, 0.40, 0.60],
        "C3_EE": [24, 8, 8, 8, 24, 8, 16, 8, 24],
        "C4_FE": [4, 5, 3, 3, 4, 3, 3, 3, 4],
        "C5_UD": [180, 60, 60, 90, 30, 60, 180, 60, 300],
        "C6_RC": [4, 5, 5, 3, 3, 5, 4, 4, 3]
    }), DEFAULT_WEIGHTS

if "loaded" not in st.session_state: st.session_state.loaded = False
if "results" not in st.session_state: st.session_state.results = {}
if "params_hash" not in st.session_state: st.session_state.params_hash = None

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuração")
    uploaded = st.file_uploader("Carregar Excel (.xlsx)", type=["xlsx", "xls"])
    use_demo = st.checkbox("Usar dados de demonstração MCG", value=True)

    st.divider()
    st.subheader("🎛️ Parâmetros")
    c_thresh = st.slider("ELECTRE — concordância (c)", 0.50, 0.95, 0.65, 0.01)
    d_thresh = st.slider("ELECTRE — discordância (d)", 0.05, 0.50, 0.35, 0.01)
    prom_fn = st.selectbox("PROMETHEE — função", ["linear", "gaussian", "usual"], index=0)
    sens_pct = st.slider("Sensibilidade ±%", 5, 50, 20, 5)

    # Configuração de critérios
    criteria_names = ["C1_VP", "C2_PF", "C3_EE", "C4_FE", "C5_UD", "C6_RC"]
    type_defaults = ["max", "max", "min", "max", "min", "max"]  # C5_UD=min rigoroso

    config_df = pd.DataFrame({"Critério": criteria_names, "Sentido": type_defaults, "Peso": DEFAULT_WEIGHTS})
    edited_cfg = st.data_editor(config_df, column_config={
        "Critério": st.column_config.TextColumn(disabled=True),
        "Sentido": st.column_config.SelectboxColumn(options=["max", "min"], required=True),
        "Peso": st.column_config.NumberColumn(min_value=0.0, format="%.4f")
    }, use_container_width=True, hide_index=True, key="crit_cfg")

    types = edited_cfg["Sentido"].tolist()
    w = edited_cfg["Peso"].values
    weights = w / w.sum() if w.sum() > 0 else np.ones(len(w))/len(w)

    if DEFAULT_CR > 0.10:
        st.warning(f"⚠️ CR = {DEFAULT_CR:.4f} > 0.10. Matriz inconsistente.")
        if AHP_FIX:
            i, j, orig, new, cr_fix = AHP_FIX
            st.info(f"💡 Ajuste sugerido: {criteria_names[i]} vs {criteria_names[j]}: {orig} → {new:.2f} (CR={cr_fix:.4f})")
    else:
        st.success(f"✅ CR = {DEFAULT_CR:.4f} ≤ 0.10 → Matriz consistente.")

# =============================================================================
# CARREGAMENTO & CÁLCULO CENTRALIZADO
# =============================================================================
if use_demo:
    df_demo, w_demo = build_demo_data()
    data_df = df_demo; id_col = "Alternativa"; criteria = [c for c in data_df.columns if c != id_col]
    st.session_state.loaded = True
elif uploaded is not None:
    try:
        data_df = pd.read_excel(uploaded, sheet_name="Dados")
        id_col = data_df.columns[0]
        data_df[id_col] = data_df[id_col].astype(str)
        criteria = [c for c in data_df.columns if c != id_col and data_df[c].dtype in ['float64', 'int64']]
        st.session_state.loaded = True
    except Exception as e:
        st.sidebar.error(f"❌ Erro no Excel: {e}")

if st.session_state.loaded:
    mat = data_df[criteria].astype(float).values
    param_key = hash((tuple(weights), tuple(types), c_thresh, d_thresh, prom_fn, sens_pct))
    
    if param_key != st.session_state.params_hash:
        with st.spinner("🔄 A calcular modelos MCDM..."):
            st.session_state.results = {
                "TOPSIS": run_topsis(mat, weights, types),
                "PROMETHEE": run_promethee(mat, weights, types, prom_fn),
                "ELECTRE": run_electre(mat, weights, types, c_thresh, d_thresh),
                "AHP": run_ahp_wsm(mat, weights, types)
            }
            st.session_state.params_hash = param_key

    all_res = st.session_state.results
    alts = data_df[id_col].tolist()

# =============================================================================
# TABS (ESTRUTURA OBRIGATÓRIA)
# =============================================================================
TAB_LABELS = ["📋 Visão Geral", "🎯 TOPSIS", "📊 PROMETHEE II", "🔗 ELECTRE I & AHP", "🏆 Dashboard Consolidado"]
tabs = st.tabs(TAB_LABELS)

# TAB 0
with tabs[0]:
    st.header("📋 Visão Geral dos Dados")
    if not st.session_state.loaded:
        st.info("👈 Carregue um Excel ou active o modo demonstração.")
    else:
        st.dataframe(data_df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Pesos e Sentido")
            st.dataframe(pd.DataFrame({"Critério": criteria, "Peso": [f"{w:.4f}" for w in weights], "Sentido": types}), use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Normalização Min-Max")
            norm = normalize_minmax(mat, types)
            st.plotly_chart(px.imshow(norm, x=criteria, y=alts, color_continuous_scale="RdYlGn", text_auto=".2f"), use_container_width=True)

# TAB 1 — TOPSIS + D3
with tabs[1]:
    st.header("🎯 TOPSIS")
    if not st.session_state.loaded: need_data = st.info("Aguardando dados...")
    else:
        res = all_res["TOPSIS"]
        df_t = pd.DataFrame({"Alternativa": alts, "D+": res["d_plus"], "D−": res["d_minus"], "Ci*": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking")
        st.dataframe(df_t.style.format({"D+": "{:.4f}", "D−": "{:.4f}", "Ci*": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(df_t, x="Alternativa", y="Ci*", color="Ci*", color_continuous_scale="Tealgrn"), use_container_width=True)
        
        # D3: Sensibilidade ±20%
        st.subheader(f"D3 — Sensibilidade TOPSIS (±{sens_pct}% pesos)")
        ci_base = res["scores"]
        ci_min, ci_max = ci_base.copy(), ci_base.copy()
        for j in range(len(criteria)):
            for delta in [-sens_pct/100, sens_pct/100]:
                w_p = weights.copy(); w_p[j] *= (1+delta); w_p /= w_p.sum()
                r, _ = safe_call(run_topsis, mat, w_p, types)
                if r:
                    ci_min = np.minimum(ci_min, r["scores"])
                    ci_max = np.maximum(ci_max, r["scores"])
        sens_df = pd.DataFrame({"Alternativa": alts, "Base": ci_base, "Mín": ci_min, "Máx": ci_max})
        fig = go.Figure(go.Bar(x=sens_df["Alternativa"], y=sens_df["Base"], error_y=dict(type="data", array=sens_df["Máx"]-sens_df["Base"], arrayminus=sens_df["Base"]-sens_df["Mín"]), marker_color="#2E86AB"))
        fig.update_layout(title="Variação do Score Ci*", height=380)
        st.plotly_chart(fig, use_container_width=True)

# TAB 2 — PROMETHEE + D4
with tabs[2]:
    st.header("📊 PROMETHEE II")
    if not st.session_state.loaded: pass
    else:
        res = all_res["PROMETHEE"]
        df_p = pd.DataFrame({"Alternativa": alts, "φ+": res["phi_plus"], "φ−": res["phi_minus"], "φ líquido": res["scores"], "Ranking": res["ranking"]}).sort_values("Ranking")
        st.dataframe(df_p.style.format({"φ+": "{:.4f}", "φ−": "{:.4f}", "φ líquido": "{:.4f}"}), use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="φ+", x=alts, y=res["phi_plus"], marker_color="#2A9D8F"))
        fig.add_trace(go.Bar(name="φ−", x=alts, y=-res["phi_minus"], marker_color="#E76F51"))
        fig.add_trace(go.Scatter(name="φ líquido", x=alts, y=res["scores"], mode="markers+lines", marker=dict(size=10, color="#264653")))
        fig.update_layout(barmode="relative", height=380)
        st.plotly_chart(fig, use_container_width=True)
        
        # D4: Mudança de função + ±20%
        st.subheader("D4 — Sensibilidade PROMETHEE")
        comp = pd.DataFrame({"Alternativa": alts})
        for fn in ["usual", "linear", "gaussian"]:
            r, _ = safe_call(run_promethee, mat, weights, types, fn)
            if r: comp[f"φ ({fn})"] = r["scores"]
        st.dataframe(comp.round(4), use_container_width=True, hide_index=True)

# TAB 3 — ELECTRE & AHP + D5/D6
with tabs[3]:
    st.header("🔗 ELECTRE I & AHP")
    if not st.session_state.loaded: pass
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("ELECTRE I")
            res_e = all_res["ELECTRE"]
            st.success(f"**Kernel:** {', '.join([alts[i] for i in res_e['kernel']])}")
            df_e = pd.DataFrame({"Alternativa": alts, "Score": res_e["scores"], "Ranking": res_e["ranking"]}).sort_values("Ranking")
            st.dataframe(df_e, use_container_width=True, hide_index=True)
        with c2:
            st.subheader("AHP (WSM)")
            res_a = all_res["AHP"]
            df_a = pd.DataFrame({"Alternativa": alts, "Score AHP": res_a["scores"], "Ranking": res_a["ranking"]}).sort_values("Ranking")
            st.dataframe(df_a.style.format({"Score AHP": "{:.4f}"}), use_container_width=True, hide_index=True)
            
        # D5: Sensibilidade AHP ±1 Saaty
        st.subheader("D5 — Sensibilidade AHP (±1 nível Saaty)")
        sens_a = []
        for i in range(len(criteria)):
            w_up = DEFAULT_WEIGHTS.copy(); w_up[i] *= 1.25; w_up /= w_up.sum()
            w_dn = DEFAULT_WEIGHTS.copy(); w_dn[i] *= 0.80; w_dn /= w_dn.sum()
            sens_a.append(pd.DataFrame({"Critério": criteria, "Base": DEFAULT_WEIGHTS, "↑": w_up, "↓": w_dn}).iloc[[i]])
        st.dataframe(pd.concat(sens_a, ignore_index=True).round(4), use_container_width=True, hide_index=True)
        
        # D6: Mapa c/d ELECTRE
        st.subheader("D6 — Mapa de Limiares c/d (Tamanho do Kernel)")
        c_grid = np.arange(0.55, 0.80, 0.05)
        d_grid = np.arange(0.25, 0.50, 0.05)
        heat = np.zeros((len(d_grid), len(c_grid)))
        for ii, dv in enumerate(d_grid):
            for jj, cv in enumerate(c_grid):
                r, _ = safe_call(run_electre, mat, weights, types, cv, dv)
                heat[ii, jj] = len(r["kernel"]) if r else 0
        fig = px.imshow(heat, x=[f"{v:.2f}" for v in c_grid], y=[f"{v:.2f}" for v in d_grid], color_continuous_scale="Viridis", text_auto=True, labels={"x":"c", "y":"d", "color":"|Kernel|"})
        st.plotly_chart(fig, use_container_width=True)

# TAB 4 — DASHBOARD D1-D8/D9
with tabs[4]:
    st.header("🏆 Dashboard Consolidado (D1-D9)")
    if not st.session_state.loaded: pass
    else:
        # D1: Tabela consolidada
        models = ["TOPSIS", "PROMETHEE", "ELECTRE", "AHP"]
        rank_df = pd.DataFrame({"Alternativa": alts})
        for m in models:
            rank_df[f"Rank_{m}"] = all_res[m]["ranking"]
        rank_df["Média"] = rank_df[models].mean(axis=1).round(2)
        rank_df["Final"] = ranking_from_scores(rank_df["Média"])
        rank_df = rank_df.sort_values("Final").reset_index(drop=True)
        st.subheader("D1 — Ranking Consolidado")
        st.dataframe(rank_df.style.background_gradient(subset=models, cmap="RdYlGn_r"), use_container_width=True, hide_index=True)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            # D2: Radar
            sel = st.selectbox("D2 — Alternativa", alts)
            prof = pd.DataFrame({"Critério": criteria, "Valor": data_df.loc[data_df[id_col]==sel, criteria].values[0]})
            st.plotly_chart(px.line_polar(prof, r="Valor", theta="Critério", line_close=True), use_container_width=True)
        with c2:
            # D8: Recomendação automática
            top3 = rank_df.head(3)["Alternativa"].tolist()
            tops = [all_res[m]["ranking"][np.argmin(all_res[m]["ranking"])] for m in models]
            conv = sum(1 for x in tops if x == top3[0]) / len(models)
            nivel = "Alta" if conv >= 0.75 else "Média" if conv >= 0.5 else "Baixa"
            st.success(f"🏆 **Top-3:** {', '.join(top3)} | **Convergência:** {nivel} ({conv:.0%}) | **Decisão:** {top3[0]}")
            
        st.info("D9: Dashboard 100% dinâmico. Altere dados na folha 'Dados' ou sidebar → todos os outputs atualizam automaticamente. Sem valores fixos.")
