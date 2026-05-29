"""
MCDM Dashboard v2 — Ferramenta de Apoio à Decisão Multicritério
================================================================
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import re

# =============================================================================
# HELPER GLOBAL: limpeza robusta de strings numéricas
# =============================================================================
def clean_number_string(s):
    if pd.isna(s): return None
    s = str(s).strip()
    if not s or s.lower() in ("nan", "none", "-", "—", "n/a", "na"): return None
    for ch in ['€', '$', '£', '¥', '%', '\u20ac', '\u00a3', '\u00a5']: s = s.replace(ch, '')
    s = re.sub(r'\bR\$\s*', '', s)
    s = s.replace(' ', '').replace('\xa0', '').replace('\u202f', '').replace('\u2009', '')
    if not s: return None
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'): s = s.replace('.', '').replace(',', '.')
        else: s = s.replace(',', '')
    elif ',' in s:
        parts = s.split(',')
        if len(parts) > 2 and all(len(p) == 3 and p.isdigit() for p in parts[1:]): s = s.replace(',', '')
        else: s = s.replace(',', '.')
    elif '.' in s:
        parts = s.split('.')
        if len(parts) > 2 and all(len(p) == 3 and p.isdigit() for p in parts[1:]): s = s.replace('.', '')
    return s

def clean_numeric_column(series):
    cleaned = series.astype(str).apply(clean_number_string)
    nums = pd.to_numeric(cleaned, errors="coerce")
    original_empty = series.isna() | (series.astype(str).str.strip() == "")
    n_failed = int((nums.isna() & ~original_empty).sum())
    return nums, n_failed

# =============================================================================
# CONFIGURAÇÃO DE PÁGINA E CSS
# =============================================================================
st.set_page_config(page_title="MCDM Dashboard v2", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded") # Sidebar reposta para expanded

CSS = """
<style>
.theory-box { background: linear-gradient(135deg, #f0f7ff 0%, #e0ecff 100%) !important; border-left: 4px solid #1F4E78; padding: 16px 20px; border-radius: 6px; margin: 12px 0 20px 0; font-size: 14px; line-height: 1.5; color: #1F4E78 !important; }
.theory-box * { color: #1F4E78 !important; }
.purpose-box { background: #e8f5e9 !important; padding: 12px 18px; border-left: 4px solid #2e7d32; border-radius: 4px; margin-bottom: 16px; color: #1b5e20 !important; font-size: 15px; }
.step-header { background: #2E75B6 !important; color: white !important; padding: 8px 14px; border-radius: 4px; font-weight: 600; margin: 16px 0 8px 0; font-size: 15px; }
.sensitivity-box { background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%) !important; border: 2px solid #f57c00; padding: 18px; border-radius: 8px; margin: 24px 0 12px 0; color: #5d3a00 !important; }
.injection-active { background: #fce4ec !important; border: 2px solid #c2185b; padding: 8px 14px; border-radius: 6px; color: #c2185b !important; font-weight: 700; text-align: center; margin: 12px 0; }
.result-box { background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%) !important; border: 2px solid #2e7d32; padding: 18px; border-radius: 8px; margin: 16px 0; color: #1b5e20 !important; font-size: 16px; font-weight: 600; }
.warning-box { background: #fff3e0 !important; border-left: 3px solid #ef6c00; padding: 10px 16px; border-radius: 4px; margin: 8px 0; color: #e65100 !important; }
.data-section { background: #fafafa !important; padding: 18px 20px; border-radius: 8px; border-left: 4px solid #1F4E78; margin: 18px 0 10px 0; }
.data-section h3 { color: #1F4E78 !important; margin-top: 0; margin-bottom: 8px; font-size: 18px; }
.stTabs [data-baseweb="tab"] { padding: 10px 18px; font-weight: 600; }
.stTabs [aria-selected="true"] { background-color: #1F4E78 !important; color: white !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# ESTADO INICIAL
# =============================================================================
def init_state():
    if "criteria_df" not in st.session_state: st.session_state.criteria_df = pd.DataFrame({"Critério": [], "Tipo": [], "Peso Manual": []})
    if "matrix_df" not in st.session_state: st.session_state.matrix_df = pd.DataFrame({"Alternativa": []})
    if "engine_weights" not in st.session_state: st.session_state.engine_weights = {}
    if "sensitivity_pct" not in st.session_state: st.session_state.sensitivity_pct = 20 # Default 20% para cumprir Req D3
    if "ahp_matrix_pasted" not in st.session_state: st.session_state.ahp_matrix_pasted = None
    if "all_results" not in st.session_state: st.session_state.all_results = {}
    if "success_message" not in st.session_state: st.session_state.success_message = None
init_state()

# =============================================================================
# HELPERS GERAIS
# =============================================================================
def get_decision_matrix():
    crit_df = st.session_state.criteria_df.copy().dropna(subset=["Critério"])
    crit_df = crit_df[crit_df["Critério"].astype(str).str.strip() != ""]
    crits = crit_df["Critério"].astype(str).tolist()
    types = crit_df["Tipo"].fillna("max").astype(str).tolist()

    m_df = st.session_state.matrix_df.copy().dropna(subset=["Alternativa"])
    m_df = m_df[m_df["Alternativa"].astype(str).str.strip() != ""]
    alts = m_df["Alternativa"].astype(str).tolist()

    matrix = []
    for crit in crits:
        if crit not in m_df.columns: matrix.append([0.0] * len(alts))
        else: matrix.append(pd.to_numeric(m_df[crit], errors="coerce").fillna(0.0).values)
    matrix = np.array(matrix).T if matrix else np.zeros((0, 0))
    return matrix, alts, crits, types

def get_active_weights():
    _, _, crits, _ = get_decision_matrix()
    n = len(crits)
    if n == 0: return np.array([])
    if "AHP" in st.session_state.engine_weights and len(st.session_state.engine_weights["AHP"]) == n:
        w = np.array(st.session_state.engine_weights["AHP"], dtype=float)
        return w / w.sum() if w.sum() > 0 else np.ones(n) / n
    return np.ones(n) / n

def show_active_weights_banner():
    w = get_active_weights(); _, _, crits, _ = get_decision_matrix()
    if "AHP" in st.session_state.engine_weights and len(st.session_state.engine_weights["AHP"]) == len(crits):
        st.markdown('<div class="injection-active">🔌 Motor activo — <b>AHP</b> (pesos da Matriz Par-a-Par)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ AHP ainda não calculado. Vá à aba 📋 Dados para processar a matriz.</div>', unsafe_allow_html=True)
    cols = st.columns([3, 1])
    with cols[0]: st.dataframe(pd.DataFrame({"Critério": crits, "Peso (%)": [f"{x*100:.2f}%" for x in w]}), hide_index=True)
    with cols[1]: st.metric("Σ pesos", f"{w.sum():.5f}")

def theory_box(title, html): st.markdown(f'<div class="theory-box"><h4>📚 {title}</h4>{html}</div>', unsafe_allow_html=True)
def purpose_box(text): st.markdown(f'<div class="purpose-box"><b>📌 Para que serve esta aba:</b> {text}</div>', unsafe_allow_html=True)
def step_header(text): st.markdown(f'<div class="step-header">{text}</div>', unsafe_allow_html=True)

def check_valid_input():
    matrix, alts, crits, types = get_decision_matrix()
    if len(alts) < 2 or len(crits) < 2 or matrix.size == 0 or np.all(matrix == 0):
        st.warning("Sem dados para apresentar. Por favor preencha e processe a aba '📋 Dados'.")
        return False
    return True

# =============================================================================
# CÁLCULOS DOS MODELOS (AHP, TOPSIS, PROMETHEE II, COPRAS)
# =============================================================================
def normalize_vector(m):
    d = np.sqrt((m ** 2).sum(axis=0))
    return m / np.where(d == 0, 1, d)

def normalize_minmax(m, t):
    out = np.zeros_like(m, dtype=float)
    for j in range(m.shape[1]):
        col = m[:, j]; mn, mx = col.min(), col.max()
        if mx == mn: out[:, j] = 0.5
        elif t[j] == "max": out[:, j] = (col - mn) / (mx - mn)
        else: out[:, j] = (mx - col) / (mx - mn)
    return out

def normalize_sum(m):
    s = m.sum(axis=0)
    return m / np.where(s == 0, 1, s)

def calc_ahp_aggregate(matrix, types, weights):
    return (normalize_minmax(matrix, types) * weights).sum(axis=1)

def calc_topsis(matrix, types, weights):
    n = len(types)
    V = normalize_vector(matrix) * weights
    Ap = np.array([V[:, j].max() if types[j] == "max" else V[:, j].min() for j in range(n)])
    An = np.array([V[:, j].min() if types[j] == "max" else V[:, j].max() for j in range(n)])
    Dp = np.sqrt(((V - Ap) ** 2).sum(axis=1)); Dn = np.sqrt(((V - An) ** 2).sum(axis=1))
    return Dn / np.where(Dp + Dn == 0, 1e-9, Dp + Dn)

def calc_promethee2(matrix, types, weights):
    m_rows, n_cols = matrix.shape
    pi = np.zeros((m_rows, m_rows))
    for a in range(m_rows):
        for b in range(m_rows):
            if a == b: continue
            for j in range(n_cols):
                d = matrix[a, j] - matrix[b, j] if types[j] == "max" else matrix[b, j] - matrix[a, j]
                if d > 0: pi[a, b] += weights[j]
    denom = max(m_rows - 1, 1)
    return (pi.sum(axis=1) / denom) - (pi.sum(axis=0) / denom)

def calc_copras(matrix, types, weights):
    V = normalize_sum(matrix) * weights
    bi = [j for j, t in enumerate(types) if t == "max"]
    ci = [j for j, t in enumerate(types) if t == "min"]
    Sp = V[:, bi].sum(axis=1) if bi else np.zeros(matrix.shape[0])
    Sm = V[:, ci].sum(axis=1) if ci else np.zeros(matrix.shape[0])
    if Sm.sum() > 0 and (Sm > 0).all():
        sm = Sm.min(); ssm = Sm.sum(); si = (sm / Sm).sum()
        return Sp + (sm * ssm) / (Sm * si) if si > 0 else Sp
    return Sp

def compute_all_models(matrix, types, weights):
    out = {}
    for name, fn in [("AHP", calc_ahp_aggregate), ("TOPSIS", calc_topsis),
                     ("PROMETHEE II", calc_promethee2), ("COPRAS", calc_copras)]:
        sc = fn(matrix, types, weights)
        rk = pd.Series(sc).rank(ascending=False, method='min').astype(int).values
        out[name] = {"scores": sc, "ranks": rk}
    return out

# =============================================================================
# SENSIBILIDADE UNIVERSAL
# =============================================================================
def render_sensitivity(score_function, alts, crits, base_weights, higher_is_better=True, key_suffix=""):
    st.markdown('<div class="sensitivity-box"><h3>🎯 Análise de Sensibilidade ± X% nos Pesos</h3></div>', unsafe_allow_html=True)
    variation_pct = st.session_state.sensitivity_pct
    bw = np.array(base_weights, dtype=float)
    bw = bw / bw.sum()
    try: base_scores = np.array(score_function(bw))
    except Exception as e: st.error(f"Erro: {e}"); return
    base_ranks = pd.Series(base_scores).rank(ascending=not higher_is_better, method='min').astype(int).values
    factor_pos, factor_neg = 1 + variation_pct / 100, 1 - variation_pct / 100
    scenarios = {"Base": base_ranks}

    for j, crit in enumerate(crits):
        for sign, factor in [("+", factor_pos), ("-", factor_neg)]:
            new_w = bw.copy()
            new_w[j] = bw[j] * factor
            other_sum_old, other_sum_new = bw.sum() - bw[j], 1 - new_w[j]
            if other_sum_old > 0 and other_sum_new > 0:
                for k in range(len(new_w)):
                    if k != j: new_w[k] = bw[k] * (other_sum_new / other_sum_old)
            ws = new_w.sum()
            new_w = new_w / ws if ws > 0 else np.ones_like(new_w) / len(new_w)
            try:
                sc = np.array(score_function(new_w))
                rk = pd.Series(sc).rank(ascending=not higher_is_better, method='min').astype(int).values
            except Exception: rk = [None] * len(alts)
            scenarios[f"{crit} {sign}{variation_pct}%"] = rk

    df_sens = pd.DataFrame(scenarios, index=alts)
    def style_row(row):
        base = row["Base"]
        styles = []
        for col in row.index:
            if col == "Base": styles.append("background-color: #d0d0d0; color: #000000; font-weight: 700;")
            else:
                val = row[col]
                if pd.isna(val): styles.append("background-color: #fafafa; color: #999999;")
                elif val < base: styles.append("background-color: #C6EFCE; color: #006100; font-weight: 600;")
                elif val > base: styles.append("background-color: #FFC7CE; color: #9C0006; font-weight: 600;")
                else: styles.append("background-color: #ffffff; color: #000000;")
        return styles
    st.dataframe(df_sens.style.apply(style_row, axis=1), use_container_width=True)

# =============================================================================
# TÍTULO E TABS
# =============================================================================
st.title("📊 MCDM Dashboard")
st.markdown("**Decisão Multicritério** · Dashboard interactivo · 4 modelos · Cumprimento do Req D1 a D9")

TAB_LABELS = ["🏆 Dashboard", "📋 Dados", "🔍 AHP", "🎯 TOPSIS", "📈 PROMETHEE II", "📊 COPRAS"]
tabs = st.tabs(TAB_LABELS)

# =============================================================================
# SIDEBAR — REQUISITO D7 (Filtros e Parâmetros)
# =============================================================================
matrix, alts, crits, types = get_decision_matrix()
methods = ["AHP", "TOPSIS", "PROMETHEE II", "COPRAS"]

with st.sidebar:
    st.header("⚙️ Parâmetros e Filtros")
    st.markdown("*(Ref: Requisito D7)*")
    
    focus_model = st.selectbox("📌 Modelo em Destaque", methods, key="sidebar_focus_model")
    
    if crits:
        focus_crit = st.selectbox("🎯 Critério (Sensibilidade)", crits, key="sidebar_focus_crit")
    else:
        focus_crit = None
        st.selectbox("🎯 Critério (Sensibilidade)", ["Sem dados"])
        
    if alts:
        focus_alt = st.selectbox("🎯 Alternativa (Destaque Radar)", alts, key="sidebar_focus_alt")
    else:
        focus_alt = None
        st.selectbox("🎯 Alternativa (Destaque Radar)", ["Sem dados"])
        
    st.markdown("---")
    st.markdown("**Sensibilidade (Req D3)**")
    st.session_state.sensitivity_pct = st.slider(
        "Variação ± nos pesos (%)", 
        min_value=5, max_value=50, value=st.session_state.sensitivity_pct, step=5,
        help="Usado para calcular as bandas de sensibilidade."
    )

# =============================================================================
# TAB 1: DADOS (Req D9 - Reutilizabilidade)
# =============================================================================
with tabs[1]:
    st.header("📋 Dados de Entrada (Req D9)")
    purpose_box("Cole a matriz AHP e a tabela das alternativas. O sistema adapta-se sem mexer no código.")

    def parse_criteria_paste(text):
        rows = [[c.strip() for c in line.split("\t" if "\t" in line else ";")] for line in text.strip("\n\r").splitlines() if line.strip()]
        if not rows: raise ValueError("Sem dados")
        header_idx = next(i for i, r in enumerate(rows) if sum(1 for c in r if c and not _is_num(c)) >= 3)
        header = rows[header_idx]
        col_codes = [c.strip() for c in header if c.strip().lower() and "max" not in c.strip().lower() and "min" not in c.strip().lower()]
        n = len(col_codes)
        body = rows[header_idx + 1:]
        types_out = []; codes_row = []; ahp = []
        for r in body:
            row_code = r[0].strip()
            if not row_code or "max" in row_code.lower() or "min" in row_code.lower(): continue
            vals = [float(clean_number_string(r[1 + k])) for k in range(n)]
            tipo = "max"
            for c in r[n+1:]:
                if c.strip().lower() in ("max", "min"): tipo = c.strip().lower(); break
            codes_row.append(row_code); types_out.append(tipo); ahp.append(vals)
            if len(ahp) == n: break
        ahp = np.array(ahp, dtype=float)
        for i in range(n):
            ahp[i, i] = 1.0
            for j in range(n):
                if i < j and ahp[i, j] > 0: ahp[j, i] = 1.0 / ahp[i, j]
        return col_codes, types_out, ahp

    def parse_alts_paste(text, expected_crits=None):
        rows = [[c.strip() for c in line.split("\t" if "\t" in line else ";")] for line in text.strip("\n\r").splitlines() if line.strip()]
        header_idx = next(i for i, r in enumerate(rows) if r and (r[0].strip().lower().startswith("alt") or sum(1 for c in r if c and not _is_num(c)) >= 2))
        header = rows[header_idx]
        col_idx = [header.index(c) for c in expected_crits] if expected_crits else list(range(1, len(header)))
        alt_names = []; matrix_rows = []
        for r in rows[header_idx + 1:]:
            if not r or not r[0].strip(): continue
            vals = [float(clean_number_string(r[ci]) or 0.0) if ci < len(r) else 0.0 for ci in col_idx]
            alt_names.append(r[0].strip()); matrix_rows.append(vals)
        return alt_names, np.array(matrix_rows, dtype=float)

    def _is_num(s):
        try: float(clean_number_string(s)); return True
        except Exception: return False

    st.markdown('<div class="data-section"><h3>📥 1. Matriz AHP (Critérios)</h3></div>', unsafe_allow_html=True)
    crit_text = st.text_area("Cole a matriz AHP (com cabeçalhos + MAX/MIN):", value=st.session_state.get("crit_paste_text", ""), height=150)
    st.markdown('<div class="data-section"><h3>📥 2. Tabela das Alternativas</h3></div>', unsafe_allow_html=True)
    alts_text = st.text_area("Cole a tabela de alternativas × critérios:", value=st.session_state.get("alts_paste_text", ""), height=150)

    if st.session_state.get("success_message"):
        st.success(st.session_state.success_message)
        st.session_state.success_message = None

    if st.button("🚀 Processar Dados", type="primary", use_container_width=True):
        try:
            codes, types_parsed, ahp_matrix = parse_criteria_paste(crit_text)
            alt_names, dec_matrix = parse_alts_paste(alts_text, expected_crits=codes)
            gm = np.prod(ahp_matrix, axis=1) ** (1.0 / len(codes))
            w_ahp = gm / gm.sum()
            
            st.session_state.criteria_df = pd.DataFrame({"Critério": codes, "Tipo": types_parsed})
            st.session_state.matrix_df = pd.DataFrame(dec_matrix, columns=codes)
            st.session_state.matrix_df.insert(0, "Alternativa", alt_names)
            st.session_state.engine_weights["AHP"] = np.array(w_ahp)
            st.session_state.ahp_matrix_pasted = ahp_matrix
            st.session_state.crit_paste_text = crit_text
            st.session_state.alts_paste_text = alts_text
            st.session_state.success_message = f"✅ Sucesso! {len(alt_names)} alternativas e {len(codes)} critérios."
            st.toast("Dados carregados!", icon="✅")
            st.rerun()
        except Exception as e: st.error(f"❌ Erro: {e}")

# =============================================================================
# TAB 0: DASHBOARD CONSOLIDADO
# =============================================================================
with tabs[0]:
    if not check_valid_input(): st.stop()
    weights = get_active_weights()
    results = compute_all_models(matrix, types, weights)

    # REQ D1: Tabela de Ranking Consolidado
    df_dash = pd.DataFrame({"Alternativa": alts})
    for m in methods: df_dash[m] = results[m]["ranks"]
    df_dash["Posição Média"] = df_dash[methods].mean(axis=1).round(2)
    df_dash["Score Composto"] = (100 - (df_dash["Posição Média"] / len(alts) * 100)).round(1) # Score invertido p/ ser legível
    df_dash["Ranking Final"] = pd.Series(df_dash["Posição Média"]).rank(ascending=True, method='min').astype(int).values
    df_dash = df_dash.sort_values("Ranking Final").reset_index(drop=True)
    top3 = df_dash.head(3)["Alternativa"].tolist()

    col_rank, col_radar = st.columns([1.5, 1])

    with col_rank:
        st.markdown("##### 🏆 D1: Ranking Consolidado")
        display = df_dash.copy()
        display.insert(0, "Medalha", display["Ranking Final"].map({1:"🥇", 2:"🥈", 3:"🥉"}).fillna(""))
        st.dataframe(display.style.background_gradient(cmap="RdYlGn_r", subset=methods + ["Posição Média", "Ranking Final"]), hide_index=True, use_container_width=True)

    # REQ D2: Gráfico de Radar
    with col_radar:
        st.markdown("##### 🎯 D2: Perfil Multicritério (Radar)")
        norm_df = pd.DataFrame(normalize_minmax(matrix, types), index=alts, columns=crits)
        fig_radar = go.Figure()
        colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
        for i, alt in enumerate(top3):
            vals = list(norm_df.loc[alt]) + [norm_df.loc[alt].iloc[0]]
            fig_radar.add_trace(go.Scatterpolar(r=vals, theta=crits + [crits[0]], fill="toself", name=f"{i+1}º {alt}", line=dict(color=colors[i]), opacity=0.5))
        if focus_alt and focus_alt not in top3:
            vals = list(norm_df.loc[focus_alt]) + [norm_df.loc[focus_alt].iloc[0]]
            fig_radar.add_trace(go.Scatterpolar(r=vals, theta=crits + [crits[0]], fill="toself", name=f"Filtro: {focus_alt}", line=dict(color="#9C27B0", width=3, dash="dot")))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)), height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    
    # REQ D3 e D4: Análise de Sensibilidade Visual Dinâmica (TOPSIS e PROMETHEE)
    col_sens1, col_sens2 = st.columns(2)
    with col_sens1:
        st.markdown(f"##### 📊 Sensibilidade do Modelo: {focus_model}")
        st.caption(f"Score variando {focus_crit} em ±{st.session_state.sensitivity_pct}%")
        
        focus_crit_idx = crits.index(focus_crit)
        scorer = {"AHP": calc_ahp_aggregate, "TOPSIS": calc_topsis, "PROMETHEE II": calc_promethee2, "COPRAS": calc_copras}[focus_model]
        
        nw_p = weights.copy(); nw_p[focus_crit_idx] *= (1 + st.session_state.sensitivity_pct/100)
        nw_p = nw_p / nw_p.sum()
        nw_m = weights.copy(); nw_m[focus_crit_idx] *= (1 - st.session_state.sensitivity_pct/100)
        nw_m = nw_m / nw_m.sum()
        
        sc_base = scorer(matrix, types, weights)
        sc_p = scorer(matrix, types, nw_p)
        sc_m = scorer(matrix, types, nw_m)
        
        df_var = pd.DataFrame({"Alt": alts, "Base": sc_base, "Max": np.maximum(sc_p, sc_m), "Min": np.minimum(sc_p, sc_m)}).sort_values("Base")
        fig_var = go.Figure()
        # Bandas de variação
        for i, row in df_var.iterrows():
            fig_var.add_shape(type="line", x0=row["Min"], x1=row["Max"], y0=row["Alt"], y1=row["Alt"], line=dict(color="#ff9800", width=6))
        # Ponto Base
        fig_var.add_trace(go.Scatter(x=df_var["Base"], y=df_var["Alt"], mode="markers", marker=dict(color="#1F4E78", size=10), name="Score Original"))
        fig_var.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_var, use_container_width=True)

    with col_sens2:
        # REQ D8: Painel de Recomendação
        st.markdown("##### 💡 D8: Recomendação Automática")
        total_top3 = sum(df_dash.head(3)["Top-3 em N modelos"].values)
        conv_pct = (total_top3 / (3 * len(methods))) * 100
        verdict_color = "#2e7d32" if conv_pct >= 70 else ("#f57c00" if conv_pct >= 40 else "#c62828")

        st.markdown(f"""
        <div style="background: {verdict_color}; color: white; padding: 20px; border-radius: 8px;">
            <h4 style="margin-top:0; color:white;">Recomendação Global</h4>
            <p style="font-size: 16px;">O modelo de consenso sugere a alternativa <b>{top3[0]}</b> como a escolha ideal, 
            seguida de <b>{top3[1]}</b> e <b>{top3[2]}</b>.</p>
            <hr style="border-top: 1px solid rgba(255,255,255,0.3);">
            <b>Grau de Convergência entre Modelos: {conv_pct:.0f}%</b>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TABS DOS MODELOS INDIVIDUAIS (Simplificadas para poupar espaço)
# =============================================================================
with tabs[2]:
    st.header("🔍 AHP")
    if check_valid_input():
        w_ahp = get_active_weights()
        st.dataframe(pd.DataFrame({"Critério": crits, "Peso AHP": w_ahp, "%": [f"{x*100:.2f}%" for x in w_ahp]}), hide_index=True)

with tabs[3]:
    st.header("🎯 TOPSIS")
    if check_valid_input():
        sc = calc_topsis(matrix, types, get_active_weights())
        st.dataframe(pd.DataFrame({"Alternativa": alts, "CC*": sc}).sort_values("CC*", ascending=False), hide_index=True)

with tabs[4]:
    st.header("📈 PROMETHEE II")
    if check_valid_input():
        sc = calc_promethee2(matrix, types, get_active_weights())
        st.dataframe(pd.DataFrame({"Alternativa": alts, "Fluxo Líquido (φ)": sc}).sort_values("Fluxo Líquido (φ)", ascending=False), hide_index=True)

with tabs[5]:
    st.header("📊 COPRAS")
    if check_valid_input():
        sc = calc_copras(matrix, types, get_active_weights())
        st.dataframe(pd.DataFrame({"Alternativa": alts, "Q_i": sc}).sort_values("Q_i", ascending=False), hide_index=True)
