"""
MCDM Dashboard v2 — Ferramenta de Apoio à Decisão Multicritério
================================================================
Reestruturado:
  • Sidebar: APENAS Motor de Pesos Activo (espelhado na aba 📋 Dados)
  • Toda a entrada de dados (Demo, Manual, Quadros em bruto) na aba 📋 Dados
  • Editores de Critérios e Matriz na aba 📋 Dados (precisão 5 casas decimais)
  • Selector de Motor de Pesos na aba 📋 Dados (Manual/SWING/SMART/Entropia/CRITIC/AHP)
  • Slider de Sensibilidade na aba 📋 Dados
  • AHP na sua própria aba com iterações até CR < 0.10
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
# Aceita: "250 000 000 €", "0,462", "25%", "1,234.56", "1.234,56", "  €5.000  "
# =============================================================================
def clean_number_string(s):
    """Converte string com formatação variada para string parsável por to_numeric."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    if not s or s.lower() in ("nan", "none", "-", "—", "n/a", "na"):
        return None
    for ch in ['€', '$', '£', '¥', '%', '\u20ac', '\u00a3', '\u00a5']:
        s = s.replace(ch, '')
    s = re.sub(r'\bR\$\s*', '', s)
    s = s.replace(' ', '').replace('\xa0', '').replace('\u202f', '').replace('\u2009', '')
    s = s.strip()
    if not s:
        return None
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        parts = s.split(',')
        if len(parts) > 2 and all(len(p) == 3 and p.isdigit() for p in parts[1:]):
            s = s.replace(',', '')
        else:
            s = s.replace(',', '.')
    elif '.' in s:
        parts = s.split('.')
        if len(parts) > 2 and all(len(p) == 3 and p.isdigit() for p in parts[1:]):
            s = s.replace('.', '')
    return s


def clean_numeric_column(series):
    """Aplica clean_number_string a uma série e devolve to_numeric + contagem de falhas."""
    cleaned = series.astype(str).apply(clean_number_string)
    nums = pd.to_numeric(cleaned, errors="coerce")
    original_empty = series.isna() | (series.astype(str).str.strip() == "")
    n_failed = int((nums.isna() & ~original_empty).sum())
    return nums, n_failed

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="MCDM Dashboard v2", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
.theory-box {
    background: linear-gradient(135deg, #f0f7ff 0%, #e0ecff 100%) !important;
    border-left: 4px solid #1F4E78; padding: 16px 20px; border-radius: 6px;
    margin: 12px 0 20px 0; font-size: 14px; line-height: 1.5;
    color: #1F4E78 !important;
}
.theory-box * { color: #1F4E78 !important; }
.theory-box h4 { color: #1F4E78 !important; margin-top: 0; font-size: 16px; font-weight: 700; }
.theory-box ul, .theory-box ol { margin: 8px 0 0 20px; }
.theory-box code { background: #fff !important; padding: 2px 6px; border-radius: 3px; font-size: 13px; color: #c7254e !important; }
.theory-box b, .theory-box strong { color: #1F4E78 !important; font-weight: 700; }

.purpose-box {
    background: #e8f5e9 !important; padding: 12px 18px; border-left: 4px solid #2e7d32;
    border-radius: 4px; margin-bottom: 16px; color: #1b5e20 !important; font-size: 15px;
}
.purpose-box * { color: #1b5e20 !important; }
.purpose-box b, .purpose-box strong { color: #1b5e20 !important; font-weight: 700; }

.step-header {
    background: #2E75B6 !important; color: white !important; padding: 8px 14px; border-radius: 4px;
    font-weight: 600; margin: 16px 0 8px 0; font-size: 15px;
}

.sensitivity-box {
    background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%) !important;
    border: 2px solid #f57c00; padding: 18px; border-radius: 8px; margin: 24px 0 12px 0;
    color: #5d3a00 !important;
}
.sensitivity-box * { color: #5d3a00 !important; }
.sensitivity-box h3 { color: #e65100 !important; margin-top: 0; font-size: 18px; }
.sensitivity-box b, .sensitivity-box strong { color: #bf360c !important; font-weight: 700; }

.injection-active {
    background: #fce4ec !important; border: 2px solid #c2185b; padding: 8px 14px;
    border-radius: 6px; color: #c2185b !important; font-weight: 700; text-align: center; margin: 12px 0;
}
.injection-active * { color: #c2185b !important; }

.result-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%) !important;
    border: 2px solid #2e7d32; padding: 18px; border-radius: 8px; margin: 16px 0;
    color: #1b5e20 !important; font-size: 16px; font-weight: 600;
}
.result-box * { color: #1b5e20 !important; }
.result-box b, .result-box strong { color: #1b5e20 !important; }

.warning-box {
    background: #fff3e0 !important; border-left: 3px solid #ef6c00; padding: 10px 16px;
    border-radius: 4px; margin: 8px 0; color: #e65100 !important;
}
.warning-box * { color: #e65100 !important; }
.warning-box b, .warning-box strong { color: #bf360c !important; }

.cta-box {
    background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%) !important;
    border: 3px dashed #f9a825; padding: 24px; border-radius: 12px; margin: 20px 0;
    color: #f57f17 !important; font-size: 17px; text-align: center;
}
.cta-box * { color: #5d4037 !important; }
.cta-box h2 { color: #e65100 !important; font-size: 24px; margin-top: 0; }
.cta-box b, .cta-box strong { color: #bf360c !important; font-weight: 700; }

.data-section {
    background: #fafafa !important; padding: 18px 20px; border-radius: 8px;
    border-left: 4px solid #1F4E78; margin: 18px 0 10px 0;
}
.data-section h3 {
    color: #1F4E78 !important; margin-top: 0; margin-bottom: 8px; font-size: 18px;
}

.stTabs [data-baseweb="tab"] { padding: 10px 18px; font-weight: 600; }
.stTabs [aria-selected="true"] { background-color: #1F4E78 !important; color: white !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# DEMO PRESETS
# =============================================================================
DEMO_PRESETS = {
    "Caso MCG (9 alts × 6 crits)": {
        "criteria": pd.DataFrame({
            "Critério": ["C1_VP", "C2_PF", "C3_EE", "C4_FE", "C5_UD", "C6_RC"],
            "Tipo": ["max", "max", "min", "max", "max", "max"],
            "Peso Manual": [0.462, 0.218, 0.024, 0.114, 0.023, 0.159],
        }),
        "matrix": pd.DataFrame({
            "Alternativa": [f"A{i}" for i in range(1, 10)],
            "C1_VP": [250_000_000, 300_000, 900_000, 650_000, 5_000_000, 1_350_000, 10_500_000, 3_450_000, 15_000_000],
            "C2_PF": [0.25, 0.35, 0.50, 0.50, 0.40, 0.50, 0.40, 0.40, 0.60],
            "C3_EE": [24, 8, 8, 8, 24, 8, 16, 8, 24],
            "C4_FE": [4, 5, 3, 3, 4, 3, 3, 3, 4],
            "C5_UD": [180, 60, 60, 90, 30, 60, 180, 60, 300],
            "C6_RC": [4, 5, 5, 3, 3, 5, 4, 4, 3],
        }),
    },
    "Selecção de Fornecedor (5 × 4)": {
        "criteria": pd.DataFrame({
            "Critério": ["Custo", "Qualidade", "Prazo", "Sustentabilidade"],
            "Tipo": ["min", "max", "min", "max"],
            "Peso Manual": [0.30, 0.30, 0.20, 0.20],
        }),
        "matrix": pd.DataFrame({
            "Alternativa": ["Forn. A", "Forn. B", "Forn. C", "Forn. D", "Forn. E"],
            "Custo": [1200.0, 1500.0, 1100.0, 1300.0, 1400.0],
            "Qualidade": [8.0, 6.0, 9.0, 7.0, 5.0],
            "Prazo": [15.0, 20.0, 18.0, 12.0, 25.0],
            "Sustentabilidade": [7.0, 5.0, 8.0, 6.0, 4.0],
        }),
    },
    "Investimento (4 × 5)": {
        "criteria": pd.DataFrame({
            "Critério": ["Retorno (%)", "Risco", "Liquidez", "Maturidade (anos)", "Custo fees"],
            "Tipo": ["max", "min", "max", "min", "min"],
            "Peso Manual": [0.30, 0.25, 0.20, 0.15, 0.10],
        }),
        "matrix": pd.DataFrame({
            "Alternativa": ["Acções", "Obrigações", "Imobiliário", "ETFs"],
            "Retorno (%)": [12.0, 4.5, 7.0, 8.5],
            "Risco": [8.0, 2.0, 5.0, 4.0],
            "Liquidez": [9.0, 7.0, 3.0, 8.0],
            "Maturidade (anos)": [1.0, 10.0, 15.0, 3.0],
            "Custo fees": [1.5, 0.5, 3.0, 0.3],
        }),
    },
}

# =============================================================================
# ESTADO INICIAL
# =============================================================================
def init_state():
    if "criteria_df" not in st.session_state:
        preset = DEMO_PRESETS["Selecção de Fornecedor (5 × 4)"]
        st.session_state.criteria_df = preset["criteria"].copy()
        st.session_state.matrix_df = preset["matrix"].copy()
    if "global_injection_on" not in st.session_state:
        st.session_state.global_injection_on = False
    if "global_injection_engine" not in st.session_state:
        st.session_state.global_injection_engine = "AHP"
    if "engine_weights" not in st.session_state:
        st.session_state.engine_weights = {}
    if "sensitivity_pct" not in st.session_state:
        st.session_state.sensitivity_pct = 20
    if "ahp_history" not in st.session_state:
        st.session_state.ahp_history = []

init_state()

# =============================================================================
# HELPERS
# =============================================================================
def get_decision_matrix():
    crit_df = st.session_state.criteria_df.copy()
    crit_df = crit_df.dropna(subset=["Critério"])
    crit_df = crit_df[crit_df["Critério"].astype(str).str.strip() != ""]
    crits = crit_df["Critério"].astype(str).tolist()
    types = crit_df["Tipo"].fillna("max").astype(str).tolist()

    m_df = st.session_state.matrix_df.copy()
    m_df = m_df.dropna(subset=["Alternativa"])
    m_df = m_df[m_df["Alternativa"].astype(str).str.strip() != ""]
    alts = m_df["Alternativa"].astype(str).tolist()

    matrix = []
    for crit in crits:
        if crit not in m_df.columns:
            matrix.append([0.0] * len(alts))
        else:
            col = pd.to_numeric(m_df[crit], errors="coerce").fillna(0.0).values
            matrix.append(col)
    matrix = np.array(matrix).T if matrix else np.zeros((0, 0))
    return matrix, alts, crits, types

def get_active_weights():
    """Sempre AHP — se calculado. Caso contrário: pesos iguais 1/n (fallback)."""
    _, _, crits, _ = get_decision_matrix()
    n = len(crits)
    if n == 0:
        return np.array([])
    if "AHP" in st.session_state.engine_weights:
        w = st.session_state.engine_weights["AHP"]
        if len(w) == n:
            w = np.array(w, dtype=float)
            return w / w.sum() if w.sum() > 0 else np.ones(n) / n
    # fallback: pesos iguais
    return np.ones(n) / n

def show_active_weights_banner():
    w = get_active_weights()
    _, _, crits, _ = get_decision_matrix()
    if "AHP" in st.session_state.engine_weights and len(st.session_state.engine_weights["AHP"]) == len(crits):
        st.markdown(
            '<div class="injection-active">🔌 Motor activo — <b>AHP</b> (pesos da Matriz Par-a-Par)</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="warning-box">⚠️ AHP ainda não calculado — modelos a usar pesos iguais (1/n). '
            'Vá à aba 🔍 AHP para definir os pesos correctos.</div>',
            unsafe_allow_html=True)
    cols = st.columns([3, 1])
    with cols[0]:
        df_w = pd.DataFrame({"Critério": crits, "Peso (%)": [f"{x*100:.2f}%" for x in w]})
        st.dataframe(df_w, hide_index=True, use_container_width=False)
    with cols[1]:
        st.metric("Σ pesos", f"{w.sum():.5f}")

def theory_box(title, html):
    st.markdown(f'<div class="theory-box"><h4>📚 {title}</h4>{html}</div>', unsafe_allow_html=True)

def purpose_box(text):
    st.markdown(
        f'<div class="purpose-box"><b>📌 Para que serve esta aba:</b> {text}</div>',
        unsafe_allow_html=True
    )

def step_header(text):
    st.markdown(f'<div class="step-header">{text}</div>', unsafe_allow_html=True)

def check_valid_input():
    matrix, alts, crits, types = get_decision_matrix()
    if len(alts) < 2 or len(crits) < 2 or matrix.size == 0 or np.all(matrix == 0):
        st.markdown(
            """<div class="cta-box">
            <h2>👈 SEM DADOS — VÁ À ABA 📋 DADOS</h2>
            <p>Para usar esta aba, primeiro tem de carregar uma <b>matriz de decisão</b>.</p>
            <p>Na aba <b>📋 Dados</b> tem 3 formas:</p>
            <p>📋 <b>Demo</b> &nbsp;·&nbsp; ✏️ <b>Manual</b> &nbsp;·&nbsp;
               📥 <b>Quadros em bruto (Q1.3 + Q1.4)</b></p>
            </div>""",
            unsafe_allow_html=True
        )
        return False
    return True

# Normalizações
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

# =============================================================================
# MODEL SCORERS — REUSABLE, DETERMINISTIC, NO SIDE-EFFECTS
# Cada um devolve um vector de scores (1 por alternativa). Maior = melhor.
# =============================================================================
def calc_ahp_aggregate(matrix, types, weights):
    """AHP final score: S_i = Σ w_j · u_j(x_ij), u via normalize_minmax."""
    U = normalize_minmax(matrix, types)
    return (U * weights).sum(axis=1)

def calc_topsis(matrix, types, weights):
    """TOPSIS CC* — Hwang & Yoon 1981."""
    n = len(types)
    R = normalize_vector(matrix)
    V = R * weights
    Ap = np.array([V[:, j].max() if types[j] == "max" else V[:, j].min() for j in range(n)])
    An = np.array([V[:, j].min() if types[j] == "max" else V[:, j].max() for j in range(n)])
    Dp = np.sqrt(((V - Ap) ** 2).sum(axis=1))
    Dn = np.sqrt(((V - An) ** 2).sum(axis=1))
    denom = np.where(Dp + Dn == 0, 1e-9, Dp + Dn)
    return Dn / denom

def calc_promethee2(matrix, types, weights):
    """PROMETHEE II φ líquido — Brans 1985, Tipo I (Usual)."""
    m_rows, n_cols = matrix.shape
    pi = np.zeros((m_rows, m_rows))
    for a in range(m_rows):
        for b in range(m_rows):
            if a == b: continue
            for j in range(n_cols):
                d = matrix[a, j] - matrix[b, j] if types[j] == "max" else matrix[b, j] - matrix[a, j]
                if d > 0:
                    pi[a, b] += weights[j]
    denom = max(m_rows - 1, 1)
    phi_p = pi.sum(axis=1) / denom
    phi_n = pi.sum(axis=0) / denom
    return phi_p - phi_n

def calc_copras(matrix, types, weights):
    """COPRAS Q_i — Zavadskas & Kaklauskas 1996."""
    m_rows = matrix.shape[0]
    Xn = normalize_sum(matrix)
    V = Xn * weights
    bi = [j for j, t in enumerate(types) if t == "max"]
    ci = [j for j, t in enumerate(types) if t == "min"]
    Sp = V[:, bi].sum(axis=1) if bi else np.zeros(m_rows)
    Sm = V[:, ci].sum(axis=1) if ci else np.zeros(m_rows)
    if Sm.sum() > 0 and (Sm > 0).all():
        sm = Sm.min(); ssm = Sm.sum(); si = (sm / Sm).sum()
        Q = Sp + (sm * ssm) / (Sm * si) if si > 0 else Sp
    else:
        Q = Sp
    return Q

def compute_all_models(matrix, types, weights):
    """Devolve dict {modelo: {scores, ranks}} para os 4 modelos, com pesos dados."""
    out = {}
    for name, fn in [("AHP", calc_ahp_aggregate),
                     ("TOPSIS", calc_topsis),
                     ("PROMETHEE II", calc_promethee2),
                     ("COPRAS", calc_copras)]:
        sc = fn(matrix, types, weights)
        rk = pd.Series(sc).rank(ascending=False, method='min').astype(int).values
        out[name] = {"scores": sc, "ranks": rk}
    return out

# =============================================================================
# SENSIBILIDADE UNIVERSAL
# =============================================================================
def render_sensitivity(score_function, alts, crits, base_weights, higher_is_better=True, key_suffix=""):
    st.markdown(
        '<div class="sensitivity-box"><h3>🎯 Análise de Sensibilidade ± X% nos Pesos</h3>'
        '<p style="margin-bottom:0; color:#bf360c;">Variamos o peso de <b>cada critério isoladamente</b> '
        'em ±X% e renormalizamos os restantes. Para cada cenário recalculamos o ranking e comparamos com o Base.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    variation_pct = st.session_state.sensitivity_pct

    bw = np.array(base_weights, dtype=float)
    bw = bw / bw.sum()
    try:
        base_scores = np.array(score_function(bw))
    except Exception as e:
        st.error(f"Erro a calcular scores base: {e}")
        return
    base_ranks = pd.Series(base_scores).rank(ascending=not higher_is_better, method='min').astype(int).values

    factor_pos = 1 + variation_pct / 100
    factor_neg = 1 - variation_pct / 100
    scenarios = {"Base": base_ranks}

    for j, crit in enumerate(crits):
        for sign, factor in [("+", factor_pos), ("-", factor_neg)]:
            new_w = bw.copy()
            new_w[j] = bw[j] * factor
            other_sum_old = bw.sum() - bw[j]
            other_sum_new = 1 - new_w[j]
            if other_sum_old > 0 and other_sum_new > 0:
                for k in range(len(new_w)):
                    if k != j:
                        new_w[k] = bw[k] * (other_sum_new / other_sum_old)
            ws = new_w.sum()
            new_w = new_w / ws if ws > 0 else np.ones_like(new_w) / len(new_w)
            try:
                sc = np.array(score_function(new_w))
                rk = pd.Series(sc).rank(ascending=not higher_is_better, method='min').astype(int).values
            except Exception:
                rk = [None] * len(alts)
            scenarios[f"{crit} {sign}{variation_pct}%"] = rk

    df_sens = pd.DataFrame(scenarios, index=alts)
    df_sens.index.name = "Alternativa"

    def style_row(row):
        base = row["Base"]
        styles = []
        for col in row.index:
            if col == "Base":
                styles.append("background-color: #d0d0d0; color: #000000; font-weight: 700;")
            else:
                val = row[col]
                try:
                    if val is None or pd.isna(val):
                        styles.append("background-color: #fafafa; color: #999999;")
                    elif val < base:
                        styles.append("background-color: #C6EFCE; color: #006100; font-weight: 600;")
                    elif val > base:
                        styles.append("background-color: #FFC7CE; color: #9C0006; font-weight: 600;")
                    else:
                        styles.append("background-color: #ffffff; color: #000000;")
                except Exception:
                    styles.append("background-color: #ffffff; color: #000000;")
        return styles

    st.markdown(f"**Variação ± aplicada:** {variation_pct}% (ajustável na aba 📋 Dados)")
    st.markdown("**Legenda:** 🟢 sobe no ranking · 🔴 desce no ranking · ⚪ sem alteração")
    st.dataframe(df_sens.style.apply(style_row, axis=1), use_container_width=True)

    base_vals = df_sens["Base"].values
    others = df_sens.drop(columns=["Base"])
    n_changes = []
    for i in range(len(alts)):
        c = 0
        for col in others.columns:
            v = others.iloc[i][col]
            try:
                if v is not None and not pd.isna(v) and v != base_vals[i]:
                    c += 1
            except Exception:
                pass
        n_changes.append(c)
    df_robust = pd.DataFrame({
        "Alternativa": alts,
        "Rank Base": base_ranks,
        "Inversões": n_changes,
        "Robustez": ["🟢 ESTÁVEL" if c == 0 else ("🟡 MODERADA" if c <= 2 else "🔴 INSTÁVEL") for c in n_changes]
    })
    st.markdown("**Resumo de Robustez por Alternativa:**")
    st.dataframe(df_robust, hide_index=True, use_container_width=True)

# Guardar score-functions e rankings por método para o Dashboard/Relatório
if "all_results" not in st.session_state:
    st.session_state.all_results = {}

def store_result(method_name, scores, ranking, higher_is_better=True):
    st.session_state.all_results[method_name] = {
        "scores": np.array(scores),
        "ranking": np.array(ranking),
        "higher_is_better": higher_is_better,
    }

# =============================================================================
# TÍTULO
# =============================================================================
st.title("📊 MCDM Dashboard")
st.markdown(
    "**Decisão Multicritério** · 3 modos de entrada de dados · 3 modelos MCDM (TOPSIS, PROMETHEE II, COPRAS) · "
    "AHP como motor de pesos · Sensibilidade universal · Dashboard executivo · Relatório final · Precisão 5 decimais"
)


# =============================================================================
# SIDEBAR — INFO motor AHP (não há escolha)
# =============================================================================
with st.sidebar:
    st.header("🔌 Motor de Pesos")
    # forçar sempre AHP
    st.session_state.global_injection_on = True
    st.session_state.global_injection_engine = "AHP"

    if "AHP" in st.session_state.engine_weights:
        st.success("✓ Motor activo: **AHP** (pesos calculados)")
    else:
        st.warning(
            "⚠️ AHP **não calculado**.\n\n"
            "Vá à aba **🔍 AHP** para preencher a Matriz de Comparação Par-a-Par.\n\n"
            "Enquanto isso, modelos usam pesos iguais (1/n)."
        )

    st.markdown("---")
    st.caption(
        "💡 **Toda a configuração** (fonte de dados, paste de quadros, editor de critérios, "
        "sensibilidade) está na aba **📋 Dados**."
    )


# =============================================================================
# TABS
# =============================================================================
TAB_LABELS = [
    "🏠 Início",
    "📋 Dados",
    "🔍 AHP",
    "🎯 TOPSIS",
    "📈 PROMETHEE II",
    "📊 COPRAS",
    "🏆 Dashboard",
    "📑 Relatório Técnico",
]
tabs = st.tabs(TAB_LABELS)


# =============================================================================
# TAB 0: INÍCIO — Como funciona a aplicação
# =============================================================================
with tabs[0]:
    st.header("🏠 Bem-vindo ao MCDM Dashboard")
    st.markdown(
        """
        Esta aplicação ajuda a tomar **decisões multicritério** comparando alternativas
        (fornecedores, projectos, investimentos, etc.) segundo vários critérios (custo,
        qualidade, prazo, risco, etc.) — usando **9 modelos científicos** consagrados.
        """
    )

    st.markdown("---")

    st.markdown(
        """<div class="cta-box">
        <h2>👉 ONDE METER OS DADOS DO ENUNCIADO?</h2>
        <p>Os dados entram na <b>aba 📋 Dados</b> (segunda aba acima).</p>
        <p>Lá tem 3 formas: <b>Demo</b> · <b>Manual</b> · <b>Quadros em bruto (Q1.3 + Q1.4)</b></p>
        <p>A sidebar fica só com o <b>🔌 Motor de Pesos Activo</b> (espelhado na aba 📋 Dados).</p>
        </div>""",
        unsafe_allow_html=True
    )

    st.subheader("🚀 Como começar em 4 passos")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("### 1️⃣ Dados")
        st.markdown(
            "Aba **📋 Dados** → escolha a fonte:\n\n"
            "• Demo · Manual · Quadros em bruto\n\n"
            "Depois ajuste tipo (max/min) e pesos manuais nos editores."
        )
    with c2:
        st.markdown("### 2️⃣ Pesos")
        st.markdown(
            "Defina importância dos critérios:\n\n"
            "• **Manual** (na aba Dados)\n\n"
            "• **AHP** (aba 🔍): comparação par-a-par com escala Saaty e validação CR<0.10\n\n"
            "Único motor de pesos disponível além do manual."
        )
    with c3:
        st.markdown("### 3️⃣ Modelos")
        st.markdown(
            "Executar **3 modelos de ranking**:\n\n"
            "🎯 **TOPSIS** · 📈 **PROMETHEE II** · 📊 **COPRAS**\n\n"
            "Cada uma dá ranking + gráfico + sensibilidade."
        )
    with c4:
        st.markdown("### 4️⃣ Decisão")
        st.markdown(
            "Consolide e decida:\n\n"
            "• **🏆 Dashboard**: vista executiva consolidada\n\n"
            "• **📑 Relatório**: descarregar CSV/Excel/MD"
        )

    st.markdown("---")
    st.subheader("📑 O que faz cada aba")

    tab_descriptions = [
        ("📋 Dados", "**Hub central**: escolher fonte de dados (Demo/Manual/Quadros em bruto), "
                     "colar Q1.3 e Q1.4, editar critérios e matriz, escolher motor de pesos (Manual ou AHP), "
                     "ajustar sensibilidade. Mostra estatísticas e heatmap."),
        ("🔍 AHP", "Saaty (1980). Comparação par-a-par escala Saaty 1-9. Calcula pesos e valida CR < 0.10. "
                   "Se CR ≥ 0.10 → sugere correcção iterativa. **É o único motor de pesos disponível** "
                   "(além do manual)."),
        ("🎯 TOPSIS", "Hwang & Yoon (1981). Distância à solução ideal A⁺ e anti-ideal A⁻. "
                       "Gráfico horizontal de CC* incluído."),
        ("📈 PROMETHEE II", "Brans (1985). Fluxos de preferência par-a-par com 3 funções (Usual/Linear/Gaussiana). "
                             "Gráfico horizontal de φ líquido incluído."),
        ("📊 COPRAS", "Zavadskas & Kaklauskas (1996). Função proporcional benefícios/custos. Grau de utilidade U_i (%). "
                       "Gráfico horizontal incluído."),
        ("🏆 Dashboard", "**Vista consolidada one-page** com ranking, radar Top-3, sensibilidade, gráficos por modelo e recomendação final. "
                          "Estilo executivo MCG."),
        ("📑 Relatório Técnico", "7 capítulos + Referências APA. Downloads CSV/Excel/Markdown."),
    ]

    for label, desc in tab_descriptions:
        st.markdown(f"**{label}** — {desc}")

    st.markdown("---")
    st.info("💡 **Dica:** comece pelos casos **Demo** na aba 📋 Dados para ver a app a funcionar antes dos seus dados.")

# =============================================================================
# TAB 1: DADOS — HUB CENTRAL DE CONFIGURAÇÃO
# (Toda a entrada de dados, editores e sensibilidade vive aqui)
# =============================================================================
with tabs[1]:
    st.header("📋 Dados — Configuração Central")
    purpose_box(
        "Aqui escolhe a <b>fonte de dados</b>, edita critérios e matriz, e define a <b>variação da análise "
        "de sensibilidade</b>. Tudo o que afecta TODAS as outras abas está aqui."
    )

    # ============================================================================
    # SECÇÃO 1 — FONTE DE DADOS
    # ============================================================================
    st.markdown('<div class="data-section"><h3>📥 1. Escolha a fonte de dados</h3></div>',
                unsafe_allow_html=True)

    data_source = st.radio(
        "Como quer fornecer os dados?",
        ["📋 Demo (pré-definidos)",
         "✏️ Manual (dimensões + paste do Excel)",
         "📥 Quadros em bruto — colar Q1.3 (alts) + Q1.4 (crits)"],
        key="data_source_radio",
        horizontal=False,
        help="3 modos: Demo carrega um exemplo · Manual cria matriz vazia ou cola do Excel · "
             "Quadros em bruto é o modo ideal para os enunciados académicos (Q1.3 + Q1.4)."
    )

    # -------- MODO 1: DEMO --------
    if data_source == "📋 Demo (pré-definidos)":
        st.markdown("Escolha um dos casos pré-carregados:")
        col1, col2 = st.columns([3, 1])
        with col1:
            preset_name = st.selectbox(
                "Caso:",
                list(DEMO_PRESETS.keys()),
                key="preset_selector"
            )
        with col2:
            st.write(" ")
            st.write(" ")
            if st.button("📥 Carregar este caso", use_container_width=True, type="primary"):
                preset = DEMO_PRESETS[preset_name]
                st.session_state.criteria_df = preset["criteria"].copy()
                st.session_state.matrix_df = preset["matrix"].copy()
                st.session_state.engine_weights = {}
                st.session_state.ahp_history = []
                st.success(f"✓ Carregado: {preset_name}")
                st.rerun()

    # -------- MODO 2: MANUAL --------
    elif data_source == "✏️ Manual (dimensões + paste do Excel)":
        st.markdown("**Opção A — Definir dimensões e criar matriz vazia:**")
        col_a, col_b, col_c = st.columns([1, 1, 2])
        n_alt_input = col_a.number_input("N.º Alternativas", min_value=2, max_value=50, value=5, step=1, key="n_alt_input")
        n_crit_input = col_b.number_input("N.º Critérios", min_value=2, max_value=15, value=4, step=1, key="n_crit_input")
        with col_c:
            st.write(" ")
            st.write(" ")
            if st.button("🆕 Criar matriz vazia", use_container_width=True):
                new_crits = pd.DataFrame({
                    "Critério": [f"C{i+1}" for i in range(n_crit_input)],
                    "Tipo": ["max"] * n_crit_input,
                    "Peso Manual": [1.0 / n_crit_input] * n_crit_input,
                })
                new_matrix = pd.DataFrame({"Alternativa": [f"Alt {i+1}" for i in range(n_alt_input)]})
                for c in new_crits["Critério"]:
                    new_matrix[c] = 0.0
                st.session_state.criteria_df = new_crits
                st.session_state.matrix_df = new_matrix
                st.session_state.engine_weights = {}
                st.success(f"✓ Matriz {n_alt_input}×{n_crit_input} criada — agora preencha os editores abaixo")
                st.rerun()

        st.markdown("---")
        st.markdown("**Opção B — Colar do Excel:**")
        st.caption(
            "1) No Excel, seleccione células incluindo cabeçalhos · 2) Ctrl+C · 3) Clique aqui · 4) Ctrl+V.\n\n"
            "A 1ª coluna deve ter os nomes das alternativas e a 1ª linha os nomes dos critérios. "
            "Aceita TAB (Excel), `;` ou espaços. Decimais com `,` ou `.`"
        )
        paste_text = st.text_area(
            "Colar aqui (Ctrl+V):",
            height=180,
            placeholder="Alternativa\tCusto\tQualidade\tPrazo\nForn A\t1200\t8\t15\nForn B\t1500\t6\t20",
            key="paste_area"
        )

        if paste_text and paste_text.strip():
            sep_guess = "\t"
            first_line = paste_text.strip().split("\n")[0]
            if "\t" in first_line:
                sep_guess = "\t"
            elif ";" in first_line:
                sep_guess = ";"
            elif "," in first_line and first_line.count(",") > 1:
                sep_guess = ","
            else:
                sep_guess = r"\s{2,}"

            try:
                if sep_guess == r"\s{2,}":
                    df_preview = pd.read_csv(StringIO(paste_text), sep=sep_guess, engine="python", dtype=str)
                else:
                    df_preview = pd.read_csv(StringIO(paste_text), sep=sep_guess, dtype=str)

                total_failures = 0
                for c in df_preview.columns[1:]:
                    nums, n_failed = clean_numeric_column(df_preview[c])
                    df_preview[c] = nums.fillna(0)
                    total_failures += n_failed

                first_col = df_preview.columns[0]
                df_preview = df_preview.rename(columns={first_col: "Alternativa"})

                st.caption(f"✓ Detectado: separador `{sep_guess}` · {len(df_preview)} alts × {len(df_preview.columns)-1} crits")
                if total_failures > 0:
                    st.warning(f"⚠️ {total_failures} valores não foram convertidos para número e ficaram a **0**. "
                               f"Verifique a tabela abaixo e corrija antes de confirmar.")
                st.dataframe(df_preview, hide_index=True, use_container_width=True)

                if st.button("📋 Confirmar e carregar", use_container_width=True, type="primary"):
                    crits_list = list(df_preview.columns[1:])
                    new_crits = pd.DataFrame({
                        "Critério": crits_list,
                        "Tipo": ["max"] * len(crits_list),
                        "Peso Manual": [1.0 / len(crits_list)] * len(crits_list),
                    })
                    st.session_state.criteria_df = new_crits
                    st.session_state.matrix_df = df_preview
                    st.session_state.engine_weights = {}
                    st.success(f"✓ Carregado: {len(df_preview)} alts × {len(crits_list)} crits")
                    st.rerun()
            except Exception as e:
                st.error(
                    f"❌ Erro a ler dados: {e}\n\n"
                    "**Verifique:**\n"
                    "• 1ª linha tem cabeçalhos (nomes dos critérios)\n"
                    "• 1ª coluna tem nomes das alternativas\n"
                    "• Valores são numéricos (decimais com `,` ou `.`)\n"
                    "• Separador é TAB (do Excel), `;` ou espaços"
                )

    # -------- MODO 3: QUADROS EM BRUTO Q1.3 + Q1.4 --------
    else:
        st.markdown(
            "Cole **dois quadros separados** como vêm no enunciado:"
        )
        st.markdown(
            "• **Quadro A (Q1.3) — Alternativas com atributos**: 1ª coluna = nome da alt, "
            "restantes = atributos numéricos (critérios) OU texto (metadados como Cliente, Estado)."
        )
        st.markdown(
            "• **Quadro B (Q1.4) — Critérios com pesos**: Código, Critério, Natureza (Benefício/Custo), Peso."
        )

        with st.expander("📖 Exemplo do formato (caso MCG do enunciado)", expanded=False):
            st.markdown("**A app aceita os valores TAL COMO VÊM no enunciado:**")
            st.markdown(
                "• Espaços nos milhares: `250 000 000` &nbsp;&nbsp;"
                "• Símbolo €: `300 000 €` &nbsp;&nbsp;"
                "• Percentagem: `25%` &nbsp;&nbsp;"
                "• Vírgula decimal: `0,462`"
            )
            st.code(
                "Quadro A — Alternativas (copia do enunciado 1.3):\n"
                "#\tRef. Interna\tCliente\tValor Pot.\tProb. Fecho\tEstado\n"
                "A1\t9786\tBe\t250 000 000 €\t25%\tCotação\n"
                "A2\t9780\tZf\t300 000 €\t35%\tCotação\n"
                "A3\t9768\tFo\t900 000 €\t50%\tCotação\n"
                "A4\t9755\tAd\t650 000 €\t50%\tCotação\n"
                "A5\t9736\tKb\t5 000 000 €\t40%\tCotação\n"
                "A6\t9735\tFo\t1 350 000 €\t50%\tNegociação\n"
                "A7\t9720\tFe\t10 500 000 €\t40%\tNegociação\n"
                "A8\t9706\tSt\t3 450 000 €\t40%\tNegociação\n"
                "A9\t9537\tKb\t15 000 000 €\t60%\tNegociação\n\n"
                "Quadro B — Critérios (copia do enunciado 1.4):\n"
                "Código\tCritério\tNatureza\tPeso\n"
                "C1\tValor Potencial do Contrato (VP)\tBenefício\t30%\n"
                "C2\tProbabilidade de Fecho (PF)\tBenefício\t22%\n"
                "C3\tEsforço Estimado (EE)\tCusto\t7%\n"
                "C4\tFit Estratégico (FE)\tBenefício\t15%\n"
                "C5\tUrgência / Prazo Decisão (UD)\tBenefício\t3%\n"
                "C6\tRelacionamento c/ Cliente (RC)\tBenefício\t17%\n",
                language="text"
            )
            st.caption(
                "⚠️ **Importante:** o Quadro 1.3 do enunciado só tem 2 critérios numéricos (Valor Pot. e Prob. Fecho). "
                "Para os restantes (C3-C6) terá de adicionar colunas com os valores das Secções 4.1 e 4.2."
            )

        col_q_a, col_q_b = st.columns(2)
        with col_q_a:
            paste_alts = st.text_area(
                "**Quadro A — Alternativas (Q1.3)** — Ctrl+V:",
                height=240, key="paste_alts_raw",
                placeholder="Alt\tCliente\tValor Pot\tEsforço\tEstado\nA1\tBe\t250000000\t24\tCotação\nA2\tZf\t300000\t8\tCotação"
            )
        with col_q_b:
            paste_crits = st.text_area(
                "**Quadro B — Critérios (Q1.4)** — Ctrl+V:",
                height=240, key="paste_crits_raw",
                placeholder="Código\tCritério\tNatureza\tPeso\nC1_VP\tValor Potencial\tBenefício\t0.462\nC2_PF\tProb. Fecho\tBenefício\t0.218"
            )

        def parse_paste(text):
            if not text or not text.strip():
                return None
            first_line = text.strip().split("\n")[0]
            sep = "\t" if "\t" in first_line else (";" if ";" in first_line else r"\s{2,}")
            try:
                if sep == r"\s{2,}":
                    df = pd.read_csv(StringIO(text), sep=sep, engine="python", dtype=str)
                else:
                    df = pd.read_csv(StringIO(text), sep=sep, dtype=str)
                return df
            except Exception:
                return None

        if paste_alts and paste_crits:
            df_alts_raw = parse_paste(paste_alts)
            df_crits_raw = parse_paste(paste_crits)

            if df_alts_raw is not None and df_crits_raw is not None:
                crits_cols_lower = [c.lower().strip() for c in df_crits_raw.columns]

                def find_col(targets, default=None):
                    for i, c in enumerate(crits_cols_lower):
                        for t in targets:
                            if t in c:
                                return df_crits_raw.columns[i]
                    return default

                col_code = find_col(["código", "code", "cod"])
                col_name = find_col(["critério", "criterio", "nome"])
                col_nat = find_col(["natureza", "tipo"])
                col_peso = find_col(["peso", "weight"])

                alt_col = df_alts_raw.columns[0]
                df_alts_raw = df_alts_raw.rename(columns={alt_col: "Alternativa"})

                numeric_cols = []
                metadata_cols = []
                total_failures_alts = 0
                for c in df_alts_raw.columns[1:]:
                    nums, n_failed = clean_numeric_column(df_alts_raw[c])
                    if nums.notna().mean() > 0.5:
                        df_alts_raw[c] = nums.fillna(0)
                        numeric_cols.append(c)
                        total_failures_alts += n_failed
                    else:
                        metadata_cols.append(c)

                st.caption(f"✓ Detectados: {len(df_alts_raw)} alts, "
                           f"{len(numeric_cols)} crit numéricos, {len(metadata_cols)} metadados")
                if total_failures_alts > 0:
                    st.warning(f"⚠️ {total_failures_alts} valores no Quadro A não foram convertidos "
                               f"para número (ficaram a 0). Reveja na tabela abaixo.")
                if metadata_cols:
                    st.caption(f"📝 Metadados (não usados para cálculo, mas guardados): {', '.join(metadata_cols)}")
                st.markdown("**Preview Quadro A:**")
                st.dataframe(df_alts_raw, hide_index=True, use_container_width=True)

                if col_code and col_name:
                    st.markdown("**Preview Quadro B (critérios identificados):**")
                    st.dataframe(df_crits_raw, hide_index=True, use_container_width=True)

                if st.button("📥 Importar tudo", use_container_width=True, type="primary"):
                    new_matrix = df_alts_raw[["Alternativa"] + numeric_cols].copy()
                    crit_list = []
                    for _, row in df_crits_raw.iterrows():
                        nome = str(row[col_name]) if col_name else ""
                        codigo = str(row[col_code]) if col_code else nome
                        nat = (str(row[col_nat]).lower() if col_nat else "max")
                        tipo = "min" if any(x in nat for x in ["custo", "cost", "min"]) else "max"
                        peso_raw = row[col_peso] if col_peso else "0"
                        peso_clean = clean_number_string(peso_raw)
                        try:
                            peso = float(peso_clean) if peso_clean else 0
                            if peso > 1:
                                peso = peso / 100
                        except Exception:
                            peso = 1.0 / len(df_crits_raw)
                        crit_list.append({"Critério": codigo, "Tipo": tipo, "Peso Manual": peso})
                    new_crits_df = pd.DataFrame(crit_list)

                    s = new_crits_df["Peso Manual"].sum()
                    if s > 0:
                        new_crits_df["Peso Manual"] = new_crits_df["Peso Manual"] / s

                    if len(numeric_cols) == len(new_crits_df):
                        rename_map = dict(zip(numeric_cols, new_crits_df["Critério"].tolist()))
                        new_matrix = new_matrix.rename(columns=rename_map)

                    st.session_state["alt_metadata"] = df_alts_raw[["Alternativa"] + metadata_cols].copy() if metadata_cols else None
                    st.session_state["crit_metadata"] = df_crits_raw.copy()

                    st.session_state.matrix_df = new_matrix
                    st.session_state.criteria_df = new_crits_df
                    st.session_state.engine_weights = {}
                    st.success(f"✓ Importado: {len(new_matrix)} alts × {len(new_crits_df)} crits. "
                               f"Metadados guardados para o relatório.")
                    st.rerun()
            else:
                st.warning("⚠️ Cole **ambos** os quadros (A e B) para activar o preview e o botão de importar.")

    # ============================================================================
    # SECÇÃO 2 — EDITOR DE CRITÉRIOS
    # ============================================================================
    st.markdown('<div class="data-section"><h3>📋 2. Editor de Critérios</h3></div>',
                unsafe_allow_html=True)
    st.caption(
        "Ajuste apenas o **nome** e o **tipo** (max = benefício, min = custo) de cada critério. "
        "Os pesos vêm sempre do **AHP** (aba 🔍 AHP — Matriz de Comparação Par-a-Par)."
    )

    edited_crit = st.data_editor(
        st.session_state.criteria_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="criteria_editor_tab",
        column_order=["Critério", "Tipo"],
        column_config={
            "Critério": st.column_config.TextColumn("Critério", required=True),
            "Tipo": st.column_config.SelectboxColumn("Tipo", options=["max", "min"], required=True),
        }
    )
    if edited_crit is not None and not edited_crit.equals(st.session_state.criteria_df):
        valid = edited_crit.dropna(subset=["Critério"])
        valid = valid[valid["Critério"].astype(str).str.strip() != ""]
        # garantir que a coluna Peso Manual existe (legado, com pesos iguais)
        if "Peso Manual" not in valid.columns:
            valid["Peso Manual"] = 1.0 / max(len(valid), 1)
        new_crits = valid["Critério"].astype(str).tolist()
        old_matrix = st.session_state.matrix_df.copy()
        new_matrix = pd.DataFrame({"Alternativa": old_matrix["Alternativa"]})
        for crit in new_crits:
            if crit in old_matrix.columns:
                new_matrix[crit] = old_matrix[crit]
            else:
                new_matrix[crit] = 0.0
        st.session_state.criteria_df = valid.reset_index(drop=True)
        st.session_state.matrix_df = new_matrix
        # quando os critérios mudam, os pesos AHP antigos deixam de ser válidos
        if "AHP" in st.session_state.engine_weights:
            old_n = len(st.session_state.engine_weights["AHP"])
            if old_n != len(new_crits):
                del st.session_state.engine_weights["AHP"]
        st.rerun()

    # ============================================================================
    # SECÇÃO 2-bis — MOTOR DE PESOS (sempre AHP — não há escolha)
    # ============================================================================
    st.markdown('<div class="data-section"><h3>⚙️ 2-bis. Motor de Pesos — sempre AHP</h3></div>',
                unsafe_allow_html=True)

    # forçar AHP sempre
    st.session_state.global_injection_on = True
    st.session_state.global_injection_engine = "AHP"

    if "AHP" in st.session_state.engine_weights:
        st.success(
            "✓ **Motor activo: AHP.** Os pesos vêm da Matriz de Comparação Par-a-Par (aba 🔍 AHP). "
            "Para os recalcular ou ajustar, vá à aba 🔍 AHP."
        )
    else:
        st.warning(
            "⚠️ **AHP ainda não foi calculado.** "
            "Tem de ir à aba **🔍 AHP**, preencher a Matriz de Comparação Par-a-Par (escala Saaty 1-9, "
            "com até 5 casas decimais — ex.: 0,11111), validar **CR < 0,10** e voltar aqui. "
            "Enquanto o AHP não estiver calculado, os modelos usam **pesos iguais (1/n)** como fallback."
        )

    # ============================================================================
    # SECÇÃO 3 — SENSIBILIDADE
    # ============================================================================
    st.markdown('<div class="data-section"><h3>🎯 3. Análise de Sensibilidade — variação ±%</h3></div>',
                unsafe_allow_html=True)
    st.caption(
        "Define a variação ± aplicada aos pesos em TODAS as abas (tornado / robustez)."
    )
    st.session_state.sensitivity_pct = st.slider(
        "Variação ± nos pesos (%):",
        5, 50, st.session_state.sensitivity_pct, 5,
        key="sens_slider_tab",
        help="Aplicada em TODAS as abas. Cada peso é variado isoladamente; restantes ajustados para Σ=1."
    )
    st.metric("Variação activa", f"±{st.session_state.sensitivity_pct}%")

    # ============================================================================
    # SECÇÃO 4 — VISUALIZAÇÃO DOS DADOS ACTUAIS
    # ============================================================================
    st.markdown('<div class="data-section"><h3>👁️ 4. Vista dos dados activos</h3></div>',
                unsafe_allow_html=True)

    matrix, alts, crits, types = get_decision_matrix()
    if not check_valid_input():
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alternativas", len(alts))
    c2.metric("Critérios", len(crits))
    c3.metric("Max (benefícios)", types.count("max"))
    c4.metric("Min (custos)", types.count("min"))

    st.subheader("Matriz de Decisão")
    display_df = pd.DataFrame(matrix, index=alts, columns=crits)
    st.dataframe(display_df.style.format("{:.5f}").background_gradient(cmap="Blues", axis=0),
                use_container_width=True)

    st.subheader("Critérios e Pesos Activos")
    show_active_weights_banner()

    st.subheader("Heatmap normalizado (min-max, sentido aplicado)")
    norm = normalize_minmax(matrix, types)
    norm_df = pd.DataFrame(norm, index=alts, columns=crits)
    st.dataframe(norm_df.style.format("{:.5f}").background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True)
    st.caption("1.0 = melhor; 0.0 = pior (com inversão automática para critérios de min).")

# =============================================================================
# TAB 2: AHP (FULL — matriz Saaty + consistência + iterações)
# =============================================================================
with tabs[2]:
    st.header("🔍 AHP — Analytic Hierarchy Process (Saaty, 1980)")
    purpose_box("Calcular pesos via <b>comparação par-a-par</b> escala Saaty 1-9. Valida com CR < 0.10. <b>Se CR ≥ 0.10, a app sugere iterativamente</b> que par corrigir até atingir consistência.")

    theory_box(
        "Teoria condensada",
        """
        <p>Determina pesos via <b>comparação par-a-par</b> usando a <b>escala Saaty 1-9</b>:</p>
        <ul>
            <li>1 = igual · 3 = moderadamente · 5 = fortemente · 7 = muito fortemente · 9 = extremamente</li>
            <li>a<sub>ji</sub> = 1/a<sub>ij</sub> (recíproco automático); diagonal = 1</li>
            <li>Vector de pesos = média geométrica das linhas (normalizada)</li>
            <li><b>Validação obrigatória</b>: CR = CI/RI < 0.10</li>
        </ul>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    n = len(crits)
    RI_TABLE = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
                9: 1.45, 10: 1.49, 11: 1.51, 12: 1.54, 13: 1.56, 14: 1.57, 15: 1.59}

    step_header("Passo 1: Matriz de Comparação Par-a-Par")
    st.latex(r"A = [a_{ij}],\quad a_{ji} = 1/a_{ij},\quad a_{ii} = 1")
    st.markdown("**Edite APENAS o triângulo superior (i < j). O inferior é recíproco automático.**")

    ahp_key = f"ahp_full_{'_'.join(crits)}"
    if ahp_key not in st.session_state:
        st.session_state[ahp_key] = pd.DataFrame(np.ones((n, n)), index=crits, columns=crits)
    if st.session_state[ahp_key].shape != (n, n):
        st.session_state[ahp_key] = pd.DataFrame(np.ones((n, n)), index=crits, columns=crits)

    edited_pw = st.data_editor(
        st.session_state[ahp_key],
        use_container_width=True, key="ahp_pw_full_editor",
        column_config={c: st.column_config.NumberColumn(
            c, min_value=0.00001, max_value=99.0, step=0.00001, format="%.5f"
        ) for c in crits}
    )

    A = edited_pw.values.astype(float).copy()
    for i in range(n):
        A[i, i] = 1.0
        for j in range(n):
            if i < j and A[i, j] > 0:
                A[j, i] = 1.0 / A[i, j]
    st.session_state[ahp_key] = pd.DataFrame(A, index=crits, columns=crits)

    step_header("Passo 2: Cálculo do Vector de Pesos (média geométrica)")
    st.latex(r"w_i = \frac{(\prod_j a_{ij})^{1/n}}{\sum_k (\prod_j a_{kj})^{1/n}}")
    geomean = np.prod(A, axis=1) ** (1.0 / n)
    w_ahp = geomean / geomean.sum()
    st.dataframe(pd.DataFrame({"Critério": crits, "Peso w_j": w_ahp,
                               "%": [f"{x*100:.3f}%" for x in w_ahp]})
                  .style.format({"Peso w_j": "{:.5f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 3: Verificação de Consistência (Saaty)")
    st.latex(r"\lambda_{max},\;CI = \frac{\lambda_{max}-n}{n-1},\;CR = \frac{CI}{RI(n)}")
    Aw = A @ w_ahp
    lam_max = (Aw / np.where(w_ahp == 0, 1e-9, w_ahp)).mean()
    CI = (lam_max - n) / (n - 1) if n > 1 else 0
    RI = RI_TABLE.get(n, 1.59)
    CR = CI / RI if RI > 0 else 0

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("n", n)
    cc2.metric("λ_max", f"{lam_max:.5f}")
    cc3.metric("CI", f"{CI:.5f}")
    cc4.metric("CR", f"{CR:.5f}",
              delta="✓ Consistente" if CR < 0.10 else "✗ Inconsistente",
              delta_color="normal" if CR < 0.10 else "inverse")

    if CR >= 0.10:
        st.markdown(
            f'<div class="warning-box">'
            f'<b>⚠️ CR = {CR:.5f} ≥ 0.10 — Matriz INCONSISTENTE.</b><br><br>'
            'Iteração: a aplicação identifica o <b>par mais problemático</b> e propõe correcção.'
            '</div>',
            unsafe_allow_html=True
        )
        st.latex(r"\text{desvio}(i, j) = \left| \ln\left(\frac{a_{ij}^{\text{observado}}}{w_i / w_j}\right) \right|")

        worst_i, worst_j, worst_dev = -1, -1, 0
        suggested_value = 1.0
        ideal_value = 1.0
        for i in range(n):
            for j in range(i+1, n):
                if w_ahp[j] != 0:
                    expected = w_ahp[i] / w_ahp[j]
                    observed = A[i, j]
                    if expected > 0 and observed > 0:
                        dev = abs(np.log(observed / expected))
                        if dev > worst_dev:
                            worst_dev = dev
                            worst_i, worst_j = i, j
                            ideal_value = expected
                            saaty_scale = [1/9, 1/7, 1/5, 1/3, 1/2, 1, 2, 3, 5, 7, 9]
                            suggested_value = min(saaty_scale, key=lambda x: abs(np.log(x) - np.log(ideal_value)))

        if worst_i >= 0:
            st.markdown("#### 🔧 Sugestão de Iteração")
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Par problemático", f"{crits[worst_i]} vs {crits[worst_j]}")
            colB.metric("Valor actual", f"{A[worst_i, worst_j]:.5f}")
            colC.metric("Valor ideal", f"{ideal_value:.5f}")
            colD.metric("Sugerido (Saaty)", f"{suggested_value:.5f}",
                       delta=f"Δ = {suggested_value - A[worst_i, worst_j]:+.5f}")

            colE, colF = st.columns([3, 1])
            with colE:
                st.info(
                    f"**Interpretação:** disse que {crits[worst_i]} vale **{A[worst_i, worst_j]:.5f}×** "
                    f"{crits[worst_j]}, mas os pesos calculados sugerem ~**{ideal_value:.5f}×**. "
                    f"Para aproximar, ajuste para **{suggested_value:.5f}**."
                )
            with colF:
                if st.button(f"✏️ Aplicar sugestão", type="primary", use_container_width=True):
                    new_df = st.session_state[ahp_key].copy()
                    new_df.iloc[worst_i, worst_j] = suggested_value
                    new_df.iloc[worst_j, worst_i] = 1.0 / suggested_value
                    st.session_state[ahp_key] = new_df
                    st.session_state.ahp_history.append({
                        "iteração": len(st.session_state.ahp_history) + 1,
                        "CR antes": round(CR, 5), "par": f"{crits[worst_i]} vs {crits[worst_j]}",
                        "valor antigo": round(A[worst_i, worst_j], 5),
                        "valor novo": round(suggested_value, 5)
                    })
                    st.success("✓ Sugestão aplicada.")
                    st.rerun()
    else:
        st.markdown(
            f'<div class="result-box">✅ <b>Matriz CONSISTENTE</b> — CR = {CR:.5f} < 0.10. Pesos AHP válidos.</div>',
            unsafe_allow_html=True
        )

    if st.session_state.ahp_history:
        with st.expander("📜 Histórico de iterações AHP"):
            st.dataframe(pd.DataFrame(st.session_state.ahp_history), hide_index=True, use_container_width=True)

    st.session_state.engine_weights["AHP"] = w_ahp

    st.markdown("---")
    step_header("Passo 4: Ranking das Alternativas (usando pesos AHP)")
    st.latex(r"S_i = \sum_{j=1}^n w_j^{AHP} \cdot u_j(x_{ij})")

    U = normalize_minmax(matrix, types)
    S = (U * w_ahp).sum(axis=1)
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Score AHP": S, "% do máx": S / S.max() * 100 if S.max() > 0 else S,
                           "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"Score AHP": "{:.5f}", "% do máx": "{:.2f}%"})
                  .background_gradient(cmap="RdYlGn", subset=["Score AHP"]),
                hide_index=True, use_container_width=True)

    best = df_res.iloc[0]["Alternativa"]
    st.markdown(
        f'<div class="result-box">🏆 Melhor segundo AHP: <b>{best}</b> '
        f'(Score = {df_res.iloc[0]["Score AHP"]:.5f}) '
        f'| CR = {CR:.5f} {"✓" if CR < 0.10 else "✗"}</div>',
        unsafe_allow_html=True
    )

    store_result("AHP", S, rank, higher_is_better=True)

    # ===== Gráficos AHP: pesos (vertical) + ranking (horizontal) =====
    step_header("📊 Gráficos AHP")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**Pesos dos Critérios (w_j)**")
        df_w_chart = pd.DataFrame({"Critério": crits, "Peso": w_ahp}).sort_values("Peso", ascending=False)
        colors_w = ["#1F4E78" if i < 2 else "#5B9BD5" for i in range(len(df_w_chart))]
        fig_w = go.Figure(go.Bar(
            x=df_w_chart["Critério"], y=df_w_chart["Peso"],
            marker=dict(color=colors_w),
            text=[f"{v*100:.2f}%" for v in df_w_chart["Peso"]], textposition="outside",
        ))
        fig_w.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="", yaxis_title="Peso w_j", showlegend=False,
        )
        st.plotly_chart(fig_w, use_container_width=True)
    with col_g2:
        st.markdown("**Ranking das Alternativas (Score AHP)**")
        df_chart = df_res.copy()
        colors_r = ["#1F4E78" if i < 3 else ("#5B9BD5" if i < 6 else "#F08080")
                    for i in range(len(df_chart))]
        fig_r = go.Figure(go.Bar(
            x=df_chart["Score AHP"], y=df_chart["Alternativa"], orientation="h",
            marker=dict(color=colors_r),
            text=[f"{v:.5f}" for v in df_chart["Score AHP"]], textposition="outside",
        ))
        fig_r.update_layout(
            height=max(280, 36 * len(alts)),
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis_title="Score AHP",
            yaxis=dict(autorange="reversed"), showlegend=False,
        )
        st.plotly_chart(fig_r, use_container_width=True)
    st.caption(f"**CR = {CR:.5f}** {'✅ Consistência OK' if CR < 0.10 else '❌ Rever julgamentos'}")

    def ahp_score_fn(w):
        U = normalize_minmax(matrix, types)
        return (U * w).sum(axis=1)
    render_sensitivity(ahp_score_fn, alts, crits, w_ahp, higher_is_better=True, key_suffix="ahp")

# =============================================================================
# TAB 3: TOPSIS
# =============================================================================
with tabs[3]:
    st.header("🎯 TOPSIS")
    purpose_box("Aplicar o método TOPSIS — mede a <b>distância à solução ideal</b> e ranqueia as alternativas.")
    theory_box("Teoria (Hwang & Yoon, 1981)",
        """<p>Método compensatório baseado em <b>distâncias</b> à solução ideal A⁺ e anti-ideal A⁻.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    def topsis_calc(W):
        R = normalize_vector(matrix); V = R * W
        Ap = np.array([V[:, j].max() if types[j] == "max" else V[:, j].min() for j in range(len(crits))])
        An = np.array([V[:, j].min() if types[j] == "max" else V[:, j].max() for j in range(len(crits))])
        Dp = np.sqrt(((V - Ap) ** 2).sum(axis=1)); Dn = np.sqrt(((V - An) ** 2).sum(axis=1))
        denom = np.where(Dp + Dn == 0, 1e-9, Dp + Dn)
        return Dn / denom, R, V, Ap, An, Dp, Dn

    CC, R, V, Ap, An, Dp, Dn = topsis_calc(weights)

    step_header("Passo 1 & 2: Matriz Normalizada (vectorial Euclidiana)")
    st.latex(r"r_{ij} = x_{ij} / \sqrt{\sum_k x_{kj}^2}")
    st.dataframe(pd.DataFrame(R, index=alts, columns=crits).style.format("{:.5f}"), use_container_width=True)

    step_header("Passo 3: Matriz Ponderada")
    st.latex(r"v_{ij} = w_j \cdot r_{ij}")
    st.dataframe(pd.DataFrame(V, index=alts, columns=crits).style.format("{:.5f}"), use_container_width=True)

    step_header("Passo 4: Soluções Ideal A⁺ e Anti-Ideal A⁻")
    st.dataframe(pd.DataFrame({"Critério": crits, "Tipo": types, "A⁺": Ap, "A⁻": An})
                  .style.format({"A⁺": "{:.5f}", "A⁻": "{:.5f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 5 & 6: Distâncias e CC*")
    st.latex(r"D_i^{\pm} = \sqrt{\sum_j (v_{ij} - A_j^{\pm})^2};\quad CC_i = D_i^- / (D_i^+ + D_i^-)")
    rank = pd.Series(CC).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "D⁺": Dp, "D⁻": Dn, "CC*": CC, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"D⁺": "{:.5f}", "D⁻": "{:.5f}", "CC*": "{:.5f}"})
                  .background_gradient(cmap="RdYlGn", subset=["CC*"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo TOPSIS: <b>{best}</b> (CC* = {df_res.iloc[0]["CC*"]:.5f})</div>', unsafe_allow_html=True)
    store_result("TOPSIS", CC, rank, True)

    # ===== Gráfico horizontal — Score de Proximidade Relativa =====
    step_header("📊 Gráfico — Score de Proximidade Relativa (CC*)")
    df_chart = df_res.copy()
    colors_t = ["#1F4E78" if i < 3 else ("#5B9BD5" if i < 6 else "#F08080")
                for i in range(len(df_chart))]
    fig_top = go.Figure(go.Bar(
        x=df_chart["CC*"], y=df_chart["Alternativa"], orientation="h",
        marker=dict(color=colors_t),
        text=[f"{v:.5f}" for v in df_chart["CC*"]], textposition="outside",
    ))
    fig_top.update_layout(
        height=max(280, 36 * len(alts)),
        margin=dict(l=10, r=60, t=20, b=10),
        xaxis_title="CC* (Coeficiente de Proximidade)",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig_top, use_container_width=True)
    st.caption("🟦 Azul escuro = Top-3 · 🟦 Azul claro = posições 4-6 · 🟥 Vermelho = últimos. Maior CC* = melhor.")

    render_sensitivity(lambda w: topsis_calc(w)[0], alts, crits, weights, True, "topsis")


# =============================================================================
# TAB 4: PROMETHEE II
# =============================================================================
with tabs[4]:
    st.header("📈 PROMETHEE II")
    purpose_box("Método de <b>fluxos de preferência par-a-par</b>. Permite 3 funções (Usual, Linear, Gaussiana).")
    theory_box("Teoria (Brans, 1985)",
        """<p>Método <b>não-compensatório</b> baseado em fluxos de preferência par-a-par.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()
    m = len(alts); n = len(crits)

    pref_type = st.radio("Função de preferência:", ["Tipo I (Usual)", "Tipo V (Linear)", "Tipo VI (Gaussiana)"],
                         horizontal=True, key="prom_pref")

    def pref(d, ftype, p=1.0, sigma=1.0):
        if d <= 0: return 0.0
        if ftype.startswith("Tipo I"): return 1.0
        elif ftype.startswith("Tipo V"): return min(d / p, 1.0)
        else: return 1.0 - np.exp(-d**2 / (2 * sigma**2))

    def prom_calc(W):
        params_p = [(matrix[:, j].max() - matrix[:, j].min()) * 0.5 if matrix[:, j].max() > matrix[:, j].min() else 1.0 for j in range(n)]
        params_s = [(matrix[:, j].max() - matrix[:, j].min()) * 0.3 if matrix[:, j].max() > matrix[:, j].min() else 1.0 for j in range(n)]
        pi = np.zeros((m, m))
        for a in range(m):
            for b in range(m):
                if a == b: continue
                for j in range(n):
                    d = matrix[a, j] - matrix[b, j] if types[j] == "max" else matrix[b, j] - matrix[a, j]
                    pi[a, b] += W[j] * pref(d, pref_type, params_p[j], params_s[j])
        phi_p = pi.sum(axis=1) / (m - 1) if m > 1 else pi.sum(axis=1)
        phi_n = pi.sum(axis=0) / (m - 1) if m > 1 else pi.sum(axis=0)
        return phi_p - phi_n, pi, phi_p, phi_n

    phi, pi, phi_p, phi_n = prom_calc(weights)

    step_header("Matriz π(a, b) — preferência agregada")
    st.dataframe(pd.DataFrame(pi, index=alts, columns=alts).style.format("{:.5f}")
                  .background_gradient(cmap="Greens"), use_container_width=True)

    step_header("Fluxos φ⁺, φ⁻ e φ líquido")
    rank = pd.Series(phi).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "φ⁺": phi_p, "φ⁻": phi_n, "φ líquido": phi, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"φ⁺": "{:.5f}", "φ⁻": "{:.5f}", "φ líquido": "{:.5f}"})
                  .background_gradient(cmap="RdYlGn", subset=["φ líquido"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo PROMETHEE II: <b>{best}</b> (φ = {df_res.iloc[0]["φ líquido"]:.5f})</div>', unsafe_allow_html=True)
    store_result("PROMETHEE II", phi, rank, True)

    # ===== Gráfico horizontal — Fluxo Líquido φ (waterfall style) =====
    step_header("📊 Gráfico — Fluxo Líquido φ")
    df_chart = df_res.copy()
    colors_p = ["#1F4E78" if v >= 0 else "#C00000" for v in df_chart["φ líquido"]]
    fig_prom = go.Figure(go.Bar(
        x=df_chart["φ líquido"], y=df_chart["Alternativa"], orientation="h",
        marker=dict(color=colors_p),
        text=[f"{v:+.5f}" for v in df_chart["φ líquido"]], textposition="outside",
    ))
    fig_prom.add_vline(x=0, line_dash="solid", line_color="#333", line_width=1)
    fig_prom.update_layout(
        height=max(280, 36 * len(alts)),
        margin=dict(l=10, r=60, t=20, b=10),
        xaxis_title="φ líquido (φ⁺ − φ⁻)",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig_prom, use_container_width=True)
    st.caption("🟦 Azul = fluxo positivo (preferida) · 🟥 Vermelho = fluxo negativo (preterida).")

    render_sensitivity(lambda w: prom_calc(w)[0], alts, crits, weights, True, "prom")


# =============================================================================
# TAB 5: COPRAS
# =============================================================================
with tabs[5]:
    st.header("📊 COPRAS")
    purpose_box("Função proporcional entre benefícios (S⁺) e custos (S⁻). Grau de utilidade U_i (%).")
    theory_box("Teoria (Zavadskas & Kaklauskas, 1996)", """<p>Avalia alternativas como função proporcional.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    def copras_calc(W):
        Xn = normalize_sum(matrix); V = Xn * W
        bi = [j for j, t in enumerate(types) if t == "max"]
        ci = [j for j, t in enumerate(types) if t == "min"]
        Sp = V[:, bi].sum(axis=1) if bi else np.zeros(len(alts))
        Sm = V[:, ci].sum(axis=1) if ci else np.zeros(len(alts))
        if Sm.sum() > 0 and (Sm > 0).all():
            sm = Sm.min(); ssm = Sm.sum(); si = (sm / Sm).sum()
            Q = Sp + (sm * ssm) / (Sm * si) if si > 0 else Sp
        else:
            Q = Sp
        U = (Q / Q.max() * 100) if Q.max() > 0 else Q * 0
        return Q, U, Xn, V, Sp, Sm

    Q, U, Xn, V, Sp, Sm = copras_calc(weights)

    step_header("S⁺ (Benefícios) e S⁻ (Custos)")
    st.dataframe(pd.DataFrame({"Alternativa": alts, "S⁺": Sp, "S⁻": Sm})
                  .style.format({"S⁺": "{:.5f}", "S⁻": "{:.5f}"}),
                hide_index=True, use_container_width=True)

    step_header("Q_i e U_i (%)")
    st.latex(r"Q_i = S_i^+ + \frac{S_{\min}^- \cdot \sum_i S_i^-}{S_i^- \cdot \sum_i (S_{\min}^-/S_i^-)}")
    rank = pd.Series(Q).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "S⁺": Sp, "S⁻": Sm, "Q_i": Q, "U_i (%)": U, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"S⁺": "{:.5f}", "S⁻": "{:.5f}", "Q_i": "{:.5f}", "U_i (%)": "{:.2f}"})
                  .background_gradient(cmap="RdYlGn", subset=["U_i (%)"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo COPRAS: <b>{best}</b> (U = {df_res.iloc[0]["U_i (%)"]:.2f}%)</div>', unsafe_allow_html=True)
    store_result("COPRAS", Q, rank, True)

    # ===== Gráfico horizontal — Grau de Utilidade U_i (%) =====
    step_header("📊 Gráfico — Grau de Utilidade U_i (%)")
    df_chart = df_res.copy()
    colors_c = ["#1F4E78" if i < 3 else ("#5B9BD5" if i < 6 else "#F08080")
                for i in range(len(df_chart))]
    fig_cop = go.Figure(go.Bar(
        x=df_chart["U_i (%)"], y=df_chart["Alternativa"], orientation="h",
        marker=dict(color=colors_c),
        text=[f"{v:.2f}%" for v in df_chart["U_i (%)"]], textposition="outside",
    ))
    fig_cop.update_layout(
        height=max(280, 36 * len(alts)),
        margin=dict(l=10, r=60, t=20, b=10),
        xaxis_title="U_i (%) — grau de utilidade (100% = melhor)",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig_cop, use_container_width=True)
    st.caption("🟦 Azul escuro = Top-3 · 🟦 Azul claro = posições 4-6 · 🟥 Vermelho = últimos. Maior U = melhor.")

    render_sensitivity(lambda w: copras_calc(w)[0], alts, crits, weights, True, "copras")

# TAB 14: DASHBOARD CONSOLIDADO
# =============================================================================
# =============================================================================
# TAB 6: DASHBOARD CONSOLIDADO — TUDO RECALCULADO DINAMICAMENTE
# Não depende de st.session_state.all_results. Cada render usa os dados actuais
# (matriz + tipos + pesos AHP/fallback) e recalcula os 4 modelos de raiz.
# =============================================================================
with tabs[6]:
    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()

    # ===== HEADER BAR =====
    ahp_status = "AHP" if "AHP" in st.session_state.engine_weights and len(st.session_state.engine_weights["AHP"]) == len(crits) else "Pesos iguais (fallback)"
    st.markdown(
        f"""<div style="background: linear-gradient(90deg, #1F4E78 0%, #2E75B6 100%);
        color: white; padding: 14px 22px; border-radius: 8px; margin-bottom: 14px;
        display: flex; justify-content: space-between; align-items: center;">
          <div>
            <span style="font-size: 20px; font-weight: 700;">📊 MCDM Dashboard</span>
            <span style="font-size: 16px; opacity: 0.85;"> | Priorização de Alternativas Multicritério</span>
          </div>
          <div style="font-size: 12px; opacity: 0.85; text-align: right;">
            Pesos: <b>{ahp_status}</b> · Variação SA ±{st.session_state.sensitivity_pct}%
          </div>
        </div>""",
        unsafe_allow_html=True
    )

    if ahp_status != "AHP":
        st.warning(
            "⚠️ AHP ainda não foi calculado — Dashboard a usar **pesos iguais (1/n)** como fallback. "
            "Vá à aba 🔍 AHP para definir os pesos correctos."
        )

    # ============================================================================
    # CÁLCULO DINÂMICO DE TODOS OS 4 MODELOS COM OS PESOS ACTUAIS
    # ============================================================================
    results = compute_all_models(matrix, types, weights)
    methods = ["AHP", "TOPSIS", "PROMETHEE II", "COPRAS"]

    # ===== Construir DataFrame consolidado =====
    df_dash = pd.DataFrame({"Alternativa": alts})
    for m in methods:
        df_dash[m] = results[m]["ranks"]
    df_dash["Posição Média"] = df_dash[methods].mean(axis=1).round(2)
    df_dash["Top-3 em N modelos"] = (df_dash[methods] <= 3).sum(axis=1)
    df_dash["Ranking Final"] = pd.Series(df_dash["Posição Média"]).rank(ascending=True, method='min').astype(int).values
    df_dash = df_dash.sort_values("Ranking Final").reset_index(drop=True)

    top3 = df_dash.head(3)["Alternativa"].tolist()
    top1 = top3[0] if top3 else "—"
    top1_idx = alts.index(top1) if top1 in alts else 0

    # ============================================================================
    # LINHA 1: Filtros · Ranking · Radar · Sensibilidade
    # ============================================================================
    col_filt, col_rank, col_radar, col_sens = st.columns([1.1, 2.2, 1.6, 1.6])

    # -------- Coluna 1: FILTROS & PARÂMETROS --------
    with col_filt:
        st.markdown("##### 🔧 Filtros & Parâmetros")
        st.markdown("**Modelo destacado:**")
        focus_model = st.radio(
            "Modelo destacado", methods,
            key="dash_focus_model", label_visibility="collapsed"
        )
        st.markdown("**Critério (sensibilidade):**")
        focus_crit = st.selectbox(
            "Critério", crits, key="dash_focus_crit", label_visibility="collapsed"
        )
        st.markdown("**Alternativa (radar):**")
        focus_alt = st.selectbox(
            "Alternativa", alts, key="dash_focus_alt", label_visibility="collapsed"
        )
        st.metric("Σ pesos", f"{weights.sum():.5f}")
        st.caption(f"Sensibilidade ±{st.session_state.sensitivity_pct}% (ajusta na aba 📋 Dados)")

    # -------- Coluna 2: RANKING CONSOLIDADO --------
    with col_rank:
        st.markdown("##### 🏆 Ranking Consolidado das Alternativas")

        def medalha(r):
            return "🥇" if r == 1 else ("🥈" if r == 2 else ("🥉" if r == 3 else ""))

        display = df_dash.copy()
        display["Medalha"] = display["Ranking Final"].apply(medalha)
        # adicionar score do modelo destacado para referência
        sc_focus = results[focus_model]["scores"]
        sc_by_alt = dict(zip(alts, sc_focus))
        display[f"Score {focus_model}"] = display["Alternativa"].map(sc_by_alt)

        compact = display[["Medalha", "Alternativa"] + methods + [f"Score {focus_model}", "Posição Média", "Ranking Final"]]
        st.dataframe(
            compact.style
                   .background_gradient(cmap="RdYlGn_r", subset=methods + ["Posição Média", "Ranking Final"])
                   .background_gradient(cmap="RdYlGn", subset=[f"Score {focus_model}"])
                   .format({"Posição Média": "{:.2f}", f"Score {focus_model}": "{:.5f}"}),
            hide_index=True, use_container_width=True,
            height=min(380, 60 + 35 * len(alts))
        )
        st.caption(f"🥇 = {top1} · Posição média = {df_dash.iloc[0]['Posição Média']:.2f} de {len(methods)} modelos")

    # -------- Coluna 3: RADAR PERFIL MULTICRITÉRIO --------
    with col_radar:
        st.markdown(f"##### 🎯 Perfil Multicritério — Top-3 + {focus_alt}")
        norm = normalize_minmax(matrix, types)
        norm_df = pd.DataFrame(norm, index=alts, columns=crits)

        fig_radar = go.Figure()
        colors_radar = ["#FFD700", "#C0C0C0", "#CD7F32"]
        labels_radar = ["🥇 1º", "🥈 2º", "🥉 3º"]
        for i, alt in enumerate(top3):
            vals = list(norm_df.loc[alt]) + [norm_df.loc[alt].iloc[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=crits + [crits[0]], fill="toself",
                name=f"{labels_radar[i]} {alt}",
                line=dict(color=colors_radar[i], width=2), opacity=0.55
            ))
        if focus_alt not in top3:
            vals = list(norm_df.loc[focus_alt]) + [norm_df.loc[focus_alt].iloc[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=crits + [crits[0]], fill="toself",
                name=f"⭐ {focus_alt}",
                line=dict(color="#9C27B0", width=3, dash="dot")
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True, legend=dict(font=dict(size=10), orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # -------- Coluna 4: SENSIBILIDADE — efeito de variar UM critério no MODELO DESTACADO --------
    with col_sens:
        st.markdown(f"##### 🌪️ Sensibilidade — {focus_crit} ±{st.session_state.sensitivity_pct}%")
        focus_crit_idx = crits.index(focus_crit)
        sens_pct = st.session_state.sensitivity_pct
        scorer_focus = {"AHP": calc_ahp_aggregate, "TOPSIS": calc_topsis,
                        "PROMETHEE II": calc_promethee2, "COPRAS": calc_copras}[focus_model]

        # peso +X% no critério escolhido, restantes ajustados para Σ=1
        nw_p = weights.copy()
        nw_p[focus_crit_idx] = weights[focus_crit_idx] * (1 + sens_pct/100)
        other_old = weights.sum() - weights[focus_crit_idx]
        other_new = 1 - nw_p[focus_crit_idx]
        if other_old > 0 and other_new > 0:
            for k in range(len(nw_p)):
                if k != focus_crit_idx:
                    nw_p[k] = weights[k] * (other_new / other_old)
        nw_p = nw_p / nw_p.sum()
        sc_p = scorer_focus(matrix, types, nw_p)

        df_s = pd.DataFrame({"Alt": alts, "Score": sc_p}).sort_values("Score", ascending=False)
        colors_s = ["#1F4E78" if a in top3 else "#5B9BD5" for a in df_s["Alt"]]
        fig_s = go.Figure(go.Bar(
            x=df_s["Score"], y=df_s["Alt"], orientation="h",
            marker=dict(color=colors_s),
            text=[f"{v:.4f}" for v in df_s["Score"]], textposition="outside",
        ))
        fig_s.update_layout(
            height=320, margin=dict(l=10, r=50, t=10, b=10),
            xaxis_title=f"Score {focus_model} com peso de {focus_crit} +{sens_pct}%",
            yaxis=dict(autorange="reversed"), showlegend=False,
        )
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")

    # ============================================================================
    # LINHA 2: Gráficos por modelo (4 colunas) — TUDO DINÂMICO
    # ============================================================================
    st.markdown("##### 📈 Scores por Modelo (calculados com os pesos actuais)")
    cols_models = st.columns(len(methods))
    label_x_map = {
        "TOPSIS": "CC* (Proximidade)",
        "PROMETHEE II": "φ líquido",
        "AHP": "Score AHP (Σ w·u)",
        "COPRAS": "Q_i"
    }
    for i, m in enumerate(methods):
        with cols_models[i]:
            sc = results[m]["scores"]
            rk = results[m]["ranks"]
            df_m = pd.DataFrame({"Alt": alts, "Score": sc, "Rank": rk}).sort_values("Rank")
            colors_m = ["#1F4E78" if r <= 3 else ("#5B9BD5" if r <= 6 else "#F08080") for r in df_m["Rank"]]
            if m == "PROMETHEE II":
                colors_m = ["#1F4E78" if v >= 0 else "#C00000" for v in df_m["Score"]]
            fig_m = go.Figure(go.Bar(
                x=df_m["Score"], y=df_m["Alt"], orientation="h",
                marker=dict(color=colors_m),
                text=[f"{v:+.4f}" if m == "PROMETHEE II" else f"{v:.4f}" for v in df_m["Score"]],
                textposition="outside",
            ))
            fig_m.update_layout(
                title=dict(text=f"<b>{m}</b> · 🥇 {df_m.iloc[0]['Alt']}", font=dict(size=13), x=0.5),
                height=max(220, 28 * len(alts)),
                margin=dict(l=10, r=50, t=40, b=10),
                xaxis_title=label_x_map.get(m, "Score"),
                yaxis=dict(autorange="reversed"), showlegend=False,
            )
            if m == "PROMETHEE II":
                fig_m.add_vline(x=0, line_dash="solid", line_color="#333", line_width=1)
            st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("---")

    # ============================================================================
    # LINHA 3: Sensibilidade por critério — robustez do CONSENSO Top-1
    # Para cada critério: perturba peso ±X%, recalcula TODOS os 4 modelos,
    # reconsolida o ranking final, e verifica se o Top-1 ainda é {top1}.
    # ============================================================================
    st.markdown(f"##### 🎯 Análise de Sensibilidade por Critério (±{st.session_state.sensitivity_pct}%) — robustez do Top-1 consensual **{top1}**")

    sens_pct = st.session_state.sensitivity_pct
    crit_status = []
    for j, crit in enumerate(crits):
        invertido = False
        details = []
        for sign, f in [("+", 1 + sens_pct/100), ("-", 1 - sens_pct/100)]:
            nw = weights.copy()
            nw[j] = weights[j] * f
            os_old = weights.sum() - weights[j]
            os_new = 1 - nw[j]
            if os_old > 0 and os_new > 0:
                for k in range(len(nw)):
                    if k != j:
                        nw[k] = weights[k] * (os_new / os_old)
            nw = nw / nw.sum()
            # recomputa TODOS os modelos + consolida
            new_results = compute_all_models(matrix, types, nw)
            df_test = pd.DataFrame({"Alternativa": alts})
            for mm in methods:
                df_test[mm] = new_results[mm]["ranks"]
            df_test["PM"] = df_test[methods].mean(axis=1)
            new_top1 = df_test.sort_values("PM").iloc[0]["Alternativa"]
            if new_top1 != top1:
                invertido = True
                details.append(f"{sign}: novo Top-1 = {new_top1}")
        crit_status.append({"Critério": crit, "Inverteu": invertido, "Detalhe": " · ".join(details) if details else "—"})

    sens_cols = st.columns(len(crits))
    for i, info in enumerate(crit_status):
        with sens_cols[i]:
            cor = "#C00000" if info["Inverteu"] else "#2E7D32"
            label = "❌ Top-1 mudou" if info["Inverteu"] else "✅ Top-1 estável"
            classif = "🔴 Sensível" if info["Inverteu"] else "🟢 Robusto"
            st.markdown(
                f"""<div style="background: white; border: 2px solid {cor}; border-radius: 8px;
                padding: 10px 8px; text-align: center; min-height: 120px;">
                  <div style="font-weight: 700; color: #1F4E78; font-size: 13px;">{info['Critério']}</div>
                  <div style="font-size: 10px; color: #666;">±{sens_pct}%</div>
                  <div style="margin: 6px 0; color: {cor}; font-weight: 600; font-size: 12px;">{label}</div>
                  <div style="font-size: 11px; color: {cor};">{classif}</div>
                  <div style="font-size: 10px; color: #666; margin-top: 4px;">{info['Detalhe']}</div>
                </div>""",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ============================================================================
    # LINHA 4: Recomendação & Notas + Critérios/Pesos
    # ============================================================================
    col_reco, col_crit = st.columns([1.4, 1.0])

    with col_reco:
        st.markdown("##### 🎯 Recomendação & Notas")
        total_top3 = sum(df_dash.head(3)["Top-3 em N modelos"].values)
        max_conv = 3 * len(methods)
        conv_pct = (total_top3 / max_conv * 100) if max_conv else 0
        if conv_pct >= 70:
            verdict_color = "#2e7d32"; verdict_label = "🟢 ALTA"
        elif conv_pct >= 40:
            verdict_color = "#f57c00"; verdict_label = "🟡 MODERADA"
        else:
            verdict_color = "#c62828"; verdict_label = "🔴 BAIXA"

        st.markdown(
            f"""<div style="background: linear-gradient(135deg, #1F4E78 0%, #2E75B6 100%);
            color: white; padding: 18px; border-radius: 10px;">
              <div style="font-size: 11px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px;">TOP-3 MCDM (consenso)</div>
              <div style="font-size: 22px; font-weight: 700; margin-top: 4px;">🥇 {top3[0] if len(top3)>0 else '—'}</div>
              <div style="font-size: 14px; opacity: 0.95;">
                🥈 {top3[1] if len(top3)>1 else '—'} · 🥉 {top3[2] if len(top3)>2 else '—'}
              </div>
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            f"""<div style="background: {verdict_color}; color: white; padding: 10px;
            border-radius: 6px; margin-top: 8px; text-align: center; font-weight: 600;">
              Convergência {verdict_label}: {conv_pct:.0f}% ({total_top3}/{max_conv} aparições no Top-3)
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown(f"**Modelos avaliados ({len(methods)}):** {', '.join(methods)}")
        sens_robust_count = sum(1 for s in crit_status if not s["Inverteu"])
        st.markdown(
            f"**Robustez SA ±{sens_pct}%:** {sens_robust_count}/{len(crits)} critérios mantêm {top1} como Top-1 consensual."
        )
        if sens_robust_count == len(crits):
            st.success(f"✅ Decisão MUITO ROBUSTA — {top1} é Top-1 em todos os cenários de SA.")
        elif sens_robust_count >= len(crits) * 0.7:
            st.info(f"🟡 Decisão MODERADAMENTE ROBUSTA — {top1} é Top-1 na maioria dos cenários.")
        else:
            st.warning(f"⚠️ Decisão SENSÍVEL — {top1} muda em vários cenários. Reavaliar pesos AHP.")

    with col_crit:
        st.markdown("##### 📊 Critérios e Pesos Activos")
        st.caption(f"Fonte: **{ahp_status}**")
        df_cwp = pd.DataFrame({
            "Crit.": crits, "Tipo": types,
            "Peso": weights, "%": [f"{w*100:.2f}%" for w in weights]
        }).sort_values("Peso", ascending=False)
        st.dataframe(
            df_cwp.style.format({"Peso": "{:.5f}"})
                  .background_gradient(cmap="Blues", subset=["Peso"]),
            hide_index=True, use_container_width=True
        )



# =============================================================================
# TAB 7: RELATÓRIO TÉCNICO
# =============================================================================
with tabs[7]:
    st.header("📑 Relatório Técnico — Estrutura dos 7 Capítulos")
    purpose_box(
        "Gera <b>relatório técnico completo</b> cumprindo a estrutura do <b>Capítulo 4</b> do enunciado."
    )

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()

    if not st.session_state.all_results:
        st.warning("⚠️ Execute pelo menos um modelo antes de gerar o relatório.")
        st.stop()

    methods = list(st.session_state.all_results.keys())
    df_dash = pd.DataFrame({"Alternativa": alts})
    for m in methods:
        df_dash[m] = st.session_state.all_results[m]["ranking"]
    df_dash["Posição Média"] = df_dash[methods].mean(axis=1).round(2)
    df_dash["Ranking Final"] = pd.Series(df_dash["Posição Média"]).rank(ascending=True, method='min').astype(int).values
    df_dash["Top-3 em N modelos"] = (df_dash[methods] <= 3).sum(axis=1)
    df_dash = df_dash.sort_values("Ranking Final")

    top1 = df_dash.iloc[0]["Alternativa"]
    top1_top3_count = df_dash.iloc[0]["Top-3 em N modelos"]
    conv_pct_top1 = (top1_top3_count / len(methods) * 100) if methods else 0
    top3 = df_dash.head(3)["Alternativa"].tolist()

    st.markdown(
        f"""<div style="background: linear-gradient(135deg, #1F4E78 0%, #2E75B6 100%);
        color: white; padding: 24px; border-radius: 10px; margin: 16px 0; text-align: center;">
        <div style="font-size: 12px; opacity: 0.9; text-transform: uppercase; letter-spacing: 2px;">RECOMENDAÇÃO MCDM FINAL</div>
        <div style="font-size: 42px; font-weight: 700; margin: 8px 0;">🏆 {top1}</div>
        <div style="font-size: 16px; opacity: 0.95;">
            Top-3: 🥇 {top3[0] if len(top3) > 0 else '—'} · 🥈 {top3[1] if len(top3) > 1 else '—'} · 🥉 {top3[2] if len(top3) > 2 else '—'}
        </div>
        <div style="font-size: 14px; opacity: 0.9; margin-top:6px;">
            Convergência: <b>{conv_pct_top1:.0f}%</b> · Modelos: <b>{len(methods)}</b>
        </div>
        </div>""",
        unsafe_allow_html=True
    )

    with st.expander("**📖 Capítulo 1 — Introdução** (1-2 pp)", expanded=True):
        st.markdown(f"""
        ### 1.1 Contexto
        Aplicação de MCDM a um problema real de priorização de **{len(alts)} alternativas** segundo **{len(crits)} critérios**.

        ### 1.2 Estrutura
        - **Cap. 2** — Dados · **Cap. 3** — Aplicação dos {len(methods)} modelos MCDM
        - **Cap. 4** — Sensibilidade · **Cap. 5** — Dashboard · **Cap. 6** — Comparação · **Cap. 7** — Conclusões
        """)

    with st.expander("**📊 Capítulo 2 — Dados e Pré-processamento** (3-5 pp)", expanded=False):
        st.markdown("### 2.1 Alternativas")
        alt_meta = st.session_state.get("alt_metadata", None)
        if alt_meta is not None and not alt_meta.empty:
            st.dataframe(alt_meta, hide_index=True, use_container_width=True)
        else:
            st.dataframe(pd.DataFrame({"Alternativa": alts}), hide_index=True, use_container_width=True)

        st.markdown("### 2.2 Critérios")
        crit_meta = st.session_state.get("crit_metadata", None)
        if crit_meta is not None and not crit_meta.empty:
            st.dataframe(crit_meta, hide_index=True, use_container_width=True)
        else:
            df_c = pd.DataFrame({"Código": crits, "Tipo": types, "Peso": weights,
                                 "%": [f"{w*100:.2f}%" for w in weights]})
            st.dataframe(df_c.style.format({"Peso": "{:.5f}"}), hide_index=True, use_container_width=True)

        st.markdown("### 2.3 Pesos Activos")
        eng_src = "Manual" if not st.session_state.global_injection_on else f"Motor: {st.session_state.global_injection_engine}"
        st.markdown(f"**Fonte:** {eng_src}")
        if st.session_state.ahp_history:
            st.markdown("**Iterações AHP aplicadas:**")
            st.dataframe(pd.DataFrame(st.session_state.ahp_history), hide_index=True, use_container_width=True)

        st.markdown("### 2.4 Matriz de Decisão")
        st.dataframe(pd.DataFrame(matrix, index=alts, columns=crits).style.format("{:.5f}")
                      .background_gradient(cmap="Blues", axis=0),
                    use_container_width=True)

    with st.expander(f"**🧮 Capítulo 3 — Aplicação dos {len(methods)} Modelos MCDM** (8-12 pp)", expanded=False):
        for m in methods:
            res = st.session_state.all_results[m]
            df_m = pd.DataFrame({
                "Alternativa": alts, "Score": res["scores"], "Ranking": res["ranking"],
            }).sort_values("Ranking")
            st.markdown(f"**3.{methods.index(m)+1} {m}** — top-1: {df_m.iloc[0]['Alternativa']} (score={df_m.iloc[0]['Score']:.5f})")
            st.dataframe(df_m.style.format({"Score": "{:.5f}"})
                          .background_gradient(cmap="RdYlGn", subset=["Score"]),
                        hide_index=True, use_container_width=True)

    with st.expander("**🎯 Capítulo 4 — Análise de Sensibilidade** (4-6 pp)", expanded=False):
        sp = st.session_state.sensitivity_pct
        st.markdown(f"### Variação ±{sp}% nos pesos")

        def quick_topsis(W):
            R = normalize_vector(matrix); V = R * W
            Ap = np.array([V[:, j].max() if types[j] == "max" else V[:, j].min() for j in range(len(crits))])
            An = np.array([V[:, j].min() if types[j] == "max" else V[:, j].max() for j in range(len(crits))])
            Dp = np.sqrt(((V - Ap) ** 2).sum(axis=1)); Dn = np.sqrt(((V - An) ** 2).sum(axis=1))
            return Dn / np.where(Dp + Dn == 0, 1e-9, Dp + Dn)
        base = quick_topsis(weights)
        base_rk = pd.Series(base).rank(ascending=False, method='min').astype(int).values
        n_inv = []
        for i_alt in range(len(alts)):
            count = 0
            for j in range(len(crits)):
                for f in [1 + sp/100, 1 - sp/100]:
                    nw = weights.copy(); nw[j] *= f
                    other_sum_old = weights.sum() - weights[j]
                    other_sum_new = 1 - nw[j]
                    if other_sum_old > 0 and other_sum_new > 0:
                        for k in range(len(nw)):
                            if k != j: nw[k] = weights[k] * (other_sum_new / other_sum_old)
                    nw = nw / nw.sum()
                    sc = quick_topsis(nw)
                    rk = pd.Series(sc).rank(ascending=False, method='min').astype(int).values
                    if rk[i_alt] != base_rk[i_alt]: count += 1
            n_inv.append(count)
        max_cenarios = 2 * len(crits)
        df_rob = pd.DataFrame({
            "Alternativa": alts, "Rank Base": base_rk,
            "Inversões": n_inv, "Total cenários": [max_cenarios] * len(alts),
            "Robustez (%)": [(1 - c/max_cenarios) * 100 for c in n_inv],
            "Classificação": ["🟢 ESTÁVEL" if c == 0 else ("🟡 MODERADA" if c <= 3 else "🔴 INSTÁVEL") for c in n_inv]
        }).sort_values("Rank Base")

        st.dataframe(df_rob.style.format({"Robustez (%)": "{:.1f}"})
                      .background_gradient(cmap="RdYlGn", subset=["Robustez (%)"]),
                    hide_index=True, use_container_width=True)

        estaveis = df_rob[df_rob["Inversões"] == 0]["Alternativa"].tolist()
        instaveis = df_rob[df_rob["Inversões"] > 3]["Alternativa"].tolist()
        if estaveis:
            st.success(f"✅ ROBUSTAS: {', '.join(estaveis)}")
        if instaveis:
            st.warning(f"⚠️ SENSÍVEIS: {', '.join(instaveis)}")

    with st.expander("**🎛️ Capítulo 5 — Dashboard e Reutilizabilidade** (3-5 pp)", expanded=False):
        st.markdown(f"""
        ### Arquitectura
        {len(methods)} modelo(s) MCDM (TOPSIS, PROMETHEE II, COPRAS) + AHP como motor de pesos.
        Estrutura em 8 abas: 🏠 Início · 📋 Dados · 🔍 AHP · 🎯 TOPSIS · 📈 PROMETHEE II · 📊 COPRAS · 🏆 Dashboard · 📑 Relatório.

        ### Guia de Utilização
        1. Aba 📋 Dados → escolher fonte (Demo / Manual / Quadros em bruto Q1.3+Q1.4)
        2. Aba 📋 Dados → editar critérios e matriz (precisão 5 decimais)
        3. Aba 📋 Dados (secção 2-bis) → escolher motor de pesos (Manual ou AHP)
        4. Aba 🔍 AHP → se escolheu AHP, preencher matriz par-a-par e validar CR<0,10
        5. Executar abas 🎯 TOPSIS, 📈 PROMETHEE II e 📊 COPRAS
        6. Aba 🏆 Dashboard → vista executiva consolidada
        7. Aba 📑 Relatório → descarregar CSV/Excel/Markdown

        ### Reutilizabilidade
        - Nada hardcoded — suporta até 50 alts × 15 crits.
        - Tipo (max/min) propagado a todos os modelos. MAX/MIN respeitado em todas as fórmulas.
        """)

    with st.expander("**⚖️ Capítulo 6 — Comparação e Recomendação** (3-4 pp)", expanded=False):
        st.dataframe(df_dash.style.background_gradient(cmap="RdYlGn_r",
                      subset=methods + ["Posição Média", "Ranking Final"]),
                    hide_index=True, use_container_width=True)
        st.markdown(f"### Recomendação Final: **{top1}** (convergência {conv_pct_top1:.0f}%)")

    with st.expander("**🎓 Capítulo 7 — Conclusões** (1-2 pp)", expanded=False):
        st.markdown(f"""
        - Aplicaram-se **{len(methods)} modelos MCDM**.
        - Top-1 consensual: **{top1}** (convergência {conv_pct_top1:.0f}%).
        - {len(df_rob[df_rob['Inversões'] == 0])} alternativas robustas em ±{st.session_state.sensitivity_pct}%.
        """)

    with st.expander("**📚 Referências (APA 7ª)**", expanded=False):
        st.markdown("""
        - Brans, J.-P., & Vincke, P. (1985). A Preference Ranking Organisation Method (The PROMETHEE Method for Multiple Criteria Decision-Making). *Management Science*, 31(6), 647-656.
        - Hwang, C.-L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer-Verlag.
        - Saaty, T. L. (1980). *The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation*. McGraw-Hill.
        - Zavadskas, E. K., & Kaklauskas, A. (1996). *Multiple Criteria Evaluation of Buildings* (COPRAS Method). Vilnius Technika.
        """)

    st.markdown("---")
    st.subheader("📥 Exportar Relatório")

    def df_to_md(df):
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = []
        for _, row in df.iterrows():
            cells = [f"{v:.5f}" if isinstance(v, float) else str(v) for v in row]
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, sep] + rows)

    sp_md = st.session_state.sensitivity_pct
    eng_src_md = "Manual" if not st.session_state.global_injection_on else f"Motor: {st.session_state.global_injection_engine}"
    df_w_md = pd.DataFrame({"Critério": crits, "Tipo": types, "Peso": weights})

    md_lines = [
        f"# Relatório Técnico MCDM",
        f"\n**Data:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"\n## 🏆 Recomendação Final: {top1}",
        f"- Top-3: {', '.join(top3)}",
        f"- Convergência: {conv_pct_top1:.0f}%",
        f"- Modelos: {len(methods)} — {', '.join(methods)}",
        f"\n## Capítulo 2 — Dados",
        f"\n### Pesos (fonte: {eng_src_md})",
        df_to_md(df_w_md),
        f"\n### Matriz de Decisão",
        df_to_md(pd.DataFrame(matrix, index=alts, columns=crits).reset_index().rename(columns={'index': 'Alt'})),
    ]

    md_lines.append(f"\n## Capítulo 3 — Modelos MCDM")
    for m in methods:
        res = st.session_state.all_results[m]
        df_m = pd.DataFrame({"Alt": alts, "Score": res["scores"], "Rank": res["ranking"]}).sort_values("Rank")
        md_lines.append(f"\n### {m}")
        md_lines.append(df_to_md(df_m))

    md_lines.append(f"\n## Capítulo 4 — Sensibilidade (±{sp_md}%)")
    md_lines.append(df_to_md(df_rob))

    md_lines.append(f"\n## Capítulo 6 — Recomendação")
    md_lines.append(df_to_md(df_dash))

    md_lines.append(f"\n## Capítulo 7 — Conclusões")
    md_lines.append(f"Top-1: **{top1}** ({conv_pct_top1:.0f}%).")

    md_report = "\n".join(md_lines)

    csv_buffer = StringIO()
    df_dash.to_csv(csv_buffer, index=False)

    excel_buf = BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        df_dash.to_excel(writer, sheet_name="Cap6_Rankings", index=False)
        pd.DataFrame(matrix, index=alts, columns=crits).to_excel(writer, sheet_name="Cap2_Matriz")
        df_w_md.to_excel(writer, sheet_name="Cap2_Pesos", index=False)
        df_rob.to_excel(writer, sheet_name="Cap4_Robustez", index=False)
        sc_data = {"Alternativa": alts}
        for m in methods:
            sc_data[m] = st.session_state.all_results[m]["scores"]
        pd.DataFrame(sc_data).to_excel(writer, sheet_name="Cap3_Scores", index=False)
        if st.session_state.ahp_history:
            pd.DataFrame(st.session_state.ahp_history).to_excel(writer, sheet_name="Cap2_AHPIter", index=False)
    excel_buf.seek(0)

    ec1, ec2, ec3 = st.columns(3)
    ec1.download_button("📥 CSV (Cap.6 Rankings)", csv_buffer.getvalue(), "rel_tecnico_cap6.csv", "text/csv",
                        use_container_width=True)
    ec2.download_button("📥 Excel (todos capítulos)", excel_buf.getvalue(), "rel_tecnico_completo.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True)
    ec3.download_button("📥 Markdown (relatório completo)", md_report.encode("utf-8"),
                        "rel_tecnico_completo.md", "text/markdown", use_container_width=True)

    st.caption("💡 O markdown pode ser convertido para PDF com Pandoc.")
