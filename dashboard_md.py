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
        st.session_state.criteria_df = pd.DataFrame({
            "Critério": [], "Tipo": [], "Peso Manual": []
        })
    if "matrix_df" not in st.session_state:
        st.session_state.matrix_df = pd.DataFrame({"Alternativa": []})
    if "global_injection_on" not in st.session_state:
        st.session_state.global_injection_on = True   # sempre AHP
    if "global_injection_engine" not in st.session_state:
        st.session_state.global_injection_engine = "AHP"
    if "engine_weights" not in st.session_state:
        st.session_state.engine_weights = {}
    if "sensitivity_pct" not in st.session_state:
        st.session_state.sensitivity_pct = 20
    if "ahp_history" not in st.session_state:
        st.session_state.ahp_history = []
    if "ahp_matrix_pasted" not in st.session_state:
        st.session_state.ahp_matrix_pasted = None
    if "all_results" not in st.session_state:
        st.session_state.all_results = {}

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
            <p>Esta aba precisa de uma <b>matriz de decisão</b> + <b>matriz AHP par-a-par</b>.</p>
            <p>Na aba <b>📋 Dados</b>:</p>
            <p>1. Cole a <b>matriz AHP dos critérios</b> (com coluna MAX/MIN)<br>
               2. Cole a <b>tabela das alternativas</b> × critérios<br>
               3. Prima <b>Processar pastes</b></p>
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
    "**Decisão Multicritério** · Dashboard interactivo · 4 modelos (AHP, TOPSIS, PROMETHEE II, COPRAS) · "
    "Pesos vindos do AHP automaticamente · Sensibilidade universal · Precisão 5 decimais"
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
    "🏆 Dashboard",
    "📋 Dados",
    "🔍 AHP",
    "🎯 TOPSIS",
    "📈 PROMETHEE II",
    "📊 COPRAS",
]
tabs = st.tabs(TAB_LABELS)


# =============================================================================
# TAB 1: DADOS — INPUT ÚNICO POR PASTE (Critérios+AHP + Alternativas)
# =============================================================================
with tabs[1]:
    st.header("📋 Dados — Configuração")
    purpose_box(
        "Cole 2 quadros aqui: (1) a <b>matriz AHP par-a-par dos critérios</b> com a coluna MAX/MIN; "
        "(2) a <b>tabela das alternativas</b> × critérios. O AHP é calculado <b>automaticamente</b> e "
        "alimenta TOPSIS, PROMETHEE II e COPRAS."
    )

    # ============================================================================
    # PARSER ROBUSTO
    # ============================================================================
    def split_paste_rows(text):
        rows = []
        # IMPORTANTE: só remover newlines, NÃO whitespace inicial (senão perdemos o canto vazio da matriz AHP)
        for line in text.strip("\n\r").splitlines():
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
            # tab é o separador natural quando vem do Excel
            if "\t" in line:
                cells = line.split("\t")
            elif ";" in line:
                cells = line.split(";")
            else:
                cells = [c for c in line.split() if c]
            cells = [c.strip() for c in cells]
            rows.append(cells)
        return rows

    def parse_criteria_paste(text):
        """Foto 2 — matriz AHP par-a-par com header (códigos) e coluna MAX/MIN.
        Devolve (codes, types, ahp_matrix) ou levanta ValueError.

        Layout esperado do corpo: row_code | val_1 | val_2 | ... | val_n | max/min
        Layout esperado do header: [vazio opcional] | code_1 | code_2 | ... | code_n | MAX/MIN
        """
        rows = split_paste_rows(text)
        if not rows:
            raise ValueError("Sem dados")

        # 1) detectar header — 1ª linha com >=3 strings não numéricas
        header_idx = None
        for i, r in enumerate(rows):
            non_num = sum(1 for c in r if c.strip() and not _is_num(c))
            if non_num >= 3:
                header_idx = i; break
        if header_idx is None:
            raise ValueError("Não encontrei linha de cabeçalho com códigos de critérios.")

        header = rows[header_idx]
        # 2) extrair códigos do header — ignorar vazios e células do tipo max/min/tipo
        col_codes = []
        for c in header:
            cl = c.strip().lower()
            if not cl: continue
            if "max" in cl or "min" in cl or "tipo" in cl or "natureza" in cl: continue
            col_codes.append(c.strip())
        if len(col_codes) < 2:
            raise ValueError(f"Cabeçalho de critérios precisa de ≥2 códigos; encontrei {col_codes}")
        n = len(col_codes)

        # 3) corpo — para cada linha, r[0]=código, r[1..n]=valores, resto=max/min
        body = rows[header_idx + 1:]
        types_out = []
        codes_row = []
        ahp = []
        for r in body:
            if not r or all(not c.strip() for c in r):
                continue
            row_code = r[0].strip()
            if not row_code or "max" in row_code.lower() or "min" in row_code.lower():
                continue
            # valores: posições 1 a n
            if len(r) < n + 1:
                raise ValueError(f"Linha '{row_code}' tem só {len(r)} células; esperava ≥{n+1}.")
            try:
                vals = [float(clean_number_string(r[1 + k])) for k in range(n)]
            except Exception as e:
                raise ValueError(f"Linha '{row_code}': erro a converter valores → {e}")
            # tipo: procurar max/min nas células restantes (após os n valores)
            tipo = "max"  # default
            for c in r[n+1:]:
                cl = c.strip().lower()
                if cl in ("max", "min"):
                    tipo = cl; break
                if "benef" in cl: tipo = "max"; break
                if "cust" in cl:  tipo = "min"; break
            codes_row.append(row_code)
            types_out.append(tipo)
            ahp.append(vals)
            if len(ahp) == n:
                break

        if len(codes_row) != n:
            raise ValueError(f"Esperava {n} linhas de critérios (= colunas), encontrei {len(codes_row)}: {codes_row}")

        # 4) validar que códigos de linha == códigos de coluna
        if codes_row != col_codes:
            if set(codes_row) == set(col_codes):
                # mesma ordem é importante para a matriz AHP — reordenar para coincidir com col_codes
                order = [codes_row.index(c) for c in col_codes]
                ahp = [ahp[i] for i in order]
                types_out = [types_out[i] for i in order]
                codes_row = col_codes[:]
            else:
                raise ValueError(
                    f"Códigos das linhas {codes_row} ≠ códigos das colunas {col_codes}. "
                    "Verifique que a matriz é quadrada e usa os mesmos códigos em linhas e colunas."
                )

        # 5) garantir reciprocidade (upper triangle wins)
        ahp = np.array(ahp, dtype=float)
        for i in range(n):
            ahp[i, i] = 1.0
            for j in range(n):
                if i < j and ahp[i, j] > 0:
                    ahp[j, i] = 1.0 / ahp[i, j]
        return col_codes, types_out, ahp

    def parse_alts_paste(text, expected_crits=None):
        """Foto 3 — alternativas × critérios. Aceita rótulos 'Critérios Quantitativos/Qualitativos'
        antes da linha de cabeçalho. Devolve (alt_names, matrix)."""
        rows = split_paste_rows(text)
        if not rows:
            raise ValueError("Sem dados")

        # encontrar a linha de cabeçalho: 1ª linha cuja célula 1 começa por 'alt' OU
        # cujas colunas seguintes contenham códigos como expected_crits.
        header_idx = None
        for i, r in enumerate(rows):
            if not r: continue
            c0 = r[0].strip().lower()
            if c0.startswith("alt"):
                header_idx = i; break
            if expected_crits:
                tail = [c.strip() for c in r[1:]]
                if any(c in tail for c in expected_crits):
                    header_idx = i; break

        if header_idx is None:
            # fallback: primeira linha com >=2 strings não numéricas
            for i, r in enumerate(rows):
                non_num = sum(1 for c in r if c.strip() and not _is_num(c))
                if non_num >= 2:
                    header_idx = i; break

        if header_idx is None:
            raise ValueError("Não encontrei linha de cabeçalho das alternativas.")

        header = rows[header_idx]
        # detectar quais colunas correspondem aos critérios esperados
        if expected_crits:
            col_idx = []
            for c in expected_crits:
                if c in header:
                    col_idx.append(header.index(c))
                else:
                    raise ValueError(
                        f"Critério '{c}' (que está na matriz AHP) não aparece na tabela de alternativas. "
                        f"Cabeçalho recebido: {header}"
                    )
            crit_codes = list(expected_crits)
        else:
            col_idx = list(range(1, len(header)))
            crit_codes = header[1:]

        body = rows[header_idx + 1:]
        alt_names = []
        matrix_rows = []
        for r in body:
            if not r or all(not c.strip() for c in r): continue
            alt_name = r[0].strip()
            if not alt_name: continue
            try:
                vals = []
                for ci in col_idx:
                    if ci < len(r):
                        v = clean_number_string(r[ci])
                        vals.append(float(v) if v else 0.0)
                    else:
                        vals.append(0.0)
                alt_names.append(alt_name)
                matrix_rows.append(vals)
            except Exception:
                continue
        if not alt_names:
            raise ValueError("Não encontrei linhas de alternativas válidas.")
        return alt_names, np.array(matrix_rows, dtype=float)

    def _is_num(s):
        try:
            float(clean_number_string(s)); return True
        except Exception:
            return False

    def compute_ahp_weights_and_cr(A):
        """Geomean + CR. Devolve (weights, CR)."""
        n = A.shape[0]
        gm = np.prod(A, axis=1) ** (1.0 / n)
        w = gm / gm.sum()
        Aw = A @ w
        lam_max = (Aw / w).mean()
        CI = (lam_max - n) / (n - 1) if n > 1 else 0.0
        RI_table = {1:0.0, 2:0.0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41,
                    9:1.45, 10:1.49, 11:1.51, 12:1.48, 13:1.56, 14:1.57, 15:1.59}
        RI = RI_table.get(n, 1.59)
        CR = CI / RI if RI > 0 else 0.0
        return w, lam_max, CI, CR

    # ============================================================================
    # SECÇÃO 1 — PASTE DO QUADRO DE CRITÉRIOS (FOTO 2)
    # ============================================================================
    st.markdown(
        '<div class="data-section"><h3>📥 1. Matriz AHP dos Critérios (par-a-par + MAX/MIN)</h3></div>',
        unsafe_allow_html=True
    )
    with st.expander("ℹ️ Como preparar este paste (exemplo Foto 2)", expanded=False):
        st.markdown(
            """
            **Estrutura esperada** — copie do Excel a sua matriz par-a-par dos critérios, **com cabeçalhos** e a coluna **MAX/MIN** no fim:

            ```
                     C1_VP    C2_PF    C3_EE    C4_FE    C5_UD    C6_RC    MAX/MIN ?
            C1_VP    1.0000   4.0000   8.0000   6.0000   9.0000   5.0000   max
            C2_PF    0.2500   1.0000   8.0000   3.0000   9.0000   2.0000   max
            C3_EE    0.1250   0.1250   1.0000   0.1111   1.0000   0.1111   min
            C4_FE    0.1667   0.3333   9.0000   1.0000   9.0000   0.1667   max
            C5_UD    0.1111   0.1111   1.0000   0.1111   1.0000   0.1111   max
            C6_RC    0.2000   0.5000   9.0000   6.0000   9.0000   1.0000   max
            ```
            • Escala Saaty: **1 = igual**, **3 = moderada**, **5 = forte**, **7 = muito forte**, **9 = extrema**; recíprocos como **0.11111 (=1/9)**, **0.1667 (=1/6)** etc.
            • A coluna **MAX/MIN ?** indica o tipo do critério: `max` = benefício, `min` = custo.
            • Pode colar até 5 casas decimais (`0.11111`).
            """
        )

    default_crit_paste = """\tC1_VP\tC2_PF\tC3_EE\tC4_FE\tC5_UD\tC6_RC\tMAX/MIN ?
C1_VP\t1.0000\t4.0000\t8.0000\t6.0000\t9.0000\t5.0000\tmax
C2_PF\t0.2500\t1.0000\t8.0000\t3.0000\t9.0000\t2.0000\tmax
C3_EE\t0.1250\t0.1250\t1.0000\t0.1111\t1.0000\t0.1111\tmin
C4_FE\t0.1667\t0.3333\t9.0000\t1.0000\t9.0000\t0.1667\tmax
C5_UD\t0.1111\t0.1111\t1.0000\t0.1111\t1.0000\t0.1111\tmax
C6_RC\t0.2000\t0.5000\t9.0000\t6.0000\t9.0000\t1.0000\tmax"""

    crit_text = st.text_area(
        "Cole aqui a matriz AHP (com cabeçalhos + coluna MAX/MIN ?):",
        value=st.session_state.get("crit_paste_text", default_crit_paste),
        height=200, key="crit_paste_area"
    )
    if crit_text != st.session_state.get("crit_paste_text", ""):
        st.session_state.crit_paste_text = crit_text

    # ============================================================================
    # SECÇÃO 2 — PASTE DAS ALTERNATIVAS (FOTO 3)
    # ============================================================================
    st.markdown(
        '<div class="data-section"><h3>📥 2. Tabela das Alternativas × Critérios</h3></div>',
        unsafe_allow_html=True
    )
    with st.expander("ℹ️ Como preparar este paste (exemplo Foto 3)", expanded=False):
        st.markdown(
            """
            **Estrutura esperada** — copie do Excel a tabela com `Alternativa` na 1ª coluna e os mesmos códigos de critérios da matriz AHP:

            ```
            Alternativa  C1_VP        C2_PF   C3_EE   C4_FE   C5_UD   C6_RC
            A1           250,000,000  0.25    24      4       180     4
            A2           300,000      0.35    8       5       60      5
            A3           900,000      0.50    8       3       60      5
            ...          ...          ...     ...     ...     ...     ...
            ```
            • Aceita valores com **vírgula, ponto, "€", "%"** ou espaços nos milhares (`250 000 000`).
            • Os códigos dos critérios devem ser **exactamente os mesmos** da matriz AHP acima.
            • Aceita rótulos como "Critérios Quantitativos / Qualitativos" acima do cabeçalho — são ignorados.
            """
        )

    default_alts_paste = """Alternativa\tC1_VP\tC2_PF\tC3_EE\tC4_FE\tC5_UD\tC6_RC
A1\t250000000\t0.25\t24\t4\t180\t4
A2\t300000\t0.35\t8\t5\t60\t5
A3\t900000\t0.50\t8\t3\t60\t5
A4\t650000\t0.50\t8\t3\t90\t3
A5\t5000000\t0.40\t24\t4\t30\t3
A6\t1350000\t0.50\t8\t3\t60\t5
A7\t10500000\t0.40\t16\t3\t180\t4
A8\t3450000\t0.40\t8\t3\t60\t4
A9\t15000000\t0.60\t24\t4\t300\t3"""

    alts_text = st.text_area(
        "Cole aqui a tabela das alternativas:",
        value=st.session_state.get("alts_paste_text", default_alts_paste),
        height=240, key="alts_paste_area"
    )
    if alts_text != st.session_state.get("alts_paste_text", ""):
        st.session_state.alts_paste_text = alts_text

    # ============================================================================
    # SECÇÃO 3 — PROCESSAR (parse + AHP + matriz)
    # ============================================================================
    st.markdown('<div class="data-section"><h3>⚙️ 3. Processar dados</h3></div>', unsafe_allow_html=True)
    if st.button("🚀 Processar pastes (parse + calcular AHP + carregar)", type="primary", use_container_width=True):
        try:
            codes, types_parsed, ahp_matrix = parse_criteria_paste(crit_text)
            alt_names, dec_matrix = parse_alts_paste(alts_text, expected_crits=codes)
            if dec_matrix.shape[1] != len(codes):
                raise ValueError(f"A tabela de alternativas tem {dec_matrix.shape[1]} colunas de critério, "
                                 f"mas a matriz AHP define {len(codes)}.")
            w_ahp, lam_max, CI, CR = compute_ahp_weights_and_cr(ahp_matrix)

            # commit ao session_state
            st.session_state.criteria_df = pd.DataFrame({
                "Critério": codes, "Tipo": types_parsed,
                "Peso Manual": [1.0/len(codes)] * len(codes),  # legado, não usado
            })
            st.session_state.matrix_df = pd.DataFrame(dec_matrix, columns=codes)
            st.session_state.matrix_df.insert(0, "Alternativa", alt_names)
            st.session_state.engine_weights["AHP"] = np.array(w_ahp)
            st.session_state.ahp_matrix_pasted = ahp_matrix
            st.session_state.ahp_cr = CR
            st.session_state.ahp_lam_max = lam_max
            st.session_state.ahp_ci = CI
            st.session_state.global_injection_on = True
            st.session_state.global_injection_engine = "AHP"
            st.session_state.all_results = {}  # invalidar caches dos modelos

            st.success(
                f"✅ Carregado: **{len(alt_names)} alternativas × {len(codes)} critérios**. "
                f"AHP calculado — **CR = {CR:.5f}** "
                f"({'✓ consistente' if CR < 0.10 else '✗ inconsistente (> 0.10)'})."
            )
            st.rerun()
        except Exception as e:
            st.error(f"❌ Erro no parsing: {e}")

    # ============================================================================
    # SECÇÃO 4 — SENSIBILIDADE GLOBAL
    # ============================================================================
    st.markdown('<div class="data-section"><h3>🎯 4. Variação para Análise de Sensibilidade</h3></div>',
                unsafe_allow_html=True)
    st.session_state.sensitivity_pct = st.slider(
        "Variação ± nos pesos (%):",
        5, 50, st.session_state.sensitivity_pct, 5,
        key="sens_slider_tab",
        help="Aplicada em todas as análises de sensibilidade do Dashboard e abas dos modelos."
    )
    st.metric("Variação activa", f"±{st.session_state.sensitivity_pct}%")

    # ============================================================================
    # SECÇÃO 5 — PRÉ-VISUALIZAÇÃO
    # ============================================================================
    st.markdown('<div class="data-section"><h3>👁️ 5. Pré-visualização dos dados activos</h3></div>',
                unsafe_allow_html=True)

    matrix, alts, crits, types = get_decision_matrix()
    if not check_valid_input():
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alternativas", len(alts))
    c2.metric("Critérios", len(crits))
    c3.metric("Max (benefícios)", types.count("max"))
    c4.metric("Min (custos)", types.count("min"))

    st.subheader("Critérios + tipo + peso AHP")
    if "AHP" in st.session_state.engine_weights and len(st.session_state.engine_weights["AHP"]) == len(crits):
        w_display = st.session_state.engine_weights["AHP"]
        cr_display = st.session_state.get("ahp_cr", None)
        df_ct = pd.DataFrame({
            "Critério": crits, "Tipo": types,
            "Peso AHP": w_display, "%": [f"{x*100:.2f}%" for x in w_display],
        })
        st.dataframe(df_ct.style.format({"Peso AHP": "{:.5f}"})
                       .background_gradient(cmap="Blues", subset=["Peso AHP"]),
                    hide_index=True, use_container_width=True)
        if cr_display is not None:
            if cr_display < 0.10:
                st.success(f"✓ CR = **{cr_display:.5f}** < 0.10 — matriz AHP consistente. Pesos válidos.")
            else:
                st.warning(f"⚠️ CR = **{cr_display:.5f}** ≥ 0.10 — matriz AHP inconsistente. "
                           f"Vá à aba 🔍 AHP para ver a sugestão de correcção.")
    else:
        st.info("Pesos AHP ainda não calculados. Carregue os pastes acima e prima Processar.")

    st.subheader("Matriz de Decisão")
    display_df = pd.DataFrame(matrix, index=alts, columns=crits)
    st.dataframe(display_df.style.format("{:.5f}").background_gradient(cmap="Blues", axis=0),
                use_container_width=True)

    st.subheader("Heatmap normalizado (min-max, sentido aplicado)")
    norm = normalize_minmax(matrix, types)
    norm_df = pd.DataFrame(norm, index=alts, columns=crits)
    st.dataframe(norm_df.style.format("{:.5f}").background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True)
    st.caption("1.0 = melhor; 0.0 = pior (com inversão automática para critérios de custo).")

# =============================================================================
# TAB 2: AHP — DISPLAY-ONLY (matriz vem da aba 📋 Dados)
# =============================================================================
with tabs[2]:
    st.header("🔍 AHP — Analytic Hierarchy Process (Saaty, 1980)")
    purpose_box(
        "Análise da <b>matriz par-a-par AHP</b> colada na aba <b>📋 Dados</b>. "
        "Esta aba mostra a matriz, os pesos calculados, a verificação de consistência (CR < 0.10) "
        "e — se a matriz for inconsistente — uma <b>sugestão de correcção</b>."
    )

    theory_box(
        "Teoria condensada",
        """
        <p>AHP determina pesos via <b>comparação par-a-par</b> usando a <b>escala Saaty 1-9</b>:</p>
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

    if "ahp_matrix_pasted" not in st.session_state or st.session_state.ahp_matrix_pasted is None:
        st.warning(
            "⚠️ Ainda não foi carregada nenhuma matriz AHP. "
            "Vá à aba **📋 Dados**, cole a matriz par-a-par (Foto 2) e prima **Processar pastes**."
        )
        st.stop()

    A = np.array(st.session_state.ahp_matrix_pasted, dtype=float)
    if A.shape != (n, n):
        st.warning(
            f"⚠️ A matriz AHP carregada tem dimensão {A.shape} mas os critérios actuais têm {n}. "
            "Volte à aba 📋 Dados e cole novamente para sincronizar."
        )
        st.stop()

    step_header("Passo 1: Matriz de Comparação Par-a-Par (carregada da aba 📋 Dados)")
    st.latex(r"A = [a_{ij}],\quad a_{ji} = 1/a_{ij},\quad a_{ii} = 1")
    st.caption("Para alterar valores: vá à aba 📋 Dados, edite o paste, e prima Processar.")
    st.dataframe(
        pd.DataFrame(A, index=crits, columns=crits).style.format("{:.5f}")
            .background_gradient(cmap="Blues", axis=None),
        use_container_width=True
    )

    step_header("Passo 2: Vector de Pesos (média geométrica)")
    st.latex(r"w_i = \frac{(\prod_j a_{ij})^{1/n}}{\sum_k (\prod_j a_{kj})^{1/n}}")
    geomean = np.prod(A, axis=1) ** (1.0 / n)
    w_ahp = geomean / geomean.sum()
    st.dataframe(pd.DataFrame({"Critério": crits, "Tipo": types, "Peso w_j": w_ahp,
                               "%": [f"{x*100:.3f}%" for x in w_ahp]})
                  .style.format({"Peso w_j": "{:.5f}"})
                  .background_gradient(cmap="Blues", subset=["Peso w_j"]),
                hide_index=True, use_container_width=True)

    step_header("Passo 3: Verificação de Consistência")
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
            'A app identifica o <b>par mais problemático</b> e propõe um valor da escala Saaty. '
            'Aplique esse valor na sua matriz na aba 📋 Dados e processe novamente.'
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
            st.markdown("#### 🔧 Sugestão para tornar a matriz consistente")
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Par problemático", f"{crits[worst_i]} vs {crits[worst_j]}")
            colB.metric("Valor actual", f"{A[worst_i, worst_j]:.5f}")
            colC.metric("Valor ideal", f"{ideal_value:.5f}")
            colD.metric("Sugerido (Saaty)", f"{suggested_value:.5f}",
                       delta=f"Δ = {suggested_value - A[worst_i, worst_j]:+.5f}")

            colE, colF = st.columns([3, 1])
            with colE:
                st.info(
                    f"**Interpretação:** está a dizer que **{crits[worst_i]}** vale "
                    f"**{A[worst_i, worst_j]:.5f}×** {crits[worst_j]}, mas os pesos calculados "
                    f"sugerem ~**{ideal_value:.5f}×**. Aplicar substitui o valor por **{suggested_value:.5f}** "
                    f"(e o recíproco simétrico = **{1/suggested_value:.5f}**)."
                )
            with colF:
                if st.button("✏️ Aplicar sugestão", type="primary", use_container_width=True,
                             key="ahp_apply_suggestion"):
                    # 1) aplicar na matriz numérica (já com reciprocidade)
                    new_A = A.copy()
                    old_val = new_A[worst_i, worst_j]
                    new_A[worst_i, worst_j] = suggested_value
                    new_A[worst_j, worst_i] = 1.0 / suggested_value
                    st.session_state.ahp_matrix_pasted = new_A

                    # 2) recalcular pesos + CR
                    gm_new = np.prod(new_A, axis=1) ** (1.0 / n)
                    w_new = gm_new / gm_new.sum()
                    Aw_new = new_A @ w_new
                    lam_new = (Aw_new / np.where(w_new == 0, 1e-9, w_new)).mean()
                    CI_new = (lam_new - n) / (n - 1) if n > 1 else 0
                    CR_new = CI_new / RI if RI > 0 else 0
                    st.session_state.engine_weights["AHP"] = w_new
                    st.session_state.ahp_cr = CR_new
                    st.session_state.ahp_lam_max = lam_new
                    st.session_state.ahp_ci = CI_new

                    # 3) invalidar caches dos modelos (vão recomputar com pesos novos)
                    st.session_state.all_results = {}

                    # 4) reconstruir o paste-text para manter a aba 📋 Dados sincronizada
                    types_now = st.session_state.criteria_df["Tipo"].astype(str).tolist()
                    lines = ["\t" + "\t".join(crits) + "\tMAX/MIN ?"]
                    for i_row in range(n):
                        vals_str = "\t".join(f"{new_A[i_row, j_col]:.5f}" for j_col in range(n))
                        lines.append(f"{crits[i_row]}\t{vals_str}\t{types_now[i_row]}")
                    st.session_state.crit_paste_text = "\n".join(lines)

                    # 5) registar no histórico
                    st.session_state.ahp_history.append({
                        "iteração": len(st.session_state.ahp_history) + 1,
                        "par": f"{crits[worst_i]} vs {crits[worst_j]}",
                        "valor antigo": round(old_val, 5),
                        "valor novo": round(suggested_value, 5),
                        "CR antes": round(CR, 5),
                        "CR depois": round(CR_new, 5),
                    })

                    if CR_new < 0.10:
                        st.success(
                            f"✅ Sugestão aplicada. **CR = {CR_new:.5f} < 0.10** — matriz agora consistente!"
                        )
                    else:
                        st.success(
                            f"✓ Sugestão aplicada. CR baixou de {CR:.5f} → {CR_new:.5f}. "
                            f"Ainda ≥ 0.10 — clique de novo se quiser continuar a iterar."
                        )
                    st.rerun()
    else:
        st.markdown(
            f'<div class="result-box">✅ <b>Matriz CONSISTENTE</b> — CR = {CR:.5f} < 0.10. Pesos AHP válidos.</div>',
            unsafe_allow_html=True
        )

    # Histórico de iterações
    if st.session_state.ahp_history:
        with st.expander(f"📜 Histórico de iterações AHP ({len(st.session_state.ahp_history)})", expanded=False):
            st.dataframe(
                pd.DataFrame(st.session_state.ahp_history),
                hide_index=True, use_container_width=True
            )
            if st.button("🗑️ Limpar histórico", key="ahp_clear_history"):
                st.session_state.ahp_history = []
                st.rerun()

    # garantir que os pesos AHP no session_state estão sincronizados
    st.session_state.engine_weights["AHP"] = w_ahp
    st.session_state.ahp_cr = CR

    st.markdown("---")
    step_header("Passo 4: Ranking das Alternativas (usando pesos AHP)")
    st.latex(r"S_i = \sum_{j=1}^n w_j^{AHP} \cdot u_j(x_{ij})")

    U = normalize_minmax(matrix, types)
    S = (U * w_ahp).sum(axis=1)
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Score AHP": S,
                           "% do máx": S / S.max() * 100 if S.max() > 0 else S,
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

    # ===== Gráficos AHP =====
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

# TAB 0: DASHBOARD CONSOLIDADO — HOMEPAGE (recalculado dinamicamente)
# =============================================================================
with tabs[0]:
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
    st.markdown(f"##### 🎯 Análise de Sensibilidade por Critério — robustez do Top-1 consensual **{top1}** (varrimento até ±50%)")

    sens_pct = st.session_state.sensitivity_pct

    def consensus_from_weights(W):
        """Devolve (top1, posição_média_top1, rank_de_{top1_actual})."""
        r = compute_all_models(matrix, types, W)
        df_t = pd.DataFrame({"Alt": alts})
        for mm in methods:
            df_t[mm] = r[mm]["ranks"]
        df_t["PM"] = df_t[methods].mean(axis=1)
        df_t = df_t.sort_values("PM").reset_index(drop=True)
        new_t1 = df_t.iloc[0]["Alt"]
        # rank do top1 original neste novo ranking
        try:
            rk_orig_top1 = int(df_t.index[df_t["Alt"] == top1].tolist()[0]) + 1
        except Exception:
            rk_orig_top1 = -1
        return new_t1, df_t.iloc[0]["PM"], rk_orig_top1

    def perturbed_weights(W, j, factor):
        """W com peso j multiplicado por factor, restantes renormalizados para Σ=1."""
        nw = W.copy()
        nw[j] = W[j] * factor
        oo = W.sum() - W[j]
        on = 1 - nw[j]
        if oo > 0 and on > 0:
            for k in range(len(nw)):
                if k != j:
                    nw[k] = W[k] * (on / oo)
        s = nw.sum()
        return nw / s if s > 0 else np.ones(len(nw)) / len(nw)

    crit_status = []
    SCAN_PCT = 50  # varrimento até ±50% para encontrar a margem de segurança
    STEP = 5

    for j, crit in enumerate(crits):
        # Estado no ponto de teste sens_pct (resultado principal)
        primary_changes = []      # detalhes do que acontece a ±sens_pct
        for sign, factor in [("+", 1 + sens_pct/100), ("-", 1 - sens_pct/100)]:
            nw = perturbed_weights(weights, j, factor)
            new_t1, _, rk_orig = consensus_from_weights(nw)
            if new_t1 != top1:
                primary_changes.append(f"{sign}{sens_pct}% → **{new_t1}**")
            else:
                primary_changes.append(f"{sign}{sens_pct}% → {top1} (#1)")
        primary_invertido = any("**" in x for x in primary_changes)

        # Varrimento para encontrar a MARGEM DE SEGURANÇA
        threshold_plus = None; new_t1_plus = None
        threshold_minus = None; new_t1_minus = None
        for pct in range(STEP, SCAN_PCT + 1, STEP):
            if threshold_plus is None:
                nw = perturbed_weights(weights, j, 1 + pct/100)
                nt1, _, _ = consensus_from_weights(nw)
                if nt1 != top1:
                    threshold_plus = pct; new_t1_plus = nt1
            if threshold_minus is None:
                nw = perturbed_weights(weights, j, 1 - pct/100)
                nt1, _, _ = consensus_from_weights(nw)
                if nt1 != top1:
                    threshold_minus = pct; new_t1_minus = nt1
            if threshold_plus is not None and threshold_minus is not None:
                break

        crit_status.append({
            "Critério": crit,
            "Inverteu_primary": primary_invertido,
            "primary_detail": "<br>".join(primary_changes),
            "threshold_plus": threshold_plus,
            "new_t1_plus": new_t1_plus,
            "threshold_minus": threshold_minus,
            "new_t1_minus": new_t1_minus,
        })

    sens_cols = st.columns(len(crits))
    for i, info in enumerate(crit_status):
        with sens_cols[i]:
            invertido = info["Inverteu_primary"]
            cor = "#C00000" if invertido else "#2E7D32"
            classif = "🔴 Sensível" if invertido else "🟢 Robusto"

            # construir texto da margem de segurança
            if info["threshold_plus"] is None:
                msg_plus = f"<span style='color:#2E7D32'>+>{SCAN_PCT}% estável</span>"
            else:
                msg_plus = (f"<span style='color:#C00000'>+{info['threshold_plus']}% → "
                            f"<b>{info['new_t1_plus']}</b></span>")
            if info["threshold_minus"] is None:
                msg_minus = f"<span style='color:#2E7D32'>->{SCAN_PCT}% estável</span>"
            else:
                msg_minus = (f"<span style='color:#C00000'>-{info['threshold_minus']}% → "
                             f"<b>{info['new_t1_minus']}</b></span>")

            st.markdown(
                f"""<div style="background: white; border: 2px solid {cor}; border-radius: 8px;
                padding: 10px 8px; min-height: 170px;">
                  <div style="font-weight: 700; color: #1F4E78; font-size: 13px; text-align: center;">{info['Critério']}</div>
                  <div style="margin: 4px 0; color: {cor}; font-weight: 600; font-size: 11px; text-align: center;">
                    ±{sens_pct}%: {classif}
                  </div>
                  <div style="font-size: 10px; color: #444; line-height: 1.4; margin-top: 6px; border-top: 1px solid #eee; padding-top: 6px;">
                    <b style="color:#1F4E78;">Margem de segurança:</b><br>
                    {msg_plus}<br>
                    {msg_minus}
                  </div>
                </div>""",
                unsafe_allow_html=True
            )

    st.caption(
        f"📖 **Leitura:** *±{sens_pct}%* (linha de cima) = resultado no nível pedido na aba 📋 Dados. "
        f"*Margem de segurança* = a partir de quanta variação isolada do peso é que o Top-1 consensual mudaria. "
        f"`+>{SCAN_PCT}% estável` significa que mesmo subindo o peso 50% o {top1} continua Top-1."
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
        sens_robust_count = sum(1 for s in crit_status if not s["Inverteu_primary"])
        st.markdown(
            f"**Robustez SA ±{sens_pct}%:** {sens_robust_count}/{len(crits)} critérios mantêm {top1} como Top-1 consensual."
        )

        # margem de segurança mínima — o "elo mais fraco"
        margens = []
        for s in crit_status:
            if s["threshold_plus"] is not None:
                margens.append(("+", s["Critério"], s["threshold_plus"], s["new_t1_plus"]))
            if s["threshold_minus"] is not None:
                margens.append(("-", s["Critério"], s["threshold_minus"], s["new_t1_minus"]))
        if margens:
            sign_m, crit_m, pct_m, new_t1_m = min(margens, key=lambda x: x[2])
            st.markdown(
                f"**Margem de segurança mínima:** ±**{pct_m}%** ({sign_m} em {crit_m}) → "
                f"a partir daí o Top-1 mudaria para **{new_t1_m}**."
            )
        else:
            st.markdown(
                f"**Margem de segurança mínima:** > ±50% em todos os critérios — "
                f"{top1} é Top-1 mesmo com variação extrema isolada de qualquer peso."
            )

        if sens_robust_count == len(crits):
            st.success(f"✅ Decisão MUITO ROBUSTA — {top1} é Top-1 em todos os cenários SA ±{sens_pct}%.")
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