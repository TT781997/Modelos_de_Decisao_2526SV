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
# CONFIGURAÇÃO DE PÁGINA E CSS
# =============================================================================
st.set_page_config(page_title="MCDM Dashboard v2", page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
/* Esconder o sidebar por defeito no CSS também, para garantir que não aparece */
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

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
        st.session_state.global_injection_on = True
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
    if "success_message" not in st.session_state:
        st.session_state.success_message = None

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
            'Vá à aba 📋 Dados para processar a matriz.</div>',
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
        st.warning("Sem dados para apresentar. Por favor preencha e processe a aba '📋 Dados'.")
        return False
    return True

# Normalizações Matemáticas (Verificadas)
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
# MODEL SCORERS — MATEMÁTICA VERIFICADA E ROBUSTA
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
    "Pesos vindos do AHP automaticamente · Sensibilidade universal · Precisão matemática rigorosa."
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
# TAB 1: DADOS — INPUT ÚNICO POR PASTE
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
        for line in text.strip("\n\r").splitlines():
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
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
        rows = split_paste_rows(text)
        if not rows: raise ValueError("Sem dados")

        header_idx = None
        for i, r in enumerate(rows):
            non_num = sum(1 for c in r if c.strip() and not _is_num(c))
            if non_num >= 3:
                header_idx = i; break
        if header_idx is None: raise ValueError("Não encontrei linha de cabeçalho.")

        header = rows[header_idx]
        col_codes = []
        for c in header:
            cl = c.strip().lower()
            if not cl: continue
            if "max" in cl or "min" in cl or "tipo" in cl or "natureza" in cl: continue
            col_codes.append(c.strip())
        if len(col_codes) < 2: raise ValueError("Cabeçalho de critérios precisa de ≥2 códigos.")
        n = len(col_codes)

        body = rows[header_idx + 1:]
        types_out = []; codes_row = []; ahp = []
        for r in body:
            if not r or all(not c.strip() for c in r): continue
            row_code = r[0].strip()
            if not row_code or "max" in row_code.lower() or "min" in row_code.lower(): continue
            if len(r) < n + 1: raise ValueError(f"Linha '{row_code}' incompleta.")
            try:
                vals = [float(clean_number_string(r[1 + k])) for k in range(n)]
            except Exception as e:
                raise ValueError(f"Linha '{row_code}': erro numérico → {e}")
            tipo = "max"
            for c in r[n+1:]:
                cl = c.strip().lower()
                if cl in ("max", "min"): tipo = cl; break
                if "benef" in cl: tipo = "max"; break
                if "cust" in cl:  tipo = "min"; break
            codes_row.append(row_code)
            types_out.append(tipo)
            ahp.append(vals)
            if len(ahp) == n: break

        if len(codes_row) != n: raise ValueError("Número de colunas e linhas não bate certo.")
        if codes_row != col_codes:
            if set(codes_row) == set(col_codes):
                order = [codes_row.index(c) for c in col_codes]
                ahp = [ahp[i] for i in order]
                types_out = [types_out[i] for i in order]
                codes_row = col_codes[:]
            else:
                raise ValueError("Códigos das linhas ≠ colunas.")

        ahp = np.array(ahp, dtype=float)
        for i in range(n):
            ahp[i, i] = 1.0
            for j in range(n):
                if i < j and ahp[i, j] > 0:
                    ahp[j, i] = 1.0 / ahp[i, j]
        return col_codes, types_out, ahp

    def parse_alts_paste(text, expected_crits=None):
        rows = split_paste_rows(text)
        if not rows: raise ValueError("Sem dados")
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
            for i, r in enumerate(rows):
                if sum(1 for c in r if c.strip() and not _is_num(c)) >= 2:
                    header_idx = i; break
        if header_idx is None: raise ValueError("Cabeçalho de alternativas não encontrado.")
        header = rows[header_idx]

        if expected_crits:
            col_idx = []
            for c in expected_crits:
                if c in header: col_idx.append(header.index(c))
                else: raise ValueError(f"Critério '{c}' não aparece na tabela.")
        else:
            col_idx = list(range(1, len(header)))

        body = rows[header_idx + 1:]
        alt_names = []; matrix_rows = []
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
                    else: vals.append(0.0)
                alt_names.append(alt_name)
                matrix_rows.append(vals)
            except Exception: continue
        if not alt_names: raise ValueError("Nenhuma alternativa válida encontrada.")
        return alt_names, np.array(matrix_rows, dtype=float)

    def _is_num(s):
        try: float(clean_number_string(s)); return True
        except Exception: return False

    def compute_ahp_weights_and_cr(A):
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
    # SECÇÕES DE TEXTO PARA OS PASTES
    # ============================================================================
    st.markdown('<div class="data-section"><h3>📥 1. Matriz AHP dos Critérios (par-a-par + MAX/MIN)</h3></div>', unsafe_allow_html=True)
    default_crit_paste = """\tC1_VP\tC2_PF\tC3_EE\tC4_FE\tC5_UD\tC6_RC\tMAX/MIN ?
C1_VP\t1.0000\t4.0000\t8.0000\t6.0000\t9.0000\t5.0000\tmax
C2_PF\t0.2500\t1.0000\t8.0000\t3.0000\t9.0000\t2.0000\tmax
C3_EE\t0.1250\t0.1250\t1.0000\t0.1111\t1.0000\t0.1111\tmin
C4_FE\t0.1667\t0.3333\t9.0000\t1.0000\t9.0000\t0.1667\tmax
C5_UD\t0.1111\t0.1111\t1.0000\t0.1111\t1.0000\t0.1111\tmax
C6_RC\t0.2000\t0.5000\t9.0000\t6.0000\t9.0000\t1.0000\tmax"""
    crit_text = st.text_area("Cole aqui a matriz AHP (com cabeçalhos + coluna MAX/MIN ?):", value=st.session_state.get("crit_paste_text", default_crit_paste), height=200)

    st.markdown('<div class="data-section"><h3>📥 2. Tabela das Alternativas × Critérios</h3></div>', unsafe_allow_html=True)
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
    alts_text = st.text_area("Cole aqui a tabela das alternativas:", value=st.session_state.get("alts_paste_text", default_alts_paste), height=240)

    # ============================================================================
    # SECÇÃO 3 — PROCESSAR DADOS
    # ============================================================================
    st.markdown('<div class="data-section"><h3>⚙️ 3. Processar dados</h3></div>', unsafe_allow_html=True)

    # Exibe mensagem de sucesso caso tenha processado no recarregamento anterior
    if st.session_state.get("success_message"):
        st.success(st.session_state.success_message)
        st.session_state.success_message = None

    if st.button("🚀 Processar pastes (parse + calcular AHP + carregar)", type="primary", use_container_width=True):
        try:
            codes, types_parsed, ahp_matrix = parse_criteria_paste(crit_text)
            alt_names, dec_matrix = parse_alts_paste(alts_text, expected_crits=codes)
            if dec_matrix.shape[1] != len(codes):
                raise ValueError(f"A tabela de alternativas tem {dec_matrix.shape[1]} colunas, mas a matriz AHP define {len(codes)}.")
            w_ahp, lam_max, CI, CR = compute_ahp_weights_and_cr(ahp_matrix)

            st.session_state.criteria_df = pd.DataFrame({"Critério": codes, "Tipo": types_parsed, "Peso Manual": [1.0/len(codes)] * len(codes)})
            st.session_state.matrix_df = pd.DataFrame(dec_matrix, columns=codes)
            st.session_state.matrix_df.insert(0, "Alternativa", alt_names)
            st.session_state.engine_weights["AHP"] = np.array(w_ahp)
            st.session_state.ahp_matrix_pasted = ahp_matrix
            st.session_state.ahp_cr = CR
            st.session_state.ahp_lam_max = lam_max
            st.session_state.ahp_ci = CI
            st.session_state.all_results = {}
            st.session_state.crit_paste_text = crit_text
            st.session_state.alts_paste_text = alts_text

            # Mensagem em memória persistente
            st.session_state.success_message = (
                f"✅ **Dados processados com sucesso!** \n\n"
                f"Carregadas **{len(alt_names)} alternativas × {len(codes)} critérios**.\n\n"
                f"AHP calculado — **CR = {CR:.5f}** "
                f"({'✓ consistente' if CR < 0.10 else '✗ inconsistente (> 0.10)'})."
            )
            st.toast("Dados carregados com sucesso!", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Erro no parsing: {e}")

    # ============================================================================
    # SECÇÃO 4 — SENSIBILIDADE
    # ============================================================================
    st.markdown('<div class="data-section"><h3>🎯 4. Variação para Análise de Sensibilidade</h3></div>', unsafe_allow_html=True)
    st.session_state.sensitivity_pct = st.slider("Variação ± nos pesos (%):", 5, 50, st.session_state.sensitivity_pct, 5)


# =============================================================================
# TAB 2: AHP
# =============================================================================
with tabs[2]:
    st.header("🔍 AHP — Analytic Hierarchy Process (Saaty, 1980)")
    purpose_box("Análise da <b>matriz par-a-par AHP</b>. Verifica a consistência (CR < 0.10) e sugere correcções se necessário.")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    n = len(crits)
    RI_TABLE = {1:0, 2:0, 3:0.58, 4:0.9, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49, 11:1.51, 12:1.54, 13:1.56, 14:1.57, 15:1.59}

    A = np.array(st.session_state.ahp_matrix_pasted, dtype=float)
    if A.shape != (n, n): st.warning("Dimensão da matriz AHP não corresponde aos critérios."); st.stop()

    step_header("Passo 1: Matriz de Comparação Par-a-Par")
    st.dataframe(pd.DataFrame(A, index=crits, columns=crits).style.format("{:.5f}").background_gradient(cmap="Blues", axis=None), use_container_width=True)

    step_header("Passo 2: Vector de Pesos (média geométrica)")
    geomean = np.prod(A, axis=1) ** (1.0 / n)
    w_ahp = geomean / geomean.sum()
    st.dataframe(pd.DataFrame({"Critério": crits, "Tipo": types, "Peso w_j": w_ahp, "%": [f"{x*100:.3f}%" for x in w_ahp]})
                  .style.format({"Peso w_j": "{:.5f}"}).background_gradient(cmap="Blues", subset=["Peso w_j"]),
                hide_index=True, use_container_width=True)

    step_header("Passo 3: Verificação de Consistência")
    Aw = A @ w_ahp
    lam_max = (Aw / np.where(w_ahp == 0, 1e-9, w_ahp)).mean()
    CI = (lam_max - n) / (n - 1) if n > 1 else 0
    RI = RI_TABLE.get(n, 1.59)
    CR = CI / RI if RI > 0 else 0

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("n", n); cc2.metric("λ_max", f"{lam_max:.5f}"); cc3.metric("CI", f"{CI:.5f}")
    cc4.metric("CR", f"{CR:.5f}", delta="✓ Consistente" if CR < 0.10 else "✗ Inconsistente", delta_color="normal" if CR < 0.10 else "inverse")

    if CR >= 0.10:
        st.markdown('<div class="warning-box"><b>⚠️ CR ≥ 0.10 — Matriz INCONSISTENTE.</b><br>Aqui tem uma sugestão de correção automática baseada na escala Saaty:</div>', unsafe_allow_html=True)
        worst_i, worst_j, worst_dev = -1, -1, 0
        suggested_value = 1.0; ideal_value = 1.0
        for i in range(n):
            for j in range(i+1, n):
                if w_ahp[j] != 0:
                    expected = w_ahp[i] / w_ahp[j]
                    observed = A[i, j]
                    if expected > 0 and observed > 0:
                        dev = abs(np.log(observed / expected))
                        if dev > worst_dev:
                            worst_dev = dev; worst_i, worst_j = i, j; ideal_value = expected
                            saaty_scale = [1/9, 1/7, 1/5, 1/3, 1/2, 1, 2, 3, 5, 7, 9]
                            suggested_value = min(saaty_scale, key=lambda x: abs(np.log(x) - np.log(ideal_value)))

        if worst_i >= 0:
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Par problemático", f"{crits[worst_i]} vs {crits[worst_j]}")
            colB.metric("Valor actual", f"{A[worst_i, worst_j]:.5f}")
            colC.metric("Sugerido (Saaty)", f"{suggested_value:.5f}")
            with colD:
                if st.button("✏️ Aplicar sugestão"):
                    new_A = A.copy()
                    new_A[worst_i, worst_j] = suggested_value
                    new_A[worst_j, worst_i] = 1.0 / suggested_value
                    st.session_state.ahp_matrix_pasted = new_A
                    # Recálculo imediato de pesos
                    gm_new = np.prod(new_A, axis=1) ** (1.0 / n)
                    st.session_state.engine_weights["AHP"] = gm_new / gm_new.sum()
                    st.session_state.all_results = {}
                    st.rerun()
    else:
        st.markdown(f'<div class="result-box">✅ <b>Matriz CONSISTENTE</b> — CR = {CR:.5f} < 0.10.</div>', unsafe_allow_html=True)

    st.session_state.engine_weights["AHP"] = w_ahp
    st.markdown("---")
    step_header("Passo 4: Ranking das Alternativas (AHP)")
    U = normalize_minmax(matrix, types)
    S = (U * w_ahp).sum(axis=1)
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Score AHP": S, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"Score AHP": "{:.5f}"}).background_gradient(cmap="RdYlGn", subset=["Score AHP"]), hide_index=True, use_container_width=True)
    store_result("AHP", S, rank, True)


# =============================================================================
# TAB 3: TOPSIS
# =============================================================================
with tabs[3]:
    st.header("🎯 TOPSIS")
    purpose_box("Mede a <b>distância à solução ideal</b>.")
    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights(); show_active_weights_banner()

    def topsis_calc(W):
        R = normalize_vector(matrix); V = R * W
        Ap = np.array([V[:, j].max() if types[j] == "max" else V[:, j].min() for j in range(len(crits))])
        An = np.array([V[:, j].min() if types[j] == "max" else V[:, j].max() for j in range(len(crits))])
        Dp = np.sqrt(((V - Ap) ** 2).sum(axis=1)); Dn = np.sqrt(((V - An) ** 2).sum(axis=1))
        return Dn / np.where(Dp + Dn == 0, 1e-9, Dp + Dn)

    CC = topsis_calc(weights)
    rank = pd.Series(CC).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "CC*": CC, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"CC*": "{:.5f}"}).background_gradient(cmap="RdYlGn", subset=["CC*"]), hide_index=True, use_container_width=True)
    store_result("TOPSIS", CC, rank, True)
    render_sensitivity(lambda w: topsis_calc(w), alts, crits, weights, True, "topsis")


# =============================================================================
# TAB 4: PROMETHEE II
# =============================================================================
with tabs[4]:
    st.header("📈 PROMETHEE II")
    purpose_box("Método de fluxos de preferência par-a-par.")
    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights(); show_active_weights_banner()

    pref_type = st.radio("Função de preferência:", ["Tipo I (Usual)", "Tipo V (Linear)", "Tipo VI (Gaussiana)"], horizontal=True)

    def prom_calc(W):
        m, n = matrix.shape
        params_p = [(matrix[:, j].max() - matrix[:, j].min()) * 0.5 if matrix[:, j].max() > matrix[:, j].min() else 1.0 for j in range(n)]
        params_s = [(matrix[:, j].max() - matrix[:, j].min()) * 0.3 if matrix[:, j].max() > matrix[:, j].min() else 1.0 for j in range(n)]
        pi = np.zeros((m, m))
        for a in range(m):
            for b in range(m):
                if a == b: continue
                for j in range(n):
                    d = matrix[a, j] - matrix[b, j] if types[j] == "max" else matrix[b, j] - matrix[a, j]
                    if d > 0:
                        v = 1.0 if pref_type.startswith("Tipo I") else (min(d / params_p[j], 1.0) if pref_type.startswith("Tipo V") else 1.0 - np.exp(-d**2 / (2 * params_s[j]**2)))
                        pi[a, b] += W[j] * v
        phi_p = pi.sum(axis=1) / max(m - 1, 1)
        phi_n = pi.sum(axis=0) / max(m - 1, 1)
        return phi_p - phi_n

    phi = prom_calc(weights)
    rank = pd.Series(phi).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "φ líquido": phi, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"φ líquido": "{:.5f}"}).background_gradient(cmap="RdYlGn", subset=["φ líquido"]), hide_index=True, use_container_width=True)
    store_result("PROMETHEE II", phi, rank, True)
    render_sensitivity(lambda w: prom_calc(w), alts, crits, weights, True, "prom")


# =============================================================================
# TAB 5: COPRAS
# =============================================================================
with tabs[5]:
    st.header("📊 COPRAS")
    purpose_box("Avalia alternativas como função proporcional entre benefícios e custos.")
    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights(); show_active_weights_banner()

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
        return Q

    Q = copras_calc(weights)
    rank = pd.Series(Q).rank(ascending=False, method='min').astype(int).values
    U = (Q / Q.max() * 100) if Q.max() > 0 else Q * 0
    df_res = pd.DataFrame({"Alternativa": alts, "Q_i": Q, "U_i (%)": U, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"Q_i": "{:.5f}", "U_i (%)": "{:.2f}"}).background_gradient(cmap="RdYlGn", subset=["U_i (%)"]), hide_index=True, use_container_width=True)
    store_result("COPRAS", Q, rank, True)
    render_sensitivity(lambda w: copras_calc(w), alts, crits, weights, True, "copras")


# =============================================================================
# TAB 0: DASHBOARD CONSOLIDADO — HOMEPAGE
# =============================================================================
with tabs[0]:
    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()

    st.markdown(
        f"""<div style="background: linear-gradient(90deg, #1F4E78 0%, #2E75B6 100%);
        color: white; padding: 14px 22px; border-radius: 8px; margin-bottom: 14px;
        display: flex; justify-content: space-between; align-items: center;">
          <div>
            <span style="font-size: 20px; font-weight: 700;">📊 MCDM Dashboard</span>
            <span style="font-size: 16px; opacity: 0.85;"> | Priorização de Alternativas Multicritério</span>
          </div>
          <div style="font-size: 12px; opacity: 0.85; text-align: right;">
            Variação SA ±{st.session_state.sensitivity_pct}%
          </div>
        </div>""",
        unsafe_allow_html=True
    )

    results = compute_all_models(matrix, types, weights)
    methods = ["AHP", "TOPSIS", "PROMETHEE II", "COPRAS"]

    df_dash = pd.DataFrame({"Alternativa": alts})
    for m in methods: df_dash[m] = results[m]["ranks"]
    df_dash["Posição Média"] = df_dash[methods].mean(axis=1).round(2)
    df_dash["Top-3 em N modelos"] = (df_dash[methods] <= 3).sum(axis=1)
    df_dash["Ranking Final"] = pd.Series(df_dash["Posição Média"]).rank(ascending=True, method='min').astype(int).values
    df_dash = df_dash.sort_values("Ranking Final").reset_index(drop=True)

    top3 = df_dash.head(3)["Alternativa"].tolist()
    top1 = top3[0] if top3 else "—"

    col_rank, col_radar = st.columns([2, 1.5])

    with col_rank:
        st.markdown("##### 🏆 Ranking Consolidado das Alternativas")
        def medalha(r): return "🥇" if r == 1 else ("🥈" if r == 2 else ("🥉" if r == 3 else ""))
        display = df_dash.copy()
        display["Medalha"] = display["Ranking Final"].apply(medalha)
        compact = display[["Medalha", "Alternativa"] + methods + ["Posição Média", "Ranking Final"]]
        st.dataframe(compact.style.background_gradient(cmap="RdYlGn_r", subset=methods + ["Posição Média", "Ranking Final"]).format({"Posição Média": "{:.2f}"}), hide_index=True, use_container_width=True)

    with col_radar:
        st.markdown("##### 🎯 Perfil Multicritério — Top-3")
        norm = normalize_minmax(matrix, types)
        norm_df = pd.DataFrame(norm, index=alts, columns=crits)
        fig_radar = go.Figure()
        colors_radar = ["#FFD700", "#C0C0C0", "#CD7F32"]
        for i, alt in enumerate(top3):
            vals = list(norm_df.loc[alt]) + [norm_df.loc[alt].iloc[0]]
            fig_radar.add_trace(go.Scatterpolar(r=vals, theta=crits + [crits[0]], fill="toself", name=f"{i+1}º {alt}", line=dict(color=colors_radar[i], width=2), opacity=0.55))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)), height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📈 Scores por Modelo")
    cols_models = st.columns(len(methods))
    for i, m in enumerate(methods):
        with cols_models[i]:
            df_m = pd.DataFrame({"Alt": alts, "Score": results[m]["scores"], "Rank": results[m]["ranks"]}).sort_values("Rank")
            colors_m = ["#1F4E78" if r <= 3 else "#5B9BD5" for r in df_m["Rank"]]
            if m == "PROMETHEE II": colors_m = ["#1F4E78" if v >= 0 else "#C00000" for v in df_m["Score"]]
            fig_m = go.Figure(go.Bar(x=df_m["Score"], y=df_m["Alt"], orientation="h", marker=dict(color=colors_m), text=[f"{v:.4f}" for v in df_m["Score"]], textposition="outside"))
            fig_m.update_layout(title=dict(text=f"<b>{m}</b>", x=0.5), height=max(220, 28 * len(alts)), margin=dict(l=10, r=50, t=30, b=10), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("---")
    col_reco, col_crit = st.columns([1.5, 1.0])

    with col_reco:
        total_top3 = sum(df_dash.head(3)["Top-3 em N modelos"].values)
        max_conv = 3 * len(methods)
        conv_pct = (total_top3 / max_conv * 100) if max_conv else 0
        verdict_color = "#2e7d32" if conv_pct >= 70 else ("#f57c00" if conv_pct >= 40 else "#c62828")

        st.markdown(f"""<div style="background: linear-gradient(135deg, #1F4E78 0%, #2E75B6 100%); color: white; padding: 18px; border-radius: 10px;">
              <div style="font-size: 11px; opacity: 0.9;">TOP-3 MCDM (CONSENSO)</div>
              <div style="font-size: 22px; font-weight: 700;">🥇 {top3[0] if len(top3)>0 else '—'}</div>
              <div style="font-size: 14px; opacity: 0.95;">🥈 {top3[1] if len(top3)>1 else '—'} · 🥉 {top3[2] if len(top3)>2 else '—'}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div style="background: {verdict_color}; color: white; padding: 10px; border-radius: 6px; margin-top: 8px; text-align: center; font-weight: 600;">
              Grau de Convergência: {conv_pct:.0f}%
            </div>""", unsafe_allow_html=True)

    with col_crit:
        st.markdown("##### 📊 Pesos AHP Activos")
        df_cwp = pd.DataFrame({"Crit.": crits, "Tipo": types, "Peso": weights, "%": [f"{w*100:.2f}%" for w in weights]}).sort_values("Peso", ascending=False)
        st.dataframe(df_cwp.style.format({"Peso": "{:.5f}"}).background_gradient(cmap="Blues", subset=["Peso"]), hide_index=True, use_container_width=True)
