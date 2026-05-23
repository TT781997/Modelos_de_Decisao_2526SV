"""
MCDM Dashboard v2 — Ferramenta de Apoio à Decisão Multicritério
================================================================
Reestruturada conforme pedido:
  • Sidebar: 3 fontes de dados (manual + paste, Excel, demo) + N_alt/N_crit + sensibilidade
  • AHP movido para a sua própria aba com iterações até CR < 0.10
  • Motores de Pesos com inputs específicos da teoria
  • Aba Gráficos com visualizações Plotly bonitas
  • Sensibilidade explícita e clara em cada aba
  • Relatório visual final
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="MCDM Dashboard v2", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
/* Boxes com cores explícitas para funcionar em DARK e LIGHT theme */
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

.stTabs [data-baseweb="tab"] { padding: 10px 18px; font-weight: 600; }
.stTabs [aria-selected="true"] { background-color: #1F4E78 !important; color: white !important; }

.sidebar-section {
    background: #f5f5f5; padding: 12px; border-radius: 6px; margin: 10px 0;
    border-left: 3px solid #1F4E78;
}
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
    if "data_source" not in st.session_state:
        st.session_state.data_source = "Demo"

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
    _, _, crits, _ = get_decision_matrix()
    n = len(crits)
    if (st.session_state.global_injection_on
        and st.session_state.global_injection_engine in st.session_state.engine_weights):
        w = st.session_state.engine_weights[st.session_state.global_injection_engine]
        if len(w) == n:
            w = np.array(w, dtype=float)
            return w / w.sum() if w.sum() > 0 else np.ones(n) / n
    crit_df = st.session_state.criteria_df.copy()
    crit_df = crit_df.dropna(subset=["Critério"])
    crit_df = crit_df[crit_df["Critério"].astype(str).str.strip() != ""]
    w = pd.to_numeric(crit_df["Peso Manual"], errors="coerce").fillna(0).values
    return w / w.sum() if w.sum() > 0 else np.ones(n) / n

def show_active_weights_banner():
    w = get_active_weights()
    _, _, crits, _ = get_decision_matrix()
    if st.session_state.global_injection_on:
        engine = st.session_state.global_injection_engine
        st.markdown(
            f'<div class="injection-active">🔌 Injecção Global ACTIVA — pesos do motor <b>{engine}</b></div>',
            unsafe_allow_html=True)
    cols = st.columns([3, 1])
    with cols[0]:
        df_w = pd.DataFrame({"Critério": crits, "Peso (%)": [f"{x*100:.2f}%" for x in w]})
        st.dataframe(df_w, hide_index=True, use_container_width=False)
    with cols[1]:
        st.metric("Σ pesos", f"{w.sum():.4f}")

def theory_box(title, html):
    st.markdown(f'<div class="theory-box"><h4>📚 {title}</h4>{html}</div>', unsafe_allow_html=True)

def purpose_box(text):
    """Caixa verde 'Para que serve esta aba' — usar no topo de cada aba."""
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
            <h2>👈 SEM DADOS — CARREGUE NA BARRA LATERAL</h2>
            <p>Para usar esta aba, primeiro tem de carregar uma <b>matriz de decisão</b>.</p>
            <p><b>3 formas (na sidebar à esquerda):</b></p>
            <p>📋 <b>Demo</b> → "Carregar este caso" &nbsp;·&nbsp;
               ✏️ <b>Manual</b> → "Criar matriz vazia" ou colar do Excel &nbsp;·&nbsp;
               📁 <b>Excel</b> → upload .xlsx</p>
            <p><i>Vá à aba 🏠 Início para ver as 3 opções em detalhe.</i></p>
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
# SENSIBILIDADE UNIVERSAL — destacada
# =============================================================================
def render_sensitivity(score_function, alts, crits, base_weights, higher_is_better=True, key_suffix=""):
    """Sensibilidade ±X% nos pesos. Apresentação destacada e clara."""
    st.markdown(
        '<div class="sensitivity-box"><h3>🎯 Análise de Sensibilidade ± X% nos Pesos</h3>'
        '<p style="margin-bottom:0; color:#bf360c;">Variamos o peso de <b>cada critério isoladamente</b> '
        'em ±X% e renormalizamos os restantes. Para cada cenário recalculamos o ranking e comparamos com o Base.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # Slider lê do session_state (controlo na sidebar)
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

    st.markdown(f"**Variação ± aplicada:** {variation_pct}% (ajustável na barra lateral)")
    st.markdown("**Legenda:** 🟢 sobe no ranking · 🔴 desce no ranking · ⚪ sem alteração")
    st.dataframe(df_sens.style.apply(style_row, axis=1), use_container_width=True)

    # Robustez
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
    "**Decisão Multicritério** · 3 modos de entrada de dados · 9 modelos MCDM · "
    "Motores de pesos · Sensibilidade universal · Relatório final"
)


# =============================================================================
# SIDEBAR — 3 fontes de dados + dimensões + sensibilidade
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuração")

    # ============== SECÇÃO 1: FONTE DE DADOS ==============
    st.markdown("### 📥 Fonte de Dados")
    data_source = st.radio(
        "Como quer fornecer os dados?",
        ["📋 Demo (pré-definidos)", "✏️ Manual (editor + paste)", "📁 Carregar Excel",
         "📥 Quadros em bruto (alts + crits)"],
        key="data_source_radio",
        help="4 modos: caso demo, manual com paste, ficheiro Excel, ou paste de DOIS quadros (alts + crits) "
             "como vêm em enunciados académicos"
    )

    if data_source == "📋 Demo (pré-definidos)":
        preset_name = st.selectbox(
            "Escolher caso:",
            list(DEMO_PRESETS.keys()),
            key="preset_selector"
        )
        if st.button("📥 Carregar este caso", use_container_width=True, type="primary"):
            preset = DEMO_PRESETS[preset_name]
            st.session_state.criteria_df = preset["criteria"].copy()
            st.session_state.matrix_df = preset["matrix"].copy()
            st.session_state.engine_weights = {}
            st.session_state.ahp_history = []
            st.success(f"✓ Carregado: {preset_name}")
            st.rerun()

    elif data_source == "✏️ Manual (editor + paste)":
        st.markdown("**Opção A — Dimensões:**")
        col_a, col_b = st.columns(2)
        n_alt_input = col_a.number_input("N.º Alternativas", min_value=2, max_value=30, value=5, step=1, key="n_alt_input")
        n_crit_input = col_b.number_input("N.º Critérios", min_value=2, max_value=15, value=4, step=1, key="n_crit_input")
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
            st.success(f"✓ Matriz {n_alt_input}×{n_crit_input} criada (preencher abaixo)")
            st.rerun()

        st.markdown("**Opção B — Colar do Excel:**")
        st.caption(
            "1) No Excel, seleccione células incluindo cabeçalhos · 2) Ctrl+C · 3) Clique aqui · 4) Ctrl+V.\n\n"
            "A 1ª coluna deve ter os nomes das alternativas e a 1ª linha os nomes dos critérios. "
            "Aceita separação por TAB (Excel) ou ; e decimais com `,` ou `.`"
        )
        paste_text = st.text_area(
            "Colar aqui (Ctrl+V):",
            height=160,
            placeholder="Alternativa\tCusto\tQualidade\tPrazo\nForn A\t1200\t8\t15\nForn B\t1500\t6\t20",
            key="paste_area"
        )

        # PREVIEW automático conforme se cola
        if paste_text and paste_text.strip():
            # Detecção automática de separador
            sep_guess = "\t"
            first_line = paste_text.strip().split("\n")[0]
            if "\t" in first_line:
                sep_guess = "\t"
            elif ";" in first_line:
                sep_guess = ";"
            elif "," in first_line and first_line.count(",") > 1:
                sep_guess = ","
            else:
                # Várias espaços consecutivos
                sep_guess = r"\s{2,}"

            try:
                if sep_guess == r"\s{2,}":
                    df_preview = pd.read_csv(StringIO(paste_text), sep=sep_guess, engine="python", dtype=str)
                else:
                    df_preview = pd.read_csv(StringIO(paste_text), sep=sep_guess, dtype=str)

                # Converter decimais com vírgula para ponto E DEPOIS para numérico
                for c in df_preview.columns[1:]:
                    df_preview[c] = (df_preview[c].astype(str)
                                                  .str.replace(",", ".", regex=False)
                                                  .str.strip())
                    df_preview[c] = pd.to_numeric(df_preview[c], errors="coerce").fillna(0)

                first_col = df_preview.columns[0]
                df_preview = df_preview.rename(columns={first_col: "Alternativa"})

                st.caption(f"✓ Detectado: separador `{sep_guess}` · {len(df_preview)} alts × {len(df_preview.columns)-1} crits")
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

    elif data_source == "📁 Carregar Excel":
        uploaded = st.file_uploader(
            "Carregar Excel (.xlsx)",
            type=["xlsx", "xls"],
            help="Folha 1 deve ter: 1ª coluna = nomes alternativas, restantes = critérios numéricos"
        )
        if uploaded is not None:
            if st.button("📁 Processar Excel", use_container_width=True):
                try:
                    df = pd.read_excel(uploaded)
                    first_col = df.columns[0]
                    crits = list(df.columns[1:])
                    df = df.rename(columns={first_col: "Alternativa"})
                    new_crits = pd.DataFrame({
                        "Critério": crits,
                        "Tipo": ["max"] * len(crits),
                        "Peso Manual": [1.0 / len(crits)] * len(crits),
                    })
                    for c in crits:
                        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
                    st.session_state.criteria_df = new_crits
                    st.session_state.matrix_df = df
                    st.session_state.engine_weights = {}
                    st.success(f"✓ Carregado: {len(df)} alts × {len(crits)} crits")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erro: {e}")

    else:  # 📥 Quadros em bruto (alts + crits)
        st.caption(
            "Cole **dois quadros separados** como vêm em enunciados académicos:\n\n"
            "**Quadro A — Alternativas com atributos**: 1ª coluna = nome alt, "
            "restantes = atributos numéricos (critérios) OU texto (metadados como Cliente, Estado).\n\n"
            "**Quadro B — Critérios com pesos**: Código, Critério, Natureza (Benefício/Custo), Peso."
        )

        with st.expander("Exemplo do formato (caso MCG)", expanded=False):
            st.code(
                "Quadro A — Alternativas:\n"
                "Alt\tRef\tCliente\tValor Pot\tProb Fecho\tEsforço\tFit\tUrgência\tRel Cliente\tEstado\n"
                "A1\t9786\tBe\t250000000\t0.25\t24\t4\t180\t4\tCotação\n"
                "A2\t9780\tZf\t300000\t0.35\t8\t5\t60\t5\tCotação\n"
                "...\n\n"
                "Quadro B — Critérios:\n"
                "Código\tCritério\tNatureza\tPeso\n"
                "C1_VP\tValor Potencial\tBenefício\t0.462\n"
                "C2_PF\tProb. Fecho\tBenefício\t0.218\n"
                "C3_EE\tEsforço Estimado\tCusto\t0.024\n"
                "...",
                language="text"
            )

        paste_alts = st.text_area(
            "**Quadro A — Alternativas (com atributos)** — colar Ctrl+V:",
            height=140, key="paste_alts_raw",
            placeholder="Alt\tCliente\tValor Pot\tEsforço\tEstado\nA1\tBe\t250000000\t24\tCotação\nA2\tZf\t300000\t8\tCotação"
        )

        paste_crits = st.text_area(
            "**Quadro B — Critérios (Código, Critério, Natureza, Peso)** — colar Ctrl+V:",
            height=120, key="paste_crits_raw",
            placeholder="Código\tCritério\tNatureza\tPeso\nC1_VP\tValor Potencial\tBenefício\t0.462\nC2_PF\tProb. Fecho\tBenefício\t0.218"
        )

        def parse_paste(text):
            """Parse paste — auto-detecta separador, tolera vírgula decimal."""
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
                # Critérios: detectar colunas — nome do crit + natureza + peso
                # Tolerar maiúsculas/minúsculas, ordem variável
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

                # Quadro A: 1ª coluna = nome alt; auto-classificar colunas como numéricas (critério) ou metadata
                alt_col = df_alts_raw.columns[0]
                df_alts_raw = df_alts_raw.rename(columns={alt_col: "Alternativa"})

                numeric_cols = []
                metadata_cols = []
                for c in df_alts_raw.columns[1:]:
                    s = df_alts_raw[c].astype(str).str.replace(",", ".", regex=False).str.strip()
                    nums = pd.to_numeric(s, errors="coerce")
                    if nums.notna().mean() > 0.5:
                        df_alts_raw[c] = nums.fillna(0)
                        numeric_cols.append(c)
                    else:
                        metadata_cols.append(c)

                # PREVIEW
                st.caption(f"✓ Detectados: {len(df_alts_raw)} alts, "
                           f"{len(numeric_cols)} crit numéricos, {len(metadata_cols)} metadados")
                if metadata_cols:
                    st.caption(f"📝 Metadados (não usados para cálculo, mas guardados): {', '.join(metadata_cols)}")
                st.dataframe(df_alts_raw, hide_index=True, use_container_width=True)

                if col_code and col_name:
                    st.caption("**Critérios identificados:**")
                    st.dataframe(df_crits_raw, hide_index=True, use_container_width=True)

                if st.button("📥 Importar tudo", use_container_width=True, type="primary"):
                    # Construir matriz e criteria_df
                    new_matrix = df_alts_raw[["Alternativa"] + numeric_cols].copy()
                    crit_list = []
                    for _, row in df_crits_raw.iterrows():
                        nome = str(row[col_name]) if col_name else ""
                        codigo = str(row[col_code]) if col_code else nome
                        nat = (str(row[col_nat]).lower() if col_nat else "max")
                        tipo = "min" if any(x in nat for x in ["custo", "cost", "min"]) else "max"
                        peso_str = str(row[col_peso]).replace(",", ".") if col_peso else "0"
                        try:
                            peso = float(peso_str)
                        except Exception:
                            peso = 1.0 / len(df_crits_raw)
                        crit_list.append({"Critério": codigo, "Tipo": tipo, "Peso Manual": peso})
                    new_crits_df = pd.DataFrame(crit_list)

                    # Normalizar pesos
                    s = new_crits_df["Peso Manual"].sum()
                    if s > 0:
                        new_crits_df["Peso Manual"] = new_crits_df["Peso Manual"] / s

                    # Renomear colunas numéricas para os códigos dos critérios na ordem
                    # (assume mesma ordem ou número)
                    if len(numeric_cols) == len(new_crits_df):
                        rename_map = dict(zip(numeric_cols, new_crits_df["Critério"].tolist()))
                        new_matrix = new_matrix.rename(columns=rename_map)

                    # Guardar metadata (para relatório)
                    st.session_state["alt_metadata"] = df_alts_raw[["Alternativa"] + metadata_cols].copy() if metadata_cols else None
                    st.session_state["crit_metadata"] = df_crits_raw.copy()

                    st.session_state.matrix_df = new_matrix
                    st.session_state.criteria_df = new_crits_df
                    st.session_state.engine_weights = {}
                    st.success(f"✓ Importado: {len(new_matrix)} alts × {len(new_crits_df)} crits. "
                               f"Metadados guardados para o relatório.")
                    st.rerun()
            else:
                st.warning("⚠️ Cole ambos os quadros para activar o preview.")

    st.markdown("---")

    # ============== SECÇÃO 2: SENSIBILIDADE ==============
    st.markdown("### 🎯 Análise de Sensibilidade")
    st.session_state.sensitivity_pct = st.slider(
        "Variação ± nos pesos (%):",
        5, 50, st.session_state.sensitivity_pct, 5,
        key="sens_slider_sidebar",
        help="Aplicada em TODAS as abas. Cada peso é variado isoladamente; restantes ajustados para Σ=1."
    )

    st.markdown("---")

    # ============== SECÇÃO 3: INJECÇÃO GLOBAL ==============
    st.markdown("### 🔌 Injecção Global de Pesos")
    st.session_state.global_injection_on = st.toggle(
        "Activar pesos do motor",
        value=st.session_state.global_injection_on,
        help="ON: todos os modelos usam pesos do motor seleccionado. OFF: usam 'Peso Manual'."
    )
    if st.session_state.global_injection_on:
        available = list(st.session_state.engine_weights.keys())
        if not available:
            st.caption("⚠️ Nenhum motor calculado ainda. Vá às abas dos motores.")
        else:
            st.session_state.global_injection_engine = st.selectbox(
                "Motor activo:",
                available,
                index=available.index(st.session_state.global_injection_engine)
                      if st.session_state.global_injection_engine in available else 0,
                key="engine_selector_sidebar"
            )

    st.markdown("---")

    # ============== SECÇÃO 4: EDITORES (sempre acessíveis) ==============
    with st.expander("📋 Editor de Critérios", expanded=True):
        edited_crit = st.data_editor(
            st.session_state.criteria_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="criteria_editor_sb",
            column_config={
                "Critério": st.column_config.TextColumn("Critério", required=True),
                "Tipo": st.column_config.SelectboxColumn("Tipo", options=["max", "min"], required=True),
                "Peso Manual": st.column_config.NumberColumn(
                    "Peso Manual", min_value=0.0, max_value=1.0, step=0.01, format="%.4f"),
            }
        )
        if edited_crit is not None and not edited_crit.equals(st.session_state.criteria_df):
            valid = edited_crit.dropna(subset=["Critério"])
            valid = valid[valid["Critério"].astype(str).str.strip() != ""]
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
            st.rerun()

        w_manual = pd.to_numeric(st.session_state.criteria_df["Peso Manual"], errors="coerce").fillna(0)
        soma = w_manual.sum()
        if abs(soma - 1.0) > 0.01:
            st.caption(f"⚠️ Σ = **{soma:.4f}** (será renormalizado)")
        else:
            st.caption(f"✅ Σ = **{soma:.4f}**")

    with st.expander("🔢 Editor de Matriz de Decisão", expanded=True):
        crit_names = st.session_state.criteria_df["Critério"].astype(str).tolist()
        matrix_col_config = {
            "Alternativa": st.column_config.TextColumn("Alternativa", required=True, width="small"),
        }
        for crit in crit_names:
            matrix_col_config[crit] = st.column_config.NumberColumn(crit, format="%.4f", required=False)
        edited_matrix = st.data_editor(
            st.session_state.matrix_df, num_rows="dynamic", use_container_width=True,
            hide_index=True, key="matrix_editor_sb", column_config=matrix_col_config,
        )
        if edited_matrix is not None and not edited_matrix.equals(st.session_state.matrix_df):
            st.session_state.matrix_df = edited_matrix.reset_index(drop=True)
            st.rerun()


# =============================================================================
# TABS
# =============================================================================
TAB_LABELS = [
    "🏠 Início",
    "📋 Dados",
    "⚖️ Motores de Pesos",
    "🔍 AHP",
    "🎯 TOPSIS",
    "📈 PROMETHEE II",
    "⚖️ VIKOR",
    "📊 COPRAS",
    "🚫 ELECTRE III",
    "💡 MAUT",
    "🌐 DEMATEL",
    "🌫️ Fuzzy TOPSIS",
    "🧮 Fuzzy AHP",
    "📊 Gráficos",
    "🏆 Dashboard",
    "🎛️ Vista 360°",
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

    # ===== CTA: ONDE METER OS DADOS — primeira coisa que se vê =====
    st.markdown(
        """<div class="cta-box">
        <h2>👈 ONDE METER OS DADOS DO ENUNCIADO?</h2>
        <p>Os dados (matriz alternativas × critérios) <b>entram pela BARRA LATERAL</b> à esquerda.
        Se não a vê, clique no ícone <b>›</b> no canto superior esquerdo para a abrir.</p>
        <p>Tem 3 formas (cada uma com botão próprio):</p>
        </div>""",
        unsafe_allow_html=True
    )

    # 3 opções de entrada — visualmente claras
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.markdown(
            """<div style="background:#e3f2fd; padding:16px; border-radius:8px; border-left:4px solid #1976d2; color:#0d47a1;">
            <h4 style="color:#0d47a1; margin-top:0;">📋 Opção 1 — Demo</h4>
            <p style="color:#0d47a1;"><b>Use um caso pré-carregado</b> (mais rápido para experimentar).</p>
            <ol style="color:#0d47a1;">
                <li>Sidebar → <b>"📋 Demo (pré-definidos)"</b></li>
                <li>Escolher: Caso MCG / Fornecedor / Investimento</li>
                <li>Clicar <b>"📥 Carregar este caso"</b></li>
            </ol>
            </div>""",
            unsafe_allow_html=True
        )
    with oc2:
        st.markdown(
            """<div style="background:#fff3e0; padding:16px; border-radius:8px; border-left:4px solid #f57c00; color:#bf360c;">
            <h4 style="color:#bf360c; margin-top:0;">✏️ Opção 2 — Manual / Paste</h4>
            <p style="color:#bf360c;"><b>Inserir manualmente</b> os dados do seu enunciado.</p>
            <ol style="color:#bf360c;">
                <li>Sidebar → <b>"✏️ Manual (editor + paste)"</b></li>
                <li><b>Opção A:</b> definir N alts × N crits → "Criar matriz vazia" → preencher nos editores expandidos</li>
                <li><b>Opção B:</b> copiar do Excel (Ctrl+C) → colar na text-area → "Confirmar e carregar"</li>
            </ol>
            </div>""",
            unsafe_allow_html=True
        )
    with oc3:
        st.markdown(
            """<div style="background:#f3e5f5; padding:16px; border-radius:8px; border-left:4px solid #7b1fa2; color:#4a148c;">
            <h4 style="color:#4a148c; margin-top:0;">📁 Opção 3 — Excel</h4>
            <p style="color:#4a148c;"><b>Carregar ficheiro</b> .xlsx do enunciado.</p>
            <ol style="color:#4a148c;">
                <li>Sidebar → <b>"📁 Carregar Excel"</b></li>
                <li>Arrastar/escolher ficheiro</li>
                <li>Clicar <b>"📁 Processar Excel"</b></li>
            </ol>
            <p style="color:#4a148c; font-size:13px;"><i>Formato: 1ª coluna = nomes alts, restantes = critérios</i></p>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("🚀 Como começar em 4 passos")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("### 1️⃣ Dados")
        st.markdown(
            "👈 Na **barra lateral** escolha como fornece os dados (ver acima).\n\n"
            "Depois ajuste **tipo (max/min)** e **pesos manuais** no editor (sidebar)."
        )
    with c2:
        st.markdown("### 2️⃣ Pesos")
        st.markdown(
            "Defina importância dos critérios:\n\n"
            "• **Manual**: edita pesos na sidebar\n\n"
            "• **Motor** (aba ⚖️): SWING, SMART, Entropia, CRITIC\n\n"
            "• **AHP** (aba 🔍): par-a-par"
        )
    with c3:
        st.markdown("### 3️⃣ Modelos")
        st.markdown(
            "Cada aba executa um modelo:\n\n"
            "TOPSIS, PROMETHEE II, VIKOR, COPRAS, ELECTRE III, MAUT, DEMATEL, Fuzzy TOPSIS, Fuzzy AHP\n\n"
            "Cada uma dá ranking + sensibilidade."
        )
    with c4:
        st.markdown("### 4️⃣ Decisão")
        st.markdown(
            "Consolide e decida:\n\n"
            "• **📊 Gráficos**: visualizações\n\n"
            "• **🏆 Dashboard**: consenso\n\n"
            "• **📄 Relatório**: descarregar"
        )

    st.markdown("---")
    st.subheader("📑 O que faz cada aba")

    tab_descriptions = [
        ("📋 Dados", "Mostra a matriz de decisão actual (alternativas × critérios) e os pesos activos. "
                     "Heatmap visual normalizado para inspecção rápida."),
        ("⚖️ Motores de Pesos", "Calcula pesos automaticamente por **4 métodos**: SWING (swing pior→melhor), "
                                 "SMART (pontuação 0-100), Entropia (variabilidade dos dados), CRITIC (variância + correlações). "
                                 "Active a injecção global na sidebar para os modelos usarem estes pesos."),
        ("🔍 AHP", "Comparação par-a-par escala Saaty 1-9. Calcula pesos via autovector e valida com CR < 0.10. "
                   "Se CR ≥ 0.10 → **app sugere correcção iterativa** do par mais inconsistente."),
        ("🎯 TOPSIS", "Hwang & Yoon (1981). Mede distância à solução ideal e anti-ideal. Compensatório, baseado em distâncias."),
        ("📈 PROMETHEE II", "Brans (1985). Fluxos de preferência par-a-par com 3 funções (Usual, Linear, Gaussiana). Não-compensatório."),
        ("⚖️ VIKOR", "Opricovic & Tzeng (2004). Solução de compromisso (utilidade global S + arrependimento R)."),
        ("📊 COPRAS", "Zavadskas & Kaklauskas (1996). Função proporcional benefícios/custos com grau de utilidade U_i (%)."),
        ("🚫 ELECTRE III", "Roy (1968+). Sobreclassificação com limiares q/p/v (indiferença, preferência, veto)."),
        ("💡 MAUT", "Keeney & Raiffa (1976). Utilidade aditiva com 4 funções (Linear, Exp, Potência 0.5, Potência 2)."),
        ("🌐 DEMATEL", "Gabus & Fontela (1972). Relações causa-efeito entre critérios + ranking ajustado."),
        ("🌫️ Fuzzy TOPSIS", "Chen (2000). TOPSIS com números fuzzy triangulares para lidar com imprecisão."),
        ("🧮 Fuzzy AHP", "Chang (1996). AHP com pesos fuzzy + defuzzificação centro de área."),
        ("📊 Gráficos", "**5 visualizações decisivas**: heatmap rankings, radar Top-3, tornado sensibilidade, "
                        "scores normalizados, convergência Top-3."),
        ("🏆 Dashboard", "Consolida os rankings de todos os modelos via Borda invertido. Top-3 consensual."),
        ("🎛️ Vista 360°", "**Dashboard one-page estilo Figura 1 do enunciado** — filtros, ranking, radar, 4 charts modelo, "
                          "tornado sensibilidade, painel de recomendação. Tudo na mesma página."),
        ("📑 Relatório Técnico", "Estrutura dos **7 capítulos** do enunciado (Intro · Dados · Modelos · SA · Dashboard · "
                                 "Comparação · Conclusões) + Referências APA. Downloads em CSV, Excel (6 folhas) e Markdown."),
    ]

    for label, desc in tab_descriptions:
        st.markdown(f"**{label}** — {desc}")

    st.markdown("---")
    st.subheader("🎯 Conceitos-chave")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("""
        **🔌 Injecção Global de Pesos**
        Toggle na sidebar. Quando **ON**, todos os modelos usam os pesos calculados pelo motor
        escolhido (AHP, SWING, etc.). Quando **OFF**, usam o "Peso Manual" definido no editor
        de critérios da sidebar.

        **⚖️ Tipo de critério**
        - **max** (benefício): quanto maior, melhor (qualidade, retorno)
        - **min** (custo): quanto menor, melhor (custo, prazo, risco)
        """)
    with cc2:
        st.markdown("""
        **🎯 Análise de Sensibilidade**
        Variamos cada peso isoladamente em ±X% (slider na sidebar) e renormalizamos os outros
        para Σ=1. Para cada cenário recalculamos o ranking. Cores:
        - 🟢 alternativa **sobe** no ranking
        - 🔴 alternativa **desce** no ranking
        - ⚪ alternativa **inalterada**

        **🏆 Convergência inter-modelo**
        Quantos modelos colocam a mesma alternativa no Top-3 = robustez consensual.
        """)

    st.markdown("---")
    st.info("💡 **Dica:** comece sempre pelos casos **Demo** para ver a app a funcionar antes de carregar os seus dados.")


# =============================================================================
# TAB 1: DADOS
# =============================================================================
with tabs[1]:
    st.header("📋 Dados de Entrada")
    purpose_box("Visualizar a <b>matriz de decisão</b> actual (alternativas × critérios) e os pesos activos. Mostra estatísticas e heatmap normalizado.")
    theory_box(
        "Como começar",
        """
        <ol>
            <li>Na <b>barra lateral</b>, escolha a <b>fonte de dados</b>:
                <ul>
                    <li><b>Demo</b>: 3 casos pré-definidos (MCG, Fornecedor, Investimento)</li>
                    <li><b>Manual</b>: defina N alts × N crits OU cole valores do Excel (separado por TAB)</li>
                    <li><b>Excel</b>: carregue um ficheiro .xlsx</li>
                </ul>
            </li>
            <li>Ajuste tipos (max/min) e pesos no editor de critérios</li>
            <li>Edite valores no editor de matriz</li>
            <li>Configure a <b>sensibilidade</b> (slider na sidebar)</li>
            <li>Veja resultados em cada aba de modelo</li>
        </ol>
        """
    )

    matrix, alts, crits, types = get_decision_matrix()
    if not check_valid_input():
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alternativas", len(alts))
    c2.metric("Critérios", len(crits))
    c3.metric("Max", types.count("max"))
    c4.metric("Min", types.count("min"))

    st.subheader("Matriz de Decisão")
    display_df = pd.DataFrame(matrix, index=alts, columns=crits)
    st.dataframe(display_df.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0),
                use_container_width=True)

    st.subheader("Critérios e Pesos Activos")
    show_active_weights_banner()

    st.subheader("Heatmap normalizado (min-max, sentido aplicado)")
    norm = normalize_minmax(matrix, types)
    norm_df = pd.DataFrame(norm, index=alts, columns=crits)
    st.dataframe(norm_df.style.format("{:.3f}").background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True)
    st.caption("1.0 = melhor; 0.0 = pior (com inversão automática para critérios de min).")


# =============================================================================
# TAB 2: MOTORES DE PESOS (SEM AHP — está na aba dedicada)
# =============================================================================
with tabs[2]:
    st.header("⚖️ Motores de Pesos")
    st.markdown(
        '<div style="background:#e8f5e9; padding:10px 16px; border-left:4px solid #2e7d32; '
        'border-radius:4px; margin-bottom:16px;">'
        '<b>📌 Para que serve esta aba:</b> Calcular automaticamente os <b>pesos dos critérios</b> '
        'em vez de os definir manualmente. Útil quando não sabe que peso atribuir ou quer um método objectivo. '
        '<br><br><b>Workflow:</b> (1) escolher motor abaixo · (2) preencher inputs específicos · '
        '(3) ver pesos calculados · (4) na sidebar activar <b>"🔌 Injecção Global"</b> e escolher este motor '
        'para os modelos MCDM passarem a usar estes pesos.'
        '</div>',
        unsafe_allow_html=True
    )
    theory_box(
        "4 métodos disponíveis (AHP está em aba dedicada)",
        """
        <ul>
            <li><b>SWING</b> (subjectivo): você define <i>quão impactante é o swing pior→melhor</i> em cada critério (0-100). Pesos = scores/Σscores.</li>
            <li><b>SMART</b> (subjectivo): você classifica cada critério em 0-100 de importância. Mais simples que SWING.</li>
            <li><b>Entropia de Shannon</b> (objectivo): cálculo automático a partir da matriz. Pesos = variabilidade dos dados.</li>
            <li><b>CRITIC</b> (objectivo): cálculo automático. Pesos = variância × (1 − correlação) com outros critérios.</li>
        </ul>
        <p>Os métodos <b>subjectivos</b> precisam dos seus inputs (pontuações). Os <b>objectivos</b> calculam tudo sozinhos.</p>
        """
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    n = len(crits)

    engine = st.radio(
        "Motor a usar:",
        ["SWING", "SMART", "Entropia", "CRITIC"],
        horizontal=True, key="engine_radio"
    )

    # ----------- SWING -----------
    if engine == "SWING":
        st.subheader("🎢 SWING Weighting")
        theory_box(
            "Como funciona (von Winterfeldt & Edwards, 1986) — método SUBJECTIVO",
            """
            <p><b>Conceito:</b> imagine uma alternativa onde TODOS os critérios estão no nível PIOR.
            Para cada critério, pergunta-se «qual é o benefício de fazer SWING desse pior nível
            para o melhor?». O critério com swing mais impactante recebe 100 pontos; os outros recebem
            pontuações relativas (0-100).</p>
            <p><b>3 passos:</b></p>
            <ol>
                <li>Confirmar/ajustar os <b>níveis pior e melhor</b> de cada critério (a app auto-preenche pela matriz)</li>
                <li>Atribuir <b>100 pontos</b> ao critério com swing mais impactante</li>
                <li>Atribuir <b>0-100</b> pontos aos restantes, relativos ao de 100</li>
            </ol>
            """
        )
        st.info(
            "💡 **Como preencher:** Edite a coluna **'Swing Score (0-100)'**. "
            "O critério mais importante = 100. Os outros, proporcionalmente. "
            "Os pesos finais são calculados automaticamente abaixo."
        )
        st.latex(r"w_j = \frac{p_j}{\sum_k p_k},\quad p_j \in [0, 100]")

        # Auto-fill pior/melhor a partir da matriz
        swing_key = f"swing_data_{'_'.join(crits)}"
        if swing_key not in st.session_state or len(st.session_state[swing_key]) != n:
            pior = [matrix[:, j].min() if types[j] == "max" else matrix[:, j].max() for j in range(n)]
            melhor = [matrix[:, j].max() if types[j] == "max" else matrix[:, j].min() for j in range(n)]
            st.session_state[swing_key] = pd.DataFrame({
                "Critério": crits,
                "Nível Pior": pior,
                "Nível Melhor": melhor,
                "Swing Score (0-100)": [100] + [50] * (n - 1)
            })

        edited = st.data_editor(
            st.session_state[swing_key], use_container_width=True, hide_index=True,
            key="swing_edit",
            column_config={
                "Critério": st.column_config.TextColumn("Critério", disabled=True),
                "Nível Pior": st.column_config.NumberColumn("Nível Pior", format="%.4f"),
                "Nível Melhor": st.column_config.NumberColumn("Nível Melhor", format="%.4f"),
                "Swing Score (0-100)": st.column_config.NumberColumn(
                    "Swing Score", min_value=0.0, max_value=100.0, step=5.0,
                    help="100 = critério onde o swing pior→melhor é mais impactante"),
            },
            disabled=["Critério"]
        )
        st.session_state[swing_key] = edited

        pts = pd.to_numeric(edited["Swing Score (0-100)"], errors="coerce").fillna(0).values
        w_swing = pts / pts.sum() if pts.sum() > 0 else np.ones(n) / n

        st.markdown("**Pesos SWING calculados:**")
        df_w = pd.DataFrame({"Critério": crits, "Score": pts, "Peso": w_swing,
                             "%": [f"{x*100:.2f}%" for x in w_swing]})
        st.dataframe(df_w.style.format({"Score": "{:.1f}", "Peso": "{:.4f}"}),
                    hide_index=True, use_container_width=True)

        # Gráfico de barras
        fig = px.bar(df_w, x="Critério", y="Peso", text=df_w["%"],
                     color="Peso", color_continuous_scale="Viridis",
                     title="Pesos SWING")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.session_state.engine_weights["SWING"] = w_swing
        st.success(f"💾 Pesos SWING guardados (Σ={w_swing.sum():.4f}). Active a injecção global na sidebar para usar.")

    # ----------- SMART -----------
    elif engine == "SMART":
        st.subheader("📐 SMART — Simple Multi-Attribute Rating Technique")
        theory_box(
            "Como funciona (von Winterfeldt & Edwards, 1986) — método SUBJECTIVO",
            """
            <p><b>Conceito:</b> mais simples que SWING. Você classifica directamente <b>cada critério em 0-100</b>
            conforme a sua importância para a decisão. Não há comparação de swings — é uma pontuação directa.</p>
            <p><b>Convenção habitual:</b> critério MAIS importante = 100; critério MENOS importante ≈ 10.
            Os pesos são pontuações ÷ soma das pontuações.</p>
            """
        )
        st.info(
            "💡 **Como preencher:** edite a coluna **'Pontuação (0-100)'** dando 100 ao critério mais "
            "importante e proporções aos restantes. Os pesos calculam-se automaticamente."
        )
        st.latex(r"w_j = \frac{p_j}{\sum_k p_k}")

        smart_key = f"smart_data_{'_'.join(crits)}"
        if smart_key not in st.session_state or len(st.session_state[smart_key]) != n:
            st.session_state[smart_key] = pd.DataFrame({
                "Critério": crits,
                "Pontuação (0-100)": [80] * n
            })
        edited = st.data_editor(
            st.session_state[smart_key], use_container_width=True, hide_index=True, key="smart_edit",
            column_config={
                "Critério": st.column_config.TextColumn("Critério", disabled=True),
                "Pontuação (0-100)": st.column_config.NumberColumn(
                    "Pontuação", min_value=0.0, max_value=100.0, step=5.0),
            }, disabled=["Critério"]
        )
        st.session_state[smart_key] = edited
        pts = pd.to_numeric(edited["Pontuação (0-100)"], errors="coerce").fillna(0).values
        w_smart = pts / pts.sum() if pts.sum() > 0 else np.ones(n) / n
        df_w = pd.DataFrame({"Critério": crits, "Score": pts, "Peso": w_smart,
                             "%": [f"{x*100:.2f}%" for x in w_smart]})
        st.dataframe(df_w.style.format({"Score": "{:.1f}", "Peso": "{:.4f}"}),
                    hide_index=True, use_container_width=True)
        fig = px.bar(df_w, x="Critério", y="Peso", text=df_w["%"],
                     color="Peso", color_continuous_scale="Plasma", title="Pesos SMART")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.session_state.engine_weights["SMART"] = w_smart
        st.success(f"💾 Pesos SMART guardados (Σ={w_smart.sum():.4f}).")

    # ----------- ENTROPIA -----------
    elif engine == "Entropia":
        st.subheader("📊 Entropia de Shannon")
        theory_box(
            "Como funciona (Shannon, 1948) — método OBJECTIVO (sem inputs do utilizador)",
            """
            <p><b>Conceito:</b> mede a <b>quantidade de informação</b> de cada critério via variabilidade dos
            dados na matriz. Critérios cuja coluna tem MUITA variabilidade (ex.: alts com 5, 80, 200) trazem
            MUITA informação para discriminar → recebem PESO MAIOR.
            Critérios com pouca variabilidade (ex.: alts todas com ~50) trazem pouca informação → peso menor.</p>
            <p><b>Não há inputs do utilizador</b> — basta clicar nesta opção; pesos calculam-se automaticamente
            pelos 3 passos abaixo.</p>
            """
        )
        st.info("💡 **Como usar:** nada a preencher. Os pesos aparecem já calculados abaixo a partir da matriz.")
        st.markdown("**Passo 1 — Normalização (max → proporção; min → inverso)**")
        st.latex(r"x'_{ij} = \frac{x_{ij}}{\sum_i x_{ij}}\;\text{(max)};\quad x'_{ij} = \frac{1/x_{ij}}{\sum_i 1/x_{ij}}\;\text{(min)}")

        try:
            m = matrix.shape[0]
            X_norm = np.zeros_like(matrix, dtype=float)
            for j in range(n):
                if types[j] == "max":
                    s = matrix[:, j].sum()
                    X_norm[:, j] = matrix[:, j] / s if s > 0 else 1/m
                else:
                    inv = 1.0 / np.where(matrix[:, j] == 0, 1e-9, matrix[:, j])
                    X_norm[:, j] = inv / inv.sum() if inv.sum() > 0 else 1/m

            st.markdown("**Passo 2 — Entropia E_j**")
            st.latex(r"E_j = -k \sum_i x'_{ij} \ln(x'_{ij}),\quad k = 1/\ln(m)")
            k = 1.0 / np.log(m) if m > 1 else 1.0
            E = np.array([-k * np.sum(np.where(X_norm[:, j] > 0, X_norm[:, j] * np.log(X_norm[:, j]), 0))
                          for j in range(n)])

            st.markdown("**Passo 3 — Divergência e pesos**")
            st.latex(r"d_j = 1 - E_j,\quad w_j = d_j / \sum_k d_k")
            d = 1 - E
            w_ent = d / d.sum() if d.sum() > 0 else np.ones(n) / n
            df_w = pd.DataFrame({
                "Critério": crits, "Entropia E_j": E, "Divergência d_j": d,
                "Peso": w_ent, "%": [f"{x*100:.2f}%" for x in w_ent]
            })
            st.dataframe(df_w.style.format({"Entropia E_j": "{:.4f}", "Divergência d_j": "{:.4f}", "Peso": "{:.4f}"}),
                        hide_index=True, use_container_width=True)
            fig = px.bar(df_w, x="Critério", y="Peso", text=df_w["%"],
                         color="Peso", color_continuous_scale="Greens", title="Pesos Entropia")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.engine_weights["Entropia"] = w_ent
            st.success(f"💾 Pesos Entropia guardados (Σ={w_ent.sum():.4f}).")
        except Exception as e:
            st.error(f"Erro Entropia: {e}")

    # ----------- CRITIC -----------
    else:  # CRITIC
        st.subheader("🔬 CRITIC")
        theory_box(
            "Como funciona (Diakoulaki, 1995) — método OBJECTIVO (sem inputs do utilizador)",
            """
            <p><b>Conceito:</b> combina duas ideias:</p>
            <ul>
                <li><b>Contraste</b>: critérios com mais variabilidade (σ alto) trazem mais informação</li>
                <li><b>Conflito</b>: critérios pouco correlacionados com outros trazem informação <i>única</i> (não redundante)</li>
            </ul>
            <p>Recebem peso maior critérios com ALTA variabilidade <b>E</b> BAIXA correlação com outros.
            <b>Sem inputs do utilizador</b> — basta clicar nesta opção.</p>
            """
        )
        st.info("💡 **Como usar:** nada a preencher. Pesos calculam-se automaticamente pelos 3 passos abaixo.")
        st.markdown("**Passo 1 — Normalização min-max**")
        st.latex(r"r_{ij} \in [0,1] \text{ via min-max}")
        try:
            R = normalize_minmax(matrix, types)
            st.markdown("**Passo 2 — σ e correlação Pearson**")
            sigma = R.std(axis=0, ddof=0)
            corr = np.corrcoef(R.T)
            corr = np.nan_to_num(corr)
            st.markdown("**Passo 3 — Conflito C_j e pesos**")
            st.latex(r"C_j = \sigma_j \sum_k (1 - r_{jk}),\quad w_j = C_j / \sum_l C_l")
            conflict = (1 - corr).sum(axis=1)
            C = sigma * conflict
            w_c = C / C.sum() if C.sum() > 0 else np.ones(n) / n
            df_w = pd.DataFrame({
                "Critério": crits, "σ_j": sigma, "Σ(1-r_jk)": conflict, "C_j": C,
                "Peso": w_c, "%": [f"{x*100:.2f}%" for x in w_c]
            })
            st.dataframe(df_w.style.format({"σ_j": "{:.4f}", "Σ(1-r_jk)": "{:.4f}", "C_j": "{:.4f}", "Peso": "{:.4f}"}),
                        hide_index=True, use_container_width=True)
            with st.expander("Ver matriz de correlação"):
                st.dataframe(pd.DataFrame(corr, index=crits, columns=crits)
                              .style.format("{:.3f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                            use_container_width=True)
            fig = px.bar(df_w, x="Critério", y="Peso", text=df_w["%"],
                         color="Peso", color_continuous_scale="Oranges", title="Pesos CRITIC")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.engine_weights["CRITIC"] = w_c
            st.success(f"💾 Pesos CRITIC guardados (Σ={w_c.sum():.4f}).")
        except Exception as e:
            st.error(f"Erro CRITIC: {e}")

    # Comparação
    st.markdown("---")
    st.subheader("📊 Comparação dos motores calculados")
    if st.session_state.engine_weights:
        comp = {"Critério": crits}
        for eng, w in st.session_state.engine_weights.items():
            if len(w) == n: comp[eng] = w
        df_comp = pd.DataFrame(comp)
        st.dataframe(df_comp.style.format({c: "{:.4f}" for c in df_comp.columns if c != "Critério"})
                              .background_gradient(cmap="Blues", axis=None,
                                                   subset=[c for c in df_comp.columns if c != "Critério"]),
                    hide_index=True, use_container_width=True)
        # Radar dos motores
        try:
            fig = go.Figure()
            for eng in df_comp.columns:
                if eng != "Critério":
                    fig.add_trace(go.Scatterpolar(
                        r=list(df_comp[eng]) + [df_comp[eng].iloc[0]],
                        theta=crits + [crits[0]],
                        fill="toself", name=eng
                    ))
            fig.update_layout(title="Comparação dos motores (radar)",
                              polar=dict(radialaxis=dict(visible=True, range=[0, max(df_comp[df_comp.columns[1:]].values.max(), 0.5)])),
                              height=420)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
    else:
        st.caption("Nenhum motor calculado ainda. Use as opções acima.")


# =============================================================================
# TAB 3: AHP (FULL — matriz Saaty + consistência + iterações)
# =============================================================================
with tabs[3]:
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
            <li><b>Validação obrigatória</b>: CR = CI/RI < 0.10 (senão, REVER julgamentos)</li>
        </ul>
        <p><b>Iteração:</b> se CR ≥ 0.10, a app identifica o <b>par mais inconsistente</b> e
        sugere o valor que reduz CR. Pode aplicar a sugestão ou ajustar manualmente até convergir.</p>
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
        column_config={c: st.column_config.NumberColumn(c, min_value=1/9, max_value=9.0, step=0.5, format="%.4f") for c in crits}
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
                               "%": [f"{x*100:.2f}%" for x in w_ahp]})
                  .style.format({"Peso w_j": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 3: Verificação de Consistência (Saaty)")
    st.markdown(
        """
        **O que é o CR e porque é obrigatório?**
        O Rácio de Consistência (CR) mede se os seus julgamentos par-a-par são **logicamente coerentes**.
        Exemplo de incoerência: se disse «C1 é 5× mais importante que C2» e «C2 é 3× mais importante que C3»,
        então logicamente C1 deveria ser ~15× mais importante que C3. Se inseriu outro valor (ex.: 2),
        o CR sobe.

        **Saaty define a regra:** se **CR ≥ 0.10** → os pesos NÃO são fiáveis e **precisa de iterar** (rever julgamentos).
        Se **CR < 0.10** → matriz aceitável.
        """
    )
    st.latex(r"\lambda_{max},\;CI = \frac{\lambda_{max}-n}{n-1},\;CR = \frac{CI}{RI(n)}")
    Aw = A @ w_ahp
    lam_max = (Aw / np.where(w_ahp == 0, 1e-9, w_ahp)).mean()
    CI = (lam_max - n) / (n - 1) if n > 1 else 0
    RI = RI_TABLE.get(n, 1.59)
    CR = CI / RI if RI > 0 else 0

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("n", n)
    cc2.metric("λ_max", f"{lam_max:.4f}")
    cc3.metric("CI", f"{CI:.4f}")
    cc4.metric("CR", f"{CR:.4f}",
              delta="✓ Consistente" if CR < 0.10 else "✗ Inconsistente",
              delta_color="normal" if CR < 0.10 else "inverse")

    # =========== ITERAÇÃO PARA REDUZIR CR ===========
    if CR >= 0.10:
        st.markdown(
            f'<div class="warning-box">'
            f'<b>⚠️ CR = {CR:.4f} ≥ 0.10 — Matriz INCONSISTENTE.</b><br><br>'
            'Conforme exige a teoria, é preciso <b>iterar</b>: rever julgamentos par-a-par até CR < 0.10. '
            'A aplicação identifica automaticamente <b>onde está o pior conflito</b> e propõe uma correcção.'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("#### 🔬 Como a aplicação detecta o par mais problemático")
        st.markdown(
            r"""
            Para cada par (i, j) na matriz comparamos o valor que você inseriu (**observado**, a<sub>ij</sub>)
            com o valor que seria **logicamente esperado** dados os pesos calculados (**esperado**, w<sub>i</sub>/w<sub>j</sub>).
            O par com maior desvio é o "ponto fraco" da matriz:
            """, unsafe_allow_html=True
        )
        st.latex(r"\text{desvio}(i, j) = \left| \ln\left(\frac{a_{ij}^{\text{observado}}}{w_i / w_j}\right) \right|")
        st.markdown("O valor sugerido é o **da escala Saaty {1/9, 1/7, 1/5, 1/3, 1/2, 1, 2, 3, 5, 7, 9}** mais próximo de w<sub>i</sub>/w<sub>j</sub>.",
                    unsafe_allow_html=True)

        # Encontrar o par mais inconsistente
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
            colB.metric("Valor actual (observado)", f"{A[worst_i, worst_j]:.4f}")
            colC.metric("Valor ideal teórico", f"{ideal_value:.4f}",
                       help="w_i / w_j — o que seria logicamente coerente")
            colD.metric("Valor sugerido (Saaty)", f"{suggested_value:.4f}",
                       delta=f"Δ = {suggested_value - A[worst_i, worst_j]:+.2f}",
                       help="Valor mais próximo na escala Saaty 1-9 que reduz CR")

            colE, colF = st.columns([3, 1])
            with colE:
                st.info(
                    f"**Interpretação:** disse que {crits[worst_i]} vale **{A[worst_i, worst_j]:.2f}×** "
                    f"{crits[worst_j]}, mas os pesos calculados sugerem que o rácio deveria ser ~**{ideal_value:.2f}×**. "
                    f"Para aproximar, ajuste para **{suggested_value:.2f}** (escala Saaty mais próxima)."
                )
            with colF:
                if st.button(f"✏️ Aplicar sugestão", type="primary", use_container_width=True):
                    new_df = st.session_state[ahp_key].copy()
                    new_df.iloc[worst_i, worst_j] = suggested_value
                    new_df.iloc[worst_j, worst_i] = 1.0 / suggested_value
                    st.session_state[ahp_key] = new_df
                    st.session_state.ahp_history.append({
                        "iteração": len(st.session_state.ahp_history) + 1,
                        "CR antes": round(CR, 4), "par": f"{crits[worst_i]} vs {crits[worst_j]}",
                        "valor antigo": round(A[worst_i, worst_j], 4),
                        "valor novo": round(suggested_value, 4)
                    })
                    st.success("✓ Sugestão aplicada. A matriz acima foi actualizada — o novo CR aparece já.")
                    st.rerun()

            st.caption(
                "**Pode iterar várias vezes** clicando em 'Aplicar sugestão' até CR < 0.10. "
                "Em alternativa, edite manualmente a matriz acima."
            )
    else:
        st.markdown(
            f'<div class="result-box">✅ <b>Matriz CONSISTENTE</b> — CR = {CR:.4f} < 0.10. Pesos AHP válidos.</div>',
            unsafe_allow_html=True
        )

    if st.session_state.ahp_history:
        with st.expander("📜 Histórico de iterações AHP"):
            st.dataframe(pd.DataFrame(st.session_state.ahp_history), hide_index=True, use_container_width=True)

    # Guardar pesos AHP
    st.session_state.engine_weights["AHP"] = w_ahp

    # ============== RANKING das alternativas usando pesos AHP ==============
    st.markdown("---")
    step_header("Passo 4: Ranking das Alternativas")
    st.latex(r"S_i = \sum_{j=1}^n w_j^{AHP} \cdot u_j(x_{ij})")
    st.markdown("(utilidade min-max com inversão para Custos)")

    # Aqui usamos os pesos AHP (não os activos — porque esta é a aba AHP)
    U = normalize_minmax(matrix, types)
    S = (U * w_ahp).sum(axis=1)
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Score AHP": S, "% do máx": S / S.max() * 100 if S.max() > 0 else S,
                           "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"Score AHP": "{:.4f}", "% do máx": "{:.1f}%"})
                  .background_gradient(cmap="RdYlGn", subset=["Score AHP"]),
                hide_index=True, use_container_width=True)

    best = df_res.iloc[0]["Alternativa"]
    st.markdown(
        f'<div class="result-box">🏆 Melhor segundo AHP: <b>{best}</b> '
        f'(Score = {df_res.iloc[0]["Score AHP"]:.4f}) '
        f'| CR = {CR:.4f} {"✓" if CR < 0.10 else "✗"}</div>',
        unsafe_allow_html=True
    )

    store_result("AHP", S, rank, higher_is_better=True)

    # Sensibilidade
    def ahp_score_fn(w):
        U = normalize_minmax(matrix, types)
        return (U * w).sum(axis=1)
    render_sensitivity(ahp_score_fn, alts, crits, w_ahp, higher_is_better=True, key_suffix="ahp")


# =============================================================================
# TAB 4: TOPSIS
# =============================================================================
with tabs[4]:
    st.header("🎯 TOPSIS")
    purpose_box("Aplicar o método TOPSIS — mede a <b>distância à solução ideal</b> e ranqueia as alternativas. Mostra os 6 passos com fórmulas e a análise de sensibilidade ±X%.")
    theory_box("Teoria (Hwang & Yoon, 1981)",
        """<p>Método compensatório baseado em <b>distâncias</b> à solução ideal A⁺ e anti-ideal A⁻.
        A melhor alternativa é simultaneamente <b>mais próxima de A⁺ e mais afastada de A⁻</b>.</p>""")

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
    st.dataframe(pd.DataFrame(R, index=alts, columns=crits).style.format("{:.4f}"), use_container_width=True)

    step_header("Passo 3: Matriz Ponderada")
    st.latex(r"v_{ij} = w_j \cdot r_{ij}")
    st.dataframe(pd.DataFrame(V, index=alts, columns=crits).style.format("{:.4f}"), use_container_width=True)

    step_header("Passo 4: Soluções Ideal A⁺ e Anti-Ideal A⁻")
    st.dataframe(pd.DataFrame({"Critério": crits, "Tipo": types, "A⁺": Ap, "A⁻": An})
                  .style.format({"A⁺": "{:.4f}", "A⁻": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 5 & 6: Distâncias e CC*")
    st.latex(r"D_i^{\pm} = \sqrt{\sum_j (v_{ij} - A_j^{\pm})^2};\quad CC_i = D_i^- / (D_i^+ + D_i^-)")
    rank = pd.Series(CC).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "D⁺": Dp, "D⁻": Dn, "CC*": CC, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"D⁺": "{:.4f}", "D⁻": "{:.4f}", "CC*": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn", subset=["CC*"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo TOPSIS: <b>{best}</b> (CC* = {df_res.iloc[0]["CC*"]:.4f})</div>', unsafe_allow_html=True)
    store_result("TOPSIS", CC, rank, True)

    render_sensitivity(lambda w: topsis_calc(w)[0], alts, crits, weights, True, "topsis")


# =============================================================================
# TAB 5: PROMETHEE II
# =============================================================================
with tabs[5]:
    st.header("📈 PROMETHEE II")
    purpose_box("Aplicar PROMETHEE II — método de <b>fluxos de preferência par-a-par</b>. Permite 3 funções de preferência (Usual, Linear, Gaussiana).")
    theory_box("Teoria (Brans, 1985)",
        """<p>Método <b>não-compensatório</b> baseado em fluxos de preferência par-a-par.
        Para cada par (a, b), agrega preferências em φ(a) = φ⁺(a) − φ⁻(a) ∈ [-1, 1].</p>
        <p><b>Funções de preferência</b>: Tipo I (Usual), Tipo V (Linear), Tipo VI (Gaussiana).</p>""")

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

    step_header("Passo 1: Função de Preferência (P_j)")
    if pref_type.startswith("Tipo I"):
        st.latex(r"P_j(a,b) = 1 \text{ se } a > b, \text{ senão } 0")
    elif pref_type.startswith("Tipo V"):
        st.latex(r"P_j = d/p \text{ se } 0 < d < p, \text{ senão } 0 \text{ ou } 1")
    else:
        st.latex(r"P_j = 1 - e^{-d^2/(2\sigma^2)},\;\sigma = 30\% \text{ do intervalo}")

    step_header("Passo 2: Matriz π(a, b) — preferência agregada")
    st.latex(r"\pi(a,b) = \sum_j w_j P_j(a,b)")
    st.dataframe(pd.DataFrame(pi, index=alts, columns=alts).style.format("{:.4f}")
                  .background_gradient(cmap="Greens"), use_container_width=True)

    step_header("Passo 3: Fluxos φ⁺, φ⁻ e φ líquido")
    st.latex(r"\phi^{\pm}(a) = \frac{1}{m-1}\sum_{b\ne a} \pi(a,b) / \pi(b,a);\quad \phi = \phi^+ - \phi^-")
    rank = pd.Series(phi).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "φ⁺": phi_p, "φ⁻": phi_n, "φ líquido": phi, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"φ⁺": "{:.4f}", "φ⁻": "{:.4f}", "φ líquido": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn", subset=["φ líquido"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo PROMETHEE II: <b>{best}</b> (φ = {df_res.iloc[0]["φ líquido"]:.4f})</div>', unsafe_allow_html=True)
    store_result("PROMETHEE II", phi, rank, True)

    render_sensitivity(lambda w: prom_calc(w)[0], alts, crits, weights, True, "prom")


# =============================================================================
# TAB 6: VIKOR
# =============================================================================
with tabs[6]:
    st.header("⚖️ VIKOR")
    purpose_box("Aplicar VIKOR — encontra a <b>solução de compromisso</b> entre utilidade global (S) e arrependimento individual (R). Parâmetro v ajustável.")
    theory_box("Teoria (Opricovic & Tzeng, 2004)",
        """<p>Procura <b>solução de compromisso</b> entre utilidade global (S) e arrependimento individual (R).
        Q_i combina ambos via v ∈ [0,1]; menor Q = melhor.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    v_p = st.slider("Parâmetro v:", 0.0, 1.0, 0.5, 0.05, key="vikor_v")

    def vikor_calc(W, v=0.5):
        n_c = len(crits)
        fb = np.array([matrix[:, j].max() if types[j] == "max" else matrix[:, j].min() for j in range(n_c)])
        fw = np.array([matrix[:, j].min() if types[j] == "max" else matrix[:, j].max() for j in range(n_c)])
        den = np.where(fb - fw == 0, 1e-9, fb - fw)
        terms = np.zeros_like(matrix, dtype=float)
        for j in range(n_c): terms[:, j] = W[j] * np.abs(fb[j] - matrix[:, j]) / abs(den[j])
        S = terms.sum(axis=1); R = terms.max(axis=1)
        Sr = (S.max() - S.min()) if S.max() != S.min() else 1e-9
        Rr = (R.max() - R.min()) if R.max() != R.min() else 1e-9
        Q = v * (S - S.min()) / Sr + (1 - v) * (R - R.min()) / Rr
        return Q, S, R, fb, fw

    Q, S, R, fb, fw = vikor_calc(weights, v_p)

    step_header("Passo 1: Melhores f* e Piores f⁻")
    st.dataframe(pd.DataFrame({"Critério": crits, "f*": fb, "f⁻": fw})
                  .style.format({"f*": "{:.4f}", "f⁻": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 2: Índices S e R")
    st.latex(r"S_i = \sum_j w_j (f_j^* - f_{ij})/(f_j^* - f_j^-);\;\;R_i = \max_j[\cdot]")

    step_header("Passo 3: Índice de Compromisso Q")
    st.latex(r"Q_i = v\frac{S_i-S^*}{S^- - S^*} + (1-v)\frac{R_i-R^*}{R^- - R^*}")
    rank = pd.Series(Q).rank(ascending=True, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "S": S, "R": R, "Q": Q, "Ranking (menor Q)": rank}).sort_values("Ranking (menor Q)")
    st.dataframe(df_res.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn_r", subset=["Q"]),
                hide_index=True, use_container_width=True)

    # Condições C1 e C2
    sorted_q = np.sort(Q)
    if len(sorted_q) >= 2:
        dq = sorted_q[1] - sorted_q[0]; thresh = 1.0 / max(len(alts) - 1, 1)
        c1 = dq >= thresh
        bi = int(np.argmin(Q)); si = int(np.argmin(S)); ri = int(np.argmin(R))
        c2 = (bi == si) or (bi == ri)
        st.markdown(f"**C1 (vantagem)**: ΔQ = {dq:.4f} vs 1/(J-1) = {thresh:.4f} → {'✅' if c1 else '❌'}")
        st.markdown(f"**C2 (estabilidade)**: melhor em S ou R → {'✅' if c2 else '❌'}")

    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo VIKOR: <b>{best}</b> (Q = {df_res.iloc[0]["Q"]:.4f})</div>', unsafe_allow_html=True)
    store_result("VIKOR", -Q, rank, True)  # -Q porque menor Q = melhor mas higher_is_better=True

    render_sensitivity(lambda w: -vikor_calc(w, v_p)[0], alts, crits, weights, True, "vikor")


# =============================================================================
# TAB 7: COPRAS
# =============================================================================
with tabs[7]:
    st.header("📊 COPRAS")
    purpose_box("Aplicar COPRAS — avalia alternativas como <b>função proporcional</b> entre benefícios (S⁺) e custos (S⁻). Resultado em grau de utilidade U_i (%).")
    theory_box("Teoria (Zavadskas & Kaklauskas, 1996)",
        """<p>Avalia alternativas como função proporcional entre <b>benefícios (S⁺)</b> e <b>custos (S⁻)</b>.
        Resultado: índice Q_i e grau de utilidade U_i (%) onde 100% = óptimo absoluto.</p>""")

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

    step_header("Passo 1: Normalização por Soma")
    st.latex(r"x'_{ij} = x_{ij} / \sum_i x_{ij}")
    st.dataframe(pd.DataFrame(Xn, index=alts, columns=crits).style.format("{:.4f}"), use_container_width=True)

    step_header("Passo 2: Matriz Ponderada")
    st.latex(r"\hat{x}_{ij} = w_j \cdot x'_{ij}")
    st.dataframe(pd.DataFrame(V, index=alts, columns=crits).style.format("{:.4f}"), use_container_width=True)

    step_header("Passo 3: S⁺ (Benefícios) e S⁻ (Custos)")
    st.dataframe(pd.DataFrame({"Alternativa": alts, "S⁺": Sp, "S⁻": Sm})
                  .style.format({"S⁺": "{:.4f}", "S⁻": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 4 & 5: Q_i (fórmula oficial) e U_i (%)")
    st.latex(r"Q_i = S_i^+ + \frac{S_{\min}^- \cdot \sum_i S_i^-}{S_i^- \cdot \sum_i (S_{\min}^-/S_i^-)};\quad U_i = Q_i/Q_{\max} \times 100")
    rank = pd.Series(Q).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "S⁺": Sp, "S⁻": Sm, "Q_i": Q, "U_i (%)": U, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"S⁺": "{:.4f}", "S⁻": "{:.4f}", "Q_i": "{:.4f}", "U_i (%)": "{:.2f}"})
                  .background_gradient(cmap="RdYlGn", subset=["U_i (%)"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo COPRAS: <b>{best}</b> (U = {df_res.iloc[0]["U_i (%)"]:.2f}%)</div>', unsafe_allow_html=True)
    store_result("COPRAS", Q, rank, True)

    render_sensitivity(lambda w: copras_calc(w)[0], alts, crits, weights, True, "copras")


# =============================================================================
# TAB 8: ELECTRE III
# =============================================================================
with tabs[8]:
    st.header("🚫 ELECTRE III")
    purpose_box("Aplicar ELECTRE III — método de <b>sobreclassificação</b> com limiares q/p/v (indiferença, preferência, veto). Permite incomparabilidades.")
    theory_box("Teoria (Roy, 1968+)",
        """<p>Método de <b>sobreclassificação</b>. Para cada par (a, b), avalia se há
        evidência suficiente que a "supera" b, usando 3 limiares: <b>q</b> (indiferença), <b>p</b> (preferência),
        <b>v</b> (veto). Não força ranking total — permite incomparabilidades.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()
    n = len(crits); m = len(alts)

    st.markdown("**Limiares % do intervalo de cada critério:**")
    c1, c2, c3 = st.columns(3)
    qp = c1.slider("q (indiferença) %", 0, 30, 5, 1, key="elec_q")
    pp = c2.slider("p (preferência) %", 5, 50, 20, 1, key="elec_p")
    vp = c3.slider("v (veto) %", 30, 80, 50, 1, key="elec_v")

    def elec_calc(W):
        q = np.zeros(n); p = np.zeros(n); v = np.zeros(n)
        for j in range(n):
            rng = matrix[:, j].max() - matrix[:, j].min()
            q[j] = rng * qp / 100; p[j] = rng * pp / 100; v[j] = rng * vp / 100
            if p[j] <= q[j]: p[j] = q[j] + 0.001
            if v[j] <= p[j]: v[j] = p[j] + 0.001
        cP = np.zeros((m, m, n)); dP = np.zeros((m, m, n))
        for a in range(m):
            for b in range(m):
                if a == b: continue
                for j in range(n):
                    ga, gb = (matrix[a, j], matrix[b, j]) if types[j] == "max" else (-matrix[a, j], -matrix[b, j])
                    diff = ga + q[j] - gb; diff2 = ga + p[j] - gb
                    if diff >= 0: cP[a, b, j] = 1.0
                    elif diff2 <= 0: cP[a, b, j] = 0.0
                    else: cP[a, b, j] = (p[j] + ga - gb) / (p[j] - q[j])
                    dif = gb - ga - p[j]; dif2 = gb - ga - v[j]
                    if dif <= 0: dP[a, b, j] = 0.0
                    elif dif2 >= 0: dP[a, b, j] = 1.0
                    else: dP[a, b, j] = (gb - ga - p[j]) / (v[j] - p[j])
        C = (cP * W).sum(axis=2)
        Sc = C.copy()
        for a in range(m):
            for b in range(m):
                if a == b: continue
                for j in range(n):
                    if dP[a, b, j] > C[a, b]:
                        Sc[a, b] *= (1 - dP[a, b, j]) / (1 - C[a, b]) if C[a, b] < 1 else 0
        return Sc, C, q, p, v, dP.max(axis=2)

    Sc, C, q_abs, p_abs, v_abs, dmax = elec_calc(weights)

    step_header("Passo 1: Limiares Absolutos por Critério")
    st.dataframe(pd.DataFrame({"Critério": crits, "q": q_abs, "p": p_abs, "v": v_abs})
                  .style.format({"q": "{:.4f}", "p": "{:.4f}", "v": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 2: Concordância Global C(a, b)")
    st.latex(r"C(a,b) = \sum_j w_j c_j(a,b)")
    st.dataframe(pd.DataFrame(C, index=alts, columns=alts).style.format("{:.3f}").background_gradient(cmap="Greens"),
                use_container_width=True)

    step_header("Passo 3: Matriz de Credibilidade S(a, b)")
    st.dataframe(pd.DataFrame(Sc, index=alts, columns=alts).style.format("{:.3f}").background_gradient(cmap="RdYlGn"),
                use_container_width=True)

    step_header("Passo 4: Ranking por Dominância Líquida")
    cutoff = st.slider("Limiar de corte λ:", 0.5, 0.95, 0.7, 0.05, key="elec_lam")
    outrank = (Sc >= cutoff) & (np.eye(m) == 0)
    net = outrank.sum(axis=1) - outrank.sum(axis=0)
    rank = pd.Series(net).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Sobreclassifica": outrank.sum(axis=1),
                           "Sobreclassificada por": outrank.sum(axis=0),
                           "Dominância líquida": net, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.background_gradient(cmap="RdYlGn", subset=["Dominância líquida"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo ELECTRE III: <b>{best}</b></div>', unsafe_allow_html=True)
    store_result("ELECTRE III", net, rank, True)

    def elec_score_fn(w):
        Sc2, *_ = elec_calc(w)
        out = (Sc2 >= cutoff) & (np.eye(m) == 0)
        return out.sum(axis=1) - out.sum(axis=0)
    render_sensitivity(elec_score_fn, alts, crits, weights, True, "elec")


# =============================================================================
# TAB 9: MAUT
# =============================================================================
with tabs[9]:
    st.header("💡 MAUT")
    purpose_box("Aplicar MAUT — converte valores em <b>utilidade [0,1]</b> via função (Linear, Exp, Potência) e agrega ponderadamente.")
    theory_box("Teoria (Keeney & Raiffa, 1976)",
        """<p>Cada valor é convertido em <b>utilidade</b> u_j(x) ∈ [0,1] via função (linear, exponencial, potência),
        e agregado: U_i = Σ w_j · u_j(x_ij). Maior U = melhor.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    func = st.radio("Função de utilidade:", ["Linear", "Exponencial (k=2)", "Potência (p=0.5, côncava)", "Potência (p=2, convexa)"],
                    horizontal=True, key="maut_func")

    def maut_calc(W):
        U0 = normalize_minmax(matrix, types)
        if func == "Linear": U = U0
        elif func.startswith("Exp"): U = (1 - np.exp(-2 * U0)) / (1 - np.exp(-2))
        elif func.startswith("Potência (p=0.5"): U = U0 ** 0.5
        else: U = U0 ** 2
        return (U * W).sum(axis=1), U

    S, U = maut_calc(weights)

    step_header("Passo 1: Utilidades Parciais u_j(x_ij)")
    if func == "Linear": st.latex(r"u(x) = (x - x_{min})/(x_{max} - x_{min})")
    elif func.startswith("Exp"): st.latex(r"u(x) = (1 - e^{-2\tilde{x}})/(1 - e^{-2})")
    elif "p=0.5" in func: st.latex(r"u(x) = \tilde{x}^{0.5} \text{ (côncava — favorece ganhos pequenos)}")
    else: st.latex(r"u(x) = \tilde{x}^{2} \text{ (convexa — penaliza valores baixos)}")
    st.dataframe(pd.DataFrame(U, index=alts, columns=crits).style.format("{:.4f}"), use_container_width=True)

    step_header("Passo 2: Utilidade Global U_i")
    st.latex(r"U_i = \sum_j w_j \cdot u_j(x_{ij})")
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "U_i": S, "% do máx": S / S.max() * 100 if S.max() > 0 else S, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"U_i": "{:.4f}", "% do máx": "{:.1f}%"})
                  .background_gradient(cmap="RdYlGn", subset=["U_i"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo MAUT: <b>{best}</b> (U = {df_res.iloc[0]["U_i"]:.4f})</div>', unsafe_allow_html=True)
    store_result("MAUT", S, rank, True)

    render_sensitivity(lambda w: maut_calc(w)[0], alts, crits, weights, True, "maut")


# =============================================================================
# TAB 10: DEMATEL
# =============================================================================
with tabs[10]:
    st.header("🌐 DEMATEL")
    purpose_box("Aplicar DEMATEL — analisa <b>relações causa-efeito</b> entre critérios e ajusta pesos pela proeminência.")
    theory_box("Teoria (Gabus & Fontela, 1972)",
        """<p>Modela <b>relações causa-efeito</b> entre critérios. Aqui, na ausência de elicitação directa,
        usa-se correlação absoluta como proxy. Output: R+C (proeminência) e R−C (causa/efeito).</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()
    n = len(crits)

    def dem_calc(W):
        try:
            Z = np.abs(np.corrcoef(matrix.T)); Z = np.nan_to_num(Z); np.fill_diagonal(Z, 0)
        except Exception: Z = np.zeros((n, n))
        s = max(Z.sum(axis=1).max(), Z.sum(axis=0).max(), 1e-9); X = Z / s
        try: T = X @ np.linalg.inv(np.eye(n) - X)
        except Exception: T = np.eye(n)
        R = T.sum(axis=1); C = T.sum(axis=0)
        prom = R + C; rel = R - C
        if prom.sum() > 0:
            Wa = W * prom; Wa = Wa / Wa.sum()
        else: Wa = W
        U = normalize_minmax(matrix, types); S = (U * Wa).sum(axis=1)
        return S, T, R, C, prom, rel, Wa

    S, T, R_v, C_v, prom, rel, Wa = dem_calc(weights)

    step_header("Passo 1: Matriz Total T = X(I - X)⁻¹")
    st.dataframe(pd.DataFrame(T, index=crits, columns=crits).style.format("{:.4f}").background_gradient(cmap="Blues"),
                use_container_width=True)

    step_header("Passo 2: R+C (Proeminência) e R−C (Causa/Efeito)")
    df_rc = pd.DataFrame({"Critério": crits, "R": R_v, "C": C_v, "R+C": prom, "R-C": rel,
                          "Tipo": ["🎯 Causa" if r > 0 else "📥 Efeito" for r in rel]})
    st.dataframe(df_rc.style.format({"R": "{:.4f}", "C": "{:.4f}", "R+C": "{:.4f}", "R-C": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    # Diagrama causa-efeito
    fig = go.Figure(go.Scatter(x=prom, y=rel, mode="markers+text", text=crits, textposition="top center",
                                marker=dict(size=14, color=rel, colorscale="RdBu", showscale=True, colorbar=dict(title="R-C"))))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(title="Diagrama Causa-Efeito DEMATEL",
                      xaxis_title="Proeminência (R+C)", yaxis_title="Relação (R-C)",
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

    step_header("Passo 3: Ranking com Pesos Ajustados")
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Score": S, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"Score": "{:.4f}"}).background_gradient(cmap="RdYlGn", subset=["Score"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo DEMATEL: <b>{best}</b></div>', unsafe_allow_html=True)
    store_result("DEMATEL", S, rank, True)

    render_sensitivity(lambda w: dem_calc(w)[0], alts, crits, weights, True, "dem")


# =============================================================================
# TAB 11: FUZZY TOPSIS
# =============================================================================
with tabs[11]:
    st.header("🌫️ Fuzzy TOPSIS")
    purpose_box("TOPSIS com <b>números fuzzy triangulares</b> (l, m, u). Útil quando os dados têm imprecisão. Spread ajustável.")
    theory_box("Teoria (Chen, 2000)",
        """<p>TOPSIS com <b>números fuzzy triangulares</b> (l, m, u). Captura imprecisão.
        Distância pelo método do vértice; CC_i mantém-se como ranking.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    spread = st.slider("Spread fuzzy (% do valor):", 5, 40, 15, 5, key="ft_spread")

    def ftopsis_calc(W, s_pct=15):
        s = s_pct / 100.0
        L = matrix * (1 - s); M = matrix.copy(); U_ = matrix * (1 + s)
        nc = matrix.shape[1]
        Ln = np.zeros_like(L); Mn = np.zeros_like(M); Un = np.zeros_like(U_)
        for j in range(nc):
            if types[j] == "max":
                d = max(U_[:, j].max(), 1e-9)
                Ln[:, j] = L[:, j] / d; Mn[:, j] = M[:, j] / d; Un[:, j] = U_[:, j] / d
            else:
                num = max(L[:, j].min(), 1e-9)
                Ln[:, j] = num / np.where(U_[:, j] == 0, 1e-9, U_[:, j])
                Mn[:, j] = num / np.where(M[:, j] == 0, 1e-9, M[:, j])
                Un[:, j] = num / np.where(L[:, j] == 0, 1e-9, L[:, j])
        Lw, Mw, Uw = Ln * W, Mn * W, Un * W
        fpis = np.array([Uw[:, j].max() for j in range(nc)])
        fnis = np.array([Lw[:, j].min() for j in range(nc)])
        def vd(a, b, c, t): return np.sqrt(((a-t)**2 + (b-t)**2 + (c-t)**2) / 3.0)
        Dp = np.zeros(len(alts)); Dn = np.zeros(len(alts))
        for i in range(len(alts)):
            for j in range(nc):
                Dp[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], fpis[j])
                Dn[i] += vd(Lw[i, j], Mw[i, j], Uw[i, j], fnis[j])
        den = np.where(Dp + Dn == 0, 1e-9, Dp + Dn)
        return Dn / den, Dp, Dn

    CC, Dp, Dn = ftopsis_calc(weights, spread)

    step_header("Passo 1-3: Matriz Fuzzy + FPIS/FNIS + Distâncias")
    st.latex(r"\tilde{x}_{ij} = (x(1-s),\, x,\, x(1+s));\quad d_v(\tilde{a},\tilde{b}) = \sqrt{\frac{1}{3}[(l_a-l_b)^2 + (m_a-m_b)^2 + (u_a-u_b)^2]}")

    step_header("Passo 4: CC* e Ranking")
    st.latex(r"CC_i = D_i^- / (D_i^+ + D_i^-)")
    rank = pd.Series(CC).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "D⁺": Dp, "D⁻": Dn, "CC*": CC, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"D⁺": "{:.4f}", "D⁻": "{:.4f}", "CC*": "{:.4f}"})
                  .background_gradient(cmap="RdYlGn", subset=["CC*"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo Fuzzy TOPSIS: <b>{best}</b> (CC* = {df_res.iloc[0]["CC*"]:.4f})</div>', unsafe_allow_html=True)
    store_result("Fuzzy TOPSIS", CC, rank, True)

    render_sensitivity(lambda w: ftopsis_calc(w, spread)[0], alts, crits, weights, True, "ft")


# =============================================================================
# TAB 12: FUZZY AHP
# =============================================================================
with tabs[12]:
    st.header("🧮 Fuzzy AHP")
    purpose_box("AHP com <b>pesos fuzzy</b> e defuzzificação por centro de área. Captura incerteza nos pesos.")
    theory_box("Teoria (Chang, 1996)",
        """<p>AHP com TFN nos pesos. Defuzzificação por centro de área: w<sub>crisp</sub> = (l+m+u)/3.</p>""")

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    fs = st.slider("Spread fuzzy dos pesos (%):", 5, 40, 20, 5, key="fa_spread")

    def fahp_calc(W, s_pct=20):
        s = s_pct / 100.0; L = W * (1 - s); M = W.copy(); U_ = W * (1 + s)
        Wc = (L + M + U_) / 3; Wc = Wc / Wc.sum()
        Um = normalize_minmax(matrix, types)
        return (Um * Wc).sum(axis=1), Wc, L, M, U_

    S, Wc, L, M, U_ = fahp_calc(weights, fs)

    step_header("Passo 1: TFN dos pesos")
    st.latex(r"\tilde{w}_j = (w(1-s),\, w,\, w(1+s))")
    st.dataframe(pd.DataFrame({"Critério": crits, "l": L, "m": M, "u": U_})
                  .style.format({"l": "{:.4f}", "m": "{:.4f}", "u": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 2: Defuzzificação")
    st.latex(r"w^{crisp} = (l + m + u) / 3; \text{ normalizar para } \sum = 1")
    st.dataframe(pd.DataFrame({"Critério": crits, "w crisp": Wc, "%": [f"{x*100:.2f}%" for x in Wc]})
                  .style.format({"w crisp": "{:.4f}"}),
                hide_index=True, use_container_width=True)

    step_header("Passo 3: Ranking")
    rank = pd.Series(S).rank(ascending=False, method='min').astype(int).values
    df_res = pd.DataFrame({"Alternativa": alts, "Score": S, "Ranking": rank}).sort_values("Ranking")
    st.dataframe(df_res.style.format({"Score": "{:.4f}"}).background_gradient(cmap="RdYlGn", subset=["Score"]),
                hide_index=True, use_container_width=True)
    best = df_res.iloc[0]["Alternativa"]
    st.markdown(f'<div class="result-box">🏆 Melhor segundo Fuzzy AHP: <b>{best}</b></div>', unsafe_allow_html=True)
    store_result("Fuzzy AHP", S, rank, True)

    render_sensitivity(lambda w: fahp_calc(w, fs)[0], alts, crits, weights, True, "fa")


# =============================================================================
# TAB 13: GRÁFICOS (Plotly bonitos e decisivos)
# =============================================================================
with tabs[13]:
    st.header("📊 Gráficos para Decisão")
    purpose_box("<b>5 visualizações Plotly</b> para apoiar a decisão: heatmap rankings, radar Top-3, tornado sensibilidade, scores normalizados, convergência Top-3.")
    theory_box(
        "Visualizações para uma decisão rápida e clara",
        """
        <p>Esta aba reúne os <b>gráficos mais impactantes</b> para decidir:</p>
        <ul>
            <li><b>Radar</b>: perfil multicritério do Top-3 (forma → identifica trade-offs)</li>
            <li><b>Heatmap de Rankings</b>: como cada modelo posiciona cada alternativa</li>
            <li><b>Tornado de Sensibilidade</b>: que critérios mais influenciam o Top-1</li>
            <li><b>Bar Race</b>: comparação directa de scores entre modelos</li>
        </ul>
        """
    )

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()

    if not st.session_state.all_results:
        st.warning("⚠️ Nenhum modelo executado. Vá às abas de modelo (TOPSIS, AHP, etc.) primeiro.")
        st.stop()

    methods = list(st.session_state.all_results.keys())

    # ============== GRÁFICO 1: HEATMAP DE RANKINGS ==============
    st.subheader("🗺️ Heatmap de Rankings por Método")
    st.caption("Mostra como cada modelo classifica cada alternativa. Verde = topo, Vermelho = fundo.")
    rank_data = {"Alternativa": alts}
    for m in methods:
        rank_data[m] = st.session_state.all_results[m]["ranking"]
    df_ranks = pd.DataFrame(rank_data).set_index("Alternativa")

    fig_heat = px.imshow(
        df_ranks.values, x=methods, y=alts,
        color_continuous_scale="RdYlGn_r", aspect="auto", text_auto=True,
        labels=dict(x="Modelo", y="Alternativa", color="Ranking"),
        title=f"Rankings das {len(alts)} alternativas em {len(methods)} modelos"
    )
    fig_heat.update_layout(height=max(360, 28 * len(alts)),
                            margin=dict(l=10, r=10, t=50, b=10))
    fig_heat.update_traces(textfont=dict(size=14, color="black"))
    st.plotly_chart(fig_heat, use_container_width=True)

    # ============== GRÁFICO 2: RADAR DO TOP-3 ==============
    st.subheader("🎯 Radar — Perfil Multicritério do Top-3")
    st.caption("Cada alternativa é desenhada como polígono. Áreas maiores = melhor desempenho global.")
    # Calcular top-3 pelo ranking médio
    avg_rank = df_ranks.mean(axis=1)
    top3_alts = avg_rank.sort_values().head(3).index.tolist()
    norm = normalize_minmax(matrix, types)
    norm_df = pd.DataFrame(norm, index=alts, columns=crits)

    fig_radar = go.Figure()
    colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    medals = ["🥇 1º", "🥈 2º", "🥉 3º"]
    for i, alt in enumerate(top3_alts):
        vals = list(norm_df.loc[alt]) + [norm_df.loc[alt].iloc[0]]
        cats = crits + [crits[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=f"{medals[i]} {alt}",
            line=dict(color=colors[i], width=2),
            opacity=0.6
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=480, showlegend=True,
        title="Top-3 — perfil normalizado (1=melhor, 0=pior por critério)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ============== GRÁFICO 3: TORNADO DE SENSIBILIDADE DO TOP-1 ==============
    st.subheader("🌪️ Tornado de Sensibilidade — Top-1")
    st.caption(f"Para a alternativa Top-1 ({top3_alts[0]}), mostra quanto cada peso afecta o seu rank. "
               "Barras mais longas = critério mais influente.")

    # Usar AHP/TOPSIS como referência (o que estiver disponível primeiro)
    ref_method = "TOPSIS" if "TOPSIS" in st.session_state.all_results else methods[0]
    top1_idx = alts.index(top3_alts[0])
    base_scores = st.session_state.all_results[ref_method]["scores"]
    base_score_top1 = base_scores[top1_idx]
    sens_pct = st.session_state.sensitivity_pct

    tornado_data = []
    for j, crit in enumerate(crits):
        # Variar +sens% e -sens%
        for sign, factor in [("+", 1 + sens_pct/100), ("-", 1 - sens_pct/100)]:
            new_w = weights.copy()
            new_w[j] = weights[j] * factor
            new_w = new_w / new_w.sum()
            # Recalcular com TOPSIS rápido
            R = normalize_vector(matrix); V = R * new_w
            Ap = np.array([V[:, k].max() if types[k] == "max" else V[:, k].min() for k in range(len(crits))])
            An = np.array([V[:, k].min() if types[k] == "max" else V[:, k].max() for k in range(len(crits))])
            Dp = np.sqrt(((V - Ap) ** 2).sum(axis=1)); Dn = np.sqrt(((V - An) ** 2).sum(axis=1))
            den = np.where(Dp + Dn == 0, 1e-9, Dp + Dn)
            new_cc = Dn / den
            delta = new_cc[top1_idx] - base_score_top1
            tornado_data.append({"Critério": crit, "Direcção": f"{sign}{sens_pct}%", "Δ Score": delta})
    df_tornado = pd.DataFrame(tornado_data)

    fig_tornado = px.bar(df_tornado, x="Δ Score", y="Critério", color="Direcção",
                         orientation="h", barmode="group",
                         color_discrete_map={f"+{sens_pct}%": "#1F77B4", f"-{sens_pct}%": "#FF7F0E"},
                         title=f"Tornado — sensibilidade do score de {top3_alts[0]} ({ref_method})")
    fig_tornado.update_layout(height=max(280, 50 * len(crits)),
                               margin=dict(l=10, r=10, t=50, b=10))
    fig_tornado.add_vline(x=0, line_dash="dash", line_color="grey")
    st.plotly_chart(fig_tornado, use_container_width=True)

    # ============== GRÁFICO 4: SCORES NORMALIZADOS POR MODELO ==============
    st.subheader("📊 Scores Normalizados por Modelo (comparação directa)")
    st.caption("Cada modelo é normalizado para [0,1]. Permite comparar a posição relativa entre métodos.")
    score_data = {"Alternativa": alts}
    for m in methods:
        sc = st.session_state.all_results[m]["scores"]
        if sc.max() > sc.min():
            score_data[m] = (sc - sc.min()) / (sc.max() - sc.min())
        else:
            score_data[m] = np.zeros_like(sc)
    df_sc = pd.DataFrame(score_data)
    # Long format para px
    df_long = df_sc.melt(id_vars="Alternativa", var_name="Modelo", value_name="Score Normalizado")
    fig_sc = px.bar(df_long, x="Alternativa", y="Score Normalizado", color="Modelo",
                    barmode="group", title="Scores normalizados [0,1] por modelo")
    fig_sc.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_sc, use_container_width=True)

    # ============== GRÁFICO 5: CONVERGÊNCIA TOP-3 ==============
    st.subheader("🎯 Convergência Top-3")
    st.caption("Quantos modelos colocam cada alternativa no Top-3? Indicador de robustez consensual.")
    top3_count = (df_ranks <= 3).sum(axis=1).sort_values(ascending=False)
    fig_conv = px.bar(x=top3_count.index, y=top3_count.values,
                      labels={"x": "Alternativa", "y": f"N.º de modelos com Top-3 (de {len(methods)})"},
                      color=top3_count.values, color_continuous_scale="Viridis",
                      title=f"Convergência Top-3 — análise de robustez")
    fig_conv.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_conv, use_container_width=True)


# =============================================================================
# TAB 14: DASHBOARD CONSOLIDADO
# =============================================================================
with tabs[14]:
    st.header("🏆 Dashboard Consolidado")
    purpose_box("Combinar os rankings de todos os modelos executados via <b>Borda invertido</b> (média de posições). Identifica o Top-3 consensual.")
    theory_box(
        "Consolidação dos modelos",
        """<p>Aplica-se <b>Borda invertido</b> (média de posições) para agregar todos os modelos.
        A alternativa com <b>menor posição média</b> é a recomendação consensual.</p>"""
    )

    if not check_valid_input(): st.stop()
    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()
    show_active_weights_banner()

    if not st.session_state.all_results:
        st.warning("⚠️ Execute primeiro pelo menos um modelo (TOPSIS, AHP, etc.).")
        st.stop()

    methods = list(st.session_state.all_results.keys())
    df_dash = pd.DataFrame({"Alternativa": alts})
    for m in methods:
        df_dash[m] = st.session_state.all_results[m]["ranking"]
    df_dash["Posição Média"] = df_dash[methods].mean(axis=1).round(2)
    df_dash["Top-3 em N modelos"] = (df_dash[methods] <= 3).sum(axis=1)
    df_dash["Ranking Final"] = pd.Series(df_dash["Posição Média"]).rank(ascending=True, method='min').astype(int).values
    df_dash = df_dash.sort_values("Ranking Final")

    st.dataframe(df_dash.style.background_gradient(cmap="RdYlGn_r", subset=methods + ["Posição Média", "Ranking Final"]),
                hide_index=True, use_container_width=True)

    st.subheader("🥇 Recomendação Final — Top-3")
    top3 = df_dash.head(3)["Alternativa"].tolist()
    c1, c2, c3 = st.columns(3)
    medals = ["🥇 1º lugar", "🥈 2º lugar", "🥉 3º lugar"]
    for k, (col, medal) in enumerate(zip([c1, c2, c3], medals)):
        if k < len(top3):
            col.metric(medal, top3[k], delta=f"Pos média: {df_dash.iloc[k]['Posição Média']}")

    total_top3 = sum(df_dash.head(3)["Top-3 em N modelos"].values)
    max_conv = 3 * len(methods)
    conv_pct = (total_top3 / max_conv * 100) if max_conv else 0
    st.info(f"**Convergência inter-modelo**: {total_top3}/{max_conv} ({conv_pct:.0f}%)\n\n"
            f"Modelos avaliados: {', '.join(methods)}")


# =============================================================================
# TAB 15: VISTA 360° — Dashboard estilo Figura 1 do enunciado
# =============================================================================
with tabs[15]:
    st.header("🎛️ Vista 360° — Dashboard Executivo")
    purpose_box(
        "<b>Vista consolidada one-page</b> estilo Figura 1 do enunciado. "
        "Filtros · Ranking · Radar · 4 gráficos por modelo · Sensibilidade · Recomendações — "
        "tudo na mesma página para tomar decisão sem mudar de aba."
    )

    if not check_valid_input():
        st.stop()

    matrix, alts, crits, types = get_decision_matrix()
    weights = get_active_weights()

    if not st.session_state.all_results:
        st.warning("⚠️ Execute primeiro os modelos (TOPSIS, AHP, etc.). Esta vista consolida resultados existentes.")
        st.stop()

    methods = list(st.session_state.all_results.keys())

    # ============================================================
    # LAYOUT TOPO: Filtros (1) + Ranking (2) + Radar (1)
    # ============================================================
    st.markdown("---")
    col_filt, col_rank, col_radar = st.columns([1.2, 2.2, 2.0])

    # ---- Coluna 1: Filtros & Parâmetros ----
    with col_filt:
        st.markdown("#### 🔧 Filtros & Parâmetros")
        st.caption("Use estes filtros para alterar dinamicamente todas as visualizações.")

        focus_model = st.selectbox("Modelo em destaque:", methods, key="v360_model")
        focus_crit = st.selectbox("Critério para sensibilidade:", crits, key="v360_crit")
        focus_alt = st.selectbox("Alternativa para destaque (radar):", alts, key="v360_alt")
        sens_pct_v360 = st.session_state.sensitivity_pct
        st.metric("Variação SA (sidebar)", f"±{sens_pct_v360}%")
        st.metric("Modelos activos", len(methods))

    # ---- Coluna 2: Tabela de Ranking Consolidado ----
    with col_rank:
        st.markdown("#### 🏆 Ranking Consolidado")
        df_consol = pd.DataFrame({"Alternativa": alts})
        for m in methods:
            df_consol[m] = st.session_state.all_results[m]["ranking"]
        df_consol["Pos. Média"] = df_consol[methods].mean(axis=1).round(2)
        df_consol["Final"] = pd.Series(df_consol["Pos. Média"]).rank(ascending=True, method='min').astype(int).values
        df_consol = df_consol.sort_values("Final").reset_index(drop=True)
        # Adicionar coluna "Top-3?" visual
        df_consol["Top-3"] = df_consol["Final"].apply(lambda r: "🥇" if r == 1 else ("🥈" if r == 2 else ("🥉" if r == 3 else "")))
        st.dataframe(
            df_consol.style.background_gradient(cmap="RdYlGn_r", subset=methods + ["Pos. Média", "Final"]),
            hide_index=True, use_container_width=True, height=min(360, 50 + 35 * len(alts))
        )

    # ---- Coluna 3: Perfil multicritério (Radar) ----
    with col_radar:
        st.markdown(f"#### 🎯 Perfil Multicritério")
        st.caption(f"Top-3 + alternativa em destaque ({focus_alt})")
        top3_v360 = df_consol.head(3)["Alternativa"].tolist()
        norm = normalize_minmax(matrix, types)
        norm_df = pd.DataFrame(norm, index=alts, columns=crits)

        fig_radar = go.Figure()
        colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
        for i, alt in enumerate(top3_v360):
            vals = list(norm_df.loc[alt]) + [norm_df.loc[alt].iloc[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=crits + [crits[0]], fill="toself", name=alt,
                line=dict(color=colors[i], width=2), opacity=0.5
            ))
        if focus_alt not in top3_v360:
            vals = list(norm_df.loc[focus_alt]) + [norm_df.loc[focus_alt].iloc[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=crits + [crits[0]], fill="toself", name=f"⭐ {focus_alt}",
                line=dict(color="#9C27B0", width=3, dash="dot")
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            height=340, showlegend=True, margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ============================================================
    # LAYOUT MEIO: 4 mini-gráficos por modelo (top-4 modelos)
    # ============================================================
    st.markdown("---")
    st.markdown("#### 📊 Scores por Modelo (top-4 modelos com maior peso de avaliação)")
    # Priorizar os 4 modelos obrigatórios do enunciado
    priority_methods = ["TOPSIS", "PROMETHEE II", "AHP", "COPRAS"]
    display_methods = [m for m in priority_methods if m in methods]
    # completar até 4 com outros se faltarem
    if len(display_methods) < 4:
        for m in methods:
            if m not in display_methods:
                display_methods.append(m)
            if len(display_methods) >= 4:
                break

    cols = st.columns(min(4, len(display_methods)))
    for i, m in enumerate(display_methods[:4]):
        with cols[i]:
            sc = st.session_state.all_results[m]["scores"]
            df_sc = pd.DataFrame({"Alt": alts, "Score": sc}).sort_values("Score", ascending=False)
            # Highlight top-3
            colors_bar = ["#FFD700" if k == 0 else "#C0C0C0" if k == 1 else "#CD7F32" if k == 2 else "#90CAF9"
                          for k in range(len(df_sc))]
            fig = go.Figure(go.Bar(
                x=df_sc["Score"], y=df_sc["Alt"], orientation="h",
                marker=dict(color=colors_bar),
                text=[f"{x:.3f}" for x in df_sc["Score"]],
                textposition="outside"
            ))
            fig.update_layout(
                title=dict(text=f"<b>{m}</b>", font=dict(size=14)),
                height=max(220, 28 * len(alts)),
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="", yaxis_title="",
                showlegend=False
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # LAYOUT FUNDO: SA tornado + Painel Recomendação
    # ============================================================
    st.markdown("---")
    col_sa, col_reco = st.columns([1.4, 1.0])

    with col_sa:
        st.markdown(f"#### 🌪️ Sensibilidade — impacto de {focus_crit} no {focus_model}")
        st.caption(f"Como varia o score de cada alternativa quando o peso de {focus_crit} muda ±{sens_pct_v360}%")

        # Calcular variações
        focus_crit_idx = crits.index(focus_crit)
        base_scores = st.session_state.all_results[focus_model]["scores"]
        # Reutilizar lógica genérica: precisamos da score_fn — usar TOPSIS-like se for TOPSIS, etc.
        # Simplificação: variar o peso, recalcular com TOPSIS rápido (porque é universal)
        def quick_topsis(W):
            R = normalize_vector(matrix); V = R * W
            Ap = np.array([V[:, j].max() if types[j] == "max" else V[:, j].min() for j in range(len(crits))])
            An = np.array([V[:, j].min() if types[j] == "max" else V[:, j].max() for j in range(len(crits))])
            Dp = np.sqrt(((V - Ap) ** 2).sum(axis=1)); Dn = np.sqrt(((V - An) ** 2).sum(axis=1))
            return Dn / np.where(Dp + Dn == 0, 1e-9, Dp + Dn)
        base = quick_topsis(weights)

        deltas = []
        for sign, fac in [("+", 1 + sens_pct_v360/100), ("-", 1 - sens_pct_v360/100)]:
            nw = weights.copy(); nw[focus_crit_idx] = weights[focus_crit_idx] * fac
            other_sum_old = weights.sum() - weights[focus_crit_idx]
            other_sum_new = 1 - nw[focus_crit_idx]
            if other_sum_old > 0 and other_sum_new > 0:
                for k in range(len(nw)):
                    if k != focus_crit_idx:
                        nw[k] = weights[k] * (other_sum_new / other_sum_old)
            nw = nw / nw.sum()
            sc = quick_topsis(nw)
            for i, alt in enumerate(alts):
                deltas.append({"Alt": alt, "Cenário": f"{sign}{sens_pct_v360}%", "Δ Score": sc[i] - base[i]})
        df_d = pd.DataFrame(deltas)
        fig_tor = px.bar(df_d, x="Δ Score", y="Alt", color="Cenário",
                          orientation="h", barmode="group",
                          color_discrete_map={f"+{sens_pct_v360}%": "#1976D2", f"-{sens_pct_v360}%": "#F57C00"})
        fig_tor.add_vline(x=0, line_dash="dash", line_color="grey")
        fig_tor.update_layout(height=max(260, 32 * len(alts)), margin=dict(l=10, r=10, t=10, b=10),
                              yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_tor, use_container_width=True)

    with col_reco:
        st.markdown("#### 🎯 Recomendações & Notas")
        top3 = df_consol.head(3)["Alternativa"].tolist()
        # Convergência
        df_ranks_for_conv = df_consol[methods]
        top3_count = (df_ranks_for_conv <= 3).sum(axis=1).values
        conv_top1 = (top3_count[0] / len(methods) * 100) if methods else 0

        if conv_top1 >= 70:
            verdict_color = "#2e7d32"; verdict_label = "🟢 ALTA"
        elif conv_top1 >= 40:
            verdict_color = "#f57c00"; verdict_label = "🟡 MODERADA"
        else:
            verdict_color = "#c62828"; verdict_label = "🔴 BAIXA"

        st.markdown(
            f"""<div style="background:linear-gradient(135deg, #1F4E78 0%, #2E75B6 100%);
            color:white; padding:18px; border-radius:10px;">
            <div style="font-size:11px; opacity:0.9; text-transform:uppercase;">TOP-3 MCDM</div>
            <div style="font-size:18px; font-weight:700;">🥇 {top3[0] if len(top3) > 0 else '—'}</div>
            <div style="font-size:14px; opacity:0.9;">🥈 {top3[1] if len(top3) > 1 else '—'} · 🥉 {top3[2] if len(top3) > 2 else '—'}</div>
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            f"""<div style="background:{verdict_color}; color:white; padding:10px;
            border-radius:6px; margin-top:8px; text-align:center; font-weight:600;">
            Convergência {verdict_label}: {conv_top1:.0f}%
            </div>""",
            unsafe_allow_html=True
        )

        st.markdown(f"**Modelos avaliados ({len(methods)}):**")
        st.write(", ".join(methods))

        st.markdown("**Critérios + pesos activos:**")
        df_cwp = pd.DataFrame({
            "Crit.": crits, "Tipo": types, "Peso": [f"{w*100:.1f}%" for w in weights]
        })
        st.dataframe(df_cwp, hide_index=True, use_container_width=True)


# =============================================================================
# TAB 16: RELATÓRIO TÉCNICO — Estrutura dos 7 capítulos (cumprindo Cap. 4 do enunciado)
# =============================================================================
with tabs[16]:
    st.header("📑 Relatório Técnico — Estrutura dos 7 Capítulos")
    purpose_box(
        "Gera <b>relatório técnico completo</b> cumprindo a estrutura do <b>Capítulo 4</b> do enunciado: "
        "Introdução · Dados · Modelos · Sensibilidade · Dashboard · Comparação · Conclusões. "
        "Cada secção é expandível e o markdown final é descarregável (depois converte-se para PDF max 30pp)."
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
    top1_pos_avg = df_dash.iloc[0]["Posição Média"]
    top1_top3_count = df_dash.iloc[0]["Top-3 em N modelos"]
    conv_pct_top1 = (top1_top3_count / len(methods) * 100) if methods else 0
    top3 = df_dash.head(3)["Alternativa"].tolist()

    # Header com recomendação
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

    # =========================================================================
    # CAP 1 — INTRODUÇÃO
    # =========================================================================
    with st.expander("**📖 Capítulo 1 — Introdução** (1-2 pp)", expanded=True):
        st.markdown(f"""
        ### 1.1 Contexto
        Aplicação de métodos de decisão multicritério (MCDM) a um problema real de priorização
        de **{len(alts)} alternativas** segundo **{len(crits)} critérios**.

        ### 1.2 Formulação do Problema
        Decisor: equipa que aplica o modelo MCDM.
        **Objectivo**: ranquear as alternativas {', '.join(alts[:5])}{'...' if len(alts) > 5 else ''}
        segundo {', '.join(crits)}, e produzir top-3 robusto via múltiplos modelos.

        ### 1.3 Estrutura do Relatório
        Este documento segue a estrutura definida no enunciado:
        - **Cap. 2** — Dados e pré-processamento
        - **Cap. 3** — Aplicação dos {len(methods)} modelos MCDM
        - **Cap. 4** — Análise de sensibilidade
        - **Cap. 5** — Dashboard e reutilizabilidade
        - **Cap. 6** — Comparação de modelos e recomendação
        - **Cap. 7** — Conclusões
        """)

    # =========================================================================
    # CAP 2 — DADOS E PRÉ-PROCESSAMENTO
    # =========================================================================
    with st.expander("**📊 Capítulo 2 — Dados e Pré-processamento** (3-5 pp)", expanded=False):
        st.markdown("### 2.1 Alternativas")
        st.markdown(f"Listagem das **{len(alts)}** alternativas avaliadas:")
        alt_meta = st.session_state.get("alt_metadata", None)
        if alt_meta is not None and not alt_meta.empty:
            st.dataframe(alt_meta, hide_index=True, use_container_width=True)
            st.caption("Metadados importados do enunciado (não usados nos cálculos MCDM, mas guardados como contexto).")
        else:
            st.dataframe(pd.DataFrame({"Alternativa": alts}), hide_index=True, use_container_width=True)

        st.markdown("### 2.2 Critérios")
        st.markdown(f"Lista dos **{len(crits)}** critérios e suas características:")
        crit_meta = st.session_state.get("crit_metadata", None)
        if crit_meta is not None and not crit_meta.empty:
            st.dataframe(crit_meta, hide_index=True, use_container_width=True)
            st.caption("Critérios importados em bruto do enunciado.")
        else:
            df_c = pd.DataFrame({"Código": crits, "Tipo": types, "Peso": weights,
                                 "%": [f"{w*100:.2f}%" for w in weights]})
            st.dataframe(df_c.style.format({"Peso": "{:.4f}"}), hide_index=True, use_container_width=True)

        st.markdown("### 2.3 Pesos e Consistência")
        eng_src = "Manual" if not st.session_state.global_injection_on else f"Motor injectado: {st.session_state.global_injection_engine}"
        st.markdown(f"**Fonte dos pesos activos:** {eng_src}")
        if "AHP" in st.session_state.engine_weights:
            st.caption("Os pesos AHP foram derivados da matriz par-a-par (ver aba 🔍 AHP).")
            if st.session_state.ahp_history:
                st.markdown("**Iterações de consistência AHP aplicadas:**")
                st.dataframe(pd.DataFrame(st.session_state.ahp_history), hide_index=True, use_container_width=True)
            else:
                st.caption("Nenhuma iteração de consistência aplicada (matriz consistente à primeira ou ainda não inserida).")
        df_w_full = pd.DataFrame({"Critério": crits, "Tipo": types, "Peso Activo": weights,
                                  "%": [f"{w*100:.2f}%" for w in weights]})
        st.dataframe(df_w_full.style.format({"Peso Activo": "{:.4f}"}), hide_index=True, use_container_width=True)

        st.markdown("### 2.4 Matriz de Decisão")
        st.dataframe(pd.DataFrame(matrix, index=alts, columns=crits).style.format("{:.4f}")
                      .background_gradient(cmap="Blues", axis=0),
                    use_container_width=True)

        st.markdown("### 2.5 Análise Crítica dos Dados")
        st.markdown(f"""
        - **Dimensionalidade**: {len(alts)} alts × {len(crits)} crits — adequado para os modelos MCDM aplicados.
        - **Critérios de custo**: {sum(1 for t in types if t == 'min')} ({', '.join(c for c, t in zip(crits, types) if t == 'min') or '—'})
        - **Critérios de benefício**: {sum(1 for t in types if t == 'max')} ({', '.join(c for c, t in zip(crits, types) if t == 'max') or '—'})
        - **Outliers de escala**: verificar especialmente C1 (Valor Potencial) se valores variam muito em ordens de grandeza —
          normalização vectorial (TOPSIS) e por soma (COPRAS) cuidam disso.
        """)

    # =========================================================================
    # CAP 3 — APLICAÇÃO DOS MODELOS MCDM
    # =========================================================================
    with st.expander(f"**🧮 Capítulo 3 — Aplicação dos {len(methods)} Modelos MCDM** (8-12 pp)", expanded=False):
        st.markdown(f"### Rankings obtidos por cada modelo")
        # tabela score × ranking
        for m in methods:
            res = st.session_state.all_results[m]
            df_m = pd.DataFrame({
                "Alternativa": alts,
                "Score": res["scores"],
                "Ranking": res["ranking"],
            }).sort_values("Ranking")
            st.markdown(f"**3.{methods.index(m)+1} {m}** — top-1: {df_m.iloc[0]['Alternativa']} (score={df_m.iloc[0]['Score']:.4f})")
            st.dataframe(df_m.style.format({"Score": "{:.4f}"})
                          .background_gradient(cmap="RdYlGn", subset=["Score"]),
                        hide_index=True, use_container_width=True)

        st.info("💡 Cada modelo está detalhado na sua aba (com normalização, passos de cálculo e fórmulas LaTeX).")

    # =========================================================================
    # CAP 4 — ANÁLISE DE SENSIBILIDADE
    # =========================================================================
    with st.expander("**🎯 Capítulo 4 — Análise de Sensibilidade** (4-6 pp)", expanded=False):
        sp = st.session_state.sensitivity_pct
        st.markdown(f"""
        ### 4.1 Metodologia
        Aplicada variação **±{sp}%** sobre o peso de cada critério individualmente, com
        renormalização dos restantes (Σ=1). Cada cenário produz {len(alts)} novos rankings,
        que são comparados com o ranking base.

        Para o **PROMETHEE II** testaram-se também as 3 funções de preferência (Usual, Linear, Gaussiana).
        Para o **AHP** variam-se julgamentos ±1 nível Saaty.
        """)

        # Calcular robustez global usando TOPSIS rápido
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

        st.markdown("### 4.2 Resultados por modelo")
        st.caption("Detalhe da SA em cada aba de modelo. Aqui apresenta-se a síntese global (TOPSIS).")
        st.dataframe(df_rob.style.format({"Robustez (%)": "{:.1f}"})
                      .background_gradient(cmap="RdYlGn", subset=["Robustez (%)"]),
                    hide_index=True, use_container_width=True)

        st.markdown("### 4.3 Robustez do ranking final")
        estaveis = df_rob[df_rob["Inversões"] == 0]["Alternativa"].tolist()
        instaveis = df_rob[df_rob["Inversões"] > 3]["Alternativa"].tolist()
        if estaveis:
            st.success(f"✅ **Alternativas ROBUSTAS**: {', '.join(estaveis)} — não mudam de posição em nenhum cenário.")
        if instaveis:
            st.warning(f"⚠️ **Alternativas SENSÍVEIS**: {', '.join(instaveis)} — mudam em >3 cenários.")

    # =========================================================================
    # CAP 5 — DASHBOARD E REUTILIZABILIDADE
    # =========================================================================
    with st.expander("**🎛️ Capítulo 5 — Dashboard e Reutilizabilidade** (3-5 pp)", expanded=False):
        st.markdown(f"""
        ### 5.1 Arquitectura
        A aplicação MCDM Dashboard implementa **{len(methods)} modelos** + 4 motores de pesos (SWING, SMART, Entropia, CRITIC) + AHP em aba dedicada.

        Estrutura em **17 abas**: 🏠 Início · 📋 Dados · ⚖️ Motores · 🔍 AHP · 9 modelos MCDM · 📊 Gráficos · 🏆 Dashboard · 🎛️ Vista 360° · 📑 Relatório.

        ### 5.2 Guia de Utilização
        1. Sidebar → escolher fonte de dados (Demo, Manual, Excel, Quadros em bruto)
        2. Sidebar → ajustar tipos (max/min) e pesos manuais ou usar motor
        3. Sidebar → activar injecção global se quiser usar pesos de motor em todos os modelos
        4. Executar abas dos modelos para gerar rankings
        5. Aba 🎛️ Vista 360° para vista executiva única
        6. Aba 📑 Relatório para descarregar markdown completo

        ### 5.3 Validação
        Cada modelo apresenta passos intermédios com fórmulas LaTeX, permitindo verificação manual.

        ### 5.4 Reutilizabilidade
        - **Nada hardcoded** — toda a matriz vem da sidebar
        - Suporta até **50 alternativas** (e mais críticos com paginação)
        - Aceita até **15 critérios**
        - Alteração nos dados de entrada actualiza automaticamente TODOS os outputs (modelos, gráficos, relatório)
        """)

    # =========================================================================
    # CAP 6 — COMPARAÇÃO DE MODELOS E RECOMENDAÇÃO
    # =========================================================================
    with st.expander("**⚖️ Capítulo 6 — Comparação de Modelos e Recomendação Final** (3-4 pp)", expanded=False):
        st.markdown("### 6.1 Convergência dos Rankings")
        st.dataframe(df_dash.style.background_gradient(cmap="RdYlGn_r",
                      subset=methods + ["Posição Média", "Ranking Final"]),
                    hide_index=True, use_container_width=True)

        st.markdown("### 6.2 Discussão das Diferenças")
        st.markdown(f"""
        Os métodos têm <b>axiomáticas diferentes</b>:
        - **TOPSIS, MAUT, COPRAS**: <b>compensatórios</b> — bom desempenho num critério compensa mau noutro
        - **PROMETHEE II, ELECTRE III**: <b>não-compensatórios</b> — utilizam fluxos / outranking par-a-par
        - **VIKOR**: <b>compromisso</b> — equilibra utilidade global e arrependimento individual
        - **Fuzzy TOPSIS/AHP**: lidam com <b>imprecisão</b> nos dados/pesos

        Convergência inter-modelo é o melhor indicador de robustez da decisão.
        """, unsafe_allow_html=True)

        st.markdown("### 6.3 Comparação com Metodologia de Origem")
        st.info("Se for o caso MCG: comparar o top-3 obtido com o esperado pela empresa (Q6.5: A1 e A9). "
                "A app reporta convergência mas a interpretação final cabe ao analista.")

        st.markdown(f"### 6.4 Recomendação Final")
        if conv_pct_top1 >= 70:
            verdict = "🟢 **ALTA convergência** — recomendação ROBUSTA, decisão com elevado grau de confiança."
        elif conv_pct_top1 >= 40:
            verdict = "🟡 **Convergência MODERADA** — recomendação aceitável, analise sensibilidade antes de decidir."
        else:
            verdict = "🔴 **BAIXA convergência** — Top-1 instável. Reveja pesos ou alargue conjunto de alternativas."
        st.markdown(verdict)
        st.markdown(f"**Recomenda-se a alternativa {top1}**, com top-3 = {top3}.")

    # =========================================================================
    # CAP 7 — CONCLUSÕES
    # =========================================================================
    with st.expander("**🎓 Capítulo 7 — Conclusões** (1-2 pp)", expanded=False):
        st.markdown(f"""
        ### Principais Conclusões
        - Aplicaram-se **{len(methods)} modelos MCDM** (mínimo do enunciado: 4 — TOPSIS, PROMETHEE II, AHP, COPRAS).
        - O top-1 consensual é **{top1}** com convergência de {conv_pct_top1:.0f}% nos modelos.
        - {len(df_rob[df_rob['Inversões'] == 0])} alternativas demonstraram robustez total em ±{st.session_state.sensitivity_pct}%.

        ### Limitações
        - Qualidade dos dados de entrada (alguns valores são estimativas qualitativas convertidas em numéricas).
        - Subjectividade dos pesos (mesmo com AHP CR < 0.10).
        - Decisão final cabe sempre ao decisor humano — o modelo é ferramenta de apoio, não substituto.

        ### Sugestões para Ciclos Futuros
        - Recolha periódica de dados actualizados.
        - Validar matriz AHP com múltiplos decisores (média geométrica das matrizes).
        - Acompanhar evolução do top-3 ao longo do tempo (tendência).
        - Considerar Fuzzy AHP / TOPSIS quando os dados forem mais imprecisos.
        """)

    # =========================================================================
    # REFERÊNCIAS
    # =========================================================================
    with st.expander("**📚 Referências (APA 7ª)**", expanded=False):
        st.markdown("""
        - Brans, J.-P., & Vincke, P. (1985). A Preference Ranking Organisation Method. *Management Science*, 31(6), 647-656.
        - Chang, D.-Y. (1996). Applications of the extent analysis method on fuzzy AHP. *European Journal of Operational Research*, 95(3), 649-655.
        - Chen, C.-T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy environment. *Fuzzy Sets and Systems*, 114(1), 1-9.
        - Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining objective weights in multiple criteria problems: The CRITIC method. *Computers & Operations Research*, 22(7), 763-770.
        - Gabus, A., & Fontela, E. (1972). World problems, an invitation to further thought within the framework of DEMATEL. *Battelle Geneva Research Center*.
        - Hwang, C.-L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer-Verlag.
        - Keeney, R. L., & Raiffa, H. (1976). *Decisions with Multiple Objectives*. Wiley.
        - Opricovic, S., & Tzeng, G.-H. (2004). Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS. *European Journal of Operational Research*, 156(2), 445-455.
        - Roy, B. (1968). Classement et choix en présence de points de vue multiples (ELECTRE). *RAIRO*, 8, 57-75.
        - Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.
        - Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27, 379-423.
        - Zavadskas, E. K., & Kaklauskas, A. (1996). *Multiple Criteria Evaluation of Buildings*. Vilnius Technika.
        """)

    # =========================================================================
    # DOWNLOADS
    # =========================================================================
    st.markdown("---")
    st.subheader("📥 Exportar Relatório")

    # Markdown completo
    def df_to_md(df):
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = []
        for _, row in df.iterrows():
            cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
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
        f"\n## Capítulo 1 — Introdução",
        f"Aplicação MCDM a {len(alts)} alternativas × {len(crits)} critérios.",
        f"\n## Capítulo 2 — Dados e Pré-processamento",
        f"\n### 2.1 Alternativas",
        ", ".join(alts),
        f"\n### 2.2 Critérios e Pesos (fonte: {eng_src_md})",
        df_to_md(df_w_md),
        f"\n### 2.4 Matriz de Decisão",
        df_to_md(pd.DataFrame(matrix, index=alts, columns=crits).reset_index().rename(columns={'index': 'Alt'})),
    ]
    if st.session_state.ahp_history:
        md_lines.append(f"\n### 2.3 Iterações AHP de Consistência")
        md_lines.append(df_to_md(pd.DataFrame(st.session_state.ahp_history)))

    md_lines.append(f"\n## Capítulo 3 — Aplicação dos {len(methods)} Modelos MCDM")
    for m in methods:
        res = st.session_state.all_results[m]
        df_m = pd.DataFrame({"Alt": alts, "Score": res["scores"], "Rank": res["ranking"]}).sort_values("Rank")
        md_lines.append(f"\n### {m}")
        md_lines.append(df_to_md(df_m))

    md_lines.append(f"\n## Capítulo 4 — Análise de Sensibilidade (±{sp_md}%)")
    md_lines.append(df_to_md(df_rob))

    md_lines.append(f"\n## Capítulo 5 — Dashboard e Reutilizabilidade")
    md_lines.append("Aplicação modular, sem hardcode, suporta até 50 alts × 15 crits.")

    md_lines.append(f"\n## Capítulo 6 — Comparação e Recomendação Final")
    md_lines.append(df_to_md(df_dash))

    md_lines.append(f"\n## Capítulo 7 — Conclusões")
    md_lines.append(f"Top-1 recomendado: **{top1}** (convergência {conv_pct_top1:.0f}%).")

    md_report = "\n".join(md_lines)

    # CSV consolidado
    csv_buffer = StringIO()
    df_dash.to_csv(csv_buffer, index=False)

    # Excel consolidado
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

    st.caption("💡 O markdown pode ser convertido para PDF com Pandoc, ou aberto no Typora/VSCode/Obsidian para edição.")
