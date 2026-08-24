import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path


def load_css(path: str):
    css_path = Path(path)
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Motore di simulazione
# --------------------------------------------------------------------------------------
def nice_number(x: float) -> float:
    if x <= 0:
        return 1.0
    exp = np.floor(np.log10(x))
    f = x / 10 ** exp
    if f < 1.5:
        nf = 1
    elif f < 3:
        nf = 2
    elif f < 7:
        nf = 5
    else:
        nf = 10
    return nf * 10 ** exp


def make_y_ticks(max_val: float, n_ticks: int = 8):
    step = nice_number(max_val / n_ticks)
    top = np.ceil(max_val / step) * step
    ticks = np.arange(0, top + step / 2, step)
    ticktext = [f"{int(round(v / 1000))}k" for v in ticks]
    return ticks, ticktext


def simulate(anni, rendimento_annuo_pct, sigma_annuo_pct, contributo_mensile, n_scenari, capitale_iniziale, seed=None):
    """
    Genera n_scenari traiettorie mensili di rendimento che, pur essendo casuali
    (stessa volatilità), condividono TUTTE lo stesso rendimento composto finale
    (isolando così il rischio di sequenza/timing dei rendimenti).
    """
    rng = np.random.default_rng(seed)
    months = int(anni * 12)
    r = rendimento_annuo_pct / 100.0
    sigma = sigma_annuo_pct / 100.0
    sigma_m = sigma / np.sqrt(12)

    target_total_log_return = anni * np.log(1 + r)

    eps = rng.normal(loc=0.0, scale=sigma_m, size=(n_scenari, months))
    adjustment = (target_total_log_return - eps.sum(axis=1, keepdims=True)) / months
    log_returns = eps + adjustment  # ogni riga somma esattamente a target_total_log_return

    cum_log = np.cumsum(log_returns, axis=1)
    cum_factor = np.exp(cum_log)
    cum_factor_full = np.hstack([np.ones((n_scenari, 1)), cum_factor])  # shape (n, months+1)

    # PIC: capitale investito in un'unica soluzione al tempo 0
    pic_path = capitale_iniziale * cum_factor_full

    # PAC: contributo mensile investito all'inizio di ogni mese, capitalizzato fino al tempo t
    inv_cum = 1.0 / cum_factor_full
    full_cumsum = np.cumsum(inv_cum, axis=1)
    shifted = np.hstack([np.zeros((n_scenari, 1)), full_cumsum[:, :-1]])
    pac_path = contributo_mensile * cum_factor_full * shifted

    x_years = np.linspace(0, anni, months + 1)
    return x_years, pic_path, pac_path


def build_trajectories_figure(x_years, paths, title, line_color="rgba(31,119,180,0.18)"):
    n_scenari = paths.shape[0]

    x_all, y_all = [], []
    for i in range(n_scenari):
        x_all.extend(x_years.tolist())
        x_all.append(None)
        y_all.extend(paths[i].tolist())
        y_all.append(None)

    mean_path = paths.mean(axis=0)
    max_val = paths.max()
    yticks, yticktext = make_y_ticks(max_val)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x_all, y=y_all,
        mode="lines",
        line=dict(color=line_color, width=1),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_years, y=mean_path,
        mode="lines",
        line=dict(color="#e11d2e", width=3),
        name="Media",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16, color="#111827")),
        xaxis_title="Anni di investimento",
        yaxis_title="Valore (€)",
        yaxis=dict(tickvals=yticks, ticktext=yticktext, gridcolor="#e5e7eb", zeroline=False),
        xaxis=dict(gridcolor="#e5e7eb", zeroline=False),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=20, t=50, b=50),
        height=460,
    )
    return fig


def fmt_euro(v: float) -> str:
    return f"€{v:,.0f}".replace(",", ".")


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="PIC vs PAC — Simulazione Montecarlo",
        layout="wide",
    )
    load_css("style.css")

    st.markdown(
        '<div class="app-title">PIC VS PAC. Simulazione PAC con metodo montecarlo e analisi dei profili '
        'di rendimento nei diversi casi</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subtitle">Ammettiamo di conoscere in anticipo il rendimento di un titolo o di un '
        'portafoglio da qui a 30 anni (ipotesi già impossibile, perché nessuno può prevedere il futuro). '
        'Avendo questo dato, è possibile prevedere il valore finale del PAC?</div>',
        unsafe_allow_html=True,
    )

    # ---- Riga 1: parametri ----
    col_vol, col_anni, col_rend, col_contr, col_scen = st.columns([1.3, 1, 1, 1, 1])

    with col_vol:
        st.markdown('<div class="field-label">Volatilità annua (σ) %</div>', unsafe_allow_html=True)
        sigma_annuo_pct = st.slider(
            "Volatilità annua", min_value=0, max_value=50, value=13, step=1,
            label_visibility="collapsed",
        )
        st.markdown(f'<div class="slider-value">{sigma_annuo_pct}%</div>', unsafe_allow_html=True)

    with col_anni:
        st.markdown('<div class="field-label">Anni</div>', unsafe_allow_html=True)
        anni = st.number_input("Anni", min_value=1, max_value=60, value=30, step=1, label_visibility="collapsed")

    with col_rend:
        st.markdown('<div class="field-label">Rendimento annuo atteso %</div>', unsafe_allow_html=True)
        rendimento_annuo_pct = st.number_input(
            "Rendimento annuo atteso %", min_value=-20.0, max_value=30.0, value=8.0, step=0.5,
            label_visibility="collapsed",
        )

    with col_contr:
        st.markdown('<div class="field-label">Contributo mensile (€)</div>', unsafe_allow_html=True)
        contributo_mensile = st.number_input(
            "Contributo mensile (€)", min_value=1, max_value=100000, value=278, step=1,
            label_visibility="collapsed",
        )

    with col_scen:
        st.markdown('<div class="field-label">Scenari</div>', unsafe_allow_html=True)
        n_scenari = st.number_input(
            "Scenari", min_value=10, max_value=2000, value=200, step=10,
            label_visibility="collapsed",
        )

    # ---- Riga 2: capitale iniziale + simula + risultati ----
    capitale_iniziale = int(contributo_mensile * 12 * anni)

    col_pic, col_btn, col_res = st.columns([1.3, 1.3, 2.4])

    with col_pic:
        st.markdown(
            f'<div class="field-label"><b>Capitale iniziale (PIC):</b> {fmt_euro(capitale_iniziale)} '
            '<span style="color:#6b7280;font-size:0.85rem;">(calcolato per eguagliare il '
            'contributo PAC: mensile x 12 x anni)</span></div>',
            unsafe_allow_html=True,
        )

    with col_btn:
        st.markdown('<div style="height:1.9rem;"></div>', unsafe_allow_html=True)
        simula = st.button("Simula", width="stretch")

    if simula or "sim_result" not in st.session_state:
        seed = np.random.randint(0, 1_000_000)
        x_years, pic_path, pac_path = simulate(
            anni, rendimento_annuo_pct, sigma_annuo_pct, contributo_mensile,
            int(n_scenari), capitale_iniziale, seed=seed,
        )
        st.session_state["sim_result"] = (x_years, pic_path, pac_path)

    x_years, pic_path, pac_path = st.session_state["sim_result"]

    pac_final = pac_path[:, -1]
    media = np.mean(pac_final)
    mediana = np.median(pac_final)
    p2_5, p97_5 = np.percentile(pac_final, [2.5, 97.5])

    pic_final = pic_path[:, -1]
    pic_media = np.mean(pic_final)
    pic_mediana = np.median(pic_final)
    pic_p2_5, pic_p97_5 = np.percentile(pic_final, [2.5, 97.5])

    with col_res:
        st.markdown('<div style="height:1.9rem;"></div>', unsafe_allow_html=True)
        st.markdown(
            f'''
            <table class="result-table">
                <tr>
                    <td><b>PAC (finale):</b></td>
                    <td>media {fmt_euro(media)}</td>
                    <td>mediana {fmt_euro(mediana)}</td>
                    <td>95% interval {fmt_euro(p2_5)} - {fmt_euro(p97_5)}</td>
                </tr>
                <tr>
                    <td><b>PIC (finale):</b></td>
                    <td>media {fmt_euro(pic_media)}</td>
                    <td>mediana {fmt_euro(pic_mediana)}</td>
                    <td>95% interval {fmt_euro(pic_p2_5)} - {fmt_euro(pic_p97_5)}</td>
                </tr>
            </table>
            ''',
            unsafe_allow_html=True,
        )

    # ---- Box informativo ----
    st.markdown(
        '''
        <div class="info-box">
            <h4>Cosa mostrano i grafici e perché leggerli con attenzione</h4>
            <p>I grafici riportano molte traiettorie ipotetiche del mercato per lo stesso orizzonte temporale:
            l'obiettivo è confrontare due strategie di investimento e comprendere le loro vulnerabilità.</p>
            <ul>
                <li><b>PIC</b>: valore di un investimento iniziale unico nel tempo su ogni traiettoria
                (linee blu); la <b>linea rossa</b> rappresenta la media.</li>
                <li><b>PAC</b>: valore ottenuto versando un contributo mensile costante su ogni traiettoria
                (linee blu); la <b>linea rossa</b> rappresenta la media.</li>
            </ul>
            <p class="implicazioni">Implicazioni: anche con lo stesso rendimento medio atteso, la volatilità e
            la sequenza dei rendimenti producono ampia dispersione nei risultati. Il PAC riduce il rischio di
            timing, ma non annulla l'incertezza legata a mercati molto volatili.</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # ---- Grafici ----
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_pic = build_trajectories_figure(
            x_years, pic_path, "PIC: simulazioni di traiettorie con stesso rendimento finale"
        )
        st.plotly_chart(fig_pic, width="stretch", config={"displayModeBar": False})

    with col_g2:
        fig_pac = build_trajectories_figure(
            x_years, pac_path, "PAC: contribuzioni mensili su ciascuna traiettoria"
        )
        st.plotly_chart(fig_pac, width="stretch", config={"displayModeBar": False})

    st.markdown(
        '<div class="footer-note">Linee blu sottili per mettere in evidenza la dispersione; '
        'la linea rossa rappresenta la media.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="footer-note moral"><b>Morale:</b> il risultato di un PAC è altamente sensibile alle '
        'variazioni del rendimento atteso e della volatilità. A parità di rendimento finale del mercato in un '
        'certo periodo di tempo, la traiettoria con cui si arriva a quel rendimento — cioè l\'ordine in cui si '
        'susseguono i rendimenti mensili — influenza pesantemente il valore finale del PAC, molto più di quanto '
        'influenzi il PIC.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="footer-note">Il PIC è una somma unica investita al tempo 0 e mai toccata: il suo valore '
        'finale è capitale × (prodotto di tutti i rendimenti mensili). Poiché la moltiplicazione è commutativa, '
        'quel prodotto non dipende dall\'ordine in cui i rendimenti si susseguono — solo dal loro prodotto '
        'totale. Infatti nel nostro simulatore, a parità di rendimento finale imposto, tutte le traiettorie PIC '
        'finiscono esattamente sullo stesso valore (lo vedi anche nel grafico: le linee blu, larghe a metà '
        'percorso, si "stringono" tutte verso lo stesso punto alla fine).<br><br>'
        'Il PAC invece versa capitale in momenti diversi nel tempo: ogni rata è esposta solo ai rendimenti '
        'successivi al suo versamento. Due sequenze con lo stesso rendimento totale ma ordine diverso (es. anni '
        'forti all\'inizio vs. anni forti alla fine) fanno arrivare capitale diverso a ogni rata, quindi il '
        'totale finale cambia — anche se il mercato, nel complesso, ha reso "la stessa cosa".</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="footer-note">Questo fenomeno ha un nome consolidato in finanza quantitativa: '
        '<b>sequence-of-returns risk</b> (rischio di sequenza dei rendimenti) — è reale e ben documentato, non '
        'solo nell\'accumulo (PAC) ma soprattutto nei prelievi in fase di decumulo/pensione.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
