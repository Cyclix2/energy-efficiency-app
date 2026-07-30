"""
Run:  streamlit run app.py
"""
import json
import pathlib
import pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

#constants
HERE = pathlib.Path(__file__).parent
MODEL_DIR = HERE / "models"

SHEET = "#EDEFF2"      # drafting paper
INK = "#1B2430"        # drawing ink
GRAPHITE = "#5A6673"   # pencil / secondary text
HEAT = "#C4452D"       # burnt sienna -- heating
COOL = "#2E6F8E"       # steel blue -- cooling
RULE = "#C8CDD4"       # dimension line
PAPER = "#F7F8FA"      # card fill

ORIENT_NAMES = {2: "North", 3: "East", 4: "South", 5: "West"}
GLAZING_LAYOUTS = {
    0: "None (unglazed)",
    1: "Uniform — 25% on each face",
    2: "North-weighted — 55% north",
    3: "East-weighted — 55% east",
    4: "South-weighted — 55% south",
    5: "West-weighted — 55% west",
}
FEATURE_LABELS = {
    "RelCompactness": "Relative compactness", "WallArea": "Wall area",
    "RoofArea": "Roof / footprint area", "Height": "Overall height",
    "Orientation": "Orientation", "GlazingArea": "Glazing ratio",
    "GlazingDistrib": "Glazing layout", "GlazedArea_abs": "Glazed area (m²)",
    "WallToRoof": "Wall-to-roof ratio", "SurfaceToVolume": "Surface-to-volume ratio",
    "IsTwoStorey": "Two-storey", "NoGlazing": "Unglazed",
}

st.set_page_config(page_title="Thermal Envelope Explorer",
                   page_icon="◧", layout="wide",
                   initial_sidebar_state="expanded")


#style
def inject_css() -> None:
    """Inject the stylesheet.

    Uses st.html(), not st.markdown(): Markdown ends a raw HTML block at the first
    blank line, which would spill the rest of the stylesheet onto the page as visible
    text and leave those rules unapplied. Blank lines are stripped as well, so the
    block stays safe regardless of which renderer handles it.
    """
    css = f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
      /* --- theme lock -------------------------------------------------------
         Streamlit follows the viewer's OS dark-mode preference unless
         .streamlit/config.toml pins it. That file is dot-prefixed and is easily
         lost when a repo is uploaded through a web UI, which would leave this
         drafting-sheet palette painted over dark Streamlit chrome -- dark text on
         dark ground. These rules make the light theme hold on their own, so a
         missing config.toml degrades the app's polish rather than its legibility.
         Scoped to Streamlit's own containers and text nodes: the colour-coded
         numbers below set their colours inline and must not be overridden. */
      :root, html {{ color-scheme: light !important; }}
      html, body, .stApp, [data-testid="stApp"],
      [data-testid="stAppViewContainer"], [data-testid="stMain"],
      [data-testid="stHeader"] {{ background: {SHEET} !important; }}
      [data-testid="stSidebar"], [data-testid="stSidebarContent"],
      [data-testid="stSidebarUserContent"] {{ background: {PAPER} !important; }}
      [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
      [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
      [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
      [data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
      .stRadio label p, .stCheckbox label p,
      [data-baseweb="select"] div {{ color: {INK} !important; }}
      [data-testid="stSlider"] [role="slider"],
      [data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background"] {{
          background-color: {COOL} !important; }}
      [data-baseweb="radio"] div[aria-checked="true"],
      [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{
          color: {GRAPHITE} !important; }}
      input, textarea, [data-baseweb="input"], [data-baseweb="select"] > div {{
          background: #FFFFFF !important; color: {INK} !important; }}

      .stApp {{ background: {SHEET}; }}
      html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', system-ui, sans-serif;
                                    color: {INK}; }}
      h1, h2, h3, h4 {{ font-family: 'IBM Plex Sans Condensed', sans-serif;
                        letter-spacing: .01em; color: {INK}; }}
      #MainMenu, footer, header {{ visibility: hidden; }}
      .block-container {{ padding-top: 1.6rem; max-width: 1320px; }}

      /* --- drawing title block --- */
      .titleblock {{ border: 1.5px solid {INK}; background: {PAPER};
                     display: grid; grid-template-columns: 1fr auto auto auto;
                     margin-bottom: 1.4rem; }}
      .titleblock > div {{ padding: .7rem 1.1rem; border-left: 1px solid {RULE}; }}
      .titleblock > div:first-child {{ border-left: none; }}
      .tb-name {{ font-family: 'IBM Plex Sans Condensed', sans-serif;
                  font-size: 1.45rem; font-weight: 700; line-height: 1.1;
                  color: {INK}; }}
      .tb-sub {{ font-size: .78rem; color: {GRAPHITE}; margin-top: .15rem; }}
      .tb-label {{ font-family: 'IBM Plex Mono', monospace; font-size: .6rem;
                   letter-spacing: .13em; color: {GRAPHITE}; text-transform: uppercase; }}
      .tb-val {{ font-family: 'IBM Plex Mono', monospace; font-size: .95rem;
                 font-weight: 500; margin-top: .2rem; }}

      /* --- step headers in the sidebar --- */
      .step {{ font-family: 'IBM Plex Mono', monospace; font-size: .62rem;
               letter-spacing: .14em; color: {GRAPHITE}; text-transform: uppercase;
               border-bottom: 1px solid {RULE}; padding-bottom: .3rem;
               margin: 1.1rem 0 .5rem; }}
      .step b {{ color: {INK}; }}

      /* --- result cards --- */
      .card {{ border: 1.5px solid {INK}; background: {PAPER}; padding: 1rem 1.2rem; }}
      .card-heat {{ border-left: 6px solid {HEAT}; }}
      .card-cool {{ border-left: 6px solid {COOL}; }}
      .card-label {{ font-family: 'IBM Plex Mono', monospace; font-size: .64rem;
                     letter-spacing: .13em; text-transform: uppercase; color: {GRAPHITE}; }}
      .card-value {{ font-family: 'IBM Plex Sans Condensed', sans-serif;
                     font-size: 2.9rem; font-weight: 700; line-height: 1; margin: .3rem 0 .1rem; }}
      .card-unit {{ font-family: 'IBM Plex Mono', monospace; font-size: .78rem;
                    color: {GRAPHITE}; }}

      .note {{ font-size: .82rem; color: {GRAPHITE}; line-height: 1.5; }}
      .badge {{ display: inline-block; font-family: 'IBM Plex Mono', monospace;
                font-size: .66rem; letter-spacing: .1em; text-transform: uppercase;
                padding: .3rem .6rem; border: 1px solid; }}
      .badge-ok {{ color: #1D6B45; border-color: #1D6B45; background: #E8F3ED; }}
      .badge-warn {{ color: #8A5A00; border-color: #8A5A00; background: #FBF2DF; }}

      [data-testid="stSidebar"] {{ background: {PAPER}; border-right: 1.5px solid {INK}; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1.5px solid {INK}; }}
      .stTabs [data-baseweb="tab"] {{ font-family: 'IBM Plex Sans Condensed', sans-serif;
                                      font-weight: 600; font-size: .95rem;
                                      border-radius: 0; padding: .55rem 1.1rem; }}
      .stTabs [aria-selected="true"] {{ background: {INK}; color: {SHEET}; }}
      div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Sans Condensed', sans-serif; }}
    </style>"""
    st.html("\n".join(line for line in css.splitlines() if line.strip()))


#model load
def _unpickle(path: pathlib.Path):
    """Load a pickled scikit-learn Pipeline exported by Section 7 of the notebook."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    """Load both models once per session.

    The .pkl files hold the complete Pipeline -- scaler, one-hot encoder and boosted
    model together -- so the app cannot preprocess differently from the notebook.
    """
    try:
        heating = _unpickle(MODEL_DIR / "heating_model.pkl")
        cooling = _unpickle(MODEL_DIR / "cooling_model.pkl")
        with open(MODEL_DIR / "metadata.json") as fh:
            meta = json.load(fh)
    except FileNotFoundError as exc:
        st.error(
            f"Could not find **{exc.filename}**.\n\n"
            "The app needs `heating_model.pkl`, `cooling_model.pkl` and `metadata.json` "
            "in the `models/` folder. Run Section 7 of the notebook, or "
            "`python train_models.py`, to generate them."
        )
        st.stop()
    return heating, cooling, meta


def engineer(row: dict) -> dict:
    """Mirrors add_engineered_features() from the notebook, exactly."""
    r = dict(row)
    r["GlazedArea_abs"] = r["GlazingArea"] * r["RoofArea"]
    r["WallToRoof"] = r["WallArea"] / r["RoofArea"]
    r["SurfaceToVolume"] = r["SurfaceArea"] / (r["RoofArea"] * r["Height"])
    r["IsTwoStorey"] = int(r["Height"] > 5)
    r["NoGlazing"] = int(r["GlazingArea"] == 0)
    r.pop("SurfaceArea")
    return r


def predict(shape: dict, glazing_area: float, glazing_distrib: int,
            orientation: int) -> tuple[float, float]:
    row = {**{k: shape[k] for k in
              ["RelCompactness", "SurfaceArea", "WallArea", "RoofArea", "Height"]},
           "Orientation": orientation, "GlazingArea": glazing_area,
           "GlazingDistrib": glazing_distrib}
    X = pd.DataFrame([engineer(row)])
    return float(HEATING.predict(X)[0]), float(COOLING.predict(X)[0])


HEATING, COOLING, META = load_artifacts()
SHAPES = META["shapes"]
inject_css()


def shape_label(s: dict) -> str:
    storeys = "2-storey" if s["Height"] > 5 else "1-storey"
    return f'RC {s["RelCompactness"]:.2f} · {storeys} · {s["RoofArea"]:.0f} m² footprint'


#title block
st.html(f"""
<div class="titleblock">
  <div>
    <div class="tb-name">Thermal Envelope Explorer</div>
    <div class="tb-sub">Design-stage heating &amp; cooling load prediction for residential buildings</div>
  </div>
  <div><div class="tb-label">Heating ±</div>
       <div class="tb-val" style="color:{HEAT}">{META['metrics']['HeatingLoad']['RMSE']:.2f}</div></div>
  <div><div class="tb-label">Cooling ±</div>
       <div class="tb-val" style="color:{COOL}">{META['metrics']['CoolingLoad']['RMSE']:.2f}</div></div>
  <div><div class="tb-label">Basis</div>
       <div class="tb-val">768 EnergyPlus runs</div></div>
</div>""")

#sidebar
with st.sidebar:
    st.markdown('<div class="step">Step <b>01</b> - Massing '
                '<span style="float:right">drives both loads</span></div>',
                unsafe_allow_html=True)
    shape_idx = st.selectbox(
        "Building shape", range(len(SHAPES)), index=10,
        format_func=lambda i: shape_label(SHAPES[i]),
        help="The 12 massings simulated in EnergyPlus. Relative compactness (RC) "
             "rises as the block becomes more compact for the same enclosed volume.")
    shape = SHAPES[shape_idx]

    st.markdown('<div class="step">Step <b>02</b> - Glazing '
                '<span style="float:right">mainly cooling</span></div>',
                unsafe_allow_html=True)
    glazing_area = st.select_slider(
        "Glazing ratio (share of floor area)", options=META["glazing_areas"],
        value=0.25, format_func=lambda v: f"{v:.0%}")
    if glazing_area == 0:
        glazing_distrib = 0
        st.caption("Unglazed — no layout to choose.")
    else:
        glazing_distrib = st.selectbox(
            "Glazing layout", [d for d in META["glazing_distribs"] if d != 0],
            index=2, format_func=lambda d: GLAZING_LAYOUTS[d])

    st.markdown('<div class="step">Step <b>03</b> - Orientation '
                '<span style="float:right">mainly cooling</span></div>',
                unsafe_allow_html=True)
    orientation = st.radio("Building faces", META["orientations"], index=2,
                           format_func=lambda o: ORIENT_NAMES[o], horizontal=True)

    st.markdown(f'<div class="step">Shortlist</div>', unsafe_allow_html=True)
    n_saved = len(st.session_state.get("shortlist", []))
    scheme_name = st.text_input("Scheme name", value=f"Scheme {chr(65 + n_saved)}",
                                key=f"scheme_name_{n_saved}")
    if st.button("Add to shortlist", use_container_width=True):
        h, c = predict(shape, glazing_area, glazing_distrib, orientation)
        st.session_state.setdefault("shortlist", []).append({
            "Scheme": scheme_name, "Shape": shape_label(shape),
            "Glazing": f"{glazing_area:.0%}", "Layout": GLAZING_LAYOUTS[glazing_distrib],
            "Faces": ORIENT_NAMES[orientation],
            "Heating": round(h, 2), "Cooling": round(c, 2),
            "Total": round(h + c, 2), "Balance": round(c - h, 2)})
        st.success(f"{scheme_name} added.")

heat, cool = predict(shape, glazing_area, glazing_distrib, orientation)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Design explorer", "Compare schemes", "Sensitivity", "Model card"])

#TAB 1 (explorer)
with tab1:
    left, right = st.columns([1.05, 1])

    with left:
        c1, c2 = st.columns(2)
        c1.markdown(f"""<div class="card card-heat">
            <div class="card-label">Heating load</div>
            <div class="card-value" style="color:{HEAT}">{heat:.1f}</div>
            <div class="card-unit">kWh/m² · ± {META['metrics']['HeatingLoad']['RMSE']:.2f}</div>
            </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="card card-cool">
            <div class="card-label">Cooling load</div>
            <div class="card-value" style="color:{COOL}">{cool:.1f}</div>
            <div class="card-unit">kWh/m² · ± {META['metrics']['CoolingLoad']['RMSE']:.2f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Seasonal balance")
        balance = cool - heat
        dominant = "Cooling-dominated" if balance > 0 else "Heating-dominated"

        fig = go.Figure()
        fig.add_trace(go.Bar(y=["balance"], x=[-heat], orientation="h",
                             marker_color=HEAT, name="Heating",
                             hovertemplate="Heating %{customdata:.1f} kWh/m²<extra></extra>",
                             customdata=[heat]))
        fig.add_trace(go.Bar(y=["balance"], x=[cool], orientation="h",
                             marker_color=COOL, name="Cooling",
                             hovertemplate="Cooling %{x:.1f} kWh/m²<extra></extra>"))
        fig.add_vline(x=0, line_width=2, line_color=INK)
        fig.update_layout(
            barmode="relative", height=140, showlegend=False,
            margin=dict(l=0, r=0, t=6, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[-50, 50], zeroline=False, gridcolor=RULE,
                       tickvals=[-40, -20, 0, 20, 40],
                       ticktext=["40", "20", "0", "20", "40"],
                       title="◄ heating   kWh/m²   cooling ►",
                       title_font=dict(size=11, color=GRAPHITE)),
            yaxis=dict(showticklabels=False),
            font=dict(family="IBM Plex Mono", color=INK))
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown(
            f'<div class="note"><b>{dominant}</b> by '
            f'<b>{abs(balance):.1f} kWh/m²</b>. Combined annual load '
            f'<b>{heat + cool:.1f} kWh/m²</b>. The balance decides which plant sizes '
            f'the system and where the fabric budget earns most.</div>',
            unsafe_allow_html=True)

    with right:
        st.markdown("#### Specification")
        spec = pd.DataFrame({
            "Parameter": ["Relative compactness", "Surface area", "Wall area",
                          "Roof / footprint area", "Overall height", "Storeys",
                          "Glazing ratio", "Glazing layout", "Orientation",
                          "Glazed area"],
            "Value": [f'{shape["RelCompactness"]:.2f}',
                      f'{shape["SurfaceArea"]:.1f} m²', f'{shape["WallArea"]:.1f} m²',
                      f'{shape["RoofArea"]:.2f} m²', f'{shape["Height"]:.1f} m',
                      "2" if shape["Height"] > 5 else "1",
                      f"{glazing_area:.0%}", GLAZING_LAYOUTS[glazing_distrib],
                      ORIENT_NAMES[orientation],
                      f'{glazing_area * shape["RoofArea"]:.1f} m²']})
        st.dataframe(spec, hide_index=True, use_container_width=True, height=388)

        st.markdown(
            f'<span class="badge badge-ok">Inside validated envelope</span>'
            f'<div class="note" style="margin-top:.5rem">Every input above is one the '
            f'model was trained on, so the quoted error bars apply. The tool only '
            f'offers the 12 simulated massings for this reason, see the model card '
            f'for what happens outside them.</div>', unsafe_allow_html=True)

#TAB 2 (compare)
with tab2:
    shortlist = st.session_state.get("shortlist", [])
    if not shortlist:
        st.markdown("#### No schemes yet")
        st.markdown('<div class="note">Set a design in the sidebar and choose '
                    '<b>Add to shortlist</b>. Add two or more to compare them here '
                    'the tool ranks by combined load and shows the seasonal split for '
                    'each.</div>', unsafe_allow_html=True)
    else:
        sl = pd.DataFrame(shortlist)
        best = sl["Total"].idxmin()

        st.markdown("#### Shortlist")
        st.dataframe(sl, hide_index=True, use_container_width=True)

        c1, c2 = st.columns([1.4, 1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sl["Scheme"], y=sl["Heating"], name="Heating",
                                 marker_color=HEAT))
            fig.add_trace(go.Bar(x=sl["Scheme"], y=sl["Cooling"], name="Cooling",
                                 marker_color=COOL))
            fig.update_layout(
                barmode="group", height=380, margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="kWh/m²", gridcolor=RULE),
                xaxis=dict(showgrid=False),
                font=dict(family="IBM Plex Sans", color=INK),
                legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            st.markdown("#### Lowest combined load")
            st.markdown(
                f'<div class="card"><div class="card-label">Recommended</div>'
                f'<div class="card-value" style="font-size:1.9rem">'
                f'{sl.loc[best, "Scheme"]}</div>'
                f'<div class="card-unit">{sl.loc[best, "Total"]:.1f} kWh/m² combined'
                f'</div></div>', unsafe_allow_html=True)
            spread = sl["Total"].max() - sl["Total"].min()
            err = (META["metrics"]["HeatingLoad"]["RMSE"]
                   + META["metrics"]["CoolingLoad"]["RMSE"])
            st.markdown(
                f'<div class="note" style="margin-top:.8rem">Spread across the '
                f'shortlist is <b>{spread:.1f} kWh/m²</b> against a combined error bar '
                f'of ±{err:.2f}. '
                + ("The ranking is well clear of model error."
                   if spread > 2 * err else
                   "<b>These schemes are within model error of each other</b> treat "
                   "the ranking as inconclusive and simulate to separate them.")
                + "</div>", unsafe_allow_html=True)
            if st.button("Clear shortlist", use_container_width=True):
                st.session_state["shortlist"] = []
                st.rerun()

#TAB 3 (sensitivity)
with tab3:
    st.markdown("#### Vary one parameter, hold the rest")
    st.markdown('<div class="note">Sweeps the chosen parameter across every value the '
                'model was trained on, keeping the current design fixed otherwise. '
                'This is the fastest way to see which levers actually move each '
                'load.</div>', unsafe_allow_html=True)

    sweep = st.radio("Parameter to sweep",
                     ["Glazing ratio", "Orientation", "Glazing layout", "Building shape"],
                     horizontal=True)

    rows = []
    if sweep == "Glazing ratio":
        for g in META["glazing_areas"]:
            d = 0 if g == 0 else glazing_distrib if glazing_distrib != 0 else 3
            h, c = predict(shape, g, d, orientation)
            rows.append({"x": f"{g:.0%}", "Heating": h, "Cooling": c})
    elif sweep == "Orientation":
        for o in META["orientations"]:
            h, c = predict(shape, glazing_area, glazing_distrib, o)
            rows.append({"x": ORIENT_NAMES[o], "Heating": h, "Cooling": c})
    elif sweep == "Glazing layout":
        for d in META["glazing_distribs"]:
            if d == 0:
                continue
            h, c = predict(shape, max(glazing_area, 0.1), d, orientation)
            rows.append({"x": GLAZING_LAYOUTS[d].split(" - ")[0],
                         "Heating": h, "Cooling": c})
    else:
        for s in SHAPES:
            h, c = predict(s, glazing_area, glazing_distrib, orientation)
            rows.append({"x": f'{s["RelCompactness"]:.2f}', "Heating": h, "Cooling": c})

    sw = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sw["x"], y=sw["Heating"], name="Heating",
                             mode="lines+markers", line=dict(color=HEAT, width=3),
                             marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=sw["x"], y=sw["Cooling"], name="Cooling",
                             mode="lines+markers", line=dict(color=COOL, width=3),
                             marker=dict(size=9)))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      yaxis=dict(title="kWh/m²", gridcolor=RULE),
                      xaxis=dict(title=sweep, showgrid=False),
                      font=dict(family="IBM Plex Sans", color=INK),
                      legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="note">Heating swing across this sweep: '
                f'<b style="color:{HEAT}">'
                f'{sw["Heating"].max() - sw["Heating"].min():.2f} kWh/m²</b></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="note">Cooling swing across this sweep: '
                f'<b style="color:{COOL}">'
                f'{sw["Cooling"].max() - sw["Cooling"].min():.2f} kWh/m²</b></div>',
                unsafe_allow_html=True)

#TAB 4 (model card)
with tab4:
    c1, c2 = st.columns([1, 1.25])

    with c1:
        st.markdown("#### Accuracy on held-out designs")
        perf = pd.DataFrame([
            {"Load": "Heating", "Model": META["metrics"]["HeatingLoad"]["model_name"],
             "RMSE": round(META["metrics"]["HeatingLoad"]["RMSE"], 3),
             "R²": round(META["metrics"]["HeatingLoad"]["R2"], 4),
             "Typical error": f'{META["metrics"]["HeatingLoad"]["MAPE"]:.1f}%'},
            {"Load": "Cooling", "Model": META["metrics"]["CoolingLoad"]["model_name"],
             "RMSE": round(META["metrics"]["CoolingLoad"]["RMSE"], 3),
             "R²": round(META["metrics"]["CoolingLoad"]["R2"], 4),
             "Typical error": f'{META["metrics"]["CoolingLoad"]["MAPE"]:.1f}%'}])
        st.dataframe(perf, hide_index=True, use_container_width=True)
        st.markdown(
            f'<div class="note">Measured on {META["n_test"]} designs held out of '
            f'training. Two different algorithms are used because they were each the '
            f'best on their own load, forcing one on both cost 13% accuracy on '
            f'cooling.</div>', unsafe_allow_html=True)

        st.markdown("#### Where the tool stops working")
        ex = META["extrapolation"]
        st.markdown(
            f'<span class="badge badge-warn">Validity envelope</span>'
            f'<div class="note" style="margin-top:.5rem">Asked to predict a massing it '
            f'has never seen, error rises to <b>{ex["HeatingLoad"]["grouped_rmse"]:.1f}'
            f'</b> kWh/m² for heating and <b>{ex["CoolingLoad"]["grouped_rmse"]:.1f}'
            f'</b> for cooling its roughly ten times worse, and no longer useful. That is '
            f'why the shape selector offers only the 12 validated massings. Take a novel '
            f'massing to a full EnergyPlus run.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="note" style="margin-top:.8rem">All 768 training simulations '
            'used a single climate file, so orientation effects here are specific to '
            'that location and do not transfer to another latitude.</div>',
            unsafe_allow_html=True)

    with c2:
        st.markdown("#### What drives each load")
        imp = (pd.DataFrame(META["importance"])
                 .rename(index=FEATURE_LABELS)
                 .sort_values("CoolingLoad"))
        fig = go.Figure()
        fig.add_trace(go.Bar(y=imp.index, x=imp["HeatingLoad"], name="Heating",
                             orientation="h", marker_color=HEAT))
        fig.add_trace(go.Bar(y=imp.index, x=imp["CoolingLoad"], name="Cooling",
                             orientation="h", marker_color=COOL))
        fig.update_layout(barmode="group", height=520,
                          margin=dict(l=0, r=0, t=30, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(title="Loss of accuracy if unknown (kWh/m²)",
                                     gridcolor=RULE),
                          yaxis=dict(showgrid=False),
                          font=dict(family="IBM Plex Sans", color=INK, size=12),
                          legend=dict(orientation="h", y=1.07, x=0))
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(
            '<div class="note"><b>The two loads answer to different levers.</b> '
            'Orientation and glazing layout matter roughly 4–7× more for cooling than '
            'for heating, the signature of solar gain. Heating is set by envelope '
            'geometry instead. Practically: fix the massing first, since it drives '
            'both, then tune orientation and glazing for summer performance, where '
            'they are nearly free in winter terms.</div>', unsafe_allow_html=True)
