# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.2.2",
#     "marimo>=0.23.16",
#     "numpy==2.5.2",
#     "pandas==3.0.5",
#     "pyarrow==25.0.1",
#     "sympy==1.14.0",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="Foster to Cauer Conversion")

with app.setup(hide_code=True):
    import marimo as mo


@app.cell(hide_code=True)
def _():
    import altair as alt
    import numpy as np
    import pandas as pd
    import sympy as sp

    return alt, np, pd, sp


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Foster to Cauer Model Conversion

    Most power semiconductor datasheets provide their transient thermal resistance coefficients as Foster coefficients. While the Foster model is easier to fit mathematically, it is less physically representative than the Cauer model, so it is desirable to convert between the two.

    The default device is a [Dynex DCR4420H65](https://www.dynexsemi.com/Portals/0/assets/downloads/DNX_DCR4420H65.pdf) thyristor, with the double-side coefficients used.
    """)
    return


@app.cell(hide_code=True)
def _(pd):
    foster_df = pd.DataFrame(
        columns=['time', 'rth'],
        data=[[0.67, 1.248], [0.146, 0.833], [0.02, 0.606], [1.287, 1.568]]
    )
    get_data, set_data = mo.state(foster_df)
    return foster_df, get_data, set_data


@app.cell(hide_code=True)
def _(get_data, set_data):
    data_edited = mo.ui.data_editor(
        data=get_data(),
        on_change=lambda df: set_data(df)
    )
    return (data_edited,)


@app.cell(hide_code=True)
def _(foster_df, set_data, set_rth_c_kw):
    restore_defaults = mo.ui.button(
        on_click=lambda _: (set_data(foster_df), set_rth_c_kw(True)),
        label='Restore defaults'
    )
    return (restore_defaults,)


@app.cell(hide_code=True)
def _():
    get_rth_c_kw, set_rth_c_kw = mo.state(True)
    return get_rth_c_kw, set_rth_c_kw


@app.cell(hide_code=True)
def _(get_rth_c_kw, set_rth_c_kw):
    rth_c_kw = mo.ui.checkbox(
        label='Rth in ºC/kW',
        value=get_rth_c_kw(),
        on_change=lambda v: set_rth_c_kw(v)
    )
    return (rth_c_kw,)


@app.cell(hide_code=True)
def _(data_edited, restore_defaults, rth_c_kw):
    mo.vstack([
        data_edited,
        restore_defaults,
        rth_c_kw
    ])
    return


@app.cell(hide_code=True)
def _(data_edited, pd):
    edited_df = pd.DataFrame(data_edited.value)
    return (edited_df,)


@app.cell(hide_code=True)
def _(edited_df, np):
    min_pow_10 = np.floor(np.log10(edited_df['time'].min())) - 1
    max_pow_10 = np.ceil(np.log10(edited_df['time'].max())) + 1

    times = np.logspace(min_pow_10, max_pow_10, 500)
    return (times,)


@app.cell(hide_code=True)
def _(pd, times):
    points_df = pd.DataFrame(columns=['time'], data=times)
    return (points_df,)


@app.cell(hide_code=True)
def _(edited_df, np, pd, points_df, rth_c_kw):
    if rth_c_kw.value:
        calc_df = edited_df.assign(rth=edited_df['rth'] / 1000)
    else:
        calc_df = edited_df

    _t = points_df['time'].values     # <-- the 500-point sweep, not calc_df
    _rth = calc_df['rth'].values      # the (possibly scaled) Foster resistances
    _tau = calc_df['time'].values     # the Foster time constants — also should come from calc_df for consistency, though time/tau isn't affected by the unit toggle

    terms = _rth[None, :] * (1 - np.exp(-_t[:, None] / _tau[None, :]))

    _term_cols = pd.DataFrame(
        terms,
        columns=[f'R{i+1} (τ={tau_i:.3g}s)' for i, tau_i in enumerate(_tau)]
    )

    plot_df = points_df.assign(Rthjc=terms.sum(axis=1))
    plot_df = pd.concat([plot_df, _term_cols], axis=1)
    return calc_df, plot_df


@app.cell(hide_code=True)
def _(alt, plot_df):
    term_cols = [c for c in plot_df.columns if c not in ('time', 'Rthjc')]
    terms_long = plot_df.melt(
        id_vars='time', value_vars=term_cols,
        var_name='component', value_name='Rth'
    )

    individual_lines = alt.Chart(terms_long).mark_line(strokeDash=[4, 2], opacity=0.6).encode(
        x=alt.X('time:Q', scale=alt.Scale(type='log'), title='Time (s)'),
        y=alt.Y('Rth:Q', title='Rth (°C/W)'),
        color=alt.Color('component:N', title='Foster term'),
        tooltip=[
            alt.Tooltip('time:Q', title='Time (s)', format='.3e'),
            alt.Tooltip('Rth:Q', title='Rth (°C/W)', format='.3e'),
            alt.Tooltip('component:N', title='Foster term')
        ]
    )

    total_line = alt.Chart(plot_df).mark_line(color='black', strokeWidth=3, tooltip=True).encode(
        x=alt.X('time:Q', scale=alt.Scale(type='log'), title='Time (s)'),
        y=alt.Y('Rthjc:Q', title='Rth (°C/W)'),
        tooltip=[
            alt.Tooltip('time:Q', title='Time (s)', format='.3e'),
            alt.Tooltip('Rthjc:Q', title='Rth (°C/W)', format='.3e')
        ]
    )

    chart = (individual_lines + total_line).properties(
        title='Foster Network - Individual Terms and Total'
    )
    mo.ui.altair_chart(chart)
    return


@app.cell(hide_code=True)
def _(calc_df, pd, sp):
    cauer_df = pd.DataFrame(columns=['r', 'c'])
    cauer_list = []
    cauer_idx = []
    s = sp.symbols('s')

    # Calculate the initial Foster transfer function
    transfer_fcn = 0
    for count, row in enumerate(calc_df.itertuples()):
        transfer_fcn += row.rth / (row.time * s + 1)

    # Simplify into a single function and calculate the numerator/denominator
    numerator, denominator = sp.fraction(sp.simplify(sp.together(transfer_fcn)))

    for i in range(count + 1, 0, -1):
        cauer_idx.append(i)
        quotient, remainder = sp.div(denominator, numerator, s)

        kn = quotient.coeff(s, 0)
        resistance = 1 / kn
        capacitance = quotient.coeff(s, 1)
        numerator, denominator = sp.fraction(sp.simplify(-(remainder / kn) / (kn * numerator + remainder)))
        cauer_list.append([float(resistance), float(capacitance)])

    cauer_df = pd.DataFrame(columns=['r', 'c'], data=cauer_list, index=cauer_idx)
    cauer_df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### References

    [1] T. G. Subhash Joshi and V. John, ‘Combined transient thermal impedance estimation for pulse-power applications’, in 2017 National Power Electronics Conference (NPEC), Pune: IEEE, Dec. 2017, pp. 42–47. doi: [10.1109/NPEC.2017.8310432](https://ieeexplore.ieee.org/document/8310432).
    """)
    return


if __name__ == "__main__":
    app.run()
