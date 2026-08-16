import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import altair as alt
    import numpy as np
    import pandas as pd

    return alt, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Foster to Cauer Model Conversion
    """)
    return


@app.cell
def _(np):
    times = np.logspace(-3, 2, 500)
    return (times,)


@app.cell
def _(mo, pd):
    foster_df = pd.DataFrame(
        columns=['time', 'rth'],
        data=[[1.23, 6.22e-3], [0.235, 2.465e-3], [0.061, 0.723e-3], [0.0092, 0.607e-3]]
    )
    get_data, set_data = mo.state(foster_df)
    return foster_df, get_data, set_data


@app.cell
def _(get_data, mo, set_data):
    data_edited = mo.ui.data_editor(
        data=get_data(),
        on_change=lambda df: set_data(df)
    )
    return (data_edited,)


@app.cell
def _(foster_df, mo, set_data):
    restore_defaults = mo.ui.button(
        on_click=lambda _: set_data(foster_df),
        label='Restore defaults'
    )
    return (restore_defaults,)


@app.cell
def _(data_edited, mo, restore_defaults):
    mo.vstack([
        data_edited,
        restore_defaults
    ])
    return


@app.cell
def _(data_edited, pd):
    edited_df = pd.DataFrame(data_edited.value)
    return (edited_df,)


@app.cell
def _(pd, times):
    points_df = pd.DataFrame(columns=['time'], data=times)
    return (points_df,)


@app.cell
def _(edited_df, np, pd, points_df):
    _t = points_df['time'].values
    _tau = edited_df['time'].values
    _rth = edited_df['rth'].values
    terms = _rth[None, :] * (1 - np.exp(-_t[:, None] / _tau[None, :]))

    # wide: one column per Foster term
    _term_cols = pd.DataFrame(
        terms,
        columns=[f'R{i+1} (τ={tau_i:.3g}s)' for i, tau_i in enumerate(_tau)]
    )

    plot_df = points_df.assign(Rthjc=terms.sum(axis=1))
    plot_df = pd.concat([plot_df, _term_cols], axis=1)
    return (plot_df,)


@app.cell
def _(alt, mo, plot_df):
    term_cols = [c for c in plot_df.columns if c not in ('time', 'Rthjc')]
    terms_long = plot_df.melt(
        id_vars='time', value_vars=term_cols,
        var_name='component', value_name='Rth'
    )

    individual_lines = alt.Chart(terms_long).mark_line(strokeDash=[4, 2], opacity=0.6).encode(
        x=alt.X('time:Q', scale=alt.Scale(type='log'), title='Time (s)'),
        y=alt.Y('Rth:Q', title='Rth (°C/kW)'),
        color=alt.Color('component:N', title='Foster term')
    )

    total_line = alt.Chart(plot_df).mark_line(color='black', strokeWidth=3).encode(
        x=alt.X('time:Q', scale=alt.Scale(type='log')),
        y='Rthjc:Q'
    )

    chart = (individual_lines + total_line).properties(
        title='Foster Network - Individual Terms and Total'
    )
    mo.ui.altair_chart(chart)
    return


if __name__ == "__main__":
    app.run()
