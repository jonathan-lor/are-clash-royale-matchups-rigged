from MatchupTable import MatchupTable
from dotenv import load_dotenv
import os
import plotly.express as px

def make_labels(n, mapping):
    # Build a list of labels for indices [0..n-1], falling back to the index if missing.
    return [mapping.get(i, str(i)) for i in range(n)]

def main():
    load_dotenv()

    api_key = os.getenv("API_KEY")
    base_url = os.getenv("API_BASE_URL")

    mt = MatchupTable(api_key=api_key, base_url=base_url)

    mt.load_from_csv('card_matchups_from_recent_ranked_games_of_08_2025_top_200_on_10_02_2025.csv')

    winrates = mt.get_winrates_table()

    idx_to_name = mt.index_to_card_name

    n_rows, n_cols = winrates.shape
    row_labels = make_labels(n_rows, idx_to_name)
    col_labels = make_labels(n_cols, idx_to_name)

    # Figure

    fig = px.imshow(
        winrates,
        origin="upper",        # (0,0) at top-left
        aspect="equal",        # square cells
        labels=dict(x="Column", y="Row", color="Value"),
        x=col_labels,          # bottom labels
        y=row_labels,          # left labels
    )

    # Show all labels (make the canvas big + fonts small)
    fig.update_xaxes(
        tickmode="array",
        tickvals=col_labels,
        ticktext=col_labels,
        tickangle=45,          # tilt for readability
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=row_labels,
        ticktext=row_labels,
        autorange="reversed",  # matches origin="upper"
    )

    # Hover shows string labels + the value
    fig.update_traces(
        #hovertemplate="Row %{y}<br>Col %{x}<br>Value %{z:.4f}<extra></extra>"
        hovertemplate="%{y} has<br>%{z:.4f} winrate<br>vs %{x}"
    )

    # tune these to change size
    FIG_SIZE = 2000
    TICK_FONT_SIZE = 8

    fig.update_layout(
        width=FIG_SIZE,
        height=FIG_SIZE,
        margin=dict(l=180, r=40, t=40, b=180),
    )

    fig.update_xaxes(tickfont=dict(size=TICK_FONT_SIZE))
    fig.update_yaxes(tickfont=dict(size=TICK_FONT_SIZE))

    fig.update_layout(coloraxis_colorbar=dict(title="Winrate", tickfont=dict(size=10)))

    fig.write_html("card_matchups_from_recent_ranked_games_of_08_2025_top_200_on_10_02_2025_heatmap.html", include_plotlyjs=True, full_html=True)


if __name__ == "__main__":
    main()