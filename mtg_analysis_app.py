import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------
st.set_page_config(page_title="MTG Stats Analysis", layout="wide")

# ----------------------------------------------------
# Data Load Function
# ----------------------------------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')  # Adjust if needed
    return df

DATA_FILE = 'mtg_data.xlsx'  # Ensure this file is in the same directory


# ----------------------------------------------------
# Summary Stats Page
# ----------------------------------------------------
def summary_stats(df, chart2_view, chart2_selected_players):
    st.title("Summary Stats")

    # Enhanced Summary Boxes (original logic)
    def create_summary_boxes_enhanced(filtered_df):
        # Calculate average positions per player
        avg_positions = filtered_df.groupby('player')['position'].mean().reset_index()
        avg_positions = avg_positions.sort_values('position').reset_index(drop=True)
        top4 = avg_positions.head(4)

        box_colors = ['skyblue', 'orange', 'silver', 'gold']
        box_icons  = ['4️⃣', '🥉', '🥈', '🥇']

        cols = st.columns(4)
        # Reverse so 1st place is on the right
        top4 = top4[::-1].reset_index(drop=True)
        ordered_players = list(top4['player'])

        for i, col in enumerate(cols):
            if i < len(top4):
                player = top4.iloc[i]['player']
                avg_pos = round(top4.iloc[i]['position'], 2)
                color   = box_colors[i]
                icon    = box_icons[i]
                with col:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {color};
                            color: black;
                            padding: 20px;
                            border-radius: 10px;
                            text-align: center;
                            font-size: 18px;
                            font-weight: bold;
                        ">
                            {icon} {player}<br>
                            Avg Position: {avg_pos}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                color = box_colors[i]
                icon  = box_icons[i]
                with col:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {color};
                            color: black;
                            padding: 20px;
                            border-radius: 10px;
                            text-align: center;
                            font-size: 18px;
                            font-weight: bold;
                        ">
                            {icon} N/A<br>
                            Avg Position: N/A
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        return ordered_players

    # Show summary boxes
    ordered_players = create_summary_boxes_enhanced(df)
    st.markdown("---")

    st.header("Detailed Analysis (Ranks Distribution)")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    # === Chart 1: Distribution of Ranks 1-4 by Player ===
    chart1_df = df[df['position'].isin([1, 2, 3, 4])].copy()
    rank_map  = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th'}
    chart1_df['rank_char'] = chart1_df['position'].map(rank_map)

    chart1_grouped = chart1_df.groupby(['player','rank_char']).size().reset_index(name='count')
    chart1_grouped['percentage'] = chart1_grouped.groupby('player')['count'].transform(lambda x: (x / x.sum()) * 100)

    # Enforce player order from summary boxes
    chart1_grouped['player'] = pd.Categorical(chart1_grouped['player'], categories=ordered_players, ordered=True)

    rank_colors = {'1st': 'gold','2nd': 'silver','3rd': 'orange','4th': 'skyblue'}

    fig1 = px.bar(
        chart1_grouped,
        x='player',
        y='percentage',
        color='rank_char',
        color_discrete_map=rank_colors,
        category_orders={'rank_char': ['1st','2nd','3rd','4th'], 'player': ordered_players},
        barmode='stack',
        title="Distribution of Ranks 1-4 by Player",
        labels={'percentage': 'Percentage (%)', 'player': 'Player', 'rank_char': 'Rank'},
        hover_data={'player': True,'rank_char': True,'percentage': ':.2f'}
    )
    fig1.update_layout(
        legend_title_text='Rank',
        yaxis=dict(range=[0, 100], dtick=20),
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    for trace in fig1.data:
        trace.text = [f"{val:.1f}%" if val > 0 else "" for val in trace.y]
        trace.textposition = 'inside'

    # === Chart 2: Average Position OR Distribution of Decks ===
    avg_position_deck = df.groupby('deck')['position'].mean().reset_index()
    avg_position_deck['avg_position'] = avg_position_deck['position'].round(2)
    total_games_deck_chart2 = df.groupby('deck').size().reset_index(name='total_games')
    avg_position_deck = avg_position_deck.merge(total_games_deck_chart2, on='deck', how='left')
    avg_position_deck = avg_position_deck.sort_values('avg_position', ascending=False).reset_index(drop=True)
    deck_order = list(avg_position_deck['deck'])

    if len(chart2_selected_players) == 0:
        chart2_warning = True
    else:
        chart2_warning = False
        chart2_filtered_df = df[df['player'].isin(chart2_selected_players)]

        if chart2_view == "Average Position":
            avg_position_deck_chart2 = chart2_filtered_df.groupby('deck')['position'].mean().reset_index()
            avg_position_deck_chart2['avg_position'] = avg_position_deck_chart2['position'].round(2)

            total_games_deck_chart2_selected = chart2_filtered_df.groupby('deck').size().reset_index(name='total_games')
            avg_position_deck_chart2 = avg_position_deck_chart2.merge(total_games_deck_chart2_selected, on='deck', how='left')
            avg_position_deck_chart2 = avg_position_deck_chart2.sort_values('avg_position', ascending=False).reset_index(drop=True)

            fig2 = px.bar(
                avg_position_deck_chart2,
                x='deck',
                y='avg_position',
                title="Average Position per Deck",
                labels={'avg_position': 'Average Position','deck': 'Deck'},
                hover_data={'deck': True,'avg_position': ':.2f'},
                color_discrete_sequence=['skyblue']*len(avg_position_deck_chart2),
            )
            fig2.update_layout(
                yaxis=dict(range=[1, 4], dtick=0.5),
                uniformtext_minsize=8,
                uniformtext_mode='hide',
                showlegend=False
            )
            fig2.update_traces(
                text=avg_position_deck_chart2['avg_position'],
                textposition='outside'
            )
            annotations = []
            for idx, row in avg_position_deck_chart2.iterrows():
                annotations.append(dict(
                    x=row['deck'],
                    y=row['avg_position'] + 0.1,
                    text=f"n={row['total_games']}",
                    showarrow=False,
                    font=dict(size=12, color="black")
                ))
            fig2.update_layout(annotations=annotations)

        else:  # "Distribution"
            chart2_distribution_df = chart2_filtered_df[chart2_filtered_df['position'].isin([1,2,3,4])].copy()
            chart2_distribution_df['rank_char'] = chart2_distribution_df['position'].map(rank_map)

            chart2_grouped = chart2_distribution_df.groupby(['deck','rank_char']).size().reset_index(name='count')
            chart2_grouped['percentage'] = chart2_grouped.groupby('deck')['count'].transform(lambda x: (x / x.sum()) * 100)
            chart2_grouped['deck'] = pd.Categorical(chart2_grouped['deck'], categories=deck_order, ordered=True)

            rank_colors_distribution = {'1st':'gold','2nd':'silver','3rd':'orange','4th':'skyblue'}

            fig2 = px.bar(
                chart2_grouped,
                x='deck',
                y='percentage',
                color='rank_char',
                color_discrete_map=rank_colors_distribution,
                category_orders={'rank_char': ['1st','2nd','3rd','4th'], 'deck': deck_order},
                barmode='stack',
                title="Distribution of Ranks 1-4 by Deck",
                labels={'percentage': 'Percentage (%)','deck': 'Deck','rank_char': 'Rank'},
                hover_data={'deck': True,'rank_char': True,'percentage': ':.2f'}
            )
            fig2.update_layout(
                legend_title_text='Rank',
                yaxis=dict(range=[0, 100], dtick=20),
                uniformtext_minsize=8,
                uniformtext_mode='hide'
            )
            for trace in fig2.data:
                trace.text = [f"{val:.1f}%" if val > 0 else "" for val in trace.y]
                trace.textposition = 'inside'

    # Display side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        if chart2_warning:
            st.warning("Please select at least one player to view Chart 2.")
        else:
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("© 2025 MTG Stats Analysis | Built with ❤️ using Streamlit")


# ----------------------------------------------------
# Detailed Stats Page
# ----------------------------------------------------

def detailed_stats(filtered_df):
    st.title("Detailed Stats")

    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selections.")
        return

    # Define stats info
    stats_info = {
        'damage_dealt': {
            'icon': '⚔️',
            'color': '#FFB3B3',  # Light Red
            'label': 'Damage Dealt'
        },
        'creature_killed': {
            'icon': '🐲',
            'color': '#FFE0B3',  # Light Orange
            'label': 'Creatures Killed'
        },
        'health_gained': {
            'icon': '❤️',
            'color': '#B3FFB3',  # Light Green
            'label': 'Health Gained'
        },
        'vip_kill': {
            'icon': '🎯',
            'color': '#B3D9FF',  # Light Blue
            'label': 'VIP Kills'
        },
        'damage_received': {
            'icon': '🛡️',
            'color': '#FFCCE0',  # Pink
            'label': 'Damage Received'
        },
        'creature_lost': {
            'icon': '💀',
            'color': '#FFE6B3',  # Light Peach
            'label': 'Creatures Lost'
        },
        'health_lost': {
            'icon': '💔',
            'color': '#B3FFE6',  # Light Teal
            'label': 'Health Lost'
        },
        'vip_loss': {
            'icon': '😵',
            'color': '#E6B3FF',  # Light Purple
            'label': 'VIP Losses'
        },
    }

    # Keep them in the exact order needed for display
    metrics = [
        'damage_dealt', 'creature_killed', 'health_gained', 'vip_kill',
        'damage_received', 'creature_lost', 'health_lost', 'vip_loss'
    ]

    # Sum up these metrics by player for the summary boxes
    sums_df = filtered_df.groupby('player')[metrics].sum(numeric_only=True).reset_index()

    # Create the first row of 4 boxes with horizontal gap and margin in style
    row1 = st.columns(4, gap="medium")
    for i in range(4):
        m      = metrics[i]
        icon   = stats_info[m]['icon']
        color  = stats_info[m]['color']
        label  = stats_info[m]['label']
        top_id = sums_df[m].idxmax()
        top_player = sums_df.loc[top_id, 'player']
        top_value  = sums_df.loc[top_id, m]
        with row1[i]:
            st.markdown(f"""
                <div style="
                    background-color: {color};
                    color: black;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 10px;
                ">
                    {icon} {label}<br>
                    {top_player}: {int(top_value)}
                </div>
            """, unsafe_allow_html=True)

    # Add a vertical gap between rows
    st.markdown("<br>", unsafe_allow_html=True)

    # Create the second row of 4 boxes
    row2 = st.columns(4, gap="medium")
    for i in range(4, 8):
        m      = metrics[i]
        icon   = stats_info[m]['icon']
        color  = stats_info[m]['color']
        label  = stats_info[m]['label']
        top_id = sums_df[m].idxmax()
        top_player = sums_df.loc[top_id, 'player']
        top_value  = sums_df.loc[top_id, m]
        with row2[i - 4]:
            st.markdown(f"""
                <div style="
                    background-color: {color};
                    color: black;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 10px;
                ">
                    {icon} {label}<br>
                    {top_player}: {int(top_value)}
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Metric Comparisons")

    # New toggle for selecting aggregation mode: Total vs. Average
    agg_mode = st.radio("Select Aggregation Mode", ["Total", "Average"], horizontal=True)

    # Helper function to create grouped bar charts
    def create_grouped_bar_chart(data, metrics_pair, title, mode="Total"):
        if mode == "Total":
            grouped = data.groupby('player')[metrics_pair].sum(numeric_only=True).reset_index()
            y_label = "Total"
            text_template = '%{text:d}'
        else:
            grouped = data.groupby('player')[metrics_pair].mean(numeric_only=True).reset_index()
            y_label = "Average"
            text_template = '%{text:.2f}'
        melted = grouped.melt(id_vars='player', var_name='metric', value_name='value')
        fig = px.bar(
            melted,
            x='player',
            y='value',
            color='metric',
            barmode='group',
            title=title,
            text='value'  # This adds the text label on top of each bar
        )
        fig.update_layout(xaxis_title="Player", yaxis_title=y_label)
        fig.update_traces(texttemplate=text_template, textposition='outside')
        return fig

    # Create side-by-side bar charts for metric comparisons using the selected mode
    col1, col2 = st.columns(2)
    with col1:
        fig1 = create_grouped_bar_chart(filtered_df, ['damage_dealt', 'damage_received'],
                                        "Damage Dealt vs. Received", mode=agg_mode)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = create_grouped_bar_chart(filtered_df, ['health_gained', 'health_lost'],
                                        "Health Gained vs. Lost", mode=agg_mode)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = create_grouped_bar_chart(filtered_df, ['creature_killed', 'creature_lost'],
                                        "Creatures Killed vs. Lost", mode=agg_mode)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = create_grouped_bar_chart(filtered_df, ['vip_kill', 'vip_loss'],
                                        "VIP Kills vs. Losses", mode=agg_mode)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("© 2025 MTG Stats Analysis | Built with ❤️ using Streamlit")


# ----------------------------------------------------
# Main App: Page Navigation
# ----------------------------------------------------
def main():
    try:
        df = load_data(DATA_FILE)
    except FileNotFoundError:
        st.error(f"Excel file '{DATA_FILE}' not found. Please ensure it's in the same directory.")
        st.stop()
        return  # Ensure no further code is executed
    except Exception as e:
        st.error(f"An error occurred while loading the Excel file: {e}")
        st.stop()
        return  # Ensure no further code is executed

    # Basic data cleanup
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    df['deck'] = df['deck'].replace({'Elven': 'Elves'})

    # ---------------
    # 1) Page Selection (FIRST in sidebar)
    # ---------------
    page = st.sidebar.selectbox("Select Page", ["Summary Stats", "Detailed Stats"])

    # ---------------
    # 2) Filter Options
    # ---------------
    st.sidebar.header("Filter Options")
    with st.sidebar.expander("Main Filters", expanded=True):
        type_options = sorted(df['type'].dropna().unique())
        selected_type = st.multiselect("Select Type", options=type_options, default=type_options)

        possible_decks = df[df['type'].isin(selected_type)]['deck'].dropna().unique()
        possible_decks = sorted(possible_decks)
        selected_deck  = st.multiselect("Select Deck", options=possible_decks, default=possible_decks)

    # Only show Chart 2 Options if on Summary Stats
    if page == "Summary Stats":
        with st.sidebar.expander("Chart 2 Options", expanded=False):
            chart2_view = st.radio(
                "Select Chart 2 View:",
                options=["Average Position", "Distribution"],
                index=0,
                horizontal=True
            )
            chart2_player_options = sorted(df['player'].dropna().unique())
            chart2_selected_players = st.multiselect(
                "Select Player(s) for Chart 2",
                options=chart2_player_options,
                default=chart2_player_options
            )
    else:
        chart2_view = None
        chart2_selected_players = []

    # Apply main filters
    filtered_df = df[
        (df['type'].isin(selected_type)) &
        (df['deck'].isin(selected_deck))
    ].copy()

    # ---------------
    # Render Page
    # ---------------
    if page == "Summary Stats":
        summary_stats(filtered_df, chart2_view, chart2_selected_players)
    else:
        detailed_stats(filtered_df)


# ----------------------------------------------------
# Run the App
# ----------------------------------------------------
if __name__ == "__main__":
    main()
