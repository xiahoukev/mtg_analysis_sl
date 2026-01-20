# -------------------- LIBRARIES --------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import numpy as np

# -------------------- CONSTANTS & CONFIG --------------------
DATA_FILE = "mtg_data.xlsx"
APP_TITLE = "Magic: The Gathering Stats Analysis"
APP_VERSION = "1.6.0"

# Rank Colours (Global Definition)
RANK_COLORS = {'1st': 'gold', '2nd': 'silver', '3rd': 'orange', '4th': 'skyblue'}
RANK_ORDER = ['1st', '2nd', '3rd', '4th']

# Consistent Player Palette (Plotly G10 + Dark24 for variety)
PLAYER_PALETTE = px.colors.qualitative.G10 + px.colors.qualitative.Dark24

st.set_page_config(page_title="MTG Stats Analysis", layout="wide", page_icon="🃏")

# -------------------- FUNCTIONS --------------------

def _file_md5(path: str) -> str:
    """Get stable file-content key (invalidates cache when file changes)."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return "MISSING"

@st.cache_data(show_spinner=False)
def load_data(file_path: str, version_key: str) -> pd.DataFrame:
    """Load data from Excel."""
    return pd.read_excel(file_path, sheet_name="Sheet1")

def get_valid_options(df, column):
    """Helper to get sorted unique options from a dataframe column."""
    return sorted(df[column].dropna().unique())

def enhance_game_ids(df):
    """
    SMART ENGINE: Splits games that share the same game_id but are actually sequential matches.
    """
    df = df.sort_values('game_id')
    match_ids = []
    current_game_id = None
    current_players = set()
    current_match_num = 0
    
    for _, row in df.iterrows():
        g_id = row['game_id']
        player = row['player']
        
        if g_id != current_game_id:
            current_game_id = g_id
            current_match_num += 1
            current_players = {player}
        else:
            if player in current_players:
                current_match_num += 1
                current_players = {player}
            else:
                current_players.add(player)
        
        match_ids.append(current_match_num)
    
    df['match_uuid'] = match_ids
    return df

def calculate_elo(df):
    """
    Calculates Elo ratings for 4-player Free-For-All.
    Starting Elo: 1200 | K-Factor: 32
    """
    df = df.sort_values('match_uuid')
    players = df['player'].unique()
    ratings = {p: 1200.0 for p in players}
    history = []
    
    for match_id, game in df.groupby('match_uuid'):
        match_res = game[['player', 'position']].to_dict('records')
        current_match_ratings = {p: ratings.get(p, 1200) for p in [r['player'] for r in match_res]}
        
        # Snapshot for history
        for p_data in match_res:
            history.append({
                'match_uuid': match_id,
                'game_id': game['game_id'].iloc[0],
                'player': p_data['player'],
                'elo': current_match_ratings[p_data['player']],
                'position': p_data['position']
            })
            
        # Update Logic
        k = 32
        for i in range(len(match_res)):
            for j in range(i + 1, len(match_res)):
                p1, pos1 = match_res[i]['player'], match_res[i]['position']
                p2, pos2 = match_res[j]['player'], match_res[j]['position']
                
                r1, r2 = current_match_ratings[p1], current_match_ratings[p2]
                
                e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
                e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
                
                if pos1 < pos2: s1, s2 = 1, 0
                elif pos1 > pos2: s1, s2 = 0, 1
                else: s1, s2 = 0.5, 0.5
                
                ratings[p1] += k * (s1 - e1)
                ratings[p2] += k * (s2 - e2)
                
    return pd.DataFrame(history), ratings

def hex_to_rgba(hex_color, opacity=0.2):
    """Helper to convert hex code to rgba string for Plotly fills."""
    hex_color = hex_color.lstrip('#')
    return f"rgba{tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (opacity,)}"

def get_best_worst(df, category_col):
    """Helper to find Best (Lowest Avg Pos) and Worst (Highest Avg Pos) categories."""
    if df.empty: return "N/A", "N/A"
    stats = df.groupby(category_col)['position'].mean().reset_index()
    if stats.empty: return "N/A", "N/A"
    
    best = stats.loc[stats['position'].idxmin()][category_col]
    worst = stats.loc[stats['position'].idxmax()][category_col]
    return best, worst

# -------------------- DATA LOADING & PRE-PROCESSING --------------------
try:
    raw_df = load_data(DATA_FILE, _file_md5(DATA_FILE))
    
    # Cleaning
    df_obj = raw_df.select_dtypes(['object'])
    raw_df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    raw_df['position'] = pd.to_numeric(raw_df['position'], errors='coerce')
    raw_df['deck'] = raw_df['deck'].replace({'Elven': 'Elves'}) 
    raw_df['color_simple'] = raw_df['primary_mana'].astype(str).apply(lambda x: x.split(' ')[0])
    
    # Logic
    raw_df = enhance_game_ids(raw_df)
    elo_history_df, current_elo = calculate_elo(raw_df)
    
    # Defaults & Colour Mapping
    all_players_def = get_valid_options(raw_df, 'player')
    
    # GLOBAL COLOUR CONSISTENCY: Map every player to a fixed colour
    unique_players = sorted(raw_df['player'].unique())
    player_color_map = {player: PLAYER_PALETTE[i % len(PLAYER_PALETTE)] for i, player in enumerate(unique_players)}

except FileNotFoundError:
    st.error(f"Excel file '{DATA_FILE}' not found. Place it beside this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading Excel: {e}")
    st.stop()

# -------------------- SIDEBAR & FILTERS --------------------

if 'reset_trigger' not in st.session_state:
    st.session_state.reset_trigger = False

def reset_callbacks():
    st.session_state['f_player'] = all_players_def
    st.session_state['f_draw'] = get_valid_options(raw_df, 'draw_type')
    st.session_state['f_type'] = get_valid_options(raw_df, 'type')
    st.session_state['f_deck'] = get_valid_options(raw_df, 'deck')
    st.session_state['f_color'] = get_valid_options(raw_df, 'primary_mana')

st.sidebar.title("Navigation")
# CHANGED: Added Player vs Player to navigation
page = st.sidebar.radio("Go to:", ["Dashboard", "Detailed Player Analytics", "Player vs Player", "Deck & Set Analysis"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Global Filters")

if st.sidebar.button("Reset All Filters", on_click=reset_callbacks):
    pass 

with st.sidebar.expander("Filter Options", expanded=True):
    # 1. Player
    selected_players = st.multiselect("Player", options=all_players_def, default=all_players_def, key='f_player')
    df_f0 = raw_df[raw_df['player'].isin(selected_players)] if selected_players else raw_df.copy()

    # 2. Draw Type
    avail_draws = get_valid_options(df_f0, 'draw_type')
    selected_draws = st.multiselect("Draw Type", options=avail_draws, default=avail_draws, key='f_draw')
    df_f1 = df_f0[df_f0['draw_type'].isin(selected_draws)] if selected_draws else df_f0.copy()

    # 3. Format
    avail_types = get_valid_options(df_f1, 'type')
    selected_types = st.multiselect("Game Format / Type", options=avail_types, default=avail_types, key='f_type')
    df_f2 = df_f1[df_f1['type'].isin(selected_types)] if selected_types else df_f1.copy()

    # 4. Deck
    avail_decks = get_valid_options(df_f2, 'deck')
    selected_decks = st.multiselect("Deck", options=avail_decks, default=avail_decks, key='f_deck')
    df_f3 = df_f2[df_f2['deck'].isin(selected_decks)] if selected_decks else df_f2.copy()

    # 5. Colour
    avail_colors = get_valid_options(df_f3, 'primary_mana')
    selected_colors = st.multiselect("Primary Colour", options=avail_colors, default=avail_colors, key='f_color')
    shared_filtered_df = df_f3[df_f3['primary_mana'].isin(selected_colors)] if selected_colors else df_f3.copy()

# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if page == "Dashboard":
    st.title(f"{APP_TITLE} - Dashboard")
    dashboard_df = shared_filtered_df.copy()

    # --- TOP STATS ROW ---
    if not dashboard_df.empty:
        stats = dashboard_df.groupby('player')['position'].agg(
            avg_position='mean', total_games='count', wins=lambda x: (x == 1).sum()
        ).reset_index()
        stats['win_rate'] = (stats['wins'] / stats['total_games']) * 100
        stats = stats.sort_values('avg_position', ascending=True).reset_index(drop=True)
        top_players = stats.head(4)
        
        rank_styles = {0: {'c':'gold','i':'🥇'}, 1: {'c':'silver','i':'🥈'}, 2: {'c':'orange','i':'🥉'}, 3: {'c':'skyblue','i':'4️⃣'}}
        cols = st.columns(4)
        
        for i, col in enumerate(cols):
            with col:
                if i < len(top_players):
                    row = top_players.iloc[i]
                    style = rank_styles.get(i, {'c':'#ccc','i':'?'})
                    st.markdown(f"""
                        <div style="background-color:{style['c']};color:black;padding:15px;border-radius:10px;text-align:center;box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size:24px;">{style['i']}</div>
                            <div style="font-size:18px;font-weight:bold;">{row['player']}</div>
                            <div style="font-size:14px;margin-top:5px;"><b>Avg Place:</b> {row['avg_position']:.2f} | <b>Win:</b> {row['win_rate']:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
        st.markdown("---")

    if dashboard_df.empty:
        st.warning("No data found for these filters.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Heatmaps", "Trends", "Meta Stats"])

        # --- TAB 1: OVERVIEW ---
        with tab1:
            col1, col2 = st.columns(2)
            
            # Rank Distribution
            c1_df = dashboard_df[dashboard_df['position'].isin([1, 2, 3, 4])].copy()
            c1_df['rank_char'] = c1_df['position'].map({1: '1st', 2: '2nd', 3: '3rd', 4: '4th'})
            c1_grp = c1_df.groupby(['player', 'rank_char']).size().reset_index(name='count')
            c1_grp['percentage'] = c1_grp.groupby('player')['count'].transform(lambda x: (x/x.sum())*100)
            
            fig1 = px.bar(c1_grp, x='player', y='percentage', color='rank_char', 
                          color_discrete_map=RANK_COLORS, barmode='stack',
                          category_orders={'rank_char': RANK_ORDER},
                          title="Rank Distribution by Player",
                          labels={'player': 'Player', 'percentage': 'Percentage (%)', 'rank_char': 'Rank'},
                          hover_data={'count': True, 'percentage': ':.1f'}) 
            fig1.update_layout(yaxis=dict(showticklabels=False, showgrid=False), legend_title="Rank")
            fig1.update_traces(texttemplate='%{y:.1f}%', textposition='inside')
            col1.plotly_chart(fig1, use_container_width=True)

            # Deck Stats
            deck_stats = dashboard_df.groupby('deck').agg(
                position=('position', 'mean'),
                games_played=('position', 'count')
            ).reset_index().sort_values('position', ascending=False)
            deck_stats['avg_str'] = deck_stats['position'].round(2).astype(str)
            
            fig2 = px.bar(deck_stats, x='deck', y='position', title="Avg Position by Deck",
                          color_discrete_sequence=['skyblue'],
                          labels={'deck': 'Deck Name', 'position': 'Avg Position'},
                          hover_data={'games_played': True, 'position': ':.2f'})
            fig2.update_layout(yaxis=dict(range=[0, 4.5], showticklabels=False), showlegend=False)
            fig2.update_traces(text=deck_stats['avg_str'], textposition='outside')
            col2.plotly_chart(fig2, use_container_width=True)

        # --- TAB 2: HEATMAPS ---
        with tab2:
            st.subheader("Performance Heatmap")
            
            h_col1, h_col2 = st.columns([1, 3])
            dim = h_col1.selectbox("Analyze Players By:", ["type","deck","color_simple"], 
                                   format_func=lambda x: {"type":"Game Format","deck":"Deck","color_simple":"Colour"}[x])
            
            # Grid Logic
            p_view = sorted(dashboard_df['player'].unique())
            i_view = sorted(dashboard_df[dim].unique())
            full_grid = pd.DataFrame(index=pd.MultiIndex.from_product([p_view, i_view], names=['player', dim])).reset_index()
            
            stats = dashboard_df.groupby(['player', dim])['position'].agg(['mean', 'count']).reset_index()
            hm = pd.merge(full_grid, stats, on=['player', dim], how='left')
            
            hm['fill'] = hm['mean'].fillna(0)
            hm['txt'] = hm['mean'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else("-"))
            hm['hov'] = hm.apply(lambda r: f"Avg: {r['mean']:.2f}<br>Games: {int(r['count'])}" if pd.notnull(r['mean']) else "Not Played", axis=1)
            
            z = hm.pivot(index='player', columns=dim, values='fill')
            t = hm.pivot(index='player', columns=dim, values='txt')
            h = hm.pivot(index='player', columns=dim, values='hov')
            
            colorscale = [[0.0, '#e0e0e0'], [0.24, '#e0e0e0'], [0.25, '#4caf50'], [1.0, '#f44336']]
            
            fig_hm = go.Figure(go.Heatmap(
                z=z.values, x=z.columns, y=z.index, text=t.values, texttemplate="%{text}",
                hovertext=h.values, hoverinfo='text',
                colorscale=colorscale, zmin=0, zmax=4, xgap=3, ygap=3,
                colorbar=dict(title="Avg Place", tickvals=[1, 2, 3, 4], ticktext=["1st", "2nd", "3rd", "4th"])
            ))
            fig_hm.update_layout(title=f"Avg Placement: Player vs {dim.capitalize().replace('Color_simple','Colour')}", height=500)
            st.plotly_chart(fig_hm, use_container_width=True)

        # --- TAB 3: TRENDS ---
        with tab3:
            st.subheader("Career Trajectory")
            t_df = dashboard_df.sort_values('match_uuid')
            t_df['cum'] = t_df.groupby('player')['position'].expanding().mean().reset_index(0,drop=True)
            
            fig_t = px.line(t_df, x='match_uuid', y='cum', color='player', markers=True, 
                            title="Cumulative Avg Position (Lower is Better)", 
                            # COLOUR CONSISTENCY APPLIED HERE
                            color_discrete_map=player_color_map,
                            labels={'match_uuid': 'Match Sequence', 'cum': 'Cumulative Avg Position', 'player': 'Player'},
                            hover_data={'position': True})
            fig_t.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_t, use_container_width=True)

        # --- TAB 4: META STATS ---
        with tab4:
            c1, c2 = st.columns(2)
            with c1:
                # Win Rate by Colour
                col_s = dashboard_df.groupby('color_simple').agg(
                    games=('position', 'count'), wins=('position', lambda x: (x==1).sum())
                ).reset_index()
                col_s = col_s[col_s['games']>=1]
                col_s['wr'] = (col_s['wins']/col_s['games'])*100
                
                fig_c = px.bar(col_s.sort_values('wr', ascending=False), x='color_simple', y='wr', 
                               color='wr', color_continuous_scale='RdYlGn', title="Win Rate by Colour",
                               labels={'color_simple': 'Colour', 'wr': 'Win Rate (%)'},
                               hover_data={'games': True})
                fig_c.update_layout(coloraxis_showscale=False)
                fig_c.update_traces(texttemplate='%{y:.0f}%', textposition='outside')
                st.plotly_chart(fig_c, use_container_width=True)
            
            with c2:
                # Avg Position by Colour (Formatting matched to Overview Deck chart)
                col_pos = dashboard_df.groupby('color_simple')['position'].mean().reset_index()
                col_pos = col_pos.sort_values('position', ascending=False)
                col_pos['avg_str'] = col_pos['position'].round(2).astype(str)
                
                fig_avg = px.bar(col_pos, x='color_simple', y='position',
                                title="Average Position by Colour",
                                labels={'color_simple': 'Colour', 'position': 'Avg Finishing Position'},
                                color_discrete_sequence=['skyblue'],
                                hover_data={'position': ':.2f'})
                fig_avg.update_layout(yaxis=dict(range=[0, 4.5], showticklabels=False), showlegend=False)
                fig_avg.update_traces(text=col_pos['avg_str'], textposition='outside')
                st.plotly_chart(fig_avg, use_container_width=True)

    with st.expander("View Raw Data"):
        st.dataframe(dashboard_df.sort_values('match_uuid', ascending=False), use_container_width=True, hide_index=True)

# ==============================================================================
# PAGE 2: DETAILED PLAYER ANALYTICS
# ==============================================================================
elif page == "Detailed Player Analytics":
    st.title("Detailed Player Analytics")
    st.markdown("Deep dive into **Elo Skill Ratings**, **Playstyle DNA**, and **Consistency Metrics**.")
    
    analytics_df = shared_filtered_df.copy()
    
    # --- ROW 1: ELO HISTORY ---
    st.subheader("The Race for Dominance (Elo History)")
    with st.expander("ℹ️ Understanding the Elo System (Click to expand)"):
        st.markdown("* **Starting Score:** 1200\n* **K-Factor:** 32 (Speed of rank change)\n* **Zero-Sum:** Points are stolen from opponents.")
    
    elo_plot_df = elo_history_df[elo_history_df['player'].isin(selected_players)]
    
    if elo_plot_df.empty:
        st.warning("Not enough data to calculate Elo ratings.")
    else:
        final_elo = elo_plot_df.groupby('player')['elo'].last().sort_values(ascending=False)
        fig_elo = px.line(
            elo_plot_df, x='match_uuid', y='elo', color='player', markers=True,
            title="Elo Rating Over Time",
            # COLOUR CONSISTENCY APPLIED HERE
            color_discrete_map=player_color_map,
            labels={'match_uuid': 'Match Number', 'elo': 'Skill Rating', 'player': 'Player'},
            category_orders={'player': final_elo.index.tolist()}
        )
        fig_elo.update_layout(hovermode="x unified")
        st.plotly_chart(fig_elo, use_container_width=True)

    st.markdown("---")

    # --- ROW 2: PLAYER DNA ---
    st.subheader("🧬 Player DNA Analysis")
    
    dna_data = []
    for p in selected_players:
        p_df = analytics_df[analytics_df['player'] == p]
        if p_df.empty: continue
        
        # Metric Calculations
        wins = len(p_df[p_df['position'] == 1])
        games = len(p_df)
        win_rate = wins / games if games > 0 else 0
        avg_pos = p_df['position'].mean()
        consistency = (4 - avg_pos) / 3
        unique_wins = p_df[p_df['position'] == 1]['deck'].nunique()
        versatility = min(unique_wins / 5, 1.0) # Cap at 5 decks
        
        # Form (Weighted last 5 games)
        last_5 = p_df.sort_values('match_uuid', ascending=False).head(5)
        last_5_wins = len(last_5[last_5['position'] == 1])
        form = min((last_5_wins / len(last_5)) * 2, 1.0) if not last_5.empty else 0
        
        top2 = len(p_df[p_df['position'] <= 2])
        top2_rate = top2 / games if games > 0 else 0
        
        dna_data.append({
            'player': p, 'Lethality': win_rate, 'Consistency': consistency,
            'Versatility': versatility, 'Form (L5)': form, 'Top 2 Rate': top2_rate
        })
    
    if dna_data:
        categories = ['Lethality', 'Consistency', 'Versatility', 'Form (L5)', 'Top 2 Rate']
        fig_radar = go.Figure()
        
        for p_dna in dna_data:
            p_name = p_dna['player']
            values = [p_dna[cat] for cat in categories]
            values.append(values[0])
            
            # COLOUR CONSISTENCY APPLIED HERE
            p_color = player_color_map.get(p_name, '#000000')
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values, theta=categories + [categories[0]], fill='toself', name=p_name,
                line=dict(color=p_color, width=2),
                fillcolor=hex_to_rgba(p_color, 0.1)
            ))
            
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False), bgcolor='rgba(0,0,0,0)'),
            showlegend=True, title="Playstyle Comparison"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    
    # --- ROW 3: CONSISTENCY DISTRIBUTION ---
    st.subheader("🎯 Consistency Analysis")
    fig_b = px.box(analytics_df, x='player', y='position', color='player', 
                   title="Finishing Position Distribution", 
                   # COLOUR CONSISTENCY APPLIED HERE
                   color_discrete_map=player_color_map,
                   labels={'player': 'Player', 'position': 'Finishing Position'})
    fig_b.update_yaxes(autorange="reversed", dtick=1)
    st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("---")

    # --- ROW 4: DRAW TYPE MECHANICS ---
    st.subheader("🎲 Draw Mechanic Analysis")
    draw_stats = analytics_df.groupby(['player', 'draw_type']).agg(
        games=('position', 'count'), wins=('position', lambda x: (x==1).sum())
    ).reset_index()
    draw_stats['win_rate'] = (draw_stats['wins'] / draw_stats['games']) * 100
    
    fig_draw = px.bar(draw_stats, x='player', y='win_rate', color='draw_type',
                      barmode='group',
                      title="Win Rate by Draw Type",
                      labels={'player': 'Player', 'win_rate': 'Win Rate (%)', 'draw_type': 'Draw Mechanism'},
                      hover_data={'games': True})
    st.plotly_chart(fig_draw, use_container_width=True)

# ==============================================================================
# PAGE 3: PLAYER VS PLAYER (NEW!)
# ==============================================================================
elif page == "Player vs Player":
    st.title("Head-to-Head Comparison")
    st.markdown("Compare two players directly. Statistics are calculated **only from matches where both players participated**.")
    
    col1, col2 = st.columns(2)
    p_options = sorted(raw_df['player'].unique())
    
    with col1:
        p1 = st.selectbox("Select Player 1", p_options, index=0)
    with col2:
        # Default to 2nd player if available
        default_idx = 1 if len(p_options) > 1 else 0
        p2 = st.selectbox("Select Player 2", p_options, index=default_idx)
        
    if p1 == p2:
        st.warning("Please select two different players to see the comparison.")
    else:
        # 1. FIND COMMON MATCHES
        p1_matches = set(raw_df[raw_df['player'] == p1]['match_uuid'])
        p2_matches = set(raw_df[raw_df['player'] == p2]['match_uuid'])
        common_matches = p1_matches.intersection(p2_matches)
        
        if not common_matches:
            st.error(f"No matches found where {p1} and {p2} played against each other.")
        else:
            # 2. FILTER DATA TO COMMON MATCHES
            h2h_df = raw_df[raw_df['match_uuid'].isin(common_matches)]
            
            # 3. CALCULATE STATS
            total_games = len(common_matches)
            
            # Helper to calculate stats for a specific player in the H2H subset
            def get_h2h_stats(player_name, opponent_name, df):
                p_rows = df[df['player'] == player_name]
                o_rows = df[df['player'] == opponent_name]
                
                # Merge to compare positions in same match
                merged = pd.merge(p_rows[['match_uuid','position']], o_rows[['match_uuid','position']], on='match_uuid', suffixes=('_p', '_o'))
                
                wins = len(p_rows[p_rows['position'] == 1])
                finished_ahead = len(merged[merged['position_p'] < merged['position_o']])
                avg_pos = p_rows['position'].mean()
                
                best_set, worst_set = get_best_worst(p_rows, 'type')
                best_col, worst_col = get_best_worst(p_rows, 'color_simple')
                
                return {
                    "Wins": wins,
                    "Win Rate": (wins/total_games)*100,
                    "Finished Ahead": finished_ahead,
                    "Avg Position": avg_pos,
                    "Best Set": best_set,
                    "Worst Set": worst_set,
                    "Best Colour": best_col,
                    "Worst Colour": worst_col
                }

            s1 = get_h2h_stats(p1, p2, h2h_df)
            s2 = get_h2h_stats(p2, p1, h2h_df)
            
            # 4. DISPLAY TABLE
            st.subheader(f"Rivalry Statistics ({total_games} Games Played)")
            
            comp_data = {
                "Metric": [
                    "🏆 Total Wins", 
                    "📈 Win Rate", 
                    "🏃 Finished Ahead of Rival", 
                    "📊 Average Position", 
                    "🃏 Best Performing Set", 
                    "💩 Worst Performing Set", 
                    "🎨 Best Colour", 
                    "💀 Worst Colour"
                ],
                f"{p1}": [
                    s1['Wins'], 
                    f"{s1['Win Rate']:.1f}%", 
                    s1['Finished Ahead'], 
                    f"{s1['Avg Position']:.2f}", 
                    s1['Best Set'], 
                    s1['Worst Set'], 
                    s1['Best Colour'], 
                    s1['Worst Colour']
                ],
                f"{p2}": [
                    s2['Wins'], 
                    f"{s2['Win Rate']:.1f}%", 
                    s2['Finished Ahead'], 
                    f"{s2['Avg Position']:.2f}", 
                    s2['Best Set'], 
                    s2['Worst Set'], 
                    s2['Best Colour'], 
                    s2['Worst Colour']
                ]
            }
            
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            st.caption(f"Note: 'Best/Worst' stats are based on Average Position within these {total_games} head-to-head games.")

# ==============================================================================
# PAGE 4: DECK & SET ANALYSIS
# ==============================================================================
elif page == "Deck & Set Analysis":
    st.title("🎴 Deck & Set Analysis")
    df_d = shared_filtered_df.copy()
    
    if df_d.empty:
        st.warning("No data available.")
    else:
        t1, t2, t3 = st.tabs(["Popularity", "Unplayed Decks", "Set Recency"])
        
        with t1:
            st.subheader("Most & Least Played Decks")
            cnt = df_d['deck'].value_counts().reset_index()
            cnt.columns = ['deck','games']
            h = 300 + (len(cnt) * 20)
            
            fig_p = px.bar(cnt, x='games', y='deck', orientation='h', height=h, 
                           title="Games Played per Deck", 
                           labels={'games': 'Games Played', 'deck': 'Deck Name'},
                           color_discrete_sequence=['#4a90e2'])
            fig_p.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_p, use_container_width=True)

        with t2:
            st.subheader("The 'Completionist' Checklist")
            all_p = sorted(df_d['player'].unique())
            all_d = sorted(df_d['deck'].unique())
            
            missing = []
            for p in all_p:
                p_decks = set(df_d[df_d['player']==p]['deck'])
                for d in all_d:
                    if d not in p_decks: missing.append({'Player':p, 'Missing Deck':d})
            
            if not missing:
                st.success("All players have played all decks in this selection!")
            else:
                md_df = pd.DataFrame(missing)
                sel = st.selectbox("Check Missing Decks For:", all_p)
                u_m = md_df[md_df['Player']==sel]
                
                if not u_m.empty:
                    st.dataframe(u_m[['Missing Deck']], use_container_width=True, hide_index=True)
                else:
                    st.success(f"{sel} has played all decks!")
                
                with st.expander("View Full Played Matrix"):
                    mat = df_d.groupby(['deck','player']).size().unstack(fill_value=0)
                    st.dataframe(mat.style.background_gradient(cmap='Blues'), use_container_width=True)

        with t3:
            st.subheader("Set Freshness Tracker")
            curr = raw_df['match_uuid'].max()
            rec = raw_df.groupby('type').agg(lst=('match_uuid','max'), cnt=('match_uuid','nunique')).reset_index()
            rec['ago'] = curr - rec['lst']
            
            st.dataframe(
                rec.sort_values('ago').style.background_gradient(subset=['ago'], cmap='Reds'),
                column_config={
                    "type":"Game Format / Set", "lst":"Last Match ID", 
                    "cnt":"Total Plays", "ago":"Matches Ago"
                },
                hide_index=True, use_container_width=True
            )

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption(f"App Version: {APP_VERSION}")
st.markdown("© 2025 MTG Stats Analysis | Built with ❤️ using Streamlit")