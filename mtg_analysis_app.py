# -------------------- LIBRARIES --------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import numpy as np

# -------------------- CONSTANTS & CONFIG --------------------
DATA_FILE = "data.csv"
APP_TITLE = "Magic: The Gathering Stats Analysis"
APP_VERSION = "1.9.2" 

# Rank Colours (Global Definition)
RANK_COLORS = {'1st': 'gold', '2nd': 'silver', '3rd': 'orange', '4th': 'skyblue'}
RANK_ORDER = ['1st', '2nd', '3rd', '4th']

# Consistent Player Palette (Plotly G10 + Dark24 for variety)
PLAYER_PALETTE = px.colors.qualitative.G10 + px.colors.qualitative.Dark24

# Navigation Mapping
NAV_MAP = {
    "Dashboard": "📊 Dashboard",
    "Analytics": "🧬 Detailed Player Analytics",
    "PvP": "⚔️ Player vs Player",
    "Decks": "🎴 Deck & Set Analysis"
}

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
    """Load data from CSV."""
    return pd.read_csv(file_path)

def get_valid_options(df, column):
    """Helper to get sorted unique options from a dataframe column."""
    if column not in df.columns: return []
    return sorted(df[column].dropna().unique())

def enhance_game_ids(df):
    """
    SMART ENGINE: Splits games that share the same game_id but are actually sequential matches.
    """
    if 'game_id' not in df.columns: return df
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
    if 'match_uuid' not in df.columns or 'position' not in df.columns: 
        return pd.DataFrame(), {}
    
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
                'game_id': game.get('game_id', game.iloc[0]).iloc[0] if 'game_id' in game else match_id,
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
    if df.empty or category_col not in df.columns: return "N/A", "N/A"
    if df[category_col].isnull().all(): return "N/A", "N/A"
    
    stats = df.groupby(category_col)['position'].mean().reset_index()
    if stats.empty: return "N/A", "N/A"
    
    best = stats.loc[stats['position'].idxmin()][category_col]
    worst = stats.loc[stats['position'].idxmax()][category_col]
    return best, worst

def _parse_time(t):
    """Parses time strings like MM:SS or HH:MM:SS into total seconds. Returns NaN if missing."""
    if pd.isna(t) or not str(t).strip():
        return np.nan
    try:
        parts = str(t).split(':')
        if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        if len(parts) == 2: return int(parts[0])*60 + float(parts[1])
        return np.nan
    except:
        return np.nan

# -------------------- DATA LOADING & PRE-PROCESSING --------------------
try:
    raw_df = load_data(DATA_FILE, _file_md5(DATA_FILE))
    
    # Cleaning
    df_obj = raw_df.select_dtypes(['object'])
    raw_df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    
    if 'position' in raw_df.columns:
        raw_df['position'] = pd.to_numeric(raw_df['position'], errors='coerce')
    if 'deck' in raw_df.columns:
        raw_df['deck'] = raw_df['deck'].replace({'Elven': 'Elves'}) 
    if 'primary_mana' in raw_df.columns:
        raw_df['color_simple'] = raw_df['primary_mana'].astype(str).apply(lambda x: x.split(' ')[0])
    
    # Logic
    raw_df = enhance_game_ids(raw_df)
    elo_history_df, current_elo = calculate_elo(raw_df)
    
    # Defaults & Colour Mapping
    all_players_def = get_valid_options(raw_df, 'player')
    
    # GLOBAL COLOUR CONSISTENCY: Map every player to a fixed colour
    unique_players = sorted(raw_df['player'].dropna().unique())
    player_color_map = {player: PLAYER_PALETTE[i % len(PLAYER_PALETTE)] for i, player in enumerate(unique_players)}

except FileNotFoundError:
    st.error(f"CSV file '{DATA_FILE}' not found. Place it beside this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
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

selection = st.sidebar.radio(
    "Go to:", 
    list(NAV_MAP.keys()), 
    format_func=lambda x: NAV_MAP[x],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("Global Filters")

if st.sidebar.button("🔄 Reset All Filters", on_click=reset_callbacks):
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
    
    if 'primary_mana' in df_f3.columns:
        shared_filtered_df = df_f3[df_f3['primary_mana'].isin(selected_colors)] if selected_colors else df_f3.copy()
    else:
        shared_filtered_df = df_f3.copy()

# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if selection == "Dashboard":
    st.title(f"{APP_TITLE} - Dashboard")
    dashboard_df = shared_filtered_df.copy()

    # --- TOP STATS ROW ---
    if not dashboard_df.empty and 'position' in dashboard_df.columns:
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

    # --- GAME SUPERLATIVES: 12-METRIC SUITE ---
    damage_cols = [c for c in dashboard_df.columns if str(c).startswith('damage_')]
    kill_cols = [c for c in dashboard_df.columns if str(c).startswith('kill_')]
    has_time = 'Avg Time' in dashboard_df.columns

    if (damage_cols or kill_cols or has_time) and not dashboard_df.empty:
        st.markdown("**Key Facts**") 
        
        adv_df = dashboard_df.copy()
        
        # Normalize damage and kill columns
        if damage_cols:
            for c in damage_cols: adv_df[c] = pd.to_numeric(adv_df[c], errors='coerce').fillna(0)
            adv_df['total_damage'] = adv_df[damage_cols].sum(axis=1)
        else:
            adv_df['total_damage'] = 0
            
        if kill_cols:
            for c in kill_cols: adv_df[c] = pd.to_numeric(adv_df[c], errors='coerce').fillna(0)
            adv_df['total_kills'] = adv_df[kill_cols].sum(axis=1)
        else:
            adv_df['total_kills'] = 0
            
        if has_time:
            adv_df['avg_time_sec'] = adv_df['Avg Time'].apply(_parse_time)
        else:
            adv_df['avg_time_sec'] = np.nan

        # Aggregate player-level stats
        adv_stats = adv_df.groupby('player').agg(
            total_damage=('total_damage', 'sum'),
            total_kills=('total_kills', 'sum'),
            avg_time=('avg_time_sec', 'mean'),
            avg_pos=('position', 'mean') if 'position' in adv_df.columns else (lambda x: 3.0),
            num_decks=('deck', 'nunique') if 'deck' in adv_df.columns else (lambda x: 0),
            games=('player', 'count')
        ).reset_index()
        
        # Add wins count
        if 'position' in adv_df.columns:
            adv_stats['wins'] = adv_df[adv_df['position'] == 1].groupby('player').size().reindex(adv_stats['player'], fill_value=0).values
        else:
            adv_stats['wins'] = 0

        # Helper function to render card with description
        def render_card(emoji, title, player_name, detail, description, title_color="#ff4b4b"):
            if player_name == "N/A":
                return f"""
                <div style="background-color:#2d2d44;border:1px solid #555;padding:20px;border-radius:12px;text-align:center;margin:8px;">
                    <div style="font-size:28px;margin-bottom:8px;">{emoji}</div>
                    <div style="font-size:15px;font-weight:bold;color:{title_color};margin-bottom:8px;">{title}</div>
                    <div style="font-size:16px;color:white;margin-bottom:12px;"><b>N/A</b></div>
                    <div style="font-size:13px;color:#ccc;margin-bottom:8px;">{detail}</div>
                    <div style="font-size:12px;color:#999;font-style:italic;">{description}</div>
                </div>
                """
            return f"""
            <div style="background-color:#2d2d44;border:1px solid #555;padding:20px;border-radius:12px;text-align:center;margin:8px;">
                <div style="font-size:28px;margin-bottom:8px;">{emoji}</div>
                <div style="font-size:15px;font-weight:bold;color:{title_color};margin-bottom:8px;">{title}</div>
                <div style="font-size:16px;color:white;margin-bottom:12px;"><b>{player_name}</b></div>
                <div style="font-size:13px;color:#ccc;margin-bottom:8px;">{detail}</div>
                <div style="font-size:12px;color:#999;font-style:italic;">{description}</div>
            </div>
            """

        # === ROW 1: CORE COMBAT EXTREMES (Opposites paired) ===
        row1_cols = st.columns(4)
        
        # 1. Most Damage 💥
        with row1_cols[0]:
            if damage_cols and adv_stats['total_damage'].sum() > 0:
                most_dmg = adv_stats.loc[adv_stats['total_damage'].idxmax()]
                st.markdown(render_card("💥", "Most Damage", most_dmg['player'], f"{int(most_dmg['total_damage'])} dmg", "Raw total damage output", "#ff4b4b"), unsafe_allow_html=True)
            else:
                st.markdown(render_card("💥", "Most Damage", "N/A", "No data", "Raw total damage output", "#ff4b4b"), unsafe_allow_html=True)
        
        # 2. Least Damage 🛡️
        with row1_cols[1]:
            if damage_cols and adv_stats['total_damage'].sum() > 0 and len(adv_stats[adv_stats['games'] >= 1]) > 0:
                least_dmg = adv_stats[adv_stats['games'] >= 1].loc[adv_stats[adv_stats['games'] >= 1]['total_damage'].idxmin()]
                st.markdown(render_card("🛡️", "Least Damage", least_dmg['player'], f"{int(least_dmg['total_damage'])} dmg", "Defensive play style", "#4b98ff"), unsafe_allow_html=True)
            else:
                st.markdown(render_card("🛡️", "Least Damage", "N/A", "No data", "Defensive play style", "#4b98ff"), unsafe_allow_html=True)
        
        # 3. Most Lethal ☠️
        with row1_cols[2]:
            if kill_cols and adv_stats['total_kills'].sum() > 0:
                most_lethal = adv_stats.loc[adv_stats['total_kills'].idxmax()]
                st.markdown(render_card("☠️", "Most Lethal", most_lethal['player'], f"{int(most_lethal['total_kills'])} kills", "Most finishing blows", "#804bff"), unsafe_allow_html=True)
            else:
                st.markdown(render_card("☠️", "Most Lethal", "N/A", "No data", "Most finishing blows", "#804bff"), unsafe_allow_html=True)
        
        # 4. Least Lethal 🧸
        with row1_cols[3]:
            if kill_cols and len(adv_stats[adv_stats['games'] >= 3]) > 0:
                least_lethal_cand = adv_stats[adv_stats['games'] >= 3]
                least_lethal = least_lethal_cand.loc[least_lethal_cand['total_kills'].idxmin()]
                st.markdown(render_card("🧸", "Least Lethal", least_lethal['player'], f"{int(least_lethal['total_kills'])} kills", "Few finishing blows (min 3 games)", "#ff9999"), unsafe_allow_html=True)
            else:
                st.markdown(render_card("🧸", "Least Lethal", "N/A", "No data", "Few finishing blows (min 3 games)", "#ff9999"), unsafe_allow_html=True)

        st.write("")  # Spacing

        # === ROW 2: COMBAT NUANCE & DYNAMICS ===
        row2_cols = st.columns(4)
        
        # 5. The Vulture 🦅
        with row2_cols[0]:
            if damage_cols and kill_cols:
                vulture_cand = adv_stats[adv_stats['total_kills'] > 0].copy()
                vulture_cand['dmg_per_kill'] = vulture_cand['total_damage'] / vulture_cand['total_kills']
                if not vulture_cand.empty:
                    vulture = vulture_cand.loc[vulture_cand['dmg_per_kill'].idxmin()]
                    st.markdown(render_card("🦅", "The Vulture", vulture['player'], f"{vulture['dmg_per_kill']:.1f} dmg/kill", "Steals kills with low damage", "#ffa500"), unsafe_allow_html=True)
                else:
                    st.markdown(render_card("🦅", "The Vulture", "N/A", "No data", "Steals kills with low damage", "#ffa500"), unsafe_allow_html=True)
            else:
                st.markdown(render_card("🦅", "The Vulture", "N/A", "No data", "Steals kills with low damage", "#ffa500"), unsafe_allow_html=True)
        
        # 6. The Grudge Bearer 🎯
        with row2_cols[1]:
            if damage_cols:
                grudge_data = {}
                for player in adv_stats['player']:
                    p_df = adv_df[adv_df['player'] == player]
                    total_p_dmg = sum([p_df[col].sum() for col in damage_cols])
                    if total_p_dmg > 0:
                        max_target = None
                        max_dmg = 0
                        for col in damage_cols:
                            col_dmg = p_df[col].sum()
                            if col_dmg > max_dmg:
                                max_dmg = col_dmg
                                max_target = col.replace('damage_', '').title()
                        fixation = (max_dmg / total_p_dmg) * 100
                        grudge_data[player] = {'fixation': fixation, 'target': max_target}
                
                if grudge_data:
                    grudge = max(grudge_data.items(), key=lambda x: x[1]['fixation'])
                    st.markdown(render_card("🎯", "Grudge Bearer", grudge[0], f"{grudge[1]['fixation']:.0f}% vs {grudge[1]['target']}", "Highest damage fixation ratio", "#00bcd4"), unsafe_allow_html=True)
                else:
                    st.markdown(render_card("🎯", "Grudge Bearer", "N/A", "No data", "Highest damage fixation ratio", "#00bcd4"), unsafe_allow_html=True)
            else:
                st.markdown(render_card("🎯", "Grudge Bearer", "N/A", "No data", "Highest damage fixation ratio", "#00bcd4"), unsafe_allow_html=True)
        
        # 7. Punching Bag 🥊
        with row2_cols[2]:
            if damage_cols:
                # Calculate total damage received per player
                dmg_received = {}
                for player in adv_stats['player']:
                    total_received = 0
                    for dcol in damage_cols:
                        dcol_name = dcol.replace('damage_', '').title()
                        if dcol_name == player:
                            total_received += adv_df[dcol].sum()
                    dmg_received[player] = total_received
                
                max_punched = max(dmg_received, key=dmg_received.get)
                st.markdown(render_card("🥊", "Punching Bag", max_punched, f"{int(dmg_received[max_punched])} received", "Gets focused most often", "#ff69b4"), unsafe_allow_html=True)
            else:
                st.markdown(render_card("🥊", "Punching Bag", "N/A", "No data", "Gets focused most often", "#ff69b4"), unsafe_allow_html=True)
        
        # 8. Glass Cannon 💣
        with row2_cols[3]:
            if damage_cols and 'position' in adv_df.columns and len(adv_stats) > 0:
                # Find player with high damage but poor placement
                cannon_cand = adv_stats[adv_stats['avg_pos'] > 2.0].copy()
                if not cannon_cand.empty:
                    cannon = cannon_cand.loc[cannon_cand['total_damage'].idxmax()]
                    st.markdown(render_card("💣", "Glass Cannon", cannon['player'], f"{int(cannon['total_damage'])} dmg (pos {cannon['avg_pos']:.1f})", "High damage, poor finishes", "#00ff00"), unsafe_allow_html=True)
                else:
                    st.markdown(render_card("💣", "Glass Cannon", "N/A", "No data", "High damage, poor finishes", "#00ff00"), unsafe_allow_html=True)
            else:
                st.markdown(render_card("💣", "Glass Cannon", "N/A", "No data", "High damage, poor finishes", "#00ff00"), unsafe_allow_html=True)

        st.write("")  # Spacing

        # === ROW 3: PLAYER PACE & DECK META ===
        row3_cols = st.columns(4)
        
        # 9. Speed Demon ⚡
        with row3_cols[0]:
            if has_time and adv_stats['avg_time'].count() > 0:
                speed = adv_stats[(adv_stats['avg_time'] > 0) & (adv_stats['avg_time'].notna())]
                if not speed.empty:
                    speed_demon = speed.loc[speed['avg_time'].idxmin()]
                    m, s = divmod(int(speed_demon['avg_time']), 60)
                    st.markdown(render_card("⚡", "Speed Demon", speed_demon['player'], f"{m:02d}:{s:02d} avg/turn", "Fastest decision-maker", "#ffeb3b"), unsafe_allow_html=True)
                else:
                    st.markdown(render_card("⚡", "Speed Demon", "N/A", "No data", "Fastest decision-maker", "#ffeb3b"), unsafe_allow_html=True)
            else:
                st.markdown(render_card("⚡", "Speed Demon", "N/A", "No data", "Fastest decision-maker", "#ffeb3b"), unsafe_allow_html=True)
        
        # 10. Most Boring 💤
        with row3_cols[1]:
            if has_time and adv_stats['avg_time'].count() > 0:
                most_boring = adv_stats.loc[adv_stats['avg_time'].idxmax()]
                if pd.notna(most_boring['avg_time']):
                    m, s = divmod(int(most_boring['avg_time']), 60)
                    st.markdown(render_card("💤", "Most Boring", most_boring['player'], f"{m:02d}:{s:02d} avg/turn", "Takes the most time per turn", "#a8a8a8"), unsafe_allow_html=True)
                else:
                    st.markdown(render_card("💤", "Most Boring", "N/A", "No data", "Takes the most time per turn", "#a8a8a8"), unsafe_allow_html=True)
            else:
                st.markdown(render_card("💤", "Most Boring", "N/A", "No data", "Takes the most time per turn", "#a8a8a8"), unsafe_allow_html=True)
        
        # 11. One-Trick Pony 🐴
        with row3_cols[2]:
            if 'deck' in adv_df.columns and len(adv_stats[adv_stats['games'] >= 3]) > 0:
                pony_cand = adv_stats[adv_stats['games'] >= 3]
                pony = pony_cand.loc[pony_cand['num_decks'].idxmin()]
                st.markdown(render_card("🐴", "One-Trick Pony", pony['player'], f"{int(pony['num_decks'])} unique deck(s)", "Least deck variety (min 3 games)", "#ee82ee"), unsafe_allow_html=True)
            else:
                st.markdown(render_card("🐴", "One-Trick Pony", "N/A", "No data", "Least deck variety (min 3 games)", "#ee82ee"), unsafe_allow_html=True)
        
        # 12. Jack of All Trades 🃏
        with row3_cols[3]:
            if 'deck' in adv_df.columns:
                trades_data = {}
                for player in adv_stats['player']:
                    p_wins = adv_df[(adv_df['player'] == player) & (adv_df['position'] == 1)]
                    if not p_wins.empty:
                        trades_data[player] = p_wins['deck'].nunique()
                    else:
                        trades_data[player] = 0
                
                if trades_data and max(trades_data.values()) > 0:
                    trades = max(trades_data.items(), key=lambda x: x[1])
                    st.markdown(render_card("🃏", "Jack of All Trades", trades[0], f"{int(trades[1])} winning deck(s)", "Most diverse 1st place finishes", "#1affa3"), unsafe_allow_html=True)
                else:
                    st.markdown(render_card("🃏", "Jack of All Trades", "N/A", "No data", "Most diverse 1st place finishes", "#1affa3"), unsafe_allow_html=True)
            else:
                st.markdown(render_card("🃏", "Jack of All Trades", "N/A", "No data", "Most diverse 1st place finishes", "#1affa3"), unsafe_allow_html=True)

        st.markdown("---")



    if dashboard_df.empty:
        st.warning("No data found for these filters.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Heatmaps", "Trends", "Meta Stats"])

        # --- TAB 1: OVERVIEW ---
        with tab1:
            col1, col2 = st.columns(2)
            
            # Rank Distribution
            if 'position' in dashboard_df.columns:
                c1_df = dashboard_df[dashboard_df['position'].isin([1, 2, 3, 4])].copy()
                c1_df['rank_char'] = c1_df['position'].map({1: '1st', 2: '2nd', 3: '3rd', 4: '4th'})
                c1_grp = c1_df.groupby(['player', 'rank_char']).size().reset_index(name='count')
                c1_grp['percentage'] = c1_grp.groupby('player')['count'].transform(lambda x: (x/x.sum())*100)
                
                fig1 = px.bar(c1_grp, x='player', y='percentage', color='rank_char', 
                              color_discrete_map=RANK_COLORS, 
                              category_orders={'rank_char': RANK_ORDER},
                              title="Rank Distribution by Player",
                              labels={'player': 'Player', 'percentage': 'Percentage (%)', 'rank_char': 'Rank'},
                              hover_data={'count': True, 'percentage': ':.1f'}) 
                
                fig1.update_layout(
                    barmode='stack',
                    yaxis=dict(showticklabels=False, showgrid=False), 
                    legend_title="Rank"
                )
                fig1.update_traces(texttemplate='%{y:.1f}%', textposition='inside')
                col1.plotly_chart(fig1, use_container_width=True)

            # Deck Stats
            if 'deck' in dashboard_df.columns and 'position' in dashboard_df.columns:
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

            # --- COMBAT INTELLIGENCE ---
            if damage_cols and kill_cols:
                st.markdown("---")
                st.subheader("⚔️ Combat Intelligence")
                
                # --- DAMAGE MATRIX ---
                st.markdown("**Damage Matrix (Attacker vs Victim)**")
                
                flow_df = dashboard_df[['player'] + damage_cols].melt(id_vars=['player'], var_name='target', value_name='damage')
                flow_df['target'] = flow_df['target'].str.replace('damage_', '').str.title()
                flow_df['damage'] = pd.to_numeric(flow_df['damage'], errors='coerce').fillna(0)
                
                damage_matrix_df = flow_df.groupby(['player', 'target'])['damage'].sum().reset_index()
                damage_pivot = damage_matrix_df.pivot(index='player', columns='target', values='damage').fillna(0)
                
                # Calculate percentages per row (% of each attacker's damage to opponents)
                damage_pct = damage_pivot.copy()
                for idx in damage_pct.index:
                    # Sum excluding self-damage (diagonal)
                    row_sum = damage_pivot.loc[idx].sum() - damage_pivot.loc[idx, idx] if idx in damage_pivot.columns else damage_pivot.loc[idx].sum()
                    if row_sum > 0:
                        damage_pct.loc[idx] = (damage_pivot.loc[idx] / row_sum) * 100
                    else:
                        damage_pct.loc[idx] = 0
                
                # Format as "number (percentage%)" and black out self-damage
                display_df = damage_pivot.astype(object)
                for idx in display_df.index:
                    for col in display_df.columns:
                        val = damage_pivot.loc[idx, col]
                        pct = damage_pct.loc[idx, col]
                        
                        # Black out self-damage (diagonal)
                        if idx == col:
                            display_df.loc[idx, col] = "██"
                        elif val == 0:
                            display_df.loc[idx, col] = "-"
                        else:
                            display_df.loc[idx, col] = f"{int(val)} ({pct:.0f}%)"
                        
                st.dataframe(
                    display_df,
                    use_container_width=True
                )
                
                # --- VULTURE METRIC & NEMESIS TRACKER ---
                col_vul, col_nem = st.columns(2)
                
                with col_vul:
                    st.markdown("**🦅 Vulture Metrics**")
                    vulture_data = []
                    for player_name in dashboard_df['player'].dropna().unique():
                        p_df = dashboard_df[dashboard_df['player'] == player_name]
                        
                        total_dmg = sum([pd.to_numeric(p_df[col], errors='coerce').fillna(0).sum() for col in damage_cols])
                        total_kills = sum([pd.to_numeric(p_df[col], errors='coerce').fillna(0).sum() for col in kill_cols])
                        
                        dmg_per_kill = total_dmg / total_kills if total_kills > 0 else 0
                        vulture_data.append({
                            'Player': player_name,
                            'Total Damage': total_dmg,
                            'Total Kills': int(total_kills),
                            'Dmg/Kill Ratio': dmg_per_kill
                        })
                    
                    if vulture_data:
                        vulture_df = pd.DataFrame(vulture_data).sort_values('Dmg/Kill Ratio', ascending=False)
                        st.dataframe(
                            vulture_df,
                            column_config={
                                "Player": st.column_config.TextColumn("Player"),
                                "Total Damage": st.column_config.NumberColumn("Damage Dealt"),
                                "Total Kills": st.column_config.NumberColumn("Kills"),
                                "Dmg/Kill Ratio": st.column_config.ProgressColumn(
                                    "Damage Per Kill",
                                    help="Higher means more damage done per kill. Lower means stealing kills (Vulture).",
                                    format="%.1f 💥",
                                    min_value=0,
                                    max_value=max(vulture_df['Dmg/Kill Ratio'].max(), 1)
                                )
                            },
                            hide_index=True, use_container_width=True
                        )

                with col_nem:
                    st.markdown("**🎯 Nemesis Tracker**")
                    nemesis_data = []
                    for player_name in dashboard_df['player'].dropna().unique():
                        p_df = dashboard_df[dashboard_df['player'] == player_name]
                        total_dmg = sum([pd.to_numeric(p_df[col], errors='coerce').fillna(0).sum() for col in damage_cols])
                        
                        if total_dmg > 0:
                            max_target = None
                            max_dmg = 0
                            max_dmg_col_idx = None
                            for i, col in enumerate(damage_cols):
                                col_dmg = pd.to_numeric(p_df[col], errors='coerce').fillna(0).sum()
                                if col_dmg > max_dmg:
                                    max_dmg = col_dmg
                                    max_target = col.replace('damage_', '').title()
                                    max_dmg_col_idx = i
                            
                            # Get corresponding kills for the nemesis
                            nemesis_kills = 0
                            if max_dmg_col_idx is not None and max_dmg_col_idx < len(kill_cols):
                                nemesis_kills = int(pd.to_numeric(p_df[kill_cols[max_dmg_col_idx]], errors='coerce').fillna(0).sum())
                            
                            fixation_pct = (max_dmg / total_dmg) * 100
                            nemesis_data.append({
                                'Attacker': player_name,
                                'Nemesis': max_target or "N/A",
                                'Damage': int(max_dmg),
                                'Kills': nemesis_kills,
                                'Fixation': fixation_pct
                            })
                    
                    if nemesis_data:
                        nemesis_df = pd.DataFrame(nemesis_data).sort_values('Fixation', ascending=False)
                        st.dataframe(
                            nemesis_df,
                            column_config={
                                "Attacker": st.column_config.TextColumn("Player"),
                                "Nemesis": st.column_config.TextColumn("Primary Target 🎯"),
                                "Damage": st.column_config.NumberColumn("Damage Focused"),
                                "Kills": st.column_config.NumberColumn("Kills"),
                                "Fixation": st.column_config.ProgressColumn(
                                    "Fixation %",
                                    help="Percentage of total damage dealt exclusively to this player.",
                                    format="%.1f%%",
                                    min_value=0,
                                    max_value=100
                                )
                            },
                            hide_index=True, use_container_width=True
                        )

        # --- TAB 2: HEATMAPS ---
        with tab2:
            st.subheader("Performance Heatmap")
            h_col1, h_col2 = st.columns([1, 3])
            
            # Identify valid dimensions based on available columns
            dims = []
            if 'type' in dashboard_df.columns: dims.append("type")
            if 'deck' in dashboard_df.columns: dims.append("deck")
            if 'color_simple' in dashboard_df.columns: dims.append("color_simple")
            
            if dims and 'position' in dashboard_df.columns:
                dim_labels = {"type":"Game Format","deck":"Deck","color_simple":"Colour"}
                dim = h_col1.selectbox("Analyze Players By:", dims, format_func=lambda x: dim_labels[x])
                
                p_view = sorted(dashboard_df['player'].dropna().unique())
                i_view = sorted(dashboard_df[dim].dropna().unique())
                full_grid = pd.DataFrame(index=pd.MultiIndex.from_product([p_view, i_view], names=['player', dim])).reset_index()
                
                stats = dashboard_df.groupby(['player', dim])['position'].agg(['mean', 'count']).reset_index()
                hm = pd.merge(full_grid, stats, on=['player', dim], how='left')
                
                hm['fill'] = hm['mean'].fillna(0)
                hm['txt'] = hm['mean'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "-")
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
                fig_hm.update_layout(title=f"Avg Placement: Player vs {dim_labels[dim]}", height=500)
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Required dimensions missing for heatmap analysis.")

        # --- TAB 3: TRENDS ---
        with tab3:
            st.subheader("Career Trajectory")
            if 'match_uuid' in dashboard_df.columns and 'position' in dashboard_df.columns:
                t_df = dashboard_df.sort_values('match_uuid')
                t_df['cum'] = t_df.groupby('player')['position'].expanding().mean().reset_index(0,drop=True)
                
                fig_t = px.line(t_df, x='match_uuid', y='cum', color='player', markers=True, 
                                title="Cumulative Avg Position (Lower is Better)", 
                                color_discrete_map=player_color_map,
                                labels={'match_uuid': 'Match Sequence', 'cum': 'Cumulative Avg Position', 'player': 'Player'},
                                hover_data={'position': True})
                fig_t.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.info("Missing 'match_uuid' or 'position' logic for trends.")

        # --- TAB 4: META STATS ---
        with tab4:
            if 'color_simple' in dashboard_df.columns and 'position' in dashboard_df.columns:
                c1, c2 = st.columns(2)
                with c1:
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
            else:
                st.info("Missing colour or position data.")

    with st.expander("View Raw Data"):
        if 'match_uuid' in dashboard_df.columns:
            st.dataframe(dashboard_df.sort_values('match_uuid', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.dataframe(dashboard_df, use_container_width=True, hide_index=True)

# ==============================================================================
# PAGE 2: DETAILED PLAYER ANALYTICS
# ==============================================================================
elif selection == "Analytics":
    st.title("Detailed Player Analytics")
    st.markdown("Deep dive into **Elo Skill Ratings**, **Playstyle DNA**, and **Consistency Metrics**.")
    
    analytics_df = shared_filtered_df.copy()
    
    # --- ROW 1: ELO HISTORY ---
    st.subheader("The Race for Dominance (Elo History)")
    with st.expander("ℹ️ Understanding the Elo System (Click to expand)"):
        st.markdown("* **Starting Score:** 1200\n* **K-Factor:** 32 (Speed of rank change)\n* **Zero-Sum:** Points are stolen from opponents.")
    
    if not elo_history_df.empty:
        elo_plot_df = elo_history_df[elo_history_df['player'].isin(selected_players)]
        
        if elo_plot_df.empty:
            st.warning("Not enough data to calculate Elo ratings.")
        else:
            final_elo = elo_plot_df.groupby('player')['elo'].last().sort_values(ascending=False)
            fig_elo = px.line(
                elo_plot_df, x='match_uuid', y='elo', color='player', markers=True,
                title="Elo Rating Over Time",
                color_discrete_map=player_color_map,
                labels={'match_uuid': 'Match Number', 'elo': 'Skill Rating', 'player': 'Player'},
                category_orders={'player': final_elo.index.tolist()}
            )
            fig_elo.update_layout(hovermode="x unified")
            st.plotly_chart(fig_elo, use_container_width=True)
    else:
        st.warning("Elo calculations require match_uuid and position mapping.")

    st.markdown("---")

    # --- ROW 2: PLAYER DNA ---
    st.subheader("🧬 Player DNA Analysis")
    with st.expander("ℹ️ How to read these metrics"):
        st.markdown("""
        * **Lethality:** Pure Win Rate %.
        * **Consistency:** Ability to avoid 4th place (Higher is better).
        * **Versatility:** Number of *unique* decks piloted to a win.
        * **Form:** Win Rate in the last 5 games.
        * **Top 2 Rate:** Percentage of games finishing 1st or 2nd.
        """)
    
    if 'position' in analytics_df.columns:
        dna_data = []
        for p in selected_players:
            p_df = analytics_df[analytics_df['player'] == p]
            if p_df.empty: continue
            
            wins = len(p_df[p_df['position'] == 1])
            games = len(p_df)
            win_rate = wins / games if games > 0 else 0
            avg_pos = p_df['position'].mean()
            consistency = (4 - avg_pos) / 3
            
            unique_wins = p_df[p_df['position'] == 1]['deck'].nunique() if 'deck' in p_df.columns else 0
            versatility = min(unique_wins / 5, 1.0) 
            
            if 'match_uuid' in p_df.columns:
                last_5 = p_df.sort_values('match_uuid', ascending=False).head(5)
            else:
                last_5 = p_df.head(5)
                
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
                       color_discrete_map=player_color_map,
                       labels={'player': 'Player', 'position': 'Finishing Position'})
        fig_b.update_yaxes(autorange="reversed", dtick=1)
        st.plotly_chart(fig_b, use_container_width=True)
    else:
        st.info("Missing positional data for DNA mapping.")

    st.markdown("---")

    # --- ROW 4: DRAW TYPE MECHANICS ---
    if 'draw_type' in analytics_df.columns and 'position' in analytics_df.columns:
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
# PAGE 3: PLAYER VS PLAYER
# ==============================================================================
elif selection == "PvP":
    st.title("⚔️ Head-to-Head Comparison")
    st.markdown("Compare two players directly. Statistics are calculated **only from matches where both players participated**.")
    
    if 'player' not in raw_df.columns:
        st.error("Missing player columns.")
    else:
        col1, col2 = st.columns(2)
        p_options = sorted(raw_df['player'].dropna().unique())
        
        with col1:
            p1 = st.selectbox("Select Player 1", p_options, index=0)
        with col2:
            default_idx = 1 if len(p_options) > 1 else 0
            p2 = st.selectbox("Select Player 2", p_options, index=default_idx)
            
        if p1 == p2:
            st.warning("Please select two different players to see the comparison.")
        elif 'match_uuid' not in raw_df.columns:
            st.error("Missing match tracking required for head-to-head analysis.")
        else:
            p1_matches = set(raw_df[raw_df['player'] == p1]['match_uuid'])
            p2_matches = set(raw_df[raw_df['player'] == p2]['match_uuid'])
            common_matches = p1_matches.intersection(p2_matches)
            
            if not common_matches:
                st.error(f"No matches found where {p1} and {p2} played against each other.")
            else:
                h2h_df = raw_df[raw_df['match_uuid'].isin(common_matches)]
                total_games = len(common_matches)
                
                def get_h2h_stats(player_name, opponent_name, df):
                    p_rows = df[df['player'] == player_name]
                    o_rows = df[df['player'] == opponent_name]
                    
                    if 'position' in df.columns:
                        merged = pd.merge(p_rows[['match_uuid','position']], o_rows[['match_uuid','position']], on='match_uuid', suffixes=('_p', '_o'))
                        wins = len(p_rows[p_rows['position'] == 1])
                        finished_ahead = len(merged[merged['position_p'] < merged['position_o']])
                        avg_pos = p_rows['position'].mean()
                    else:
                        wins, finished_ahead, avg_pos = 0, 0, 0
                        
                    best_set, worst_set = get_best_worst(p_rows, 'type')
                    best_col, worst_col = get_best_worst(p_rows, 'color_simple')
                    
                    return {
                        "Wins": wins,
                        "Win Rate": (wins/total_games)*100 if total_games > 0 else 0,
                        "Finished Ahead": finished_ahead,
                        "Avg Position": avg_pos,
                        "Best Set": best_set,
                        "Worst Set": worst_set,
                        "Best Colour": best_col,
                        "Worst Colour": worst_col
                    }

                s1 = get_h2h_stats(p1, p2, h2h_df)
                s2 = get_h2h_stats(p2, p1, h2h_df)
                
                st.subheader(f"Rivalry Statistics ({total_games} Games Played)")
                
                comp_data = {
                    "Metric": [
                        "🏆 Total Wins", "📈 Win Rate", "🏃 Finished Ahead of Rival", 
                        "📊 Average Position", "🃏 Best Performing Set", "💩 Worst Performing Set", 
                        "🎨 Best Colour", "💀 Worst Colour"
                    ],
                    f"{p1}": [
                        s1['Wins'], f"{s1['Win Rate']:.1f}%", s1['Finished Ahead'], 
                        f"{s1['Avg Position']:.2f}", s1['Best Set'], s1['Worst Set'], 
                        s1['Best Colour'], s1['Worst Colour']
                    ],
                    f"{p2}": [
                        s2['Wins'], f"{s2['Win Rate']:.1f}%", s2['Finished Ahead'], 
                        f"{s2['Avg Position']:.2f}", s2['Best Set'], s2['Worst Set'], 
                        s2['Best Colour'], s2['Worst Colour']
                    ]
                }
                
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                st.caption(f"Note: 'Best/Worst' stats are based on Average Position within these {total_games} head-to-head games.")

# ==============================================================================
# PAGE 4: DECK & SET ANALYSIS
# ==============================================================================
elif selection == "Decks":
    st.title("🎴 Deck & Set Analysis")
    df_d = shared_filtered_df.copy()
    
    if df_d.empty:
        st.warning("No data available.")
    else:
        t1, t2, t3 = st.tabs(["Popularity", "Unplayed Decks", "Set Recency"])
        
        with t1:
            st.subheader("Most & Least Played Decks")
            if 'deck' in df_d.columns:
                cnt = df_d['deck'].value_counts().reset_index()
                cnt.columns = ['deck','games']
                h = 300 + (len(cnt) * 20)
                
                fig_p = px.bar(cnt, x='games', y='deck', orientation='h', height=h, 
                               title="Games Played per Deck", 
                               labels={'games': 'Games Played', 'deck': 'Deck Name'},
                               color_discrete_sequence=['#4a90e2'])
                fig_p.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info("No deck configuration found.")

        with t2:
            st.subheader("The 'Completionist' Checklist")
            if 'deck' in df_d.columns and 'player' in df_d.columns:
                all_p = sorted(df_d['player'].dropna().unique())
                all_d = sorted(df_d['deck'].dropna().unique())
                
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
            else:
                st.info("Missing player or deck logic to compute checklist.")

        with t3:
            st.subheader("Set Freshness Tracker")
            if 'match_uuid' in raw_df.columns and 'type' in raw_df.columns:
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
            else:
                st.info("Missing match UUIDs to compute tracking metrics.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption(f"App Version: {APP_VERSION}")
st.markdown("© 2026 MTG Stats Analysis | Built with ❤️ using Streamlit")