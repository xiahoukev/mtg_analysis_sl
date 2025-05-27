# -------------------- LIBRARIES --------------------
import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------- CONSTANTS --------------------
mana_colors = {
    'Blue': "#3f17d3",
    'Red': "#f30505",
    'Green': "#076b07",
    'White': '#FFFFFF',
    'Black': "#000000",
    'Colorless': '#7f7f7f',
    'Unknown': '#cccccc'
}

# Set page configuration
st.set_page_config(page_title="MTG Stats Analysis", layout="wide")

# App Title
st.title("Magic: The Gathering Stats Analysis")

# Load data from Excel
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')  # Replace 'Sheet1' if necessary
    return df

# Define the path to your Excel file
DATA_FILE = 'mtg_data.xlsx'  # Ensure this file exists in the same directory

# Load the data with error handling
try:
    df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(f"Excel file '{DATA_FILE}' not found. Please ensure it's in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the Excel file: {e}")
    st.stop()

# Data preprocessing
df['position'] = pd.to_numeric(df['position'], errors='coerce')
df['deck'] = df['deck'].replace({'Elven': 'Elves'})  # Ensure consistency

# Sidebar filters using Expanders
st.sidebar.header("Filter Options")

with st.sidebar.expander("Main Filters", expanded=True):

    # Slice for 'draw method'
    draw_options = sorted(df['draw_type'].unique())
    draw_type = st.multiselect("Draw Type", options=draw_options, default=draw_options)

    # Slicer for 'type'
    type_options = sorted(df['type'].unique())
    selected_type = st.multiselect("Select Type", options=type_options, default=type_options)

    # Slicer for 'deck'
    filtered_decks = df[df['type'].isin(selected_type)]['deck'].unique()
    filtered_decks = sorted(filtered_decks)
    selected_deck = st.multiselect("Select Deck", options=filtered_decks, default=filtered_decks)

    # Slicer for 'color'
    filtered_color = df[df['type'].isin(selected_type)]['primary_mana'].unique()
    filtered_color = sorted(filtered_color)
    selected_color = st.multiselect("Select Colour", options=filtered_color, default=filtered_color)
           

with st.sidebar.expander("Chart 2 Options", expanded=False):
    # Toggle for Chart 2 view
    chart2_view = st.radio(
        "Select Chart 2 View:",
        options=["Average Position", "Distribution"],
        index=0,
        horizontal=True
    )

    # Player filter for Chart 2
    chart2_player_options = sorted(df['player'].unique())
    chart2_selected_players = st.multiselect(
        "Select Player(s) for Chart 2",
        options=chart2_player_options,
        default=chart2_player_options
    )

# Apply filters to data
filtered_df = df[
    (df['draw_type'].isin(draw_type)) &
    (df['type'].isin(selected_type)) &
    (df['deck'].isin(selected_deck)) &
    (df['primary_mana'].isin(selected_color))
]

# Enhanced Summary Boxes
def create_summary_boxes_enhanced(filtered_df):
    # Calculate average positions per player based on filtered data
    avg_positions = filtered_df.groupby('player')['position'].mean().reset_index()
    # Sort by average position (ascending: lower is better)
    avg_positions = avg_positions.sort_values('position').reset_index(drop=True)
    # Select top 5 players
    top4 = avg_positions.head(4)
    
    # Assign colors and icons for ranks 4th to 1st
    # Changing 4th to 'skyblue'
    box_colors = ['skyblue', 'orange', 'silver', 'gold']  # 5th to 1st
    box_icons = ['4️⃣', '🥉', '🥈', '🥇']  # 4th to 1st

    # Create four columns
    cols = st.columns(4)
    
    # Reverse the order to have 5th on the left and 1st on the right
    top4 = top4[::-1].reset_index(drop=True)
    
    # Prepare ordered_players list (only actual players, no N/A)
    ordered_players = list(top4['player'])  # Ordered from 5th to 1st
    
    for i, col in enumerate(cols):
        if i < len(top4):
            player = top4.iloc[i]['player']
            avg_pos = round(top4.iloc[i]['position'], 2)
            color = box_colors[i]
            icon = box_icons[i]
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
            # Remaining boxes are N/A
            color = box_colors[i]
            icon = box_icons[i]
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
    
    # Return the ordered list of players for consistent ordering in charts
    return ordered_players

# Call the enhanced summary boxes and get the ordered players
ordered_players = create_summary_boxes_enhanced(filtered_df)

st.markdown("---")  # Separator

st.header("Detailed Analysis")

if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your selections.")
else:
    # === Chart 1: Distribution of Ranks 1-4 by Player ===
    
    # Filter ranks to include only 1, 2, 3, 4
    chart1_df = filtered_df[filtered_df['position'].isin([1, 2, 3, 4])]
    
    # Convert ranks to character representation
    rank_map = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th'}
    chart1_df['rank_char'] = chart1_df['position'].map(rank_map)
    
    # Group by player and rank
    chart1_grouped = chart1_df.groupby(['player', 'rank_char']).size().reset_index(name='count')
    
    # Calculate percentage distribution per player
    chart1_grouped['percentage'] = chart1_grouped.groupby('player')['count'].transform(lambda x: (x / x.sum()) * 100)
    
    # Ensure ordered players
    chart1_grouped['player'] = pd.Categorical(chart1_grouped['player'], categories=ordered_players, ordered=True)
    
    # Define discrete color mapping for ranks, changing '4th' to skyblue
    rank_colors = {
        '1st': 'gold',
        '2nd': 'silver',
        '3rd': 'orange',
        '4th': 'skyblue'  # Changed from 'pink' to 'skyblue'
    }
    
    # Create the stacked bar chart with category_orders to enforce player order and legend order
    fig1 = px.bar(
        chart1_grouped,
        x='player',
        y='percentage',
        color='rank_char',
        color_discrete_map=rank_colors,
        category_orders={
            'rank_char': ['1st', '2nd', '3rd', '4th'],
            'player': ordered_players
        },  # Ensure correct order
        barmode='stack',
        title="Distribution of Ranks 1-4 by Player",
        labels={'percentage': 'Percentage (%)', 'player': 'Player', 'rank_char': 'Rank'},
        hover_data={'player': True, 'rank_char': True, 'percentage': ':.2f'}
    )
    
    fig1.update_layout(
        legend_title_text='Rank',
        #yaxis=dict(range=[0, 100], dtick=20),
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    
    # Reorder legend items by specifying 'rank_char' order
    fig1.update_layout(
        legend=dict(
            itemsizing='constant',
            traceorder='normal'  # Ensures the legend follows the category_orders
        )
    )
    
    # Add percentage labels inside bar segments if percentage > 0%
    for trace in fig1.data:
        trace.text = [f"{val:.1f}%" if val > 0 else "" for val in trace.y]
        trace.textposition = 'inside'
    
    # === Chart 2: Average Position or Distribution of Decks ===
    
    # Precompute avg_position_deck and deck_order for both views
    # This ensures deck_order is available for both views and avoids NameError
    # Calculate average position per deck
    avg_position_deck = filtered_df.groupby('deck')['position'].mean().reset_index()
    avg_position_deck['avg_position'] = avg_position_deck['position'].round(2)
    
    # Calculate total games per deck
    total_games_deck_chart2 = filtered_df.groupby('deck').size().reset_index(name='total_games')
    
    # Merge to have total games information
    avg_position_deck = avg_position_deck.merge(total_games_deck_chart2, on='deck', how='left')
    
    # Sort decks by average position (descending: worst to best) to reverse the previous order
    avg_position_deck = avg_position_deck.sort_values('avg_position', ascending=False).reset_index(drop=True)
    
    # Define deck_order based on sorted avg_position_deck
    deck_order = list(avg_position_deck['deck'])  # From worst to best
    
    # Filter Chart 2 data based on selected players
    if len(chart2_selected_players) == 0:
        chart2_warning = True
    else:
        chart2_warning = False
        # Further filter 'filtered_df' based on selected players
        chart2_filtered_df = filtered_df[filtered_df['player'].isin(chart2_selected_players)]
        
        if chart2_view == "Average Position":
            # Calculate average position per deck for selected players
            avg_position_deck_chart2 = chart2_filtered_df.groupby('deck')['position'].mean().reset_index()
            avg_position_deck_chart2['avg_position'] = avg_position_deck_chart2['position'].round(2)
            
            # Calculate total games per deck for selected players
            total_games_deck_chart2_selected = chart2_filtered_df.groupby('deck').size().reset_index(name='total_games')
            
            # Merge to have total games information
            avg_position_deck_chart2 = avg_position_deck_chart2.merge(total_games_deck_chart2_selected, on='deck', how='left')
            
            # Sort decks by average position (descending: worst to best)
            avg_position_deck_chart2 = avg_position_deck_chart2.sort_values('avg_position', ascending=False).reset_index(drop=True)
            
            # Create the average position bar chart with uniform skyblue color and no legend
            fig2 = px.bar(
                avg_position_deck_chart2,
                x='deck',
                y='avg_position',
                title="Average Position per Deck",
                labels={'avg_position': 'Average Position', 'deck': 'Deck'},
                hover_data={'deck': True, 'avg_position': ':.2f'},
                color_discrete_sequence=['skyblue'] * len(avg_position_deck_chart2),  # Same skyblue color for all bars
            )
            
            fig2.update_layout(
                yaxis=dict(range=[1, 4], dtick=0.5),
                uniformtext_minsize=8,
                uniformtext_mode='hide',
                showlegend=False  # Remove legend here
            )
            
            # Add labels to bars (average position)
            fig2.update_traces(
                text=avg_position_deck_chart2['avg_position'],
                textposition='outside'
            )
            
            # Add 'n' games played as annotations on top of each deck's bar
            annotations = []
            for idx, row in avg_position_deck_chart2.iterrows():
                annotations.append(dict(
                    x=row['deck'],
                    y=row['avg_position'] + 0.1,  # Slightly above the bar
                    text=f"n={row['total_games']}",
                    showarrow=False,
                    font=dict(
                        size=12,
                        color="black"
                    )
                ))
            
            fig2.update_layout(
                annotations=annotations
            )
            
        elif chart2_view == "Distribution":
            # Calculate distribution of ranks per deck for selected players
            chart2_distribution_df = chart2_filtered_df[chart2_filtered_df['position'].isin([1, 2, 3, 4])]
            chart2_distribution_df['rank_char'] = chart2_distribution_df['position'].map(rank_map)
            
            # Group by deck and rank
            chart2_grouped = chart2_distribution_df.groupby(['deck', 'rank_char']).size().reset_index(name='count')
            
            # Calculate percentage distribution per deck
            chart2_grouped['percentage'] = chart2_grouped.groupby('deck')['count'].transform(lambda x: (x / x.sum()) * 100)
            
            # Ensure decks are ordered as per deck_order
            chart2_grouped['deck'] = pd.Categorical(chart2_grouped['deck'], categories=deck_order, ordered=True)
            
            # Define discrete color mapping for ranks, changing '4th' to skyblue
            rank_colors_distribution = {
                '1st': 'gold',
                '2nd': 'silver',
                '3rd': 'orange',
                '4th': 'skyblue'
            }
            
            # Create the stacked bar chart with category_orders to enforce deck order and legend order
            fig2 = px.bar(
                chart2_grouped,
                x='deck',
                y='percentage',
                color='rank_char',
                color_discrete_map=rank_colors_distribution,
                category_orders={
                    'rank_char': ['1st', '2nd', '3rd', '4th'],
                    'deck': deck_order
                },  # Ensure correct order
                barmode='stack',
                title="Distribution of Ranks 1-4 by Deck",
                labels={'percentage': 'Percentage (%)', 'deck': 'Deck', 'rank_char': 'Rank'},
                hover_data={'deck': True, 'rank_char': True, 'percentage': ':.2f'}
            )
            
            fig2.update_layout(
                legend_title_text='Rank',
                yaxis=dict(range=[0, 100], dtick=20),
                uniformtext_minsize=8,
                uniformtext_mode='hide'
            )
            
            # Reorder legend items by specifying 'rank_char' order
            fig2.update_layout(
                legend=dict(
                    itemsizing='constant',
                    traceorder='normal'  # Ensures the legend follows the category_orders
                )
            )
            
            # Add percentage labels inside bar segments if percentage > 0%
            for trace in fig2.data:
                trace.text = [f"{val:.1f}%" if val > 0 else "" for val in trace.y]
                trace.textposition = 'inside'
    
    # Display charts side by side
    chart1_area, chart2_area = st.columns(2)
    
    with chart1_area:
        st.plotly_chart(fig1, use_container_width=True)
    
    with chart2_area:
        if chart2_warning:
            st.warning("Please select at least one player to view Chart 2.")
        else:
            st.plotly_chart(fig2, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 MTG Stats Analysis | Built with ❤️ using Streamlit")
