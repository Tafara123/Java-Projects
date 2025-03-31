import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_candidate_ranking_chart(df, limit=10):
    """
    Create a horizontal bar chart showing candidate rankings
    
    Args:
        df (DataFrame): DataFrame with candidate scoring results
        limit (int): Maximum number of candidates to show
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if 'Overall Match (%)' not in df.columns or len(df) == 0:
        return None
    
    # Sort and limit
    sorted_df = df.sort_values('Overall Match (%)', ascending=False).head(limit)
    
    # Create chart
    fig = px.bar(
        sorted_df,
        y='Candidate Name',
        x='Overall Match (%)',
        orientation='h',
        title=f'Top {min(limit, len(sorted_df))} Candidates by Match Score',
        color='Overall Match (%)',
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={'Overall Match (%)': 'Match Score (%)'}
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis={'range': [0, 100]}
    )
    
    return fig

def create_score_distribution_chart(df):
    """
    Create a histogram showing the distribution of match scores
    
    Args:
        df (DataFrame): DataFrame with candidate scoring results
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if 'Overall Match (%)' not in df.columns or len(df) == 0:
        return None
    
    # Create histogram
    fig = px.histogram(
        df,
        x='Overall Match (%)',
        nbins=10,
        title='Distribution of Match Scores',
        color_discrete_sequence=['#3D85C6'],
        labels={'Overall Match (%)': 'Match Score (%)'}
    )
    
    # Add average line
    avg_match = df['Overall Match (%)'].mean()
    fig.add_vline(
        x=avg_match,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: {avg_match:.1f}%",
        annotation_position="top right"
    )
    
    return fig

def create_skill_category_chart(df):
    """
    Create a bar chart showing average match by skill category
    
    Args:
        df (DataFrame): DataFrame with candidate scoring results
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Find skill category columns
    skill_columns = [col for col in df.columns if 'Match (%)' in col and col != 'Overall Match (%)']
    
    if not skill_columns or len(df) == 0:
        return None
    
    # Calculate average for each category
    avg_scores = {}
    for col in skill_columns:
        category_name = col.replace(' Match (%)', '')
        avg_scores[category_name] = df[col].mean()
    
    # Create DataFrame for plotting
    avg_df = pd.DataFrame({
        'Skill Category': list(avg_scores.keys()),
        'Average Match (%)': list(avg_scores.values())
    })
    
    # Create chart
    fig = px.bar(
        avg_df.sort_values('Average Match (%)', ascending=False),
        x='Skill Category',
        y='Average Match (%)',
        title='Average Match Score by Skill Category',
        color='Average Match (%)',
        color_continuous_scale=px.colors.sequential.Viridis,
        text='Average Match (%)'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 100])
    
    return fig

def create_candidate_comparison_chart(df, candidates=None, limit=5):
    """
    Create a grouped bar chart comparing candidates across skills
    
    Args:
        df (DataFrame): DataFrame with candidate scoring results
        candidates (list): List of candidate names to compare
        limit (int): Maximum number of candidates to show if none specified
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Find skill category columns
    skill_columns = [col for col in df.columns if 'Match (%)' in col]
    
    if not skill_columns or len(df) == 0:
        return None
    
    # Filter to specific candidates or use top ones
    if candidates:
        compare_df = df[df['Candidate Name'].isin(candidates)]
    else:
        compare_df = df.sort_values('Overall Match (%)', ascending=False).head(limit)
    
    if len(compare_df) == 0:
        return None
    
    # Melt the DataFrame for plotting
    plot_df = pd.melt(
        compare_df, 
        id_vars=['Candidate Name'],
        value_vars=skill_columns,
        var_name='Skill Category',
        value_name='Match Score'
    )
    
    # Clean up category names
    plot_df['Skill Category'] = plot_df['Skill Category'].str.replace(' Match (%)', '')
    
    # Create chart
    fig = px.bar(
        plot_df,
        x='Candidate Name',
        y='Match Score',
        color='Skill Category',
        barmode='group',
        title='Candidate Comparison by Skill Categories',
        labels={'Match Score': 'Match Score (%)'}
    )
    
    fig.update_layout(yaxis_range=[0, 100])
    
    return fig

def create_skills_radar_chart(candidate_data, job_profile):
    """
    Create a radar chart showing candidate's skills vs job requirements
    
    Args:
        candidate_data (dict): Candidate profile data
        job_profile (dict): Job profile data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if not candidate_data or not job_profile or 'match_details' not in candidate_data:
        return None
    
    # Get skill matches
    skill_matches = candidate_data['match_details'].get('skill_matches', {})
    
    if not skill_matches:
        return None
    
    # Prepare data for radar chart
    categories = list(skill_matches.keys())
    values = [skill_matches[cat]['score'] * 100 for cat in categories]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=candidate_data.get('candidate_name', 'Candidate')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title=f"Skill Match Profile: {candidate_data.get('candidate_name', 'Candidate')}",
        showlegend=True
    )
    
    return fig

def create_dashboard(df, raw_results=None):
    """
    Create a comprehensive dashboard with multiple visualizations
    
    Args:
        df (DataFrame): DataFrame with candidate scoring results
        raw_results (list): List of raw candidate results
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if len(df) == 0:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Candidate Ranking', 
            'Match Score Distribution',
            'Skill Category Analysis', 
            'Experience vs Match Score'
        ),
        specs=[
            [{"type": "bar"}, {"type": "histogram"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # 1. Candidate Ranking (top 10)
    if 'Overall Match (%)' in df.columns:
        top_df = df.sort_values('Overall Match (%)', ascending=False).head(10)
        
        fig.add_trace(
            go.Bar(
                y=top_df['Candidate Name'],
                x=top_df['Overall Match (%)'],
                orientation='h',
                marker=dict(
                    color=top_df['Overall Match (%)'],
                    colorscale='Viridis'
                )
            ),
            row=1, col=1
        )
        
        fig.update_xaxes(title_text='Match %', range=[0, 100], row=1, col=1)
        fig.update_yaxes(title_text='', categoryorder='total ascending', row=1, col=1)
    
    # 2. Match Score Distribution
    if 'Overall Match (%)' in df.columns:
        fig.add_trace(
            go.Histogram(
                x=df['Overall Match (%)'],
                nbinsx=10,
                marker_color='#3D85C6'
            ),
            row=1, col=2
        )
        
        avg_match = df['Overall Match (%)'].mean()
        fig.add_vline(
            x=avg_match, 
            line_dash="dash", 
            line_color="red", 
            row=1, col=2
        )
        
        fig.update_xaxes(title_text='Match %', range=[0, 100], row=1, col=2)
        fig.update_yaxes(title_text='Count', row=1, col=2)
    
    # 3. Skill Category Analysis
    skill_columns = [col for col in df.columns if 'Match (%)' in col and col != 'Overall Match (%)']
    
    if skill_columns:
        # Calculate average for each category
        avg_scores = {}
        for col in skill_columns:
            category_name = col.replace(' Match (%)', '')
            avg_scores[category_name] = df[col].mean()
        
        # Sort by average score
        sorted_categories = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        fig.add_trace(
            go.Bar(
                x=[cat[0] for cat in sorted_categories],
                y=[cat[1] for cat in sorted_categories],
                marker=dict(
                    color=[cat[1] for cat in sorted_categories],
                    colorscale='Viridis'
                )
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text='Category', row=2, col=1)
        fig.update_yaxes(title_text='Avg Match %', range=[0, 100], row=2, col=1)
    
    # 4. Experience vs Match Score
    if 'Years Experience' in df.columns and 'Overall Match (%)' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Years Experience'],
                y=df['Overall Match (%)'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df['Overall Match (%)'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=df['Candidate Name']
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text='Years Experience', row=2, col=2)
        fig.update_yaxes(title_text='Match %', range=[0, 100], row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text='Candidate Scoring Dashboard',
        showlegend=False
    )
    
    return fig

def generate_insights(df, raw_results=None):
    """
    Generate text insights about candidate scoring results
    
    Args:
        df (DataFrame): DataFrame with candidate scoring results
        raw_results (list): List of raw candidate results
        
    Returns:
        str: Insights text
    """
    if len(df) == 0:
        return "No data available for analysis."
    
    insights = []
    
    # Basic statistics
    if 'Overall Match (%)' in df.columns:
        avg_match = df['Overall Match (%)'].mean()
        max_match = df['Overall Match (%)'].max()
        min_match = df['Overall Match (%)'].min()
        
        insights.append(f"**Overall Match Statistics**: The average match score is {avg_match:.1f}%, with the highest being {max_match:.1f}% and the lowest being {min_match:.1f}%.")
        
        # Binning candidates by match score
        high_matches = len(df[df['Overall Match (%)'] >= 75])
        medium_matches = len(df[(df['Overall Match (%)'] >= 50) & (df['Overall Match (%)'] < 75)])
        low_matches = len(df[df['Overall Match (%)'] < 50])
        
        insights.append(f"**Match Distribution**: {high_matches} candidates have high match scores (≥75%), {medium_matches} have medium match scores (50-74%), and {low_matches} have low match scores (<50%).")
    
    # Top candidates
    top_candidates = df.sort_values('Overall Match (%)', ascending=False).head(3)
    if len(top_candidates) > 0:
        top_names = ", ".join(top_candidates['Candidate Name'].tolist())
        insights.append(f"**Top Candidates**: The top 3 candidates are {top_names}.")
    
    # Experience insights
    if 'Years Experience' in df.columns:
        avg_exp = df['Years Experience'].mean()
        max_exp = df['Years Experience'].max()
        top_exp_candidate = df.loc[df['Years Experience'].idxmax(), 'Candidate Name']
        
        insights.append(f"**Experience**: The average work experience is {avg_exp:.1f} years. The most experienced candidate is {top_exp_candidate} with {max_exp:.1f} years of experience.")
    
    # Skill gap insights
    skill_columns = [col for col in df.columns if 'Match (%)' in col and col != 'Overall Match (%)']
    if skill_columns:
        avg_scores = {}
        for col in skill_columns:
            category_name = col.replace(' Match (%)', '')
            avg_scores[category_name] = df[col].mean()
        
        # Highest and lowest skill matches
        highest_skill = max(avg_scores.items(), key=lambda x: x[1])
        lowest_skill = min(avg_scores.items(), key=lambda x: x[1])
        
        insights.append(f"**Skill Analysis**: The candidate pool is strongest in {highest_skill[0]} (average {highest_skill[1]:.1f}%) and weakest in {lowest_skill[0]} (average {lowest_skill[1]:.1f}%).")
    
    # Recommendation
    if len(df) > 0 and 'Overall Match (%)' in df.columns:
        top_candidate = df.sort_values('Overall Match (%)', ascending=False).iloc[0]
        
        if top_candidate['Overall Match (%)'] >= 75:
            recommendation = f"Based on the analysis, **{top_candidate['Candidate Name']}** is an excellent match for this position with a match score of {top_candidate['Overall Match (%)']:.1f}%."
        elif top_candidate['Overall Match (%)'] >= 60:
            recommendation = f"Based on the analysis, **{top_candidate['Candidate Name']}** is a good match for this position with a match score of {top_candidate['Overall Match (%)']:.1f}%, but may require some additional training."
        else:
            recommendation = "None of the candidates are strong matches for this position. Consider expanding the search or adjusting the job requirements."
        
        insights.append(f"**Recommendation**: {recommendation}")
    
    return "\n\n".join(insights)

def export_excel_report(df, file_path):
    """
    Export a formatted Excel report of candidate scoring results
    
    Args:
        df (DataFrame): DataFrame with candidate scoring results
        file_path (str): Path to save the Excel file
        
    Returns:
        str: Path to the saved file
    """
    try:
        import xlsxwriter
        
        # Create Excel writer
        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
        
        # Write DataFrame to Excel
        df.to_excel(writer, sheet_name='Candidate Scoring', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Candidate Scoring']
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'bg_color': '#D7E4BC',
            'border': 1
        })
        
        # Set column widths
        for i, col in enumerate(df.columns):
            col_width = max(len(col) + 2, df[col].astype(str).map(len).max() + 2)
            worksheet.set_column(i, i, col_width)
        
        # Apply header format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Add conditional formatting for match percentages
        match_columns = [col_num for col_num, col in enumerate(df.columns) if '(%)' in col]
        
        for col_num in match_columns:
            worksheet.conditional_format(1, col_num, len(df) + 1, col_num, {
                'type': '3_color_scale',
                'min_color': "#FF9999",
                'mid_color': "#FFFF99",
                'max_color': "#99CC99"
            })
        
        # Save the workbook
        writer.save()
        
        return file_path
    except ImportError:
        # Fallback to basic Excel export
        df.to_excel(file_path, index=False)
        return file_path