import sqlite3
import pandas as pd
import numpy as np
from scipy import stats

def analyze_significance():
    conn = sqlite3.connect('reefcheck.db')
    
    # Get all COTS change events
    cots_changes = pd.read_sql_query("""
        SELECT 
            site_id,
            survey_id_1,
            survey_id_2,
            change,
            total_cots_1,
            total_cots_2
        FROM cots_changes
    """, conn)
    
    # Get all surveys
    all_surveys = pd.read_sql_query("""
        SELECT DISTINCT survey_id, site_id, date
        FROM belt
        ORDER BY date
    """, conn)
    
    # Identify non-COTS-change surveys (control group)
    change_surveys = set(cots_changes['survey_id_1'].tolist() + cots_changes['survey_id_2'].tolist())
    control_surveys = all_surveys[~all_surveys['survey_id'].isin(change_surveys)]
    
    def get_organism_counts(survey_id):
        # Get raw segment data
        query = """
            SELECT 
                organism_code,
                segment_1_count,
                segment_2_count,
                segment_3_count,
                segment_4_count
            FROM belt
            WHERE survey_id = ?
            AND organism_code != 'COTS'
            AND organism_code IS NOT NULL 
            AND organism_code != ''
        """
        df = pd.read_sql_query(query, conn, params=[survey_id])
        
        # Process counts properly
        result_data = []
        count_columns = ['segment_1_count', 'segment_2_count', 'segment_3_count', 'segment_4_count']
        
        for _, row in df.iterrows():
            valid_counts = []
            for col in count_columns:
                count = row[col]
                # Skip any empty strings or null values
                if pd.isna(count) or count == '':
                    continue
                try:
                    # Convert to float and validate
                    count_float = float(count)
                    valid_counts.append(count_float)
                except (ValueError, TypeError):
                    continue
            
            # Only include if we have valid counts
            if valid_counts:
                result_data.append({
                    'organism_code': row['organism_code'],
                    'total_count': sum(valid_counts),
                    'segments_measured': len(valid_counts)
                })
        
        return pd.DataFrame(result_data)
    
    # Collect organism data for each group
    def collect_group_data(survey_ids):
        all_data = []
        for survey_id in survey_ids:
            counts = get_organism_counts(survey_id)
            if not counts.empty:
                # Only use total_count where we have at least 2 segments measured
                valid_counts = counts[counts['segments_measured'] >= 2]
                if not valid_counts.empty:
                    all_data.append(valid_counts.set_index('organism_code')['total_count'])
        return pd.DataFrame(all_data)
    
    # Get data for each group
    increase_surveys = cots_changes[cots_changes['change'] > 0]['survey_id_2']
    decrease_surveys = cots_changes[cots_changes['change'] < 0]['survey_id_2']
    control_survey_ids = control_surveys['survey_id']
    
    increase_data = collect_group_data(increase_surveys)
    decrease_data = collect_group_data(decrease_surveys)
    control_data = collect_group_data(control_survey_ids)
    
    # Perform statistical analysis
    print("=== Statistical Analysis of Organism Patterns ===")
    print("Note: Analysis only includes measurements with at least 2 valid segments")
    print("      Organisms significant in both increase and decrease events are excluded\n")
    
    # First pass to identify significant organisms in each group
    significant_increases = set()
    significant_decreases = set()
    
    # Check increases
    for organism in control_data.columns:
        if organism in increase_data:
            increase_values = increase_data[organism].dropna()
            control_values = control_data[organism].dropna()
            if len(increase_values) >= 2:
                stat, p_value = stats.ttest_ind(increase_values, control_values)
                if p_value < 0.05:
                    significant_increases.add(organism)
    
    # Check decreases
    for organism in control_data.columns:
        if organism in decrease_data:
            decrease_values = decrease_data[organism].dropna()
            control_values = control_data[organism].dropna()
            if len(decrease_values) >= 2:
                stat, p_value = stats.ttest_ind(decrease_values, control_values)
                if p_value < 0.05:
                    significant_decreases.add(organism)
    
    # Remove organisms that are significant in both
    significant_in_both = significant_increases & significant_decreases
    significant_increases = significant_increases - significant_in_both
    significant_decreases = significant_decreases - significant_in_both
    
    if significant_in_both:
        print("Excluded organisms (significant in both):")
        for org in sorted(significant_in_both):
            print(f"- {org}")
        print()
    
    increase_results = []
    decrease_results = []
    
    # Process increases
    for organism in significant_increases:
        increase_values = increase_data[organism].dropna()
        control_values = control_data[organism].dropna()
        stat, p_value = stats.ttest_ind(increase_values, control_values)
        effect_size = (increase_values.mean() - control_values.mean()) / control_values.std()
        increase_results.append({
            'organism': organism,
            'p_value': p_value,
            'effect_size': effect_size,
            'presence_rate': (increase_data[organism] > 0).mean(),
            'mean_during_event': increase_values.mean(),
            'mean_control': control_values.mean(),
            'sample_size': len(increase_values)
        })
    
    # Process decreases
    for organism in significant_decreases:
        decrease_values = decrease_data[organism].dropna()
        control_values = control_data[organism].dropna()
        stat, p_value = stats.ttest_ind(decrease_values, control_values)
        effect_size = (decrease_values.mean() - control_values.mean()) / control_values.std()
        decrease_results.append({
            'organism': organism,
            'p_value': p_value,
            'effect_size': effect_size,
            'presence_rate': (decrease_data[organism] > 0).mean(),
            'mean_during_event': decrease_values.mean(),
            'mean_control': control_values.mean(),
            'sample_size': len(decrease_values)
        })
    
    # Sort and print results by group
    def print_results(results, event_type):
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(by='effect_size', ascending=False)
            
            print(f"\n=== Unique Significant Patterns During COTS {event_type} ===")
            print(f"Total patterns found: {len(results_df)}")
            
            for _, row in results_df.iterrows():
                print(f"\n{row['organism']}:")
                print(f"  Effect size: {row['effect_size']:.2f}")
                print(f"  P-value: {row['p_value']:.4f}")
                print(f"  Present in {row['presence_rate']*100:.1f}% of events")
                print(f"  Mean during {event_type.lower()}: {row['mean_during_event']:.2f}")
                print(f"  Mean during control: {row['mean_control']:.2f}")
                print(f"  Percent difference: {((row['mean_during_event'] - row['mean_control']) / row['mean_control'] * 100):.1f}%")
                print(f"  Sample size: {row['sample_size']}")
    
    print_results(increase_results, "INCREASES")
    print_results(decrease_results, "DECREASES")
    
    conn.close()

if __name__ == "__main__":
    analyze_significance()
