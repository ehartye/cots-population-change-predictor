import sqlite3
import pandas as pd
import numpy as np
from scipy import stats

def analyze_precursors():
    conn = sqlite3.connect('reefcheck.db')
    
    # Get all COTS change events
    cots_changes = pd.read_sql_query("""
        SELECT 
            site_id,
            survey_id_1,
            survey_id_2,
            change,
            total_cots_1,
            total_cots_2,
            date_1,
            date_2
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
        
        result_data = []
        count_columns = ['segment_1_count', 'segment_2_count', 'segment_3_count', 'segment_4_count']
        
        for _, row in df.iterrows():
            valid_counts = []
            for col in count_columns:
                count = row[col]
                if pd.isna(count) or count == '':
                    continue
                try:
                    count_float = float(count)
                    valid_counts.append(count_float)
                except (ValueError, TypeError):
                    continue
            
            if valid_counts:
                result_data.append({
                    'organism_code': row['organism_code'],
                    'total_count': sum(valid_counts),
                    'segments_measured': len(valid_counts)
                })
        
        return pd.DataFrame(result_data)
    
    def collect_group_data(survey_ids):
        all_data = []
        for survey_id in survey_ids:
            counts = get_organism_counts(survey_id)
            if not counts.empty:
                valid_counts = counts[counts['segments_measured'] >= 2]
                if not valid_counts.empty:
                    all_data.append(valid_counts.set_index('organism_code')['total_count'])
        return pd.DataFrame(all_data)
    
    # Get data for each group - using survey_id_1 (initial surveys)
    increase_surveys = cots_changes[cots_changes['change'] > 0]['survey_id_1']
    decrease_surveys = cots_changes[cots_changes['change'] < 0]['survey_id_1']
    control_survey_ids = control_surveys['survey_id']
    
    increase_data = collect_group_data(increase_surveys)
    decrease_data = collect_group_data(decrease_surveys)
    control_data = collect_group_data(control_survey_ids)
    
    print("=== Analysis of Initial Survey Patterns Before COTS Changes ===")
    print("Note: Analysis only includes measurements with at least 2 valid segments\n")
    
    # Identify significant organisms in each group
    significant_increases = set()
    significant_decreases = set()
    
    def analyze_organism(organism, group_data, group_name):
        if organism in group_data:
            group_values = group_data[organism].dropna()
            control_values = control_data[organism].dropna()
            if len(group_values) >= 2:
                stat, p_value = stats.ttest_ind(group_values, control_values)
                
                # Calculate effect size with validation
                control_std = control_values.std()
                if control_std > 0:  # Only calculate if std dev is non-zero
                    effect_size = (group_values.mean() - control_values.mean()) / control_std
                else:
                    # If std dev is 0, use a different effect size calculation
                    # Here we use the raw difference if all control values are the same
                    effect_size = group_values.mean() - control_values.mean()
                
                presence_rate = (group_data[organism] > 0).mean()
                return {
                    'organism': organism,
                    'group': group_name,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'presence_rate': presence_rate,
                    'mean': group_values.mean(),
                    'control_mean': control_values.mean(),
                    'sample_size': len(group_values)
                }
        return None
    
    all_results = []
    shared_organisms = set()
    
    for organism in control_data.columns:
        increase_result = analyze_organism(organism, increase_data, "Increase")
        decrease_result = analyze_organism(organism, decrease_data, "Decrease")
        
        if increase_result and increase_result['p_value'] < 0.05:
            significant_increases.add(organism)
            all_results.append(increase_result)
        
        if decrease_result and decrease_result['p_value'] < 0.05:
            significant_decreases.add(organism)
            all_results.append(decrease_result)
    
    shared_organisms = significant_increases & significant_decreases
    unique_increases = significant_increases - shared_organisms
    unique_decreases = significant_decreases - shared_organisms
    
    def print_organism_stats(results, title):
        print(f"\n=== {title} ===")
        print(f"Total patterns found: {len(results)}")
        
        for result in sorted(results, key=lambda x: abs(x['effect_size']), reverse=True):
            print(f"\n{result['organism']}:")
            print(f"  Effect size: {result['effect_size']:.2f}")
            print(f"  P-value: {result['p_value']:.4f}")
            print(f"  Present in {result['presence_rate']*100:.1f}% of initial surveys")
            print(f"  Mean before {result['group'].lower()}: {result['mean']:.2f}")
            print(f"  Mean during control: {result['control_mean']:.2f}")
            print(f"  Percent difference: {((result['mean'] - result['control_mean']) / result['control_mean'] * 100):.1f}%")
            print(f"  Sample size: {result['sample_size']}")
    
    # Print shared patterns
    if shared_organisms:
        print("\n=== Organisms Significant Before Both Increases and Decreases ===")
        shared_results = [r for r in all_results if r['organism'] in shared_organisms]
        for organism in shared_organisms:
            inc_result = next(r for r in shared_results if r['organism'] == organism and r['group'] == "Increase")
            dec_result = next(r for r in shared_results if r['organism'] == organism and r['group'] == "Decrease")
            
            print(f"\n{organism}:")
            print("  Before Increases:")
            print(f"    Effect size: {inc_result['effect_size']:.2f}")
            print(f"    P-value: {inc_result['p_value']:.4f}")
            print(f"    Present in {inc_result['presence_rate']*100:.1f}% of surveys")
            print(f"    Mean: {inc_result['mean']:.2f} (Control: {inc_result['control_mean']:.2f})")
            print(f"    Percent difference: {((inc_result['mean'] - inc_result['control_mean']) / inc_result['control_mean'] * 100):.1f}%")
            
            print("  Before Decreases:")
            print(f"    Effect size: {dec_result['effect_size']:.2f}")
            print(f"    P-value: {dec_result['p_value']:.4f}")
            print(f"    Present in {dec_result['presence_rate']*100:.1f}% of surveys")
            print(f"    Mean: {dec_result['mean']:.2f} (Control: {dec_result['control_mean']:.2f})")
            print(f"    Percent difference: {((dec_result['mean'] - dec_result['control_mean']) / dec_result['control_mean'] * 100):.1f}%")
    
    # Print unique patterns for each group
    increase_results = [r for r in all_results if r['organism'] in unique_increases]
    decrease_results = [r for r in all_results if r['organism'] in unique_decreases]
    
    print_organism_stats(increase_results, "Unique Patterns Before COTS INCREASES")
    print_organism_stats(decrease_results, "Unique Patterns Before COTS DECREASES")
    
    conn.close()

if __name__ == "__main__":
    analyze_precursors()
