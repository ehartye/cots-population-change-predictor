import sqlite3
import pandas as pd
import numpy as np
from scipy import stats

def analyze_substrate_precursors():
    conn = sqlite3.connect('reefcheck.db')
    
    # Create temporary index to speed up joins
    conn.execute("CREATE INDEX IF NOT EXISTS idx_substrate_survey ON substrate(survey_id)")
    
    # Get all COTS change events with their substrate data in one query
    change_query = """
    WITH survey_totals AS (
        SELECT 
            survey_id,
            substrate_code,
            SUM(total) as total_coverage
        FROM substrate
        WHERE substrate_code IS NOT NULL 
        AND substrate_code != ''
        AND segment_code IN ('S1', 'S2', 'S3', 'S4')
        AND total IS NOT NULL 
        AND total != ''
        GROUP BY survey_id, substrate_code
    )
    SELECT 
        c.*,
        s1.site_id as site_id_1,
        s2.site_id as site_id_2,
        st1.substrate_code,
        st1.total_coverage
    FROM cots_changes c
    LEFT JOIN substrate s1 ON c.survey_id_1 = s1.survey_id
    LEFT JOIN substrate s2 ON c.survey_id_2 = s2.survey_id
    LEFT JOIN survey_totals st1 ON c.survey_id_1 = st1.survey_id
    GROUP BY c.survey_id_1, c.survey_id_2, st1.substrate_code
    """
    
    changes_df = pd.read_sql_query(change_query, conn)
    
    # Get control survey data in one query
    control_query = """
    WITH survey_totals AS (
        SELECT 
            survey_id,
            substrate_code,
            SUM(total) as total_coverage
        FROM substrate
        WHERE substrate_code IS NOT NULL 
        AND substrate_code != ''
        AND segment_code IN ('S1', 'S2', 'S3', 'S4')
        AND total IS NOT NULL 
        AND total != ''
        GROUP BY survey_id, substrate_code
    )
    SELECT 
        s.survey_id,
        s.site_id,
        s.date,
        st.substrate_code,
        st.total_coverage
    FROM substrate s
    JOIN survey_totals st ON s.survey_id = st.survey_id
    WHERE s.survey_id NOT IN (
        SELECT survey_id_1 FROM cots_changes
        UNION
        SELECT survey_id_2 FROM cots_changes
    )
    GROUP BY s.survey_id, st.substrate_code
    """
    
    control_df = pd.read_sql_query(control_query, conn)
    
    # Pivot the data for analysis
    control_pivot = control_df.pivot_table(
        index='survey_id',
        columns='substrate_code',
        values='total_coverage',
        aggfunc='first'
    ).fillna(0)
    
    increase_data = changes_df[changes_df['change'] > 0].pivot_table(
        index='survey_id_1',
        columns='substrate_code',
        values='total_coverage',
        aggfunc='first'
    ).fillna(0)
    
    decrease_data = changes_df[changes_df['change'] < 0].pivot_table(
        index='survey_id_1',
        columns='substrate_code',
        values='total_coverage',
        aggfunc='first'
    ).fillna(0)
    
    print("=== Analysis of Initial Substrate Patterns Before COTS Changes ===")
    print("Note: Analysis includes total substrate coverage per survey\n")
    
    # Identify significant substrates in each group
    significant_increases = set()
    significant_decreases = set()
    all_results = []
    
    def analyze_substrate(substrate, group_data, group_name):
        if substrate in group_data and substrate in control_pivot:
            group_values = group_data[substrate].dropna()
            control_values = control_pivot[substrate].dropna()
            if len(group_values) >= 2:
                stat, p_value = stats.ttest_ind(group_values, control_values)
                # Handle potential division by zero in effect size calculation
                control_std = control_values.std()
                if control_std == 0:
                    effect_size = float('inf') if group_values.mean() > control_values.mean() else float('-inf')
                else:
                    effect_size = (group_values.mean() - control_values.mean()) / control_std
                presence_rate = (group_data[substrate] > 0).mean()
                return {
                    'substrate': substrate,
                    'group': group_name,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'presence_rate': presence_rate,
                    'mean': group_values.mean(),
                    'control_mean': control_values.mean(),
                    'sample_size': len(group_values)
                }
        return None
    
    # Analyze all substrates
    all_substrates = set(control_pivot.columns) | set(increase_data.columns) | set(decrease_data.columns)
    
    for substrate in all_substrates:
        increase_result = analyze_substrate(substrate, increase_data, "Increase")
        decrease_result = analyze_substrate(substrate, decrease_data, "Decrease")
        
        if increase_result and increase_result['p_value'] < 0.05:
            significant_increases.add(substrate)
            all_results.append(increase_result)
        
        if decrease_result and decrease_result['p_value'] < 0.05:
            significant_decreases.add(substrate)
            all_results.append(decrease_result)
    
    shared_substrates = significant_increases & significant_decreases
    unique_increases = significant_increases - shared_substrates
    unique_decreases = significant_decreases - shared_substrates
    
    def print_substrate_stats(results, title):
        print(f"\n=== {title} ===")
        print(f"Total patterns found: {len(results)}")
        
        for result in sorted(results, key=lambda x: abs(x['effect_size']), reverse=True):
            print(f"\n{result['substrate']}:")
            print(f"  Effect size: {result['effect_size']:.2f}")
            print(f"  P-value: {result['p_value']:.4f}")
            print(f"  Present in {result['presence_rate']*100:.1f}% of initial surveys")
            print(f"  Mean coverage before {result['group'].lower()}: {result['mean']:.2f}")
            print(f"  Mean coverage during control: {result['control_mean']:.2f}")
            print(f"  Percent difference: {((result['mean'] - result['control_mean']) / result['control_mean'] * 100):.1f}%")
            print(f"  Sample size: {result['sample_size']}")
    
    # Print shared patterns
    if shared_substrates:
        print("\n=== Substrates Significant Before Both Increases and Decreases ===")
        shared_results = [r for r in all_results if r['substrate'] in shared_substrates]
        for substrate in shared_substrates:
            inc_result = next(r for r in shared_results if r['substrate'] == substrate and r['group'] == "Increase")
            dec_result = next(r for r in shared_results if r['substrate'] == substrate and r['group'] == "Decrease")
            
            print(f"\n{substrate}:")
            print("  Before Increases:")
            print(f"    Effect size: {inc_result['effect_size']:.2f}")
            print(f"    P-value: {inc_result['p_value']:.4f}")
            print(f"    Present in {inc_result['presence_rate']*100:.1f}% of surveys")
            print(f"    Mean coverage: {inc_result['mean']:.2f} (Control: {inc_result['control_mean']:.2f})")
            print(f"    Percent difference: {((inc_result['mean'] - inc_result['control_mean']) / inc_result['control_mean'] * 100):.1f}%")
            
            print("  Before Decreases:")
            print(f"    Effect size: {dec_result['effect_size']:.2f}")
            print(f"    P-value: {dec_result['p_value']:.4f}")
            print(f"    Present in {dec_result['presence_rate']*100:.1f}% of surveys")
            print(f"    Mean coverage: {dec_result['mean']:.2f} (Control: {dec_result['control_mean']:.2f})")
            print(f"    Percent difference: {((dec_result['mean'] - dec_result['control_mean']) / dec_result['control_mean'] * 100):.1f}%")
    
    # Print unique patterns for each group
    increase_results = [r for r in all_results if r['substrate'] in unique_increases]
    decrease_results = [r for r in all_results if r['substrate'] in unique_decreases]
    
    print_substrate_stats(increase_results, "Unique Substrate Patterns Before COTS INCREASES")
    print_substrate_stats(decrease_results, "Unique Substrate Patterns Before COTS DECREASES")
    
    # Clean up temporary index
    conn.execute("DROP INDEX IF EXISTS idx_substrate_survey")
    conn.close()

if __name__ == "__main__":
    analyze_substrate_precursors()
