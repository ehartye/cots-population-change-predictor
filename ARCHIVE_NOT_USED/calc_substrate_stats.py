import sqlite3
import pandas as pd
import numpy as np
from scipy import stats

def calculate_substrate_statistics():
    # Connect to database
    conn = sqlite3.connect('reefcheck.db')
    
    # Drop existing table if it exists
    conn.execute("DROP TABLE IF EXISTS substrate_stats")
    conn.commit()
    
    # Read substrate data with explicit NULL and empty string filtering
    # Include survey_id to group by survey
    query = """
    SELECT survey_id,
           substrate_code, 
           segment_code,
           total
    FROM substrate 
    WHERE substrate_code IS NOT NULL 
    AND substrate_code != ''
    AND segment_code IN ('S1', 'S2', 'S3', 'S4')
    AND total IS NOT NULL 
    AND total != ''
    """
    df = pd.read_sql_query(query, conn)
    
    # Convert total to numeric, coercing errors to NaN
    df['total'] = pd.to_numeric(df['total'], errors='coerce')
    
    # Drop any rows where total is NaN after conversion
    df = df.dropna(subset=['total'])
    
    # Group by survey_id and substrate_code, sum the totals
    survey_totals = df.groupby(['survey_id', 'substrate_code'])['total'].sum().reset_index()
    
    # Create stats table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS substrate_stats (
        substrate_code TEXT PRIMARY KEY,
        count INTEGER,
        total REAL,
        mean REAL,
        median REAL,
        std_dev REAL,
        mode REAL,
        min REAL,
        max REAL,
        q1 REAL,
        q3 REAL,
        skewness REAL,
        kurtosis REAL
    )
    """
    
    conn.execute(create_table_sql)
    conn.commit()
    
    # Calculate statistics for each substrate type
    stats_data = []
    
    for substrate in survey_totals['substrate_code'].unique():
        # Get all totals for this substrate type
        substrate_data = survey_totals[survey_totals['substrate_code'] == substrate]
        valid_totals = substrate_data['total'].values
        
        if len(valid_totals) > 0:
            stats_dict = {
                'substrate_code': substrate,
                'count': len(valid_totals),
                'total': float(np.sum(valid_totals)),
                'mean': float(np.mean(valid_totals)),
                'median': float(np.median(valid_totals)),
                'std_dev': float(np.std(valid_totals)) if len(valid_totals) > 1 else None,
                'mode': float(stats.mode(valid_totals, keepdims=True)[0][0]),
                'min': float(np.min(valid_totals)),
                'max': float(np.max(valid_totals)),
                'q1': float(np.percentile(valid_totals, 25)),
                'q3': float(np.percentile(valid_totals, 75)),
                'skewness': float(stats.skew(valid_totals)) if len(valid_totals) > 2 else None,
                'kurtosis': float(stats.kurtosis(valid_totals)) if len(valid_totals) > 2 else None
            }
            stats_data.append(stats_dict)
    
    # Insert statistics into the table
    insert_sql = """
    INSERT OR REPLACE INTO substrate_stats 
    (substrate_code, count, total, mean, median, std_dev, mode, min, max, q1, q3, skewness, kurtosis)
    VALUES 
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    for stats_dict in stats_data:
        conn.execute(insert_sql, (
            stats_dict['substrate_code'],
            stats_dict['count'],
            stats_dict['total'],
            stats_dict['mean'],
            stats_dict['median'],
            stats_dict['std_dev'],
            stats_dict['mode'],
            stats_dict['min'],
            stats_dict['max'],
            stats_dict['q1'],
            stats_dict['q3'],
            stats_dict['skewness'],
            stats_dict['kurtosis']
        ))
    
    conn.commit()
    conn.close()
    
    print("Substrate statistics have been calculated and stored in the substrate_stats table.")
    
    # Print a sample of the results
    conn = sqlite3.connect('reefcheck.db')
    print("\nSample of calculated statistics:")
    sample_query = "SELECT substrate_code, count, total, mean, median FROM substrate_stats LIMIT 5;"
    sample_df = pd.read_sql_query(sample_query, conn)
    print(sample_df)
    conn.close()

if __name__ == "__main__":
    calculate_substrate_statistics()
