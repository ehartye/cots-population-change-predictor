import sqlite3
import pandas as pd
import numpy as np
from scipy import stats

def calculate_belt_statistics():
    # Connect to database
    conn = sqlite3.connect('reefcheck.db')
    
    # Read belt data with explicit NULL and empty string filtering
    query = """
    SELECT organism_code, 
           segment_1_count, segment_2_count, 
           segment_3_count, segment_4_count
    FROM belt 
    WHERE organism_code IS NOT NULL 
    AND organism_code != ''
    AND segment_1_count IS NOT NULL 
    AND segment_2_count IS NOT NULL 
    AND segment_3_count IS NOT NULL 
    AND segment_4_count IS NOT NULL 
    AND segment_1_count != ''
    AND segment_2_count != ''
    AND segment_3_count != ''
    AND segment_4_count != ''
    """
    df = pd.read_sql_query(query, conn)
    
    # Convert segment counts to numeric, raising error for invalid values
    count_columns = ['segment_1_count', 'segment_2_count', 'segment_3_count', 'segment_4_count']
    
    # Create stats table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS belt_stats (
        organism_code TEXT PRIMARY KEY,
        count INTEGER,
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
    
    # Calculate statistics for each organism
    stats_data = []
    
    for organism in df['organism_code'].unique():
        # Get all counts for this organism
        organism_data = df[df['organism_code'] == organism]
        
        # Process each count column separately to ensure proper validation
        valid_counts = []
        for col in count_columns:
            counts = organism_data[col].values
            for count in counts:
                # Skip any remaining empty strings or null values
                if pd.isna(count) or count == '':
                    continue
                try:
                    # Convert to float and validate
                    count_float = float(count)
                    valid_counts.append(count_float)
                except (ValueError, TypeError):
                    # Skip any values that can't be converted to float
                    continue
        
        if len(valid_counts) > 0:
            valid_counts = np.array(valid_counts)
            stats_dict = {
                'organism_code': organism,
                'count': len(valid_counts),
                'mean': float(np.mean(valid_counts)),
                'median': float(np.median(valid_counts)),
                'std_dev': float(np.std(valid_counts)) if len(valid_counts) > 1 else None,
                'mode': float(stats.mode(valid_counts, keepdims=True)[0][0]),
                'min': float(np.min(valid_counts)),
                'max': float(np.max(valid_counts)),
                'q1': float(np.percentile(valid_counts, 25)),
                'q3': float(np.percentile(valid_counts, 75)),
                'skewness': float(stats.skew(valid_counts)) if len(valid_counts) > 2 else None,
                'kurtosis': float(stats.kurtosis(valid_counts)) if len(valid_counts) > 2 else None
            }
            stats_data.append(stats_dict)
    
    # Insert statistics into the table
    insert_sql = """
    INSERT OR REPLACE INTO belt_stats 
    (organism_code, count, mean, median, std_dev, mode, min, max, q1, q3, skewness, kurtosis)
    VALUES 
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    for stats_dict in stats_data:
        conn.execute(insert_sql, (
            stats_dict['organism_code'],
            stats_dict['count'],
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
    
    print("Belt statistics have been calculated and stored in the belt_stats table.")
    
    # Print a sample of the results
    conn = sqlite3.connect('reefcheck.db')
    print("\nSample of calculated statistics:")
    sample_query = "SELECT * FROM belt_stats LIMIT 5;"
    sample_df = pd.read_sql_query(sample_query, conn)
    print(sample_df)
    conn.close()

if __name__ == "__main__":
    calculate_belt_statistics()
