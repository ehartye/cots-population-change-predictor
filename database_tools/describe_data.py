import sqlite3
from pathlib import Path
from collections import defaultdict

def connect_db():
    """Connect to the SQLite database."""
    return sqlite3.connect('reefcheck.db')

def format_number(value):
    """Safely format a number with one decimal place."""
    try:
        return f"{float(value):.1f}"
    except (ValueError, TypeError):
        return "N/A"

def describe_table_structure(cursor, table_name):
    """Describe the structure of a table."""
    print(f"\n=== {table_name.upper()} TABLE STRUCTURE ===")
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    
    print(f"Number of columns: {len(columns)}")
    print("\nColumns:")
    for col in columns:
        print(f"- {col[1]} ({col[2]})")

def get_numeric_stats(cursor, table, column):
    """Get statistical information for numeric columns."""
    cursor.execute(f"""
        SELECT 
            COUNT({column}) as count,
            MIN(CAST({column} AS FLOAT)) as min_val,
            MAX(CAST({column} AS FLOAT)) as max_val,
            AVG(CAST({column} AS FLOAT)) as mean,
            (
                SELECT {column}
                FROM {table}
                WHERE {column} IS NOT NULL
                ORDER BY CAST({column} AS FLOAT)
                LIMIT 1
                OFFSET (SELECT COUNT({column}) FROM {table} WHERE {column} IS NOT NULL) / 2
            ) as median
        FROM {table}
        WHERE {column} IS NOT NULL
    """)
    stats = cursor.fetchone()
    return {
        'count': stats[0],
        'min': format_number(stats[1]),
        'max': format_number(stats[2]),
        'mean': format_number(stats[3]),
        'median': format_number(stats[4])
    }

def get_categorical_stats(cursor, table, column, limit=10):
    """Get distribution of values for categorical columns."""
    cursor.execute(f"""
        SELECT {column}, COUNT(*) as count
        FROM {table}
        WHERE {column} IS NOT NULL
        GROUP BY {column}
        ORDER BY count DESC
        LIMIT {limit}
    """)
    return cursor.fetchall()

def describe_site_description(cursor):
    """Analyze the site_description table."""
    print("\n=== SITE DESCRIPTION ANALYSIS ===")
    
    # Basic counts
    cursor.execute("SELECT COUNT(*) FROM site_description")
    total_sites = cursor.fetchone()[0]
    print(f"\nTotal number of sites: {total_sites}")
    
    # Geographic distribution
    print("\nGeographic Distribution:")
    for country, count in get_categorical_stats(cursor, 'site_description', 'country'):
        print(f"- {country or 'Unknown'}: {count} sites")
    
    # Region distribution
    print("\nRegion Distribution:")
    for region, count in get_categorical_stats(cursor, 'site_description', 'region'):
        print(f"- {region or 'Unknown'}: {count} sites")
    
    # Depth statistics
    depth_stats = get_numeric_stats(cursor, 'site_description', 'depth')
    print("\nDepth Statistics (meters):")
    print(f"- Range: {depth_stats['min']} to {depth_stats['max']}")
    print(f"- Average: {depth_stats['mean']}")
    print(f"- Median: {depth_stats['median']}")
    
    # Temperature statistics
    temp_stats = get_numeric_stats(cursor, 'site_description', 'water_temp_surface')
    print("\nWater Temperature Statistics (°C):")
    print(f"- Range: {temp_stats['min']} to {temp_stats['max']}")
    print(f"- Average: {temp_stats['mean']}")
    print(f"- Median: {temp_stats['median']}")
    
    # Protection status
    print("\nProtection Status Distribution:")
    for status, count in get_categorical_stats(cursor, 'site_description', 'protection_status'):
        print(f"- {status or 'Unknown'}: {count} sites")
    
    # Impact level distribution
    print("\nImpact Level Distribution:")
    for impact, count in get_categorical_stats(cursor, 'site_description', 'impact_level'):
        print(f"- {impact or 'Unknown'}: {count} sites")
    
    # Date range
    cursor.execute("SELECT MIN(date), MAX(date) FROM site_description")
    min_date, max_date = cursor.fetchone()
    print(f"\nSurvey Period: {min_date} to {max_date}")

def describe_belt(cursor):
    """Analyze the belt table."""
    print("\n=== BELT SURVEY ANALYSIS ===")
    
    # Basic counts
    cursor.execute("SELECT COUNT(*) FROM belt")
    total_records = cursor.fetchone()[0]
    print(f"\nTotal number of observations: {total_records}")
    
    # Organism distribution
    print("\nTop 10 Most Observed Organisms:")
    cursor.execute("""
        SELECT 
            organism_code,
            COUNT(*) as count,
            SUM(COALESCE(CAST(segment_1_count AS INTEGER), 0) + 
                COALESCE(CAST(segment_2_count AS INTEGER), 0) + 
                COALESCE(CAST(segment_3_count AS INTEGER), 0) + 
                COALESCE(CAST(segment_4_count AS INTEGER), 0)) as total_count
        FROM belt
        GROUP BY organism_code
        ORDER BY total_count DESC
        LIMIT 10
    """)
    for org, obs_count, total_count in cursor.fetchall():
        print(f"- {org}: {obs_count} observations, {total_count} individuals")
    
    # Survey type distribution
    print("\nSurvey Type Distribution:")
    for type_name, count in get_categorical_stats(cursor, 'belt', 'type'):
        print(f"- {type_name or 'Unknown'}: {count} surveys")
    
    # Depth statistics
    depth_stats = get_numeric_stats(cursor, 'belt', 'depth')
    print("\nDepth Statistics (meters):")
    print(f"- Range: {depth_stats['min']} to {depth_stats['max']}")
    print(f"- Average: {depth_stats['mean']}")
    print(f"- Median: {depth_stats['median']}")

def describe_substrate(cursor):
    """Analyze the substrate table."""
    print("\n=== SUBSTRATE ANALYSIS ===")
    
    # Basic counts
    cursor.execute("SELECT COUNT(*) FROM substrate")
    total_records = cursor.fetchone()[0]
    print(f"\nTotal number of substrate observations: {total_records}")
    
    # Substrate type distribution
    print("\nSubstrate Type Distribution:")
    cursor.execute("""
        SELECT 
            substrate_code, 
            COUNT(*) as count, 
            SUM(COALESCE(CAST(total AS INTEGER), 0)) as total_points,
            ROUND(AVG(COALESCE(CAST(total AS INTEGER), 0)), 2) as avg_points
        FROM substrate
        GROUP BY substrate_code
        ORDER BY total_points DESC
    """)
    for substrate, count, total, avg in cursor.fetchall():
        print(f"- {substrate}: {count} observations, {total} total points (avg: {format_number(avg)} per observation)")
    
    # Segment distribution
    print("\nSegment Distribution:")
    for segment, count in get_categorical_stats(cursor, 'substrate', 'segment_code'):
        print(f"- {segment or 'Unknown'}: {count} observations")

def main():
    """Main function to describe the ReefCheck database."""
    print("=== REEFCHECK DATABASE ANALYSIS ===")
    print("Analyzing data from reefcheck.db...")
    
    conn = connect_db()
    cursor = conn.cursor()
    
    # Describe table structures
    for table in ['site_description', 'belt', 'substrate']:
        describe_table_structure(cursor, table)
    
    # Detailed analysis of each table
    describe_site_description(cursor)
    describe_belt(cursor)
    describe_substrate(cursor)
    
    conn.close()

if __name__ == "__main__":
    main()
