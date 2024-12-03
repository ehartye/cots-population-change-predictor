import sqlite3
import csv
import os
from pathlib import Path

def create_database():
    """Create the SQLite database and return a connection."""
    conn = sqlite3.connect('reefcheck.db')
    return conn

def create_tables(conn):
    """Create the necessary tables in the database."""
    cursor = conn.cursor()
    
    # Drop existing tables if they exist
    cursor.execute("DROP TABLE IF EXISTS belt")
    cursor.execute("DROP TABLE IF EXISTS site_description")
    cursor.execute("DROP TABLE IF EXISTS substrate")
    
    # Site Description table - focusing on key environmental and location data
    cursor.execute('''
        CREATE TABLE site_description (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_id TEXT,
            survey_id TEXT,
            reef_name TEXT,
            country TEXT,
            state_province_island TEXT,
            city_town TEXT,
            region TEXT,
            longitude_decimal REAL,
            latitude_decimal REAL,
            year INTEGER,
            date DATE,
            depth REAL,
            visibility REAL,
            weather TEXT,
            water_temp_surface REAL,
            protection_status TEXT,
            impact_level TEXT,
            comments TEXT,
            raw_data TEXT
        )
    ''')
    
    # Belt Survey table - for both fish and invertebrate counts
    cursor.execute('''
        CREATE TABLE belt (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_id TEXT,
            survey_id TEXT,
            reef_name TEXT,
            country TEXT,
            year INTEGER,
            date DATE,
            depth REAL,
            organism_code TEXT,
            type TEXT,
            segment_1_count INTEGER,  -- 0-20m
            segment_2_count INTEGER,  -- 25-45m
            segment_3_count INTEGER,  -- 50-70m
            segment_4_count INTEGER,  -- 75-95m
            recorded_by TEXT,
            what_errors TEXT,
            raw_data TEXT
        )
    ''')
    
    # Substrate Survey table - for point sampling data
    cursor.execute('''
        CREATE TABLE substrate (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_id TEXT,
            survey_id TEXT,
            reef_name TEXT,
            country TEXT,
            year INTEGER,
            date DATE,
            depth REAL,
            substrate_code TEXT,
            segment_code TEXT,
            total INTEGER,
            recorded_by TEXT,
            raw_data TEXT
        )
    ''')
    
    conn.commit()

def convert_to_decimal_coords(degrees, minutes, seconds, direction):
    """Convert coordinates from degrees/minutes/seconds to decimal format."""
    try:
        deg = float(degrees) if degrees else 0
        min = float(minutes) if minutes else 0
        sec = float(seconds) if seconds else 0
        
        decimal = deg + (min / 60.0) + (sec / 3600.0)
        
        if direction and direction.upper() in ['S', 'W']:
            decimal = -decimal
            
        return decimal
    except (ValueError, TypeError):
        return None

def parse_site_description(row, headers):
    """Parse a row of site description data."""
    # Get index positions for key fields
    idx = {}
    for i, header in enumerate(headers):
        idx[header.strip()] = i
    
    # Convert coordinates to decimal format
    lon = convert_to_decimal_coords(
        row[idx.get('longitude_degrees', 0)],
        row[idx.get('longitude_minutes', 0)],
        row[idx.get('longitude_seconds', 0)],
        row[idx.get('longitude_cardinal_direction', '')]
    )
    
    lat = convert_to_decimal_coords(
        row[idx.get('latitude_degrees', 0)],
        row[idx.get('latitude_minutes', 0)],
        row[idx.get('latitude_seconds', 0)],
        row[idx.get('latitude_cardinal_direction', '')]
    )
    
    return {
        'site_id': row[idx.get('site_id', 0)],
        'survey_id': row[idx.get('survey_id', 0)],
        'reef_name': row[idx.get('reef_name', 0)],
        'country': row[idx.get('country', 0)],
        'state_province_island': row[idx.get('state_province_island', 0)],
        'city_town': row[idx.get('city_town', 0)],
        'region': row[idx.get('region', 0)],
        'longitude_decimal': lon,
        'latitude_decimal': lat,
        'year': row[idx.get('year', 0)],
        'date': row[idx.get('date', 0)],
        'depth': row[idx.get('depth (m)', 0)],
        'visibility': row[idx.get('horizontal_visibility_in_water (m)', 0)],
        'weather': row[idx.get('weather', 0)],
        'water_temp_surface': row[idx.get('water_temp_at_surface (C)', 0)],
        'protection_status': row[idx.get('is_site_protected', 0)],
        'impact_level': row[idx.get('overall_anthro_impact', 0)],
        'comments': row[idx.get('site_comments', 0)],
        'raw_data': ','.join(row)
    }

def parse_belt(row, headers):
    """Parse a row of belt survey data."""
    idx = {}
    for i, header in enumerate(headers):
        idx[header.strip()] = i
    
    return {
        'site_id': row[idx.get('site_id', 0)],
        'survey_id': row[idx.get('survey_id', 0)],
        'reef_name': row[idx.get('reef_name', 0)],
        'country': row[idx.get('country', 0)],
        'year': row[idx.get('year', 0)],
        'date': row[idx.get('date', 0)],
        'depth': row[idx.get('depth (m)', 0)],
        'organism_code': row[idx.get('organism_code', 0)],
        'type': row[idx.get('type', 0)],
        'segment_1_count': row[idx.get('s1 (0-20m)', 0)],
        'segment_2_count': row[idx.get('s2 (25-45m)', 0)],
        'segment_3_count': row[idx.get('s3 (50-70m)', 0)],
        'segment_4_count': row[idx.get('s4 (75-95m)', 0)],
        'recorded_by': row[idx.get('fish_recorded_by', 0)] or row[idx.get('inverts_recorded_by', 0)],
        'what_errors': row[idx.get('what_errors', 0)],
        'raw_data': ','.join(row)
    }

def parse_substrate(row, headers):
    """Parse a row of substrate data."""
    idx = {}
    for i, header in enumerate(headers):
        idx[header.strip()] = i
    
    return {
        'site_id': row[idx.get('site_id', 0)],
        'survey_id': row[idx.get('survey_id', 0)],
        'reef_name': row[idx.get('reef_name', 0)],
        'country': row[idx.get('country', 0)],
        'year': row[idx.get('year', 0)],
        'date': row[idx.get('date', 0)],
        'depth': row[idx.get('depth (m)', 0)],
        'substrate_code': row[idx.get('substrate_code', 0)],
        'segment_code': row[idx.get('segment_code', 0)],
        'total': row[idx.get('total', 0)],
        'recorded_by': row[idx.get('substrate_recorded_by', 0)],
        'raw_data': ','.join(row)
    }

def import_csv_to_table(conn, csv_path, table_name):
    """Import data from a CSV file into the specified table."""
    cursor = conn.cursor()
    
    print(f"Importing {csv_path} into {table_name} table...")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            headers = [h.strip() for h in next(csv_reader)]
            print(f"Processing {table_name} data...")
            
            # Select appropriate parsing function
            if table_name == 'site_description':
                parse_func = parse_site_description
            elif table_name == 'belt':
                parse_func = parse_belt
            else:
                parse_func = parse_substrate
            
            # Create the parameterized SQL insert statement
            fields = list(parse_func(headers, headers).keys())
            placeholders = ','.join(['?' for _ in fields])
            insert_sql = f'INSERT INTO {table_name} ({",".join(fields)}) VALUES ({placeholders})'
            
            count = 0
            for row in csv_reader:
                if len(row) != len(headers):
                    print(f"Warning: Row {count + 1} has incorrect number of fields. Skipping...")
                    continue
                    
                parsed_data = parse_func(row, headers)
                cursor.execute(insert_sql, list(parsed_data.values()))
                count += 1
                
                if count % 10000 == 0:
                    conn.commit()
                    print(f"Processed {count} records...")
            
            conn.commit()
            print(f"Successfully imported {count} records from {csv_path}")
            
    except Exception as e:
        print(f"Error importing {csv_path}: {str(e)}")
        conn.rollback()

def main():
    """Main execution function."""
    # Create database connection
    conn = create_database()
    print("Database connection established")
    
    # Create tables
    create_tables(conn)
    print("Tables created successfully")
    
    # Define data directory and files to import
    data_dir = Path('data')
    files_to_import = {
        'Belt.csv': 'belt',
        'Site_Description.csv': 'site_description',
        'Substrate.csv': 'substrate'
    }
    
    # Import each file
    for filename, table in files_to_import.items():
        file_path = data_dir / filename
        if file_path.exists():
            import_csv_to_table(conn, str(file_path), table)
        else:
            print(f"Warning: {filename} not found in data directory")
    
    # Close connection
    conn.close()
    print("Database connection closed")

if __name__ == "__main__":
    main()
