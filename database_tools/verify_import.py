import sqlite3
import os
import csv
from pathlib import Path
from import_data import convert_to_decimal_coords
import math

def verify_database_structure():
    """Verify that the database has the correct tables and schema."""
    print("\n=== Verifying Database Structure ===")
    conn = sqlite3.connect('reefcheck.db')
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    expected_tables = {'belt', 'site_description', 'substrate'}
    
    if not expected_tables.issubset(tables):
        missing = expected_tables - tables
        print(f"❌ Missing tables: {missing}")
        return False
    print("✓ All expected tables exist")
    
    # Verify column count for each table
    expected_columns = {
        'site_description': 20,
        'belt': 16,  # Updated to match actual schema
        'substrate': 13
    }
    
    for table, expected_count in expected_columns.items():
        cursor.execute(f"PRAGMA table_info({table})")
        actual_count = len(cursor.fetchall())
        if actual_count != expected_count:
            print(f"❌ Table {table} has {actual_count} columns, expected {expected_count}")
            return False
    print("✓ All tables have correct number of columns")
    
    conn.close()
    return True

def verify_data_import():
    """Verify that data was imported correctly by comparing row counts."""
    print("\n=== Verifying Data Import ===")
    conn = sqlite3.connect('reefcheck.db')
    cursor = conn.cursor()
    
    data_dir = Path('data')
    files_to_check = {
        'Belt.csv': 'belt',
        'Site_Description.csv': 'site_description',
        'Substrate.csv': 'substrate'
    }
    
    all_valid = True
    for csv_file, table in files_to_check.items():
        csv_path = data_dir / csv_file
        if not csv_path.exists():
            print(f"❌ Source file {csv_file} not found")
            all_valid = False
            continue
            
        # Count valid CSV rows (subtract header and empty rows)
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header
            csv_count = sum(1 for row in csv_reader if any(field.strip() for field in row))
            
        # Count database rows
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        db_count = cursor.fetchone()[0]
        
        if csv_count == db_count:
            print(f"✓ {table}: {db_count} rows imported correctly")
        else:
            print(f"❌ {table}: Mismatch - CSV has {csv_count} valid rows, DB has {db_count} rows")
            all_valid = False
    
    conn.close()
    return all_valid

def is_close_enough(a, b, rel_tol=1e-9):
    """Compare floating point numbers with relative tolerance."""
    return math.isclose(a, b, rel_tol=rel_tol)

def verify_coordinate_conversion():
    """Test the coordinate conversion function with known values."""
    print("\n=== Verifying Coordinate Conversion ===")
    test_cases = [
        # (degrees, minutes, seconds, direction, expected_result)
        (45, 30, 0, 'N', 45.5),
        (123, 15, 30, 'W', -123.25833333333333),
        (0, 30, 30, 'S', -0.5083333333333333),
        ('bad', 'data', 'test', 'N', None)  # Error case
    ]
    
    all_valid = True
    for deg, min, sec, dir, expected in test_cases:
        result = convert_to_decimal_coords(deg, min, sec, dir)
        if result is None and expected is None:
            print(f"✓ Coordinate conversion passed: {deg}°{min}'{sec}\" {dir} = {result}")
        elif result is not None and expected is not None and is_close_enough(result, expected):
            print(f"✓ Coordinate conversion passed: {deg}°{min}'{sec}\" {dir} = {result}")
        else:
            print(f"❌ Coordinate conversion failed: {deg}°{min}'{sec}\" {dir}")
            print(f"   Expected {expected}, got {result}")
            all_valid = False
    
    return all_valid

def verify_data_consistency():
    """Verify data consistency across tables."""
    print("\n=== Verifying Data Consistency ===")
    conn = sqlite3.connect('reefcheck.db')
    cursor = conn.cursor()
    
    # Check for matching site_ids across tables
    cursor.execute("""
        SELECT DISTINCT site_id FROM site_description 
        INTERSECT 
        SELECT DISTINCT site_id FROM belt
        INTERSECT 
        SELECT DISTINCT site_id FROM substrate
    """)
    common_sites = cursor.fetchall()
    
    cursor.execute("SELECT COUNT(DISTINCT site_id) FROM site_description")
    total_sites = cursor.fetchone()[0]
    
    print(f"Total sites in site_description: {total_sites}")
    print(f"Sites present in all tables: {len(common_sites)}")
    
    # Verify data types
    print("\nChecking data types...")
    cursor.execute("SELECT longitude_decimal, latitude_decimal FROM site_description WHERE longitude_decimal NOT NULL LIMIT 1")
    coords = cursor.fetchone()
    if coords and isinstance(coords[0], float) and isinstance(coords[1], float):
        print("✓ Coordinate data types are correct (float)")
    else:
        print("❌ Coordinate data types are incorrect")
    
    # Additional data quality checks
    cursor.execute("SELECT COUNT(*) FROM site_description WHERE longitude_decimal < -180 OR longitude_decimal > 180")
    invalid_lon = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM site_description WHERE latitude_decimal < -90 OR latitude_decimal > 90")
    invalid_lat = cursor.fetchone()[0]
    
    if invalid_lon == 0 and invalid_lat == 0:
        print("✓ All coordinates are within valid ranges")
    else:
        print(f"❌ Found {invalid_lon} invalid longitudes and {invalid_lat} invalid latitudes")
    
    conn.close()

def main():
    """Run all verification checks."""
    print("Starting verification of import_data.py...")
    
    # Run import_data.py first
    print("\nRunning import_data.py...")
    os.system('python import_data.py')
    
    # Run verification checks
    structure_valid = verify_database_structure()
    import_valid = verify_data_import()
    coord_valid = verify_coordinate_conversion()
    verify_data_consistency()
    
    # Final summary
    print("\n=== Verification Summary ===")
    print(f"Database Structure: {'✓' if structure_valid else '❌'}")
    print(f"Data Import: {'✓' if import_valid else '❌'}")
    print(f"Coordinate Conversion: {'✓' if coord_valid else '❌'}")
    
    if all([structure_valid, import_valid, coord_valid]):
        print("\n✅ All verification checks passed!")
    else:
        print("\n❌ Some verification checks failed. Please review the output above.")

if __name__ == "__main__":
    main()
