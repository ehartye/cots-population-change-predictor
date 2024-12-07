import sqlite3
import pandas as pd
import os.path

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'reefcheck.db')

# Define key features at module level
KEY_FEATURES = [
    'Grouper > 60 cm', 'Grouper 30-40 cm', 'Grouper 40-50 cm', 'Grouper Total',
    'Giant Clam 40-50 cm', 'Giant Clam > 50 cm',
    'Bleaching (% Of Population)', 'Bleaching (% Of Colony)',
    'Coral Damage Other', 'Humphead Wrasse', 'Parrotfish', 'Tripneustes', 'has_bleaching', 'has_coral_damage',
    'has_parrotfish', 'has_wrasse', 'total_giant_clam', 'has_tripneustes'
]

def prepare_features():
    conn = sqlite3.connect(DB_PATH)
    
    # Get all surveys with their COTS change status
    cots_changes = pd.read_sql_query("""
        SELECT DISTINCT survey_id_1 as survey_id, 
               CASE 
                   WHEN change > 0 THEN 'increase'
                   WHEN change < 0 THEN 'decrease'
               END as change_type
        FROM cots_changes
    """, conn)
    
    # Get all surveys
    all_surveys = pd.read_sql_query("""
        SELECT DISTINCT survey_id, site_id, date
        FROM belt
        ORDER BY date
    """, conn)
    
    # Function to get organism counts for a survey
    def get_survey_features(survey_id):
        query = """
            SELECT 
                organism_code,
                segment_1_count,
                segment_2_count,
                segment_3_count,
                segment_4_count
            FROM belt
            WHERE survey_id = ?
            AND organism_code IS NOT NULL 
            AND organism_code != ''
        """
        df = pd.read_sql_query(query, conn, params=[survey_id])
        
        features = {}
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
            
            if len(valid_counts) >= 2:  # Only include if at least 2 valid segments
                organism = row['organism_code']
                avg_count = sum(valid_counts) / len(valid_counts)
                features[organism] = avg_count
                
                # Add derived features for key indicators
                if 'Bleaching' in organism:
                    features['has_bleaching'] = 1 if avg_count > 0 else 0
                elif 'Coral Damage' in organism:
                    features['has_coral_damage'] = 1 if avg_count > 0 else 0
                elif organism == 'Parrotfish':
                    features['has_parrotfish'] = 1 if avg_count > 0 else 0
                elif organism == 'Humphead Wrasse':
                    features['has_wrasse'] = 1 if avg_count > 0 else 0
                elif organism.startswith('Giant Clam'):
                    features['total_giant_clam'] = features.get('total_giant_clam', 0) + avg_count
                elif organism == 'Tripneustes':
                    features['has_tripneustes'] = 1 if avg_count > 0 else 0
        
        return features
    
    # Prepare dataset
    dataset = []
    
    for _, survey in all_surveys.iterrows():
        features = get_survey_features(survey['survey_id'])
        
        # Only include surveys with sufficient data
        if len(features) >= 5:  # Require at least 5 features to be present
            row = {'survey_id': survey['survey_id'], 'site_id': survey['site_id']}
            
            # Add features
            for feature in KEY_FEATURES:
                row[feature] = features.get(feature, 0)  # Use 0 if feature not present
            
            # Add target variable
            change_record = cots_changes[cots_changes['survey_id'] == survey['survey_id']]
            if not change_record.empty:
                row['target'] = change_record.iloc[0]['change_type']
            else:
                row['target'] = 'none'
            
            dataset.append(row)
    
    conn.close()
    return pd.DataFrame(dataset)
