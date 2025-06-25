import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def analyze_data_quality(filepath='data/all_price_moves.csv', output_dir='data/quality_reports'):
    """Comprehensive data quality analysis with CSV output"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(filepath)
    
    print("=== Initial Data Overview ===")
    print(f"Total records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 1. Missing Values Analysis
    print("\n=== Missing Values Analysis ===")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_report = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    print(missing_report.sort_values('Missing Percentage', ascending=False))
    
    # Save missing values report
    missing_report.to_csv(os.path.join(output_dir, 'missing_values_report.csv'))
    
    # 2. Duplicate Analysis
    print("\n=== Duplicate Records Analysis ===")
    duplicates = df.duplicated().sum()
    print(f"Exact duplicates: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # Save duplicate analysis
    duplicate_report = pd.DataFrame({
        'Metric': ['Total Records', 'Duplicates', 'Duplicate Percentage'],
        'Value': [len(df), duplicates, duplicates/len(df)*100]
    })
    duplicate_report.to_csv(os.path.join(output_dir, 'duplicate_report.csv'), index=False)
    
    # 3. Price Movement Analysis
    if 'price_change_percentage' in df.columns:
        print("\n=== Price Movement Analysis ===")
        print(df['price_change_percentage'].describe())
        
        plt.figure(figsize=(12, 6))
        sns.histplot(df['price_change_percentage'], bins=50)
        plt.title('Distribution of Price Change Percentage')
        plt.xlabel('Price Change Percentage')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'price_change_distribution.png'))
        plt.close()
        
        # Identify outliers using IQR method
        Q1 = df['price_change_percentage'].quantile(0.25)
        Q3 = df['price_change_percentage'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df['price_change_percentage'] < lower_bound) | 
                     (df['price_change_percentage'] > upper_bound)]
        print(f"\nPotential outliers (IQR method): {len(outliers)} records")
        
        # Save outlier analysis
        outlier_report = pd.DataFrame({
            'Metric': ['Q1', 'Q3', 'IQR', 'Lower Bound', 'Upper Bound', 'Outlier Count'],
            'Value': [Q1, Q3, IQR, lower_bound, upper_bound, len(outliers)]
        })
        outlier_report.to_csv(os.path.join(output_dir, 'outlier_report.csv'), index=False)
    
    # 4. Event Type Analysis
    if 'event' in df.columns:
        print("\n=== Event Type Distribution ===")
        event_counts = df['event'].value_counts(dropna=False)
        print(event_counts)
        
        # Save event distribution
        event_counts.to_csv(os.path.join(output_dir, 'event_distribution.csv'))
    
    # 5. Date Analysis
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    date_report_data = []
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            print(f"\n=== {col} Analysis ===")
            print(f"Date range: {df[col].min()} to {df[col].max()}")
            print(f"Missing dates: {df[col].isnull().sum()}")
            date_report_data.append({
                'Column': col,
                'Min Date': df[col].min(),
                'Max Date': df[col].max(),
                'Missing Dates': df[col].isnull().sum()
            })
        except:
            print(f"\nCould not parse {col} as dates")
    
    # Save date analysis
    if date_report_data:
        pd.DataFrame(date_report_data).to_csv(os.path.join(output_dir, 'date_analysis_report.csv'), index=False)
    
    # 6. Text Quality Analysis
    text_report_data = []
    text_cols = ['content', 'title', 'content_en', 'title_en']
    for col in text_cols:
        if col in df.columns:
            print(f"\n=== {col} Text Analysis ===")
            # Text length analysis
            df[f'{col}_length'] = df[col].str.len()
            avg_length = df[f'{col}_length'].mean()
            empty_count = df[col].isnull().sum()
            print(f"Average length: {avg_length:.1f} characters")
            print(f"Empty texts: {empty_count}")
            
            # Sample some short texts that might be problematic
            short_texts = df[df[f'{col}_length'] < 10][col].dropna().head(5)
            if not short_texts.empty:
                print("\nSample very short texts:")
                for text in short_texts:
                    print(f"- {text}")
            
            text_report_data.append({
                'Column': col,
                'Average Length': avg_length,
                'Empty Count': empty_count,
                'Short Text Count (<10 chars)': len(df[df[f'{col}_length'] < 10])
            })
    
    # Save text quality analysis
    if text_report_data:
        pd.DataFrame(text_report_data).to_csv(os.path.join(output_dir, 'text_quality_report.csv'), index=False)
    
    return df

# Run the analysis
if __name__ == '__main__':
    df = analyze_data_quality()