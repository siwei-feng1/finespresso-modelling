import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class DataQualityAnalyzer:
    def __init__(self, filepath='data/modeling_data.csv', output_dir='data/quality_reports'):
        self.filepath = filepath
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None

    def analyze(self):
        """Comprehensive data quality analysis with CSV output"""
        # Load data
        self.df = pd.read_csv(self.filepath)
        
        print("=== Initial Data Overview ===")
        print(f"Total records: {len(self.df)}")
        print(f"Columns: {self.df.columns.tolist()}")
        
        # 1. Missing Values Analysis
        print("\n=== Missing Values Analysis ===")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_report = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        })
        print(missing_report.sort_values('Missing Percentage', ascending=False))
        missing_report.to_csv(os.path.join(self.output_dir, 'missing_values_report.csv'))
        
        # 2. Duplicate Analysis
        print("\n=== Duplicate Records Analysis ===")
        duplicates = self.df.duplicated().sum()
        print(f"Exact duplicates: {duplicates} ({duplicates/len(self.df)*100:.2f}%)")
        duplicate_report = pd.DataFrame({
            'Metric': ['Total Records', 'Duplicates', 'Duplicate Percentage'],
            'Value': [len(self.df), duplicates, duplicates/len(self.df)*100]
        })
        duplicate_report.to_csv(os.path.join(self.output_dir, 'duplicate_report.csv'), index=False)
        
        # 3. Price Movement Analysis
        if 'price_change_percentage' in self.df.columns:
            print("\n=== Price Movement Analysis ===")
            print(self.df['price_change_percentage'].describe())
            plt.figure(figsize=(12, 6))
            sns.histplot(self.df['price_change_percentage'], bins=50)
            plt.title('Distribution of Price Change Percentage')
            plt.xlabel('Price Change Percentage')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(self.output_dir, 'price_change_distribution.png'))
            plt.close()
            Q1 = self.df['price_change_percentage'].quantile(0.25)
            Q3 = self.df['price_change_percentage'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df['price_change_percentage'] < lower_bound) | 
                         (self.df['price_change_percentage'] > upper_bound)]
            print(f"\nPotential outliers (IQR method): {len(outliers)} records")
            outlier_report = pd.DataFrame({
                'Metric': ['Q1', 'Q3', 'IQR', 'Lower Bound', 'Upper Bound', 'Outlier Count'],
                'Value': [Q1, Q3, IQR, lower_bound, upper_bound, len(outliers)]
            })
            outlier_report.to_csv(os.path.join(self.output_dir, 'outlier_report.csv'), index=False)
        
        # 4. Event Type Analysis
        if 'event' in self.df.columns:
            print("\n=== Event Type Distribution ===")
            event_counts = self.df['event'].value_counts(dropna=False)
            print(event_counts)
            event_counts.to_csv(os.path.join(self.output_dir, 'event_distribution.csv'))
        
        # 5. Date Analysis
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]
        date_report_data = []
        for col in date_cols:
            try:
                self.df[col] = pd.to_datetime(self.df[col])
                print(f"\n=== {col} Analysis ===")
                print(f"Date range: {self.df[col].min()} to {self.df[col].max()}")
                print(f"Missing dates: {self.df[col].isnull().sum()}")
                date_report_data.append({
                    'Column': col,
                    'Min Date': self.df[col].min(),
                    'Max Date': self.df[col].max(),
                    'Missing Dates': self.df[col].isnull().sum()
                })
            except:
                print(f"\nCould not parse {col} as dates")
        if date_report_data:
            pd.DataFrame(date_report_data).to_csv(os.path.join(self.output_dir, 'date_analysis_report.csv'), index=False)
        
        # 6. Text Quality Analysis
        text_report_data = []
        text_cols = ['content', 'title', 'content_en', 'title_en']
        for col in text_cols:
            if col in self.df.columns:
                print(f"\n=== {col} Text Analysis ===")
                self.df[f'{col}_length'] = self.df[col].str.len()
                avg_length = self.df[f'{col}_length'].mean()
                empty_count = self.df[col].isnull().sum()
                print(f"Average length: {avg_length:.1f} characters")
                print(f"Empty texts: {empty_count}")
                short_texts = self.df[self.df[f'{col}_length'] < 10][col].dropna().head(5)
                if not short_texts.empty:
                    print("\nSample very short texts:")
                    for text in short_texts:
                        print(f"- {text}")
                text_report_data.append({
                    'Column': col,
                    'Average Length': avg_length,
                    'Empty Count': empty_count,
                    'Short Text Count (<10 chars)': len(self.df[self.df[f'{col}_length'] < 10])
                })
        if text_report_data:
            pd.DataFrame(text_report_data).to_csv(os.path.join(self.output_dir, 'text_quality_report.csv'), index=False)
        
        return self.df

# Usage example
if __name__ == '__main__':
    analyzer = DataQualityAnalyzer()
    df = analyzer.analyze()