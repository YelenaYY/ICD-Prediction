import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_mimic_iv_data():
    # Load the datasets
    print("Loading datasets...")
    discharge_path = 'physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz'
    diagnoses_path = 'physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz'
    
    discharge_df = pd.read_csv(discharge_path, compression='gzip')
    diagnoses_df = pd.read_csv(diagnoses_path, compression='gzip')
    
    # 1. Count total clinical notes
    total_notes = len(discharge_df)
    print(f"\nTotal number of clinical notes: {total_notes}")
    
    # 2. Create combined table
    print("\nCreating combined table...")
    # Merge discharge notes with diagnoses
    combined_df = pd.merge(
        discharge_df[['hadm_id', 'text']],
        diagnoses_df[['hadm_id', 'icd_code', 'icd_version']],
        on='hadm_id',
        how='inner'
    )
    
    print("\nCombined table shape:", combined_df.shape)
    print("\nSample of combined data:")
    print(combined_df.head())
    
    # 3. Analyze ICD code distribution
    print("\nAnalyzing ICD code distribution...")
    
    # Separate ICD-9 and ICD-10 codes
    icd9_df = combined_df[combined_df['icd_version'] == 9]
    icd10_df = combined_df[combined_df['icd_version'] == 10]
    
    # Count frequency of ICD-9 codes
    icd9_freq = pd.DataFrame(
        Counter(icd9_df['icd_code']).most_common(),
        columns=['icd_code', 'frequency']
    )
    
    # Count frequency of ICD-10 codes
    icd10_freq = pd.DataFrame(
        Counter(icd10_df['icd_code']).most_common(),
        columns=['icd_code', 'frequency']
    )
    
    # Print detailed statistics for top and bottom 100 codes
    print("\n" + "="*50)
    print("TOP 100 MOST FREQUENT ICD-9 CODES")
    print("="*50)
    print(icd9_freq.head(100).to_string())
    
    print("\n" + "="*50)
    print("TOP 100 MOST FREQUENT ICD-10 CODES")
    print("="*50)
    print(icd10_freq.head(100).to_string())
    
    print("\n" + "="*50)
    print("100 LEAST FREQUENT ICD-9 CODES")
    print("="*50)
    print(icd9_freq.tail(100).to_string())
    
    print("\n" + "="*50)
    print("100 LEAST FREQUENT ICD-10 CODES")
    print("="*50)
    print(icd10_freq.tail(100).to_string())
    
    # Save detailed frequency tables to separate CSV files
    # icd9_freq.head(100).to_csv('icd9_top100_frequency.csv', index=False)
    # icd9_freq.tail(100).to_csv('icd9_bottom100_frequency.csv', index=False)
    # icd10_freq.head(100).to_csv('icd10_top100_frequency.csv', index=False)
    # icd10_freq.tail(100).to_csv('icd10_bottom100_frequency.csv', index=False)
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print("\nICD-9 Statistics:")
    print(f"Total unique ICD-9 codes: {len(icd9_freq)}")
    print("\nICD-10 Statistics:")
    print(f"Total unique ICD-10 codes: {len(icd10_freq)}")
    
    # Create frequency distribution table
    print("\n" + "="*70)
    print("FREQUENCY DISTRIBUTION TABLE")
    print("="*70)
    
    # Create a DataFrame to store the distribution
    freq_dist = pd.DataFrame({
        'Frequency': range(1, 11),
        'ICD-9 Count': [len(icd9_freq[icd9_freq['frequency'] == freq]) for freq in range(1, 11)],
        'ICD-10 Count': [len(icd10_freq[icd10_freq['frequency'] == freq]) for freq in range(1, 11)]
    })
    
    # Add percentage columns
    freq_dist['ICD-9 %'] = (freq_dist['ICD-9 Count'] / len(icd9_freq) * 100).round(2)
    freq_dist['ICD-10 %'] = (freq_dist['ICD-10 Count'] / len(icd10_freq) * 100).round(2)
    
    # Reorder columns
    freq_dist = freq_dist[['Frequency', 'ICD-9 Count', 'ICD-9 %', 'ICD-10 Count', 'ICD-10 %']]
    
    # Print the table
    print("\nDistribution of ICD codes by frequency of occurrence:")
    print(freq_dist.to_string(index=False))
    
    # Print total counts for reference
    print("\nTotal unique ICD-9 codes:", len(icd9_freq))
    print("Total unique ICD-10 codes:", len(icd10_freq))

    # Create visualizations for top 100 codes
    # Commented out for now as it's not needed for the analysis
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
    
    # Top 100 ICD-9 codes
    sns.barplot(data=icd9_freq.head(100), 
                x='icd_code', 
                y='frequency',
                ax=ax1,
                color='skyblue')
    ax1.set_title('Top 100 Most Frequent ICD-9 Codes', fontsize=12)
    ax1.set_xlabel('ICD-9 Codes', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.tick_params(axis='x', rotation=90, labelsize=6)
    
    # Bottom 100 ICD-9 codes
    sns.barplot(data=icd9_freq.tail(100), 
                x='icd_code', 
                y='frequency',
                ax=ax2,
                color='lightblue')
    ax2.set_title('100 Least Frequent ICD-9 Codes', fontsize=12)
    ax2.set_xlabel('ICD-9 Codes', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.tick_params(axis='x', rotation=90, labelsize=6)
    
    # Top 100 ICD-10 codes
    sns.barplot(data=icd10_freq.head(100), 
                x='icd_code', 
                y='frequency',
                ax=ax3,
                color='lightgreen')
    ax3.set_title('Top 100 Most Frequent ICD-10 Codes', fontsize=12)
    ax3.set_xlabel('ICD-10 Codes', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.tick_params(axis='x', rotation=90, labelsize=6)
    
    # Bottom 100 ICD-10 codes
    sns.barplot(data=icd10_freq.tail(100), 
                x='icd_code', 
                y='frequency',
                ax=ax4,
                color='palegreen')
    ax4.set_title('100 Least Frequent ICD-10 Codes', fontsize=12)
    ax4.set_xlabel('ICD-10 Codes', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.tick_params(axis='x', rotation=90, labelsize=6)
    
    plt.tight_layout()
    plt.show()
    """

    return combined_df, icd9_freq, icd10_freq

if __name__ == "__main__":
    combined_df, icd9_freq, icd10_freq = analyze_mimic_iv_data()
