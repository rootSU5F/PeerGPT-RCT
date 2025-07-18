import pandas as pd
import re
from typing import Optional

def map_sample_size_to_code(text: str) -> int:
    """Map sample size bias text to numerical code (0-9)"""
    if not text or text.strip() == "":
        return 0  # No bias detected
    
    text = text.lower()
    
    if "empty text provided" in text:
        return 1
    elif "no sample size reported" in text:
        return 2
    elif "very small sample size" in text and "high risk of underpowering and selection bias" in text:
        return 3
    elif "small per-group sample size" in text and "high risk of underpowering" in text:
        return 4
    elif "modest sample size" in text and "may be underpowered to detect moderate effects" in text:
        return 5
    elif "small sample size" in text and "may be underpowered to detect clinically meaningful effects" in text:
        return 6
    elif "no power analysis reported" in text and "unable to verify adequate study power" in text:
        return 7
    elif "large observational study" in text and "without power analysis" in text:
        return 8
    elif "very small per-group sample size" in text and "high risk of underpowering despite total" in text:
        return 9
    else:
        return 0  # Default to no bias if pattern not recognized

def map_placebo_to_code(text: str) -> int:
    """Map placebo bias text to numerical code (0-9)"""
    if not text or text.strip() == "":
        return 0  # No bias detected
    
    text = text.lower()
    
    if "open-label" in text or "unblinded study" in text:
        return 1
    elif "explicitly states no placebo used" in text:
        return 2
    elif "head-to-head trial without blinding" in text:
        return 3
    elif "control group without placebo or blinding" in text:
        return 4
    elif "blinded control without placebo mention" in text:
        return 5
    elif "compared to standard of care without placebo or blinding" in text:
        return 6
    elif "no placebo or control group mentioned" in text or "single-arm study" in text:
        return 7
    elif "claims blinding without placebo mention" in text:
        return 8
    elif "no mention of placebo" in text and "controlled trial" in text:
        return 9
    else:
        return 0  # Default to no bias

def map_blinding_to_code(text: str) -> int:
    """Map blinding bias text to numerical code (0-7)"""
    if not text or text.strip() == "":
        return 0  # No bias detected
    
    text = text.lower()
    
    if "blinding explicitly not performed" in text or "open-label" in text or "unblinded" in text:
        return 1
    elif "placebo" in text and "no blinding reported" in text:
        return 2
    elif "randomized controlled trial without blinding mention" in text:
        return 3
    elif "no mention of blinding" in text:
        return 4
    elif "single-blind design" in text:
        return 5
    elif "blinding mentioned but lacks detail" in text:
        return 6
    elif "blinding reported but no" in text and "method" in text:
        return 7
    else:
        return 0  # Default to no bias

def map_composite_outcome_to_code(text: str) -> int:
    """Map composite outcome bias text to numerical code (0-8)"""
    if not text or text.strip() == "":
        return 0  # No bias detected
    
    text = text.lower()
    
    if "composite primary outcome detected" in text and "may reduce interpretability" in text:
        return 1
    elif "composite outcome detected" in text and "verify clinical relevance" in text:
        return 2
    elif "composite primary" in text and "hard endpoints" in text:
        return 3
    elif "composite primary" in text and "soft endpoints" in text:
        return 4
    elif "composite primary" in text and "surrogate endpoints" in text:
        return 5
    elif "composite primary outcome" in text and "components" in text and "interpretability" in text:
        return 6
    elif "composite outcome" in text and "components" in text:
        return 7
    elif "standard composite outcome" in text or "mace" in text.lower():
        return 8
    else:
        return 0  # Default to no bias

def map_primary_outcome_events_to_code(text: str) -> int:
    """Map primary outcome events bias text to numerical code (0-6)"""
    if not text or text.strip() == "":
        return 0  # No bias detected
    
    text = text.lower()
    
    if "primary outcome results not clearly reported" in text:
        return 1
    elif "very low primary outcome events" in text and "minimum" in text:
        return 2
    elif "low primary outcome events in" in text and "groups" in text:
        return 3
    elif "low primary outcome events" in text and "total" in text:
        return 4
    elif "highly unbalanced event rates" in text:
        return 5
    elif "low total events" in text and "multivariate analysis" in text:
        return 6
    else:
        return 0  # Default to no bias

def process_bias_csv(input_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Process the bias detection CSV and add numerical codes for each bias type.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the output CSV (optional)
    
    Returns:
        DataFrame with added code columns
    """
    # Read the CSV
    df = pd.read_csv(input_file)
    
    # Apply mapping functions to create code columns
    df['sample_size_code'] = df['sample_size_bias'].fillna('').apply(map_sample_size_to_code)
    df['placebo_code'] = df['placebo_bias'].fillna('').apply(map_placebo_to_code)
    df['blinding_code'] = df['blinding_bias'].fillna('').apply(map_blinding_to_code)
    df['composite_outcome_code'] = df['composite_outcome_bias'].fillna('').apply(map_composite_outcome_to_code)
    df['primary_outcome_events_code'] = df['primary_outcome_events_bias'].fillna('').apply(map_primary_outcome_events_to_code)
    
    # Calculate total bias codes detected (non-zero codes)
    code_columns = ['sample_size_code', 'placebo_code', 'blinding_code', 
                   'composite_outcome_code', 'primary_outcome_events_code']
    df['total_bias_codes_detected'] = (df[code_columns] != 0).sum(axis=1)
    
    # Reorder columns to put codes next to their corresponding text
    new_column_order = []
    for col in df.columns:
        if col not in code_columns + ['total_bias_codes_detected']:
            new_column_order.append(col)
            # Add corresponding code column if it exists
            code_col = col.replace('_bias', '_code')
            if code_col in df.columns:
                new_column_order.append(code_col)
    
    # Add remaining columns
    new_column_order.extend(['total_bias_codes_detected'])
    df = df[new_column_order]
    
    # Save to file if specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"✅ Processed data saved to: {output_file}")
    
    return df

def analyze_bias_codes(df: pd.DataFrame) -> None:
    """Print summary statistics of bias codes"""
    print("\n📊 BIAS CODE ANALYSIS")
    print("=" * 50)
    
    code_columns = ['sample_size_code', 'placebo_code', 'blinding_code', 
                   'composite_outcome_code', 'primary_outcome_events_code']
    
    total_abstracts = len(df)
    
    for col in code_columns:
        bias_type = col.replace('_code', '').replace('_', ' ').title()
        
        # Count each code
        code_counts = df[col].value_counts().sort_index()
        no_bias_count = code_counts.get(0, 0)
        bias_detected_count = total_abstracts - no_bias_count
        
        print(f"\n{bias_type}:")
        print(f"  No bias (code 0): {no_bias_count} ({no_bias_count/total_abstracts*100:.1f}%)")
        print(f"  Bias detected: {bias_detected_count} ({bias_detected_count/total_abstracts*100:.1f}%)")
        
        # Show breakdown of specific bias codes
        for code in sorted(code_counts.index):
            if code != 0:
                count = code_counts[code]
                print(f"    Code {code}: {count} ({count/total_abstracts*100:.1f}%)")
# Example usage
if __name__ == "__main__":
    # Process a single CSV file
    input_file = "batch_analysis.csv"  # Your input file
    output_file = "batch_rct_analysis_with_codes.csv"
    
    try:
        # Process the CSV and add codes
        df = process_bias_csv(input_file, output_file)
        
        # Analyze the results
        analyze_bias_codes(df)
        
        print(f"\n✅ Processing complete!")
        print(f"📁 Original columns: {df.columns.tolist()[:8]}...")  # Show first 8 columns
        print(f"📊 Added code columns for model comparison")
        
        # Example: If you have results from multiple models, you can compare them
        # df_model2 = process_bias_csv("model2_results.csv")
        # compare_models(df, df_model2, "Original Model", "Model 2")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find input file '{input_file}'")
        print("📝 Please make sure the file exists and the path is correct")
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")