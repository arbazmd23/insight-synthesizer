import pandas as pd
import re

def clean_survey_responses(text):
    """
    Clean individual survey responses by:
    1. Removing special characters/encoding issues
    2. Standardizing whitespace
    3. Removing leading/trailing quotes
    4. Basic text normalization
    """
    if not isinstance(text, str):
        return ""
    
    # Fix common encoding issues
    text = text.replace('â€™', "'").replace('â€"', '-')
    
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s\'\-.,;?!]', '', text)
    
    # Standardize whitespace
    text = ' '.join(text.split())
    
    # Remove leading/trailing quotes/whitespace
    text = text.strip('"\'').strip()
    
    # Capitalize first letter
    if len(text) > 0:
        text = text[0].upper() + text[1:]
    
    return text

# Load your CSV file
df = pd.read_csv('Outlaw_ML_Assessment_Custom_Survey_Data.csv')

# Print the actual column names to verify
print("Actual columns in the CSV file:", df.columns.tolist())

# Use the correct column name from your CSV: 'survey_response'
df['cleaned_response'] = df['survey_response'].apply(clean_survey_responses)

# Remove any empty responses after cleaning
df = df[df['cleaned_response'].str.len() > 0]

# Save cleaned data to new CSV
df.to_csv('cleaned_survey_responses_testing.csv', index=False)

print(f"\nOriginal responses: {len(df)}")
print(f"Cleaned responses saved to 'cleaned_survey_responses.csv'")
print("\nSample cleaned responses:")
print(df['cleaned_response'].head().to_string(index=False))