# Survey Theme Analyzer - Technical Documentation

## Architecture Overview

The Survey Theme Analyzer application follows a modular architecture designed to process survey responses through several stages to extract a dominant theme and supporting quotes.

## Core Components

### 1. SurveyAnalyzer Class

This is the primary class responsible for analyzing survey responses and extracting insights. Key methods include:

- `load_model()`: Loads the Qwen LLM with appropriate configuration
- `analyze_sentiment()`: Performs sentiment analysis on text using TextBlob
- `extract_themes()`: Extracts the dominant theme from survey responses
- `extract_supporting_quotes()`: Finds quotes that support the identified theme
- `analyze_survey()`: Orchestrates the entire analysis workflow
- `generate_output_and_visualization()`: Creates output files in multiple formats

### 2. LLM Integration

The application uses the llama-cpp-python library to interface with the Qwen language model:

```python
self.llm = Llama(
    model_path=self.model_path,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    verbose=True
)
```

Parameters explained:
- `n_ctx`: Context window size (2048 tokens)
- `n_threads`: Number of CPU threads for inference
- `n_gpu_layers`: GPU acceleration (disabled by default)

### 3. Prompt Engineering

#### Theme Extraction Prompt
```
You are an expert at analyzing customer feedback and identifying key themes.
Your task is to analyze survey responses and extract the single most significant theme.

Sample responses from the survey:
[Sample responses here]

Instructions:
1. Analyze these responses and identify the single most important and dominant theme
2. For this theme:
   - Provide a concise 2-5 word title
   - Determine the sentiment (positive, neutral, or negative)
   - Explain why this theme is significant in 1-2 sentences

Format your response exactly as follows:
THEME: [Theme Title]
SENTIMENT: [positive/neutral/negative]
EXPLANATION: [Brief explanation]
```

#### Quote Extraction Prompt
```
You are an expert at identifying the most relevant customer quotes that support a specific theme.

Theme: "[Theme Title]"
Theme explanation: "[Theme explanation]"
Theme sentiment: [sentiment]

Your task is to identify the most representative customer quotes that support this theme from the survey responses below.
Select quotes that:
1. Clearly express the theme
2. Provide specific details or context
3. Represent different perspectives within the theme
4. Match the theme's sentiment

Format your response as a JSON array of quotes:
["Quote 1", "Quote 2", "Quote 3", ...]
```

## Processing Pipeline

1. **Data Loading**
   - Loads survey responses from CSV or JSON sources
   - Cleans and deduplicates responses

2. **Theme Extraction**
   - Generates a prompt for the LLM
   - Extracts theme information (title, sentiment, explanation)
   - Falls back to backup method if LLM extraction fails

3. **Quote Selection**
   - Generates a prompt for the LLM
   - Extracts and verifies supporting quotes
   - Falls back to backup method if LLM extraction fails

4. **Output Generation**
   - Generates JSON data file
   - Creates visualization
   - Generates text report

## Backup Methods

### Backup Theme Extraction
If the LLM approach fails, the application:
1. Calculates the dominant sentiment across all responses
2. Creates a theme based on this sentiment

### Backup Quote Selection
If the LLM approach fails, the application:
1. Filters responses by matching sentiment
2. Prioritizes longer, more detailed responses

## Error Handling

The application includes:
- Timeout mechanism for LLM operations
- Exception handling for parsing errors
- Graceful fallbacks when primary methods fail

## Performance Considerations

- Uses timeout controls to prevent LLM operations from hanging
- Samples responses to keep prompts manageable
- Verifies quotes against original responses to ensure accuracy

## Output Formats

1. **JSON Output**: Structured data including theme, quotes, and sentiment distribution
2. **Visualization**: Matplotlib figure showing sentiment distribution and theme details
3. **Text Report**: Human-readable report with theme information and supporting quotes

## Customization Options

The application can be customized by modifying:
- `MODEL_PATH`: Path to the Qwen GGUF model file
- `OUTPUT_FILE`: JSON output filename
- `VISUALIZATION_FILE`: Visualization output filename
- `TEXT_REPORT_FILE`: Text report filename