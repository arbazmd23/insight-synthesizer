import os
import json
import re
import pandas as pd
from collections import defaultdict, Counter
from llama_cpp import Llama
from textblob import TextBlob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = "Qwen3-8B-Q4_K_M.gguf"  # Use your Qwen model file
OUTPUT_FILE = "single_theme_insights.json"
VISUALIZATION_FILE = "dominant_theme_analysis.png"

def load_model():
    """Load the Qwen GGUF model with optimized settings"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    print("Loading Qwen model...")
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,  # Reduced context window
        n_threads=4,  # Use fewer CPU threads to reduce memory usage
        n_gpu_layers=0,  # Change if you have GPU
        verbose=True    # Enable verbose mode to see progress
    )

def analyze_sentiment(text):
    """Enhanced sentiment analysis with TextBlob"""
    analysis = TextBlob(text)
    # Adjusted thresholds for better classification
    if analysis.sentiment.polarity > 0.15:
        return "positive"
    elif analysis.sentiment.polarity < -0.15:
        return "negative"
    else:
        return "neutral"

def generate_prompt(responses):
    """Improved prompt for single dominant theme extraction"""
    # Use fewer examples to keep prompt shorter
    sample_responses = "\n".join(f"- {r}" for r in responses[:5])
    return f"""<|im_start|>system
Analyze these customer feedback responses to identify ONE DOMINANT theme:
{sample_responses}
[Total responses: {len(responses)}]

Instructions:
1. Identify THE SINGLE MOST IMPORTANT theme that dominates the feedback
2. Provide a clear 2-4 word title for this theme
3. Select 5-8 representative VERBATIM quotes that support this theme
4. Determine overall theme sentiment (positive/neutral/negative)
5. Explain in 2-3 sentences why this is the dominant theme

Focus on:
- Specific product feedback (not generic comments)
- Issues/requests mentioned by multiple respondents
- Themes with strong emotional sentiment
- Actionable insights

Format EXACTLY like this:
DOMINANT THEME: [Theme Name]
SENTIMENT: [positive/neutral/negative]
QUOTES:
- [Exact quote 1]
- [Exact quote 2]
...
EXPLANATION: [Brief explanation of why this is the dominant theme]<|im_end|>
<|im_start|>user
Please analyze all {len(responses)} responses and identify the single most important dominant theme.<|im_end|>
"""

def extract_single_theme(llm, responses):
    """Extract a single dominant theme from model output"""
    prompt = generate_prompt(responses)
    
    print("Generating single dominant theme...")
    print("This may take a few minutes...")
    
    # Add timeout and error handling
    try:
        import signal
        from contextlib import contextmanager
        
        @contextmanager
        def timeout(time):
            # Register a function to raise a TimeoutError on the signal.
            signal.signal(signal.SIGALRM, raise_timeout)
            # Schedule the signal to be sent after the specified time.
            signal.alarm(time)
            
            try:
                yield
            except Exception as e:
                print(f"Exception during model inference: {e}")
                raise
            finally:
                # Unregister the signal so it won't be triggered if the timeout is not reached.
                signal.signal(signal.SIGALRM, signal.SIG_IGN)
                signal.alarm(0)
                
        def raise_timeout(signum, frame):
            raise TimeoutError("Model inference timed out")
        
        # Try with timeout
        try:
            with timeout(300):  # 5 minute timeout
                output = llm(
                    prompt,
                    max_tokens=1500,  # Reduced for faster completion
                    temperature=0.3,
                    stop=["<|im_end|>"],
                    echo=False
                )
        except TimeoutError:
            print("Model inference timed out. Using fallback approach.")
            return fallback_single_theme_extraction(responses)
            
    except (AttributeError, ImportError):
        # If signal module is not available (e.g., on Windows)
        print("Timeout functionality not available, using standard call with reduced parameters")
        try:
            output = llm(
                prompt,
                max_tokens=1500,  # Reduced for faster completion
                temperature=0.3,
                stop=["<|im_end|>"],
                echo=False
            )
        except Exception as e:
            print(f"Exception during model inference: {e}")
            return fallback_single_theme_extraction(responses)
    
    try:
        dominant_theme = parse_single_theme(output['choices'][0]['text'], responses)
        
        if dominant_theme:
            print(f"Successfully extracted dominant theme: {dominant_theme['theme']}")
            print(f"Found {len(dominant_theme['quotes'])} supporting quotes")
            return dominant_theme
        else:
            print("Failed to parse single theme from model output. Using fallback.")
            return fallback_single_theme_extraction(responses)
    except Exception as e:
        print(f"Error parsing model output: {e}")
        return fallback_single_theme_extraction(responses)

def parse_single_theme(output_text, all_responses):
    """Parse the single theme from model output"""
    # Debug: Print the first part of the output
    print("Model output sample:")
    print(output_text[:200])
    
    # Create lookup for fuzzy quote matching
    response_lookup = {r.lower(): r for r in all_responses}
    
    # Regex patterns for parsing
    theme_pattern = re.compile(r'DOMINANT THEME:\s*(.*)', re.IGNORECASE)
    sentiment_pattern = re.compile(r'SENTIMENT:\s*(positive|negative|neutral)', re.IGNORECASE)
    explanation_pattern = re.compile(r'EXPLANATION:\s*(.*)', re.IGNORECASE)
    quote_pattern = re.compile(r'^-\s*(.*)')
    
    # Extract information
    theme_match = theme_pattern.search(output_text)
    sentiment_match = sentiment_pattern.search(output_text)
    explanation_match = explanation_pattern.search(output_text)
    
    if not theme_match:
        print("No theme found in model output")
        return None
    
    theme_name = theme_match.group(1).strip()
    sentiment = sentiment_match.group(1).lower() if sentiment_match else "neutral"
    explanation = explanation_match.group(1).strip() if explanation_match else ""
    
    # Extract quotes
    quotes = []
    quotes_section = False
    lines = output_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
            
        # Detect quotes section
        if line.upper() == "QUOTES:":
            quotes_section = True
            continue
            
        # End of quotes section
        if quotes_section and line.upper().startswith("EXPLANATION:"):
            quotes_section = False
            continue
            
        # Quote detection
        quote_match = quote_pattern.match(line)
        if quotes_section and quote_match:
            quote = quote_match.group(1).strip()
            
            # Try exact match first
            if quote.lower() in response_lookup:
                quotes.append(response_lookup[quote.lower()])
                continue
                
            # Try fuzzy matching next
            best_match = None
            best_score = 0
            for resp in all_responses:
                # Simple fuzzy match: if 80% of the words match
                quote_words = set(quote.lower().split())
                resp_words = set(resp.lower().split())
                if len(quote_words) > 0:
                    overlap = len(quote_words.intersection(resp_words))
                    score = overlap / len(quote_words)
                    if score > 0.8 and score > best_score:
                        best_match = resp
                        best_score = score
            
            if best_match:
                quotes.append(best_match)
    
    # If no quotes were extracted, find quotes related to the theme
    if not quotes:
        print("No quotes found, finding related quotes based on theme words...")
        theme_words = set(theme_name.lower().split())
        for resp in all_responses:
            for word in theme_words:
                if len(word) > 3 and word in resp.lower():  # Only consider meaningful words
                    quotes.append(resp)
                    if len(quotes) >= 5:  # Limit to 5 quotes
                        break
            if len(quotes) >= 5:
                break
    
    return {
        "theme": theme_name,
        "quotes": quotes,
        "sentiment": sentiment,
        "explanation": explanation,
        "quote_count": len(quotes)
    }

def fallback_single_theme_extraction(responses):
    """Extract a single dominant theme as a fallback"""
    print("Using fallback single theme extraction...")
    
    # Step 1: Analyze all responses for keywords and sentiment
    keywords = defaultdict(int)
    word_sentiment = defaultdict(list)
    
    # Common stop words to filter out
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                      'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                      'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
                      'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                      'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                      'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                      'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                      'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                      'with', 'about', 'against', 'between', 'into', 'through', 'during', 
                      'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                      'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
                      'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
                      'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
                      'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
                      'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
                      'should', 'now', 'would'])
    
    all_quotes_by_keyword = defaultdict(list)
    
    for response in responses:
        sentiment = analyze_sentiment(response)
        
        # Extract meaningful words (4+ characters)
        words = re.findall(r'\b\w{4,}\b', response.lower())
        for word in words:
            if word not in stop_words:
                keywords[word] += 1
                word_sentiment[word].append(sentiment)
                all_quotes_by_keyword[word].append(response)
    
    # Step 2: Identify possible themes based on frequency and emotional resonance
    potential_themes = []
    
    for word, count in keywords.items():
        if count >= 3:  # Word appears in at least 3 responses
            sentiments = word_sentiment[word]
            # Measure emotional resonance - more positive or negative responses
            non_neutral = [s for s in sentiments if s != "neutral"]
            emotional_weight = len(non_neutral) / len(sentiments) if sentiments else 0
            
            # Calculate theme score: frequency * emotional resonance
            theme_score = count * (1 + emotional_weight)
            
            potential_themes.append({
                "word": word,
                "score": theme_score,
                "count": count,
                "quotes": all_quotes_by_keyword[word]
            })
    
    # Step 3: Find the best theme
    if not potential_themes:
        # Fallback for no significant themes - use most common sentiment
        sentiment_counts = Counter([analyze_sentiment(r) for r in responses])
        dominant_sentiment = sentiment_counts.most_common(1)[0][0]
        
        return {
            "theme": f"General {dominant_sentiment.title()} Feedback",
            "quotes": [r for r in responses if analyze_sentiment(r) == dominant_sentiment][:5],
            "sentiment": dominant_sentiment,
            "explanation": f"Most responses ({sentiment_counts[dominant_sentiment]} out of {len(responses)}) expressed {dominant_sentiment} sentiment.",
            "quote_count": min(5, sentiment_counts[dominant_sentiment])
        }
    
    # Sort potential themes by score
    sorted_themes = sorted(potential_themes, key=lambda x: x["score"], reverse=True)
    best_theme = sorted_themes[0]
    
    # Step 4: Expand the theme name using the top related words
    # Find related words that often co-occur
    related_words = []
    for response in best_theme["quotes"]:
        words = [w for w in re.findall(r'\b\w{4,}\b', response.lower()) 
                if w not in stop_words and w != best_theme["word"]]
        related_words.extend(words)
    
    related_word_counts = Counter(related_words)
    top_related = [word for word, _ in related_word_counts.most_common(2)]
    
    # Construct a more meaningful theme name
    if top_related:
        theme_name = f"{best_theme['word'].title()} {' '.join(w.title() for w in top_related[:1])}"
    else:
        theme_name = best_theme["word"].title()
    
    # Make the theme name more descriptive
    if any(word in theme_name.lower() for word in ["use", "using", "usage"]):
        theme_name = "Ease of Use"
    elif any(word in theme_name.lower() for word in ["integrate", "integration", "connect"]):
        theme_name = "Integration Capabilities"
    elif any(word in theme_name.lower() for word in ["feature", "functionality"]):
        theme_name = "Feature Requests"
    elif any(word in theme_name.lower() for word in ["cost", "price", "pay", "payment"]):
        theme_name = "Cost Considerations"
    elif any(word in theme_name.lower() for word in ["time", "fast", "quick", "speed"]):
        theme_name = "Speed and Efficiency"
    
    # Calculate the dominant sentiment for this theme
    sentiments = [analyze_sentiment(q) for q in best_theme["quotes"]]
    sentiment_counter = Counter(sentiments)
    dominant_sentiment = sentiment_counter.most_common(1)[0][0]
    
    # Create explanation
    keyword_count = best_theme["count"]
    total_pct = (keyword_count / len(responses)) * 100
    explanation = f"This theme appeared in {keyword_count} responses ({total_pct:.1f}% of total) with predominantly {dominant_sentiment} sentiment."
    
    return {
        "theme": theme_name,
        "quotes": best_theme["quotes"][:8],  # Limit to 8 quotes
        "sentiment": dominant_sentiment,
        "explanation": explanation,
        "quote_count": min(8, len(best_theme["quotes"]))
    }

def generate_output_and_visualization(dominant_theme, responses):
    """Generate output file and visualization for single dominant theme"""
    # Analyze all responses' sentiment
    response_sentiments = [analyze_sentiment(r) for r in responses]
    sentiment_counts = dict(Counter(response_sentiments))
    
    overall_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
    
    # Create final output
    output = {
        "overall_sentiment": overall_sentiment,
        "sentiment_distribution": sentiment_counts,
        "dominant_theme": dominant_theme,
        "theme_quotes": dominant_theme["quotes"],
        "theme_sentiment": dominant_theme["sentiment"],
        "theme_explanation": dominant_theme.get("explanation", "")
    }
    
    # Save output
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Generate visualization
    plt.figure(figsize=(12, 10))
    
    # Sentiment distribution
    plt.subplot(2, 2, 1)
    colors = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
    pd.Series(sentiment_counts).plot(
        kind='pie',
        autopct='%1.1f%%',
        title='Response Sentiment',
        colors=[colors[s] for s in sentiment_counts.keys()]
    )
    
    # Theme visualization
    plt.subplot(2, 1, 2)
    
    plt.text(0.5, 0.95, f'DOMINANT THEME: {dominant_theme["theme"]}', 
             fontsize=18, ha='center', va='top', fontweight='bold')
    
    theme_color = colors[dominant_theme["sentiment"]]
    plt.text(0.5, 0.88, f'Sentiment: {dominant_theme["sentiment"].title()}', 
             fontsize=14, ha='center', va='top', 
             color=theme_color, fontweight='bold')
    
    # Display explanation
    explanation = dominant_theme.get("explanation", "")
    if explanation:
        plt.text(0.5, 0.82, explanation, 
                 fontsize=12, ha='center', va='top', 
                 wrap=True, bbox=dict(facecolor='white', alpha=0.5))
    
    # Display key quotes
    quote_y_pos = 0.74
    plt.text(0.5, quote_y_pos, "KEY SUPPORTING QUOTES:", 
             fontsize=14, ha='center', va='top', fontweight='bold')
    
    # Display quotes
    quote_text = ""
    for i, quote in enumerate(dominant_theme["quotes"][:6], 1):
        quote_text += f"{i}. {quote}\n\n"
    
    plt.text(0.5, quote_y_pos - 0.08, quote_text, 
             fontsize=11, ha='center', va='top', 
             wrap=True, bbox=dict(facecolor='lightyellow', alpha=0.5))
    
    plt.axis('off')  # Hide axes for the text display
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_FILE, dpi=300)
    
    # Create a separate text report
    with open("single_theme_report.txt", "w") as f:
        f.write(f"DOMINANT THEME: {dominant_theme['theme']}\n")
        f.write(f"Sentiment: {dominant_theme['sentiment'].title()}\n")
        if explanation:
            f.write(f"\n{explanation}\n")
        f.write(f"\nSupported by {dominant_theme['quote_count']} quotes\n\n")
        f.write("KEY QUOTES:\n")
        for i, quote in enumerate(dominant_theme["quotes"], 1):
            f.write(f"{i}. {quote}\n")

    return output

def main():
    try:
        # Check if cleaned_survey_responses.csv exists, otherwise use JSON responses
        if os.path.exists('cleaned_survey_responses.csv'):
            df = pd.read_csv('cleaned_survey_responses.csv')
            responses = df['cleaned_response'].tolist()
        else:
            # Extract responses from the existing JSON file
            with open('enhanced_insights.json', 'r') as f:
                data = json.load(f)
                # Collect all quotes from all themes and selected_quotes
                responses = []
                if 'selected_quotes' in data:
                    responses.extend(data['selected_quotes'])
                if 'themes' in data:
                    for theme in data['themes']:
                        for quote in theme.get('quotes', []):
                            if quote not in responses:
                                responses.append(quote)
        
        print(f"Loaded {len(responses)} responses")
        
        # Check if we can skip model inference
        if os.path.exists(OUTPUT_FILE):
            print(f"Found existing single theme file: {OUTPUT_FILE}")
            print("Checking if we can reuse it...")
            
            try:
                with open(OUTPUT_FILE, 'r') as f:
                    existing_output = json.load(f)
                
                if existing_output.get("dominant_theme"):
                    print("Reusing existing single theme file!")
                    print(f"Dominant theme: {existing_output['dominant_theme']['theme']}")
                    return existing_output
            except Exception as e:
                print(f"Error reading existing file: {e}. Re-analyzing...")
        
        # Extract single dominant theme
        print("Extracting single dominant theme...")
        llm = load_model()
        dominant_theme = extract_single_theme(llm, responses)
        
        # Generate output and visualization
        output = generate_output_and_visualization(dominant_theme, responses)
        
        # Print summary
        print("\n=== Single Theme Analysis Complete ===")
        print(f"DOMINANT THEME: {dominant_theme['theme']}")
        print(f"Sentiment: {dominant_theme['sentiment'].title()}")
        if dominant_theme.get("explanation"):
            print(f"Explanation: {dominant_theme['explanation']}")
        print(f"Supporting quotes: {dominant_theme['quote_count']}")
        print(f"\nResults saved to {OUTPUT_FILE}")
        print(f"Visualization saved to {VISUALIZATION_FILE}")
        print(f"Detailed report saved to single_theme_report.txt")
        
        return output
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()