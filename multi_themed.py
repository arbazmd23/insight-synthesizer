import os
import json
import re
import pandas as pd
from collections import Counter
from llama_cpp import Llama
from textblob import TextBlob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import signal
from contextlib import contextmanager
import time

# Configuration
MODEL_PATH = "Qwen3-8B-Q4_K_M.gguf"  
OUTPUT_FILE = "multiple_theme_insights.json"
VISUALIZATION_FILE = "theme_analysis.png"
TEXT_REPORT_FILE = "theme_report.txt"
MAX_THEMES = 3  # Changed from 1 to 3 to match project requirements

class SurveyAnalyzer:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.llm = None
    
    def load_model(self):
        """Load the Qwen GGUF model with optimized settings"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        print(f"Loading Qwen model from {self.model_path}...")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=True
        )
        return self.llm
    
    @staticmethod
    def analyze_sentiment(text):
        """Enhanced sentiment analysis with TextBlob"""
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0.15:
            return "positive"
        elif analysis.sentiment.polarity < -0.15:
            return "negative"
        else:
            return "neutral"
    
    def generate_themes_prompt(self, responses):
        """Generate a prompt asking the LLM to identify multiple themes"""
    
        sample_size = min(15, len(responses))
        sample_responses = "\n".join(f"- {r}" for r in responses[:sample_size])
        
        return f"""<|im_start|>system
You are an expert at analyzing customer feedback and identifying key themes. Your task is to analyze survey responses and extract the {MAX_THEMES} most significant themes.

Sample responses from the survey:
{sample_responses}

[Total survey responses: {len(responses)}]

Instructions:
1. Analyze these responses and identify the {MAX_THEMES} most important and dominant themes
2. For each theme:
   - Provide a concise 2-5 word title
   - Determine the sentiment (positive, neutral, or negative)
   - Explain why this theme is significant in 1-2 sentences

Your analysis should be insightful, objective, and focused on what customers care about most. Themes should be specific and actionable, not generic.

Format your response exactly as follows:
THEME 1: [Theme Title]
SENTIMENT 1: [positive/neutral/negative]
EXPLANATION 1: [Brief explanation]

THEME 2: [Theme Title]
SENTIMENT 2: [positive/neutral/negative]
EXPLANATION 2: [Brief explanation]

THEME 3: [Theme Title]
SENTIMENT 3: [positive/neutral/negative]
EXPLANATION 3: [Brief explanation]<|im_end|>
<|im_start|>user
Please analyze all {len(responses)} survey responses and identify the {MAX_THEMES} most significant themes.<|im_end|>
"""

    def generate_quotes_prompt(self, responses, theme):
        """Generate a prompt to extract supporting quotes for a specific theme"""
        sample_size = min(20, len(responses))
        sample_responses = "\n".join(f"- {r}" for r in responses[:sample_size])
        
        return f"""<|im_start|>system
You are an expert at identifying the most relevant customer quotes that support a specific theme.

Theme: "{theme['title']}"
Theme explanation: "{theme['explanation']}"
Theme sentiment: {theme['sentiment']}

Your task is to identify the most representative customer quotes that support this theme from the survey responses below.
Select quotes that:
1. Clearly express the theme
2. Provide specific details or context
3. Represent different perspectives within the theme
4. Match the theme's sentiment

Sample survey responses:
{sample_responses}

[Total survey responses: {len(responses)}]

Instructions:
Select 12 quotes that best represent this theme. Choose quotes that are diverse, specific, and impactful.

Format your response as a JSON array of quotes:
["Quote 1", "Quote 2", "Quote 3", ...]

Only include the exact quotes from the responses. Do not modify them. Only include the JSON array in your response, nothing else.<|im_end|>
<|im_start|>user
Please select the 12 most representative quotes for the theme "{theme['title']}".<|im_end|>
"""

    @contextmanager
    def timeout(self, seconds):
        """Context manager for timing out operations"""
        def raise_timeout(signum, frame):
            raise TimeoutError("Model inference timed out")
            
        original_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, raise_timeout)
        
        try:
            signal.alarm(seconds)
            yield
        finally:
            signal.signal(signal.SIGALRM, original_handler)
            signal.alarm(0)
    
    def extract_themes(self, responses):
        """Extract multiple themes using the LLM"""
        if self.llm is None:
            self.load_model()
            
        prompt = self.generate_themes_prompt(responses)
        print(f"Generating {MAX_THEMES} dominant themes...")
        
        try:
            try:
                with self.timeout(300): 
                    output = self.llm(
                        prompt,
                        max_tokens=1500,
                        temperature=0.3,
                        stop=["<|im_end|>"],
                        echo=False
                    )
            except (AttributeError, ImportError, NameError):
                print("Timeout functionality not available, using standard call")
                output = self.llm(
                    prompt,
                    max_tokens=1500,
                    temperature=0.3,
                    stop=["<|im_end|>"],
                    echo=False
                )
                
            themes = self.parse_themes(output['choices'][0]['text'])
            
            if themes and len(themes) > 0:
                print(f"Successfully extracted {len(themes)} themes")
                return themes
            else:
                print("Failed to parse themes from model output.")
                return self.backup_extract_themes(responses)
                
        except Exception as e:
            print(f"Error during theme extraction: {e}")
            return self.backup_extract_themes(responses)
    
    def parse_themes(self, output_text):
        """Parse multiple themes from the LLM output"""
        themes = []
        
        # Updated regex pattern to capture multiple themes
        theme_pattern = re.compile(r'THEME\s*(\d+):?\s*(.*)', re.IGNORECASE)
        sentiment_pattern = re.compile(r'SENTIMENT\s*(\d+):?\s*(positive|negative|neutral)', re.IGNORECASE)
        explanation_pattern = re.compile(r'EXPLANATION\s*(\d+):?\s*(.*?)(?=\n\nTHEME|\n\n|\Z)', re.IGNORECASE | re.DOTALL)
        
        # Find all matches
        theme_matches = theme_pattern.finditer(output_text)
        sentiment_matches = {m.group(1): m.group(2).lower() for m in sentiment_pattern.finditer(output_text)}
        explanation_matches = {m.group(1): m.group(2).strip() for m in explanation_pattern.finditer(output_text)}
        
        for match in theme_matches:
            theme_number = match.group(1)
            theme_title = match.group(2).strip()
            
            sentiment = sentiment_matches.get(theme_number, "neutral")
            explanation = explanation_matches.get(theme_number, "")
            
            themes.append({
                "title": theme_title,
                "sentiment": sentiment,
                "explanation": explanation
            })
        
        # If parsing failed but there's some text, try fallback parsing
        if not themes and output_text.strip():
            # Simple fallback: try to split by double newlines and parse each block
            blocks = output_text.split("\n\n")
            current_theme = {}
            
            for block in blocks:
                if "THEME" in block.upper():
                    if current_theme and "title" in current_theme:
                        themes.append(current_theme)
                    current_theme = {"title": block.split(":", 1)[1].strip() if ":" in block else block.strip()}
                elif "SENTIMENT" in block.upper() and current_theme:
                    sentiment = block.split(":", 1)[1].strip().lower() if ":" in block else block.strip().lower()
                    if sentiment in ["positive", "negative", "neutral"]:
                        current_theme["sentiment"] = sentiment
                    else:
                        current_theme["sentiment"] = "neutral"
                elif "EXPLANATION" in block.upper() and current_theme:
                    current_theme["explanation"] = block.split(":", 1)[1].strip() if ":" in block else block.strip()
            
            if current_theme and "title" in current_theme:
                themes.append(current_theme)
        
        # Limit to MAX_THEMES
        return themes[:MAX_THEMES]
    
    def extract_supporting_quotes(self, responses, theme):
        """Extract supporting quotes for a specific theme using the LLM"""
        if self.llm is None:
            self.load_model()
            
        prompt = self.generate_quotes_prompt(responses, theme)
        print(f"Extracting quotes for theme: {theme['title']}...")
        
        try:
            output = self.llm(
                prompt,
                max_tokens=2000,
                temperature=0.3,
                stop=["<|im_end|>"],
                echo=False
            )

            quotes_text = output['choices'][0]['text'].strip()

            json_start = quotes_text.find('[')
            json_end = quotes_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                quotes_json = quotes_text[json_start:json_end]
                try:
                    quotes = json.loads(quotes_json)
                    if isinstance(quotes, list) and all(isinstance(q, str) for q in quotes):
                        verified_quotes = []
                        for quote in quotes:
                            best_match = self.find_best_matching_response(quote, responses)
                            if best_match:
                                verified_quotes.append(best_match)
                        
                        return verified_quotes[:12]
                except json.JSONDecodeError:
                    print(f"Failed to parse quotes JSON for theme '{theme['title']}'")
            
            return self.backup_extract_quotes(responses, theme)
            
        except Exception as e:
            print(f"Error extracting quotes for theme '{theme['title']}': {e}")
            return self.backup_extract_quotes(responses, theme)
    
    def find_best_matching_response(self, quote, responses):
        """Find the original response that best matches the quote"""
        if quote in responses:
            return quote
            
        normalized_quote = ' '.join(quote.lower().split())
        normalized_responses = {' '.join(r.lower().split()): r for r in responses}
        if normalized_quote in normalized_responses:
            return normalized_responses[normalized_quote]
            
        best_match = None
        best_score = 0
        
        quote_words = set(quote.lower().split())
        
        for response in responses:
            response_words = set(response.lower().split())
            
            if len(quote_words) > 0 and len(response_words) > 0:
                intersection = len(quote_words.intersection(response_words))
                union = len(quote_words.union(response_words))
                score = intersection / union
                
                # Prioritize responses that contain all words from the quote
                if quote_words.issubset(response_words):
                    score += 0.5
                    
                # Prioritize responses that are close in length to the quote
                length_ratio = min(len(quote), len(response)) / max(len(quote), len(response))
                score += length_ratio * 0.5
                
                if score > best_score:
                    best_match = response
                    best_score = score
        
        # Return the best match if it's reasonably good
        if best_score > 0.5:
            return best_match
        
        return None
    
    def backup_extract_themes(self, responses):
        """Backup method to extract multiple themes when LLM approach fails"""
        print(f"Using backup theme extraction method for {MAX_THEMES} themes...")
        
        # Calculate sentiment for all responses
        sentiments = [self.analyze_sentiment(r) for r in responses]
        sentiment_counts = Counter(sentiments)
        
        # Get the most common sentiments
        common_sentiments = sentiment_counts.most_common(MAX_THEMES)
        
        # Create themes based on sentiments
        themes = []
        for sentiment, count in common_sentiments:
            themes.append({
                "title": f"{sentiment.title()} Feedback",
                "sentiment": sentiment,
                "explanation": f"{count} responses express {sentiment} sentiment toward the product/service, indicating significant {sentiment} reception."
            })
            
        # If we have fewer themes than needed, add generic ones
        theme_types = ["Feature Requests", "User Experience", "Technical Issues"]
        i = 0
        while len(themes) < MAX_THEMES and i < len(theme_types):
            if not any(theme_types[i] in t["title"] for t in themes):
                themes.append({
                    "title": theme_types[i],
                    "sentiment": "neutral",
                    "explanation": f"Users have provided feedback related to {theme_types[i].lower()}."
                })
            i += 1
            
        return themes[:MAX_THEMES]
    
    def backup_extract_quotes(self, responses, theme):
        """Backup method to extract quotes when LLM approach fails"""
        print(f"Using backup quote extraction for theme: {theme['title']}...")
        
        # Filter by sentiment match
        theme_sentiment = theme.get("sentiment", "neutral")
        matching_sentiment_responses = [r for r in responses if self.analyze_sentiment(r) == theme_sentiment]
        
        # If we don't have enough responses with matching sentiment, include others
        if len(matching_sentiment_responses) < 12:
            other_responses = [r for r in responses if r not in matching_sentiment_responses]
            # Sort other responses by length (prioritize longer, more detailed responses)
            other_responses.sort(key=len, reverse=True)
            matching_sentiment_responses.extend(other_responses)
            
        # Limit to 12 quotes
        return matching_sentiment_responses[:12]
    
    def analyze_survey(self, responses):
        """Main method to analyze survey responses and extract insights"""
        # Extract themes
        themes = self.extract_themes(responses)
        
        # For each theme, extract supporting quotes
        insights = []
        for theme in themes:
            quotes = self.extract_supporting_quotes(responses, theme)
            
            insights.append({
                "theme": theme["title"],
                "quotes": quotes,
                "sentiment": theme["sentiment"],
                "explanation": theme.get("explanation", "")
            })
        
        return insights
    
    def generate_output_and_visualization(self, insights, responses):
        """Generate output file and visualization for the insights"""
        # Analyze overall sentiment
        response_sentiments = [self.analyze_sentiment(r) for r in responses]
        sentiment_counts = dict(Counter(response_sentiments))
        overall_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        
        # Create output
        output = {
            "overall_sentiment": overall_sentiment,
            "sentiment_distribution": sentiment_counts,
            "themes": insights
        }
        
        # Save output JSON
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Generate visualization
        plt.figure(figsize=(15, 12))
        
        # Sentiment distribution chart
        plt.subplot(2, 2, 1)
        colors = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
        pd.Series(sentiment_counts).plot(
            kind='pie',
            autopct='%1.1f%%',
            title='Overall Sentiment Distribution',
            colors=[colors[s] for s in sentiment_counts.keys()]
        )
        
        # Theme visualization for multiple themes
        plt.subplot(2, 2, 2)
        plt.text(0.5, 0.95, 'TOP THEMES FROM SURVEY ANALYSIS', 
                 fontsize=16, ha='center', va='top', fontweight='bold')
        
        y_position = 0.85
        for i, insight in enumerate(insights):
            theme_color = colors[insight["sentiment"]]
            
            plt.text(0.5, y_position, f'{i+1}. {insight["theme"]}', 
                     fontsize=14, ha='center', va='top', 
                     color=theme_color, fontweight='bold')
            y_position -= 0.1
            
            plt.text(0.5, y_position, f'Sentiment: {insight["sentiment"].title()}', 
                     fontsize=12, ha='center', va='top', color=theme_color)
            y_position -= 0.05
        
        plt.axis('off')
        
        # Individual theme details (up to 3 themes)
        for i, insight in enumerate(insights[:3], 1):
            plt.subplot(2, 3, i+3)
            
            theme_color = colors[insight["sentiment"]]
            
            plt.text(0.5, 0.95, f'THEME {i}: {insight["theme"]}', 
                     fontsize=14, ha='center', va='top', 
                     color=theme_color, fontweight='bold')
            
            plt.text(0.5, 0.85, f'Sentiment: {insight["sentiment"].title()}', 
                     fontsize=12, ha='center', va='top', color=theme_color)
            
            if insight.get("explanation"):
                plt.text(0.5, 0.75, insight["explanation"], 
                         fontsize=11, ha='center', va='top', 
                         wrap=True, bbox=dict(facecolor='white', alpha=0.5))
            
            # Sample quotes (up to 3 for visualization)
            quote_text = ""
            for j, quote in enumerate(insight["quotes"][:3], 1):
                quote_text += f"• {quote[:80]}{'...' if len(quote) > 80 else ''}\n\n"
            
            if quote_text:
                plt.text(0.5, 0.65, quote_text, 
                         fontsize=10, ha='center', va='top', 
                         wrap=True, bbox=dict(facecolor='lightyellow', alpha=0.5))
            
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(VISUALIZATION_FILE, dpi=300)
        
        # Create text report
        with open(TEXT_REPORT_FILE, "w") as f:
            f.write("SURVEY ANALYSIS REPORT\n")
            f.write("=====================\n\n")
            f.write(f"Total Responses Analyzed: {len(responses)}\n")
            f.write(f"Overall Sentiment: {overall_sentiment.title()}\n\n")
            
            for i, insight in enumerate(insights, 1):
                f.write(f"THEME {i}: {insight['theme']}\n")
                f.write(f"Sentiment: {insight['sentiment'].title()}\n")
                
                if insight.get("explanation"):
                    f.write(f"Explanation: {insight['explanation']}\n")
                
                f.write("\nSUPPORTING QUOTES:\n")
                for j, quote in enumerate(insight["quotes"], 1):
                    f.write(f"{j}. {quote}\n")
                
                f.write("\n" + "-"*50 + "\n\n")
        
        return output


def load_responses():
    """Load survey responses from available sources"""
    if os.path.exists('cleaned_survey_responses.csv'):
        print("Loading responses from cleaned_survey_responses.csv")
        df = pd.read_csv('cleaned_survey_responses.csv')
        responses = df['cleaned_response'].dropna().tolist()
    elif os.path.exists('cleaned_survey_responses_testing.csv'):
        print("Loading responses from cleaned_survey_responses_testing.csv")
        df = pd.read_csv('cleaned_survey_responses_testing.csv')
        responses = df['cleaned_response'].dropna().tolist()
    else:
        # Extract responses from the existing JSON file
        try:
            print("Looking for insights JSON file...")
            json_files = [f for f in os.listdir('.') if f.endswith('_insights.json')]
            
            if json_files:
                with open(json_files[0], 'r') as f:
                    print(f"Loading responses from {json_files[0]}")
                    data = json.load(f)
                
                # Collect all quotes from themes
                responses = []
                if 'themes' in data:
                    for theme in data['themes']:
                        for quote in theme.get('quotes', []):
                            if quote not in responses:
                                responses.append(quote)
                elif 'dominant_theme' in data:
                    for quote in data['dominant_theme'].get('quotes', []):
                        responses.append(quote)
            else:
                print("No insight files found.")
                responses = []
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading responses: {e}")
            responses = []
    
    # Remove empty responses and duplicates
    responses = [r for r in responses if r and r.strip()]
    responses = list(set(responses))  # Remove duplicates
    
    return responses

def check_existing_output():
    """Check if we can reuse existing output file"""
    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing output file: {OUTPUT_FILE}")
        print("Checking if we can reuse it...")
        
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing_output = json.load(f)
            
            if existing_output.get("themes") and len(existing_output["themes"]) >= MAX_THEMES:
                print("Reusing existing output file!")
                return existing_output
        except Exception as e:
            print(f"Error reading existing file: {e}. Re-analyzing...")
    
    return None

def main():
    try:
        start_time = time.time()
        
        # Load responses
        responses = load_responses()
        print(f"Loaded {len(responses)} unique responses")
        
        if not responses:
            print("No responses found. Exiting.")
            return None
        
        # Check for existing output
        existing_output = check_existing_output()
        if existing_output:
            return existing_output
        
        # Initialize analyzer
        analyzer = SurveyAnalyzer()
        
        # Analyze survey
        print(f"Analyzing survey responses to extract {MAX_THEMES} themes...")
        insights = analyzer.analyze_survey(responses)
        
        # Generate output and visualization
        output = analyzer.generate_output_and_visualization(insights, responses)
        
        # Print summary
        print("\n=== Survey Analysis Complete ===")
        processing_time = time.time() - start_time
        print(f"Processing time: {processing_time:.2f} seconds")
        
        print("\nExtracted Themes:")
        for i, insight in enumerate(insights, 1):
            print(f"\n{i}. {insight['theme']} ({insight['sentiment']})")
            print(f"   {insight.get('explanation', '')}")
            print(f"   Supported by {len(insight['quotes'])} quotes")
        
        print(f"\nResults saved to {OUTPUT_FILE}")
        print(f"Visualization saved to {VISUALIZATION_FILE}")
        print(f"Detailed report saved to {TEXT_REPORT_FILE}")
        
        return output
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()