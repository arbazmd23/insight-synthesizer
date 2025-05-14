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
MODEL_PATH = "Qwen3-8B-Q4_K_M.gguf"  # Path to your Qwen model file
OUTPUT_FILE = "multi_theme_insights.json"
VISUALIZATION_FILE = "themes_analysis.png"
TEXT_REPORT_FILE = "themes_report.txt"
MAX_THEMES = 3  # Number of themes to extract

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
        # Sample a subset of responses to keep the prompt manageable
        sample_size = min(15, len(responses))
        sample_responses = "\n".join(f"- {r}" for r in responses[:sample_size])
        
        return f"""<|im_start|>system
You are an expert at analyzing customer feedback and identifying key themes. Your task is to analyze survey responses and extract the top {MAX_THEMES} most significant themes.

Sample responses from the survey:
{sample_responses}

[Total survey responses: {len(responses)}]

Instructions:
1. Analyze these responses and identify the {MAX_THEMES} most important and distinct themes
2. For each theme:
   - Provide a concise 2-5 word title
   - Determine the sentiment (positive, neutral, or negative)
   - Explain why this theme is significant in 1-2 sentences

Your analysis should be insightful, objective, and focused on what customers care about most. Themes should be specific and actionable, not generic.

Format your response exactly as follows:
THEME 1: [Theme Title]
SENTIMENT: [positive/neutral/negative]
EXPLANATION: [Brief explanation]

THEME 2: [Theme Title]
SENTIMENT: [positive/neutral/negative]
EXPLANATION: [Brief explanation]

THEME 3: [Theme Title]
SENTIMENT: [positive/neutral/negative]
EXPLANATION: [Brief explanation]<|im_end|>
<|im_start|>user
Please analyze all {len(responses)} survey responses and identify the top {MAX_THEMES} most significant themes.<|im_end|>
"""

    def generate_quotes_prompt(self, responses, theme):
        """Generate a prompt to extract supporting quotes for a specific theme"""
        # Sample a subset of responses to keep the prompt manageable
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
            
        # Register signal handler
        original_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, raise_timeout)
        
        try:
            # Set alarm
            signal.alarm(seconds)
            yield
        finally:
            # Restore original handler and cancel alarm
            signal.signal(signal.SIGALRM, original_handler)
            signal.alarm(0)
    
    def extract_themes(self, responses):
        """Extract multiple themes using the LLM"""
        # Ensure model is loaded
        if self.llm is None:
            self.load_model()
            
        prompt = self.generate_themes_prompt(responses)
        print(f"Generating {MAX_THEMES} themes...")
        
        try:
            try:
                with self.timeout(300):  # 5 minute timeout
                    output = self.llm(
                        prompt,
                        max_tokens=1500,
                        temperature=0.3,
                        stop=["<|im_end|>"],
                        echo=False
                    )
            except (AttributeError, ImportError, NameError):
                # Fallback for systems without signal support
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
        """Parse the themes from the LLM output"""
        themes = []
        
        # Regex patterns for parsing
        theme_pattern = re.compile(r'THEME\s+(\d+):\s*(.*)', re.IGNORECASE)
        sentiment_pattern = re.compile(r'SENTIMENT:\s*(positive|negative|neutral)', re.IGNORECASE)
        explanation_pattern = re.compile(r'EXPLANATION:\s*(.*?)(?=\n\n|\Z)', re.IGNORECASE | re.DOTALL)
        
        # Split output by theme sections
        theme_sections = re.split(r'THEME\s+\d+:', output_text)[1:]  # Skip the first empty item
        
        # Process each theme section
        for i, section in enumerate(theme_sections, 1):
            # Get the theme title from the previous split
            theme_title_match = theme_pattern.search(f"THEME {i}: " + section)
            theme_title = theme_title_match.group(2).strip() if theme_title_match else f"Theme {i}"
            
            # Extract sentiment and explanation
            sentiment_match = sentiment_pattern.search(section)
            explanation_match = explanation_pattern.search(section)
            
            sentiment = sentiment_match.group(1).lower() if sentiment_match else "neutral"
            explanation = explanation_match.group(1).strip() if explanation_match else ""
            
            themes.append({
                "title": theme_title,
                "sentiment": sentiment,
                "explanation": explanation
            })
        
        return themes[:MAX_THEMES]  # Ensure we don't exceed MAX_THEMES

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
            
            # Extract the JSON array of quotes
            quotes_text = output['choices'][0]['text'].strip()
            
            # Look for something that resembles a JSON array
            json_start = quotes_text.find('[')
            json_end = quotes_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                quotes_json = quotes_text[json_start:json_end]
                try:
                    quotes = json.loads(quotes_json)
                    # Ensure we have a list of strings
                    if isinstance(quotes, list) and all(isinstance(q, str) for q in quotes):
                        # Verify quotes are from the original responses 
                        verified_quotes = []
                        for quote in quotes:
                            # Find best matching original response
                            best_match = self.find_best_matching_response(quote, responses)
                            if best_match:
                                verified_quotes.append(best_match)
                        
                        # Limit to 12 quotes
                        return verified_quotes[:12]
                except json.JSONDecodeError:
                    print(f"Failed to parse quotes JSON for theme '{theme['title']}'")
            
            # If we get here, something went wrong
            return self.backup_extract_quotes(responses, theme)
            
        except Exception as e:
            print(f"Error extracting quotes for theme '{theme['title']}': {e}")
            return self.backup_extract_quotes(responses, theme)
    
    def find_best_matching_response(self, quote, responses):
        """Find the original response that best matches the quote"""
        # Exact match
        if quote in responses:
            return quote
            
        # Normalized match (lowercase, whitespace)
        normalized_quote = ' '.join(quote.lower().split())
        normalized_responses = {' '.join(r.lower().split()): r for r in responses}
        if normalized_quote in normalized_responses:
            return normalized_responses[normalized_quote]
            
        # Fuzzy match
        best_match = None
        best_score = 0
        
        # Convert quote to a set of words for comparison
        quote_words = set(quote.lower().split())
        
        for response in responses:
            response_words = set(response.lower().split())
            
            # Calculate overlap score (Jaccard similarity)
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
        
        # If no good match, return None
        return None
    
    def backup_extract_themes(self, responses):
        """Backup method to extract themes when LLM approach fails"""
        print("Using backup theme extraction method...")
        
        # Calculate sentiment for all responses
        sentiments = [self.analyze_sentiment(r) for r in responses]
        sentiment_counts = Counter(sentiments)
        
        # Get dominant sentiment
        dominant_sentiment = sentiment_counts.most_common(1)[0][0]
        
        # Create a simple set of themes based on sentiment
        themes = [
            {
                "title": f"{dominant_sentiment.title()} Feedback",
                "sentiment": dominant_sentiment,
                "explanation": f"Most responses express {dominant_sentiment} sentiment toward the product/service."
            }
        ]
        
        # Add a second theme based on length/detail of responses
        long_responses = [r for r in responses if len(r.split()) > 15]
        short_responses = [r for r in responses if len(r.split()) <= 15]
        
        if len(long_responses) >= 5:
            themes.append({
                "title": "Detailed Feedback",
                "sentiment": "neutral",
                "explanation": "These responses provide more detailed and specific feedback about the product/service."
            })
        else:
            themes.append({
                "title": "Brief Comments",
                "sentiment": "neutral",
                "explanation": "Most responses are brief comments or quick reactions rather than detailed feedback."
            })
            
        # Add a third theme based on question words (suggesting feature requests or issues)
        question_responses = [r for r in responses if "?" in r or any(w in r.lower() for w in ["how", "what", "when", "why", "would", "could"])]
        
        if len(question_responses) >= 5:
            themes.append({
                "title": "Questions & Suggestions",
                "sentiment": "neutral",
                "explanation": "These responses include questions or suggestions for improvement."
            })
        else:
            # Default third theme
            themes.append({
                "title": "General Impressions",
                "sentiment": "neutral",
                "explanation": "These responses express overall impressions and reactions."
            })
            
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
        plt.figure(figsize=(12, 10))
        
        # Sentiment distribution chart
        plt.subplot(2, 2, 1)
        colors = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
        pd.Series(sentiment_counts).plot(
            kind='pie',
            autopct='%1.1f%%',
            title='Overall Sentiment Distribution',
            colors=[colors[s] for s in sentiment_counts.keys()]
        )
        
        # Themes visualization
        plt.subplot(2, 1, 2)
        
        plt.text(0.5, 0.98, 'KEY THEMES FROM SURVEY ANALYSIS', 
                 fontsize=18, ha='center', va='top', fontweight='bold')
        
        y_pos = 0.9
        for i, insight in enumerate(insights, 1):
            theme_color = colors[insight["sentiment"]]
            
            plt.text(0.5, y_pos, f'{i}. {insight["theme"]}', 
                     fontsize=16, ha='center', va='top', 
                     color=theme_color, fontweight='bold')
            
            plt.text(0.5, y_pos - 0.04, f'Sentiment: {insight["sentiment"].title()}', 
                     fontsize=12, ha='center', va='top', color=theme_color)
            
            if insight.get("explanation"):
                plt.text(0.5, y_pos - 0.08, insight["explanation"], 
                         fontsize=12, ha='center', va='top', 
                         wrap=True, bbox=dict(facecolor='white', alpha=0.5))
            
            # Sample quotes (limited to 3 for visualization)
            quote_text = ""
            for j, quote in enumerate(insight["quotes"][:3], 1):
                quote_text += f"• {quote[:100]}{'...' if len(quote) > 100 else ''}\n\n"
            
            if quote_text:
                plt.text(0.5, y_pos - 0.14, quote_text, 
                         fontsize=10, ha='center', va='top', 
                         wrap=True, bbox=dict(facecolor='lightyellow', alpha=0.5))
            
            y_pos -= 0.28
        
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
        print(f"Extracted {len(insights)} themes:")
        
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