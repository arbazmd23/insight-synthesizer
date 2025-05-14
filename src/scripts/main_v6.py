
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
import signal
from contextlib import contextmanager

MODEL_PATH = "Qwen3-8B-Q4_K_M.gguf"
OUTPUT_FILE = "single_theme_insights.json"
VISUALIZATION_FILE = "dominant_theme_analysis.png"
TEXT_REPORT_FILE = "single_theme_report.txt"

class SurveyAnalyzer:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.llm = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.llm = Llama(model_path=self.model_path, n_ctx=2048, n_threads=4, n_gpu_layers=0, verbose=True)

    @staticmethod
    def analyze_sentiment(text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0.15:
            return "positive"
        elif analysis.sentiment.polarity < -0.15:
            return "negative"
        else:
            return "neutral"

    def generate_prompt(self, responses):
        sample_responses = "\n".join(f"- {r}" for r in responses[:10])
        return f"""<|im_start|>system
Analyze these customer feedback responses to identify ONE DOMINANT theme:
{sample_responses}

Instructions:

1. Identify ONE DOMINANT theme that covers the majority of the feedback.
2. Provide a 2-5 word theme title.
3. Select 10-15 high-quality quotes that strongly support the theme.
4. Determine overall sentiment (positive/neutral/negative).
5. Explain in 2-3 sentences why this theme represents the dataset.

Format:
DOMINANT THEME: [Theme Name]
SENTIMENT: [positive/neutral/negative]
QUOTES:
* [Quote 1]
* [Quote 2]
...
EXPLANATION: [Why this is the dominant theme]<|im_end|>
<|im_start|>user
Please analyze all {len(responses)} responses and extract the most representative theme.<|im_end|>"""

    def extract_single_theme(self, responses):
        if self.llm is None:
            self.load_model()

        prompt = self.generate_prompt(responses)
        result = self.llm(prompt, max_tokens=2000, temperature=0.3, stop=["<|im_end|>"], echo=False)
        return self.parse_single_theme(result['choices'][0]['text'], responses)

    def parse_single_theme(self, output_text, all_responses):
        response_lookup = {r.lower(): r for r in all_responses}
        theme_pattern = re.compile(r'DOMINANT THEME:\s*(.*)', re.IGNORECASE)
        sentiment_pattern = re.compile(r'SENTIMENT:\s*(positive|negative|neutral)', re.IGNORECASE)
        explanation_pattern = re.compile(r'EXPLANATION:\s*(.*)', re.IGNORECASE)
        quote_pattern = re.compile(r'^\*\s*(.*)')

        theme_match = theme_pattern.search(output_text)
        sentiment_match = sentiment_pattern.search(output_text)
        explanation_match = explanation_pattern.search(output_text)

        theme = theme_match.group(1).strip() if theme_match else "Unknown"
        sentiment = sentiment_match.group(1).strip().lower() if sentiment_match else "neutral"
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        quotes = []
        in_quotes = False
        for line in output_text.splitlines():
            if line.upper().startswith("QUOTES"):
                in_quotes = True
                continue
            if line.upper().startswith("EXPLANATION"):
                in_quotes = False
                continue
            if in_quotes:
                match = quote_pattern.match(line)
                if match:
                    quote = match.group(1).strip()
                    best_match = next((r for r in all_responses if quote.lower() in r.lower()), quote)
                    quotes.append(best_match)

        return {
            "theme": theme,
            "sentiment": sentiment,
            "explanation": explanation,
            "quotes": quotes[:15],
            "quote_count": len(quotes[:15])
        }

    def generate_output_and_visualization(self, dominant_theme, responses):
        sentiment_counts = dict(Counter([self.analyze_sentiment(r) for r in responses]))
        overall_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]

        output = {
            "overall_sentiment": overall_sentiment,
            "sentiment_distribution": sentiment_counts,
            "dominant_theme": dominant_theme
        }

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output, f, indent=2)

        plt.figure(figsize=(10, 9))

        plt.subplot(2, 2, 1)
        pd.Series(sentiment_counts).plot(
            kind='pie', autopct='%1.1f%%', title='Sentiment Distribution',
            colors=[{"positive": "#4CAF50", "neutral": "#FFC107", "negative": "#F44336"}[s] for s in sentiment_counts]
        )

        plt.subplot(2, 1, 2)
        plt.text(0.5, 0.95, f"DOMINANT THEME: {dominant_theme['theme']}", ha='center', va='top', fontsize=18)
        plt.text(0.5, 0.88, f"Sentiment: {dominant_theme['sentiment'].title()}", ha='center', fontsize=14)
        plt.text(0.5, 0.82, dominant_theme['explanation'], ha='center', wrap=True, fontsize=12, bbox=dict(facecolor='white'))
        y = 0.74
        for i, quote in enumerate(dominant_theme['quotes'][:10], 1):
            plt.text(0.5, y, f"{i}. {quote}", ha='center', wrap=True, fontsize=10)
            y -= 0.06
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(VISUALIZATION_FILE, dpi=300)

        with open(TEXT_REPORT_FILE, 'w') as f:
            f.write(f"DOMINANT THEME: {dominant_theme['theme']}\n")
            f.write(f"Sentiment: {dominant_theme['sentiment'].title()}\n")
            f.write(f"\n{dominant_theme['explanation']}\n\n")
            for i, quote in enumerate(dominant_theme['quotes'], 1):
                f.write(f"{i}. {quote}\n")

        return output

def load_responses():
    df = pd.read_csv("cleaned_survey_responses_testing.csv")
    return df['cleaned_response'].dropna().tolist()

def main():
    responses = load_responses()
    analyzer = SurveyAnalyzer()
    dominant_theme = analyzer.extract_single_theme(responses)
    output = analyzer.generate_output_and_visualization(dominant_theme, responses)
    print("\n=== Dominant Theme Extraction Complete ===")
    print(json.dumps(output['dominant_theme'], indent=2))

if __name__ == '__main__':
    main()
