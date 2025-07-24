from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
from tqdm import tqdm

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Load and prepare data
df = pd.read_csv(r"path/to/your/dataset.csv")
print(f"Original dataset shape: {df.shape}")
print(f"Working with: {df.shape}")

print("Sentiment Analysis with RoBERTa")
print("=" * 40)

# Initialize global variables
scores_dict = {}
results_dict = {}  # Store all analysis results

def polarity_scores_roberta(text):
    """
    Analyze sentiment of input text using RoBERTa model
    """
    try:
        # Handle NaN or empty text
        if pd.isna(text) or text == "":
            return None
            
        # Tokenize and encode the text
        encoded_text = tokenizer(str(text), return_tensors='pt', truncation=True, max_length=512)
        
        # Get model output
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        # Return scores dictionary
        return {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
    
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return None

def format_sentiment_results(scores):
    """
    Format and display sentiment analysis results
    """
    if not scores:
        print("No sentiment data available.")
        return
    
    sentiment_labels = {
        'roberta_neg': 'Negative',
        'roberta_neu': 'Neutral', 
        'roberta_pos': 'Positive'
    }
    
    print("\nSentiment Analysis Results:")
    print("-" * 30)
    
    for key, score in scores.items():
        label = sentiment_labels[key]
        percentage = score * 100
        print(f"{label:8}: {percentage:6.2f}%")
    
    # Find the dominant sentiment
    max_key = max(scores, key=scores.get)
    dominant_sentiment = sentiment_labels[max_key]
    confidence = scores[max_key] * 100
    
    print(f"\nPredicted: {dominant_sentiment} ({confidence:.1f}% confidence)")

def analyze_dataset():
    """
    Analyze all reviews in the dataset
    """
    global results_dict
    print("\nAnalyzing all reviews in the dataset...")
    print("This may take a few minutes...")
    
    results_list = []
    
    # Analyze each review
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
        text = row.get('Text', '')
        summary = row.get('Summary', '')
        
        # Analyze the review text
        result = polarity_scores_roberta(text)
        
        if result:
            result_entry = {
                'Id': row.get('Id', idx),
                'Summary': summary,
                'Text': text,
                'roberta_neg': result['roberta_neg'],
                'roberta_neu': result['roberta_neu'],
                'roberta_pos': result['roberta_pos']
            }
            results_list.append(result_entry)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    results_dict['analysis_results'] = results_df
    
    print(f"\nCompleted analysis of {len(results_df)} reviews!")
    return results_df

def show_positive_comments():
    """
    Display all positive comments from the dataset analysis
    """
    if 'analysis_results' not in results_dict:
        print("No dataset analysis found. Please run 'analyze' command first.")
        return
    
    results_df = results_dict['analysis_results']
    
    # Filter for positive reviews (positive score > negative and neutral, and > 60% confidence)
    positive_reviews = results_df[
        (results_df['roberta_pos'] > results_df[['roberta_neg', 'roberta_neu']].max(axis=1)) & 
        (results_df['roberta_pos'] > 0.6)
    ]
    
    if positive_reviews.empty:
        print("No strongly positive reviews found (confidence > 60%).")
        return
    
    print(f"\nAll {len(positive_reviews)} Positive Reviews:")
    print("=" * 100)
    
    for i, (idx, row) in enumerate(positive_reviews.iterrows(), 1):
        print(f"{i}. Review ID: {row['Id']}")
        print(f"   Confidence: {row['roberta_pos']:.1%}")
        print(f"   Summary: {row['Summary']}")
        print(f"   Review: {row['Text'][:200]}{'...' if len(str(row['Text'])) > 200 else ''}")
        print("-" * 100)

def show_negative_comments():
    """
    Display all negative comments from the dataset analysis
    """
    if 'analysis_results' not in results_dict:
        print("No dataset analysis found. Please run 'analyze' command first.")
        return
    
    results_df = results_dict['analysis_results']
    
    # Filter for negative reviews
    negative_reviews = results_df[
        (results_df['roberta_neg'] > results_df[['roberta_pos', 'roberta_neu']].max(axis=1)) & 
        (results_df['roberta_neg'] > 0.6)
    ]
    
    if negative_reviews.empty:
        print("No strongly negative reviews found (confidence > 60%).")
        return
    
    print(f"\nAll {len(negative_reviews)} Negative Reviews:")
    print("=" * 100)
    
    for i, (idx, row) in enumerate(negative_reviews.iterrows(), 1):
        print(f"{i}. Review ID: {row['Id']}")
        print(f"   Confidence: {row['roberta_neg']:.1%}")
        print(f"   Summary: {row['Summary']}")
        print(f"   Review: {row['Text'][:200]}{'...' if len(str(row['Text'])) > 200 else ''}")
        print("-" * 100)

def show_statistics():
    """
    Show overall statistics of the analysis
    """
    if 'analysis_results' not in results_dict:
        print("No dataset analysis found. Please run 'analyze' command first.")
        return
    
    results_df = results_dict['analysis_results']
    
    # Calculate dominant sentiments
    dominant_sentiments = []
    for _, row in results_df.iterrows():
        scores = [row['roberta_neg'], row['roberta_neu'], row['roberta_pos']]
        max_idx = scores.index(max(scores))
        sentiment_names = ['Negative', 'Neutral', 'Positive']
        dominant_sentiments.append(sentiment_names[max_idx])
    
    sentiment_counts = pd.Series(dominant_sentiments).value_counts()
    
    print("\nDataset Sentiment Statistics:")
    print("=" * 40)
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"{sentiment:8}: {count:3d} reviews ({percentage:5.1f}%)")
    
    print(f"\nTotal analyzed: {len(results_df)} reviews")

def main():
    """
    Main function to run sentiment analysis
    """
    print("Available commands:")
    print("- Enter text: Analyze individual text")
    print("- 'analyze': Analyze all reviews in dataset")
    print("- 'positive': Show positive reviews from dataset")
    print("- 'negative': Show negative reviews from dataset")
    print("- 'stats': Show dataset statistics")
    print("- 'quit', 'exit', 'q': Exit program\n")
    
    while True:
        # Get user input
        user_input = input("Enter command or text to analyze: ").strip()
        
        # Check for exit conditions
        if user_input.lower() in ['quit', 'exit', 'q', '']:
            print("Thank you for using the sentiment analyzer! Goodbye!")
            break
        
        # Check for special commands
        if user_input.lower() == 'analyze':
            analyze_dataset()
            print("\n" + "="*50 + "\n")
            continue
        
        if user_input.lower() == 'positive':
            show_positive_comments()
            print("\n" + "="*50 + "\n")
            continue
        
        if user_input.lower() == 'negative':
            show_negative_comments()
            print("\n" + "="*50 + "\n")
            continue
        
        if user_input.lower() == 'stats':
            show_statistics()
            print("\n" + "="*50 + "\n")
            continue
        
        # Run sentiment analysis on individual text
        result = polarity_scores_roberta(user_input)
        
        if result:
            # Display results
            format_sentiment_results(result)
            print("\n" + "="*50 + "\n")
        else:
            print("Failed to analyze sentiment. Please check your input and try again.\n")

# Main execution
if __name__ == "__main__":
    main()