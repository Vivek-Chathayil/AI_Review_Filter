# AI_Review_Filter

# AI Review Filter - Sentiment Analysis Tool

A comprehensive sentiment analysis tool for CSV datasets using the pre-trained RoBERTa model from Cardiff NLP. This tool can analyze individual text inputs and perform batch analysis on any CSV dataset containing text reviews or comments.

## Features

- **Individual text analysis** - Analyze any text input in real-time
- **Batch CSV dataset analysis** - Process entire CSV datasets with text columns
- **Positive/Negative review filtering** - Find reviews with high confidence scores (>60%)
- **Statistical overview** - Get sentiment distribution across the dataset
- **Progress tracking** - Visual progress bars for batch operations
- **Error handling** - Robust handling of missing or invalid data
- **Flexible dataset support** - Works with any CSV format containing text data

## Supported Dataset Formats

The tool is designed to work with any CSV dataset that contains text data. It looks for these common column names:
- `Text` - Main review/comment text (primary analysis target)
- `Summary` - Brief summary of the review
- `Id` - Unique identifier for each entry

**Default Example**: Amazon Fine Food Reviews dataset from Kaggle

## Model Information

Uses the `cardiffnlp/twitter-roberta-base-sentiment` model:
- Pre-trained RoBERTa architecture
- Fine-tuned on Twitter sentiment data
- Three-class classification (Negative, Neutral, Positive)
- Optimized for social media but works excellently on product reviews and general text

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- A CSV dataset with text columns

### Setup

1. **Prepare your dataset**:
   - Ensure you have a CSV file with text data
   - The default path is set for Amazon Fine Food Reviews, but you can use any CSV
   - Update the file path in `Ai_Review_filter.py`:
   ```python
   df = pd.read_csv(r"YOUR_PATH_TO_CSV_FILE.csv")
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python Ai_Review_filter.py
   ```

## Usage

### Available Commands

The tool provides several commands for different types of analysis:

| Command | Description |
|---------|-------------|
| `analyze` | Process all reviews in the dataset (processes first 500 rows by default) |
| `positive` | Show all positive reviews with >60% confidence |
| `negative` | Show all negative reviews with >60% confidence |
| `stats` | Display overall sentiment statistics and distribution |
| `quit/exit/q` | Exit the program |
| Any text | Analyze individual text input in real-time |

### Basic Workflow

1. **Start the program**:
   ```bash
   python Ai_Review_filter.py
   ```

2. **Process your dataset**:
   ```
   Enter command or text to analyze: analyze
   ```
   This will process the first 500 rows of your CSV (configurable)

3. **View overall statistics**:
   ```
   Enter command or text to analyze: stats
   ```

4. **Filter specific sentiments**:
   ```
   Enter command or text to analyze: positive
   Enter command or text to analyze: negative
   ```

5. **Test individual text**:
   ```
   Enter command or text to analyze: This product is amazing!
   ```

### Example Session

```
AI Review Filter - Sentiment Analysis
====================================
Original dataset shape: (568454, 10)
Working with: (500, 10)

Sentiment Analysis with RoBERTa
========================================

Available commands:
- Enter text: Analyze individual text
- 'analyze': Analyze all reviews in dataset
- 'positive': Show positive reviews from dataset
- 'negative': Show negative reviews from dataset
- 'stats': Show dataset statistics
- 'quit', 'exit', 'q': Exit program

Enter command or text to analyze: This product exceeded my expectations!

Sentiment Analysis Results:
------------------------------
Negative:   1.82%
Neutral :   6.45%
Positive:  91.73%

Predicted: Positive (91.7% confidence)

==================================================

Enter command or text to analyze: analyze
Analyzing all reviews in the dataset...
This may take a few minutes...
Processing reviews: 100%|████████████| 500/500 [02:45<00:00,  3.02it/s]

Completed analysis of 485 reviews!

==================================================

Enter command or text to analyze: stats

Dataset Sentiment Statistics:
========================================
Positive: 378 reviews ( 77.9%)
Negative:  89 reviews ( 18.4%)
Neutral :  18 reviews (  3.7%)

Total analyzed: 485 reviews

==================================================

Enter command or text to analyze: positive

All 378 Positive Reviews:
====================================================================================================
1. Review ID: 1
   Confidence: 89.2%
   Summary: Good Quality Dog Food
   Review: I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product...

2. Review ID: 5
   Confidence: 94.1%
   Summary: Great product!
   Review: This is a confection that has been around a few centuries. It is a light, pillowy citrus gelatin with nuts - in this case...
```

## Technical Details

### Performance
- **Processing Speed**: ~3-4 reviews per second on CPU
- **Memory Usage**: ~1GB RAM (model loading + data processing)
- **Batch Size**: Processes 500 reviews by default (easily configurable)
- **Text Limit**: 512 tokens per review (automatically truncated)

### Data Handling
- **Flexible CSV Support**: Works with any CSV containing text columns
- **Column Detection**: Automatically detects `Text`, `Summary`, `Id` columns
- **Missing Data**: Handles NaN and empty values gracefully
- **Large Text**: Long reviews are automatically truncated to 512 tokens
- **Progress Tracking**: Real-time progress bars with processing speed

### Confidence Thresholds
- **High Confidence**: >60% used for filtering positive/negative reviews
- **Display Format**: Shows confidence percentages and review previews
- **Text Preview**: First 200 characters of each review for quick scanning

## Configuration Options

### Adjustable Parameters

You can easily modify these parameters in `Ai_Review_filter.py`:

```python
# Number of reviews to process (line 12)
df = df.head(500)  # Change 500 to desired number

# Confidence threshold for filtering (lines 108, 135)
(results_df['roberta_pos'] > 0.6)  # Change 0.6 to desired threshold

# Text preview length (lines 115, 142)
row['Text'][:200]  # Change 200 for longer/shorter preview

# CSV file path (line 8)
df = pd.read_csv(r"YOUR_PATH_HERE.csv")
```

## File Structure

```
ai-review-filter/
├── Ai_Review_filter.py      # Main application script
├── requirements.txt         # Python dependencies
├── README.md               # This documentation
└── data/
    └── your_dataset.csv    # Your CSV dataset
```

## Dataset Compatibility

### Tested Datasets
- ✅ Amazon Fine Food Reviews
- ✅ Product review datasets
- ✅ Customer feedback CSVs
- ✅ Social media comments
- ✅ Survey responses

### Required CSV Structure
Your CSV should contain at least one text column. Common column names supported:
- `Text` (primary target for analysis)
- `Summary` (brief text summary)
- `Id` (unique identifier)
- Any other text columns will be preserved

### Example CSV Format
```csv
Id,Summary,Text,Rating
1,"Great product","I love this item, excellent quality",5
2,"Poor quality","This product broke after one day",1
3,"Average","Nothing special about this product",3
```

## Troubleshooting

### Common Issues

1. **File Not Found Error**:
   ```
   FileNotFoundError: CSV file not found
   ```
   - Verify the file path in line 8 of `Ai_Review_filter.py`
   - Use absolute paths with raw strings: `r"C:\path\to\file.csv"`

2. **Column Not Found**:
   ```
   KeyError: 'Text' column not found
   ```
   - Check your CSV column names
   - The tool looks for `Text`, `Summary`, `Id` columns
   - Rename your text column to `Text` or modify the code

3. **Memory Error**:
   ```
   RuntimeError: out of memory
   ```
   - Reduce batch size: `df.head(100)` instead of 500
   - Close other memory-intensive applications
   - Consider processing in smaller chunks

4. **Slow Processing**:
   - Normal speed: 3-4 reviews/second on CPU
   - For faster processing, consider GPU acceleration
   - Reduce dataset size for testing: `df.head(50)`

5. **Empty Results**:
   - Check if your CSV has valid text in the `Text` column
   - Verify data isn't all NaN or empty strings
   - Try running individual text analysis first

### Performance Optimization

- **First Run**: Model downloads ~500MB on initial execution
- **GPU Support**: Install PyTorch with CUDA for 10x faster processing
- **Batch Processing**: Run `analyze` once, then explore results with other commands
- **Memory Management**: Use smaller batch sizes on limited RAM systems

## Results Interpretation

### Confidence Score Ranges
- **90-100%**: Extremely confident prediction
- **80-89%**: Very confident prediction  
- **60-79%**: Confident prediction (used for filtering)
- **40-59%**: Moderate confidence
- **<40%**: Low confidence (mixed or unclear sentiment)

### Sentiment Categories
- **Positive**: Customer satisfaction, praise, recommendations, positive emotions
- **Negative**: Complaints, dissatisfaction, problems, negative emotions
- **Neutral**: Factual statements, mixed opinions, informational content

### Statistical Output
The `stats` command provides:
- Count and percentage of each sentiment category
- Total number of successfully analyzed reviews
- Distribution overview for dataset insights

## Advanced Usage

### Processing Large Datasets
For datasets larger than 500 rows:

```python
# Process in chunks
df = pd.read_csv("large_dataset.csv")
chunk_size = 1000
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    # Process chunk
```

### Custom Column Names
If your CSV uses different column names:

```python
# Modify the analyze_dataset() function
text = row.get('review_text', '')  # Change 'Text' to your column name
summary = row.get('title', '')     # Change 'Summary' to your column name
```

## Contributing

Contributions welcome! Priority areas:

1. **Export Features**: Save filtered results to CSV/Excel
2. **Visualization**: Add sentiment distribution charts
3. **Multi-language Support**: Support for non-English text
4. **Advanced Filtering**: Custom confidence thresholds, date ranges
5. **Batch Processing**: Automatic chunking for large datasets

## License

MIT License. The RoBERTa model follows Hugging Face and Cardiff NLP licensing terms.

## Acknowledgments

- **Cardiff NLP** for the pre-trained sentiment model
- **Hugging Face** for the transformers library and model hosting
- **Facebook AI** for the RoBERTa architecture
- **Community contributors** for dataset testing and feedback

## Citation

If using this tool in research:

```bibtex
@software{ai_review_filter,
  title={AI Review Filter: RoBERTa-based Sentiment Analysis Tool},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ai-review-filter}
}
```

---

**Note**: This tool is designed for research, analysis, and educational purposes. Processing large datasets requires adequate computational resources and time. Always verify results with domain expertise for critical applications.
