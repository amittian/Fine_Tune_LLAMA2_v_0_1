import pandas as pd
from keybert import KeyBERT

class KeyPhraseExtractor:
    def __init__(self, 
                 keyphrase_ngram_range=(1, 2), 
                 stop_words='english', 
                 top_n=5, 
                 min_df=1, 
                 use_maxsum=False, 
                 use_mmr=False, 
                 diversity=0.5, 
                 #use_faiss=False, 
                 vectorizer=None, 
                 highlight=False):
        """
        Initializes the KeyPhraseExtractor with KeyBERT parameters.
        
        :param keyphrase_ngram_range: tuple, range of n-grams for keyphrase extraction (min_n, max_n)
        :param stop_words: str or list, stop words to exclude from extraction
        :param top_n: int, number of top keyphrases to extract
        :param min_df: float or int, minimum document frequency for terms
        :param use_maxsum: bool, use MaxSum similarity for extracting diverse keyphrases
        :param use_mmr: bool, use Maximal Marginal Relevance (MMR) for more diversity
        :param diversity: float (0 to 1), degree of diversity in keyphrases
        :param use_faiss: bool, use Faiss for faster similarity search in large corpora
        :param vectorizer: custom vectorizer (e.g., TF-IDF, CountVectorizer)
        :param highlight: bool, highlight extracted keyphrases in the text
        """
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.stop_words = stop_words
        self.top_n = top_n
        self.min_df = min_df
        self.use_maxsum = use_maxsum
        self.use_mmr = use_mmr
        self.diversity = diversity
        #self.use_faiss = use_faiss
        self.vectorizer = vectorizer
        self.highlight = highlight
        
        # Initialize the KeyBERT model
        self.model = KeyBERT()

    def extract_key_phrases(self, text):
        """
        Extract key phrases from a given text using the KeyBERT model.
        
        :param text: str, input text from which key phrases are to be extracted
        :return: list, top key phrases
        """
        # Extract key phrases using the configured KeyBERT model
        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=self.keyphrase_ngram_range,
            stop_words=self.stop_words,
            top_n=self.top_n,
            use_maxsum=self.use_maxsum,
            use_mmr=self.use_mmr,
            diversity=self.diversity,
            vectorizer=self.vectorizer,
            #use_faiss=self.use_faiss
        )
        
        # Return just the phrases, not the scores
        return [kw[0] for kw in keywords]

    def process_dataframe(self, df, text_column):
        """
        Apply key phrase extraction to a DataFrame column.
        
        :param df: pandas DataFrame, input DataFrame containing text
        :param text_column: str, name of the column with text data
        :return: pandas DataFrame, original DataFrame with additional 'key_phrases' column
        """
        df['key_phrases'] = df[text_column].apply(self.extract_key_phrases)
        return df

# Sample usage of the KeyPhraseExtractor class
if __name__ == "__main__":
    # Sample DataFrame
    data = {'descriptions': [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models are transforming industries.",
        "Natural language processing is a complex field of AI."
    ]}
    df = pd.DataFrame(data)
    
    # Initialize the key phrase extractor with specific parameters
    extractor = KeyPhraseExtractor(
        keyphrase_ngram_range=(1, 3),  # single words and two-word phrases
        stop_words='english',          # remove English stop words
        top_n=3,                       # extract top 3 key phrases
        use_mmr=True,                  # use MMR for more diverse key phrases
        diversity=0.9                 # prioritize diversity over relevance
    )
    
    # Process the DataFrame and extract key phrases from the 'descriptions' column
    df = extractor.process_dataframe(df, text_column='descriptions')

    # Display the DataFrame with the extracted key phrases
   # print(df)

df
