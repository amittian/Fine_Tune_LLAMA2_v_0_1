import pandas as pd
import numpy as np

# Convert numpy types to native Python types
def convert_dtypes_for_postgres(df):
    for col in df.columns:
        if df[col].dtype == 'float32' or df[col].dtype == 'float64':
            df[col] = df[col].astype(float)
        elif df[col].dtype == 'int32' or df[col].dtype == 'int64':
            df[col] = df[col].astype(int)
        elif df[col].dtype == 'object':
            # If column contains lists/tuples (like your 'connect_Topic_Representation'), 
            # convert to a string that PostgreSQL can handle
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, tuple)) else x)
    return df

# Assuming df is your DataFrame
df_cleaned = convert_dtypes_for_postgres(df)

# Now df_cleaned has proper types for inserting into the PostgreSQL database




import pandas as pd
import scipy.constants as spc
from scipy.cluster import hierarchy as sch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


from bertopic import BERTopic
from umap import UMAP
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.cluster.hierarchy as sch

class TopicModeling:
    def __init__(self, df, column_name, sentence_transformer_model='sentence-transformers/bert-large-nli-stsb-mean-tokens'):
        """
        Initialize the TopicModeling class with the necessary models and data.
        :param df: DataFrame containing the text data.
        :param column_name: Name of the column containing text descriptions.
        :param sentence_transformer_model: Model to be used for sentence embeddings.
        """
        self.df = df
        self.column_name = column_name
        
        # Use a more optimized and powerful sentence transformer model if needed
        self.sentence_model = SentenceTransformer(sentence_transformer_model)

        # Set UMAP parameters for better optimization of topic modeling
        self.umap_model = UMAP(
            n_neighbors=30,        # Increased to consider a broader local neighborhood
            n_components=9,        # Reduced dimensionality to capture better topic relationships
            min_dist=0.0,          # Adjusted to allow slight separation between points
            metric="cosine",       # Cosine distance for embedding similarity
            random_state=42,  
          low_memory=False
        )

       # Initialize HDBSCAN model for clustering
        #self.hdbscan_model = HDBSCAN(min_samples=5,min_cluster_size=10, metric='euclidean', prediction_data=True)

        # Initialize BERTopic with HDBSCAN as the clustering model
        self.topic_model = BERTopic(
            #hdbscan_model=self.hdbscan_model,  # Integrating HDBSCAN for clustering
            language="english",
            umap_model=self.umap_model,
            embedding_model=self.sentence_model,
            top_n_words=20,
            n_gram_range=(1, 3),
            calculate_probabilities=True,
            nr_topics=9,  # Allow flexible number of topics
            representation_model=KeyBERTInspired()
        )

    def encode_embeddings(self):
        """
        Encodes the descriptions into embeddings using the sentence model.
        """
        descriptions = self.df[self.column_name].tolist()
        return self.sentence_model.encode(descriptions, show_progress_bar=False)

    def fit_topic_model(self, embeddings):
        """
        Fits the BERTopic model using the provided embeddings.
        """
        topics, probabilities = self.topic_model.fit_transform(self.df[self.column_name], embeddings)
        return topics, probabilities

    def get_similarity_matrix(self, probabilities):
        """
        Computes the cosine similarity matrix from topic probabilities.
        """
        topic_distributions = np.array(probabilities).T  # Transpose to get topics as rows
        return cosine_similarity(topic_distributions)

    def get_topic_info(self):
        """
        Retrieves the topic information from the BERTopic model.
        """
        return self.topic_model.get_topic_info()

    def hierarchical_clustering(self, descriptions):
        """
        Performs hierarchical clustering on the topics identified by the BERTopic model.
        """
        linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
        return self.topic_model.hierarchical_topics(descriptions, linkage_function=linkage_function)


def add_connect_bertopic_outputs_to_dataframe(df, topic_model, topics, probs):
    """
    Adds BERTopic outputs (Topic_id, Topic_Name, Topic_Representation, Probability, Document_Probabilities)
    to the original DataFrame with 'connect' prefix for each column.
    
    :param df: Original DataFrame containing the documents.
    :param topic_model: The trained BERTopic model.
    :param topics: List of topic IDs assigned to each document.
    :param probs: List of probability distributions for each document (should be a 2D list or array).
    
    :return: DataFrame with the additional BERTopic outputs.
    """
    
    # Step 1: Add the Topic ID
    df['connect_Topic_id'] = topics

    # Step 2: Add the Topic Name (Human-readable topic names)
    df['connect_Topic_Name'] = df['connect_Topic_id'].apply(lambda x: ' '.join([word for word, _ in topic_model.get_topic(x)] if x != -1 else 'Outlier'))
    
    # Step 3: Add the Topic Representation (list of words and their weights)
    df['connect_Topic_Representation'] = df['connect_Topic_id'].apply(lambda x: topic_model.get_topic(x) if x != -1 else [])
    
    # Step 4: Add the maximum Probability (confidence level in topic assignment)
    df['connect_Probability'] = [max(prob) if len(prob) > 0 else 0 for prob in probs]
    
    # Step 5: Add the full Document Probabilities (probability distribution across all topics)
    # Ensure 'probs' is a valid 2D array/list and the number of rows matches the dataframe
    # if len(probs) == len(df):
    #     df['connect_Document_Probabilities'] = probs
    # else:
    #     raise ValueError(f"Mismatch between number of documents ({len(df)}) and probability distributions ({len(probs)}).")
    
    return df



def save_connect_bertopic_model(topic_model, file_path="connect_bertopic_model_v_0_1"):
    """
    Saves the trained BERTopic model to a file using BERTopic's built-in save method.
    
    Args:
        topic_model (BERTopic): The trained BERTopic model.
        file_path (str): The file path where the model will be saved. Default is 'connect_bertopic_model'.
    """
    try:
        # Save the model
        topic_model.save(file_path)
        print(f"Connect C-Ops BERTopic model saved successfully at '{file_path}'.")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")


def load_connect_bertopic_model(file_path="connect_bertopic_model"):
    """
    Loads a previously saved BERTopic model using BERTopic's built-in load method.
    
    Args:
        file_path (str): The file path from where the model will be loaded. Default is 'connect_bertopic_model'.
    
    Returns:
        BERTopic: The loaded BERTopic model.
    """
    try:
        # Load the model
        loaded_model = BERTopic.load(file_path)
        print(f"Connect C-Ops BERTopic model loaded successfully from '{file_path}'.")
        return loaded_model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None




# Usage
if __name__ == "__main__":
   
    # Assuming 'df' is the DataFrame with your data and 'consolidated_description' is the column containing the text data

    input_connect_data = pd.read_csv("/content/drive/MyDrive/company_project/dummy_variable_data_public.csv")

    print("shape of the input dataframe" , input_connect_data.shape)

    input_connect_data = input_connect_data[['incident_number', 'app_id','created_on', 'description',
       'short_description', 'assignment_group', 'category','app_name',
       'resolution_code', 'resolution_update',
       'meta_tags']]

     # Combine the 'description' and 'short_description' columns
    input_connect_data['consolidated_description'] = input_connect_data['description'] + ' ' + input_connect_data['short_description']

    # Remove any null values
    input_connect_data.dropna(subset=['consolidated_description'], inplace=True)
  
    df = input_connect_data.copy()
    topic_modeling = TopicModeling(df, "consolidated_description")

    # Generate embeddings
    embeddings = topic_modeling.encode_embeddings()

    # Fit BERTopic model and restrict the number of topics to 9
    topics, probs = topic_modeling.fit_topic_model(embeddings)

    # Get topic information
    print(topic_modeling.get_topic_info())

    # Perform hierarchical clustering
    hierarchical_topics = topic_modeling.hierarchical_clustering(list(df["consolidated_description"]))



    # Example usage (without docs):
    df = add_connect_bertopic_outputs_to_dataframe(df, topic_modeling.topic_model, topics, probs)

    # Print the updated DataFrame with the Connect C-Ops BERTopic outputs
    
    df.head()


    # Assuming 'topic_modeling' is an instance of your TopicModeling class and 'topic_model' is trained.
    save_connect_bertopic_model(topic_modeling.topic_model, file_path="connect_bertopic_model_v_0_1")
