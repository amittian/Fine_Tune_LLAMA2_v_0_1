from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd



future_connect_data = pd.read_csv("/content/drive/MyDrive/company_project/dummy_variable_data_public.csv")



print("shape of the input dataframe" , future_connect_data.shape)

future_connect_data = future_connect_data[['incident_number', 'app_id','created_on', 'description',
    'short_description', 'assignment_group', 'category','app_name',
    'resolution_code', 'resolution_update',
    'meta_tags']]

  # Combine the 'description' and 'short_description' columns
future_connect_data['consolidated_description'] = future_connect_data['description'] + ' ' + future_connect_data['short_description']

# Remove any null values
future_connect_data.dropna(subset=['consolidated_description'], inplace=True)


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




def connect_inference_pipeline(loaded_model, new_df, column_name, sentence_transformer_model='sentence-transformers/bert-large-nli-stsb-mean-tokens'):
    """
    Inference pipeline function to make predictions on new data using the loaded BERTopic model.
    
    Args:
        loaded_model (BERTopic): The loaded BERTopic model.
        new_df (pd.DataFrame): The new dataframe containing text data for prediction.
        column_name (str): The name of the column in the dataframe that contains the text data (e.g., 'consolidated_description').
        sentence_transformer_model (str): The SentenceTransformer model to encode new data. Default is 'bert-large-nli-stsb-mean-tokens'.
    
    Returns:
        pd.DataFrame: The new dataframe with predicted topics, topic names, and probabilities added.
    """
    # Step 1: Load the sentence transformer model for encoding
    sentence_model = SentenceTransformer(sentence_transformer_model)
    
    # Step 2: Extract the text data from the dataframe
    new_docs = new_df[column_name].tolist()
    
    # Step 3: Encode the new data into embeddings
    print("Encoding the new data into embeddings...")
    new_embeddings = sentence_model.encode(new_docs, show_progress_bar=True)
    
    # Step 4: Make predictions using the loaded BERTopic model
    print("Making predictions on the new data...")
    topics, probs = loaded_model.transform(new_docs, new_embeddings)
    
    # Step 5: Get the topic representations (words that represent each topic)
    topic_info = loaded_model.get_topic_info()
    
    # Step 6: Add predicted topics, topic names, representations, and probabilities to the new dataframe
    new_df['connect_Predicted_Topic_ID'] = topics
    new_df['connect_Predicted_Topic_Name'] = new_df['connect_Predicted_Topic_ID'].apply(lambda x: topic_info.loc[topic_info['Topic'] == x, 'Name'].values[0] if x != -1 else 'Outlier')
    new_df['connect_Predicted_Topic_Representation'] = new_df['connect_Predicted_Topic_ID'].apply(lambda x: loaded_model.get_topic(x) if x != -1 else [])
    # Check if probs is a 2D array and convert to 1D if necessary by selecting the probabilities of the assigned topic for each document
    if probs.ndim == 2:
        new_df['connect_Predicted_Topic_Probability'] = [probs[i,j] for i, j in enumerate(topics)]
    else:
        new_df['connect_Predicted_Topic_Probability'] = probs
    
    return new_df


# Load the previously saved model
loaded_topic_model = load_connect_bertopic_model(file_path="connect_bertopic_model_v_0_1")

# Now you can use the loaded model as needed
if loaded_topic_model:
    print("Model loaded successfully and ready to use.")


predicted_df = connect_inference_pipeline(loaded_topic_model, future_connect_data, column_name="consolidated_description")




predicted_df["connect_Predicted_Topic_ID"].value_counts() 
