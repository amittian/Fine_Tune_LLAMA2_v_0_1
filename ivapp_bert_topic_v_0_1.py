import pandas as pd
import scipy.constants as spc
from scipy.cluster import hierarchy as sch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic.representation import KeyBERTInspired
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class TopicModeling:
    def __init__(self, df, column_name, sentence_transformer_model='sentence-transformers/bert-large-nli-stsb-mean-tokens'):

        # "bert-large-multilingual-cased",  "dunzhang/stella_en_1.5B_v"

        """
        Initialize the TopicModeling class with the necessary models and data.
        :param df: DataFrame containing the text data.
        :param column_name: Name of the column containing text descriptions.
        :param sentence_transformer_model: Model to be used for sentence embeddings.
        """
        self.df = df
        self.column_name = column_name
        self.sentence_model = SentenceTransformer(sentence_transformer_model)
        self.representation_model = KeyBERTInspired()
        self.umap_model = UMAP(n_neighbors=10, n_components=9, min_dist=0.0, metric="cosine", random_state=42)
        self.topic_model = BERTopic(
            language="english",
            umap_model=self.umap_model,
            embedding_model=self.sentence_model,
            top_n_words=20,
            n_gram_range=(1, 3),
            min_topic_size=30,
            nr_topics=None,
            calculate_probabilities=True,

            representation_model=self.representation_model
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
        return self.topic_model.fit_transform(self.df[self.column_name], embeddings)


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


# Usage
if __name__ == "__main__":
    #df = pd.read_csv("path_to_your_data.csv")  # Make sure to have the right path and data


    topic_modeling = TopicModeling(df, "consolidated_description")
    embeddings = topic_modeling.encode_embeddings()
    topics, probs = topic_modeling.fit_topic_model(embeddings)

    print(topic_modeling.get_topic_info())
    hierarchical_topics = topic_modeling.hierarchical_clustering(list(df["consolidated_description"]))

    # Calculate similarity matrix if probabilities are available

    # if probs is not None:
    #     similarity_matrix = topic_modeling.get_similarity_matrix(probs)

    #     # Visualize the similarity matrix
    #     plt.figure(figsize=(10, 8))
    #     ax = sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    #     ax.set_title('Topic Similarity Matrix')
    #     plt.show()
    # else:
    #     print("No probabilities available to calculate similarity matrix.")

    # embeddings = sentence_model.encode(docs, show_progress_bar=False)

    # # Train BERTopic
    # topic_model = BERTopic().fit(docs, embeddings)

    # Run the visualization with the original embeddings

import pandas as pd
import plotly.io as ivapp_pio

# Generate HTML strings for DataFrames
ivapp_topic_info_html = topic_modeling.get_topic_info().to_html()
ivapp_hierarchical_topics_html = hierarchical_topics.to_html()

# Generate HTML strings for Plotly visualizations
ivapp_fig_barchart = topic_modeling.topic_model.visualize_barchart()
ivapp_barchart_html = ivapp_pio.to_html(ivapp_fig_barchart, full_html=False)

ivapp_fig_heatmap = topic_modeling.topic_model.visualize_heatmap()
ivapp_heatmap_html = ivapp_pio.to_html(ivapp_fig_heatmap, full_html=False)

ivapp_fig_topics = topic_modeling.topic_model.visualize_topics()
ivapp_topics_html = ivapp_pio.to_html(ivapp_fig_topics, full_html=False)

# Combined HTML content (without the similarity matrix)
ivapp_html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive ivapp C-Ops Topic Modeling Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
        }}
        .plotly-graph-div {{
            width: 90%;
            margin: auto;
        }}
    </style>
</head>
<body>
    <h1>ivapp C-Ops Bar Chart Visualization</h1>
    {ivapp_barchart_html}
    <h1>ivapp C-Ops Plotly Heatmap Visualization</h1>
    {ivapp_heatmap_html}
    <h1>ivapp C-Ops Hierarchical Topics Visualization</h1>
    {ivapp_hierarchical_topics_html}
    <h1>ivapp C-Ops Topic Visualization</h1>
    {ivapp_topics_html}
    <h1>ivapp C-Ops Topic Information</h1>
    {ivapp_topic_info_html}
</body>
</html>
"""

# Write the combined HTML content to a file
with open('combined_ivapp_topic_visualization_set_01.html', 'w') as ivapp_file:
    ivapp_file.write(ivapp_html_content)

print("All visualizations have been saved in 'combined_ivapp_topic_visualization_set_01.html'.")

print(" ........................ stage 3 starts here ..................................................")

import pandas as pd
import plotly.io as pio

# Step 1: Calculate the topic distributions on a token-level
docs = df['consolidated_description'].tolist()
topic_distr, topic_token_distr = topic_modeling.topic_model.approximate_distribution(docs, calculate_tokens=True)

# Step 2: Visualize the token-level distributions and get a Styler object (HTML table)
ivapp_df_token_distro = topic_modeling.topic_model.visualize_approximate_distribution(docs[1], topic_token_distr[1])

# Convert the Styler object to HTML
ivapp_token_distro_html = ivapp_df_token_distro.to_html()

# Step 3: Visualize the term rank (returns a Plotly figure)
ivapp_fig_term_rank = topic_modeling.topic_model.visualize_term_rank()

# Convert the Plotly figure to an HTML string
ivapp_term_rank_html = pio.to_html(ivapp_fig_term_rank, full_html=False)

# Step 4: Combine the visualizations into a single HTML file
ivapp_html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>ivapp C-Ops Comprehensive Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .plotly-graph-div {{
            width: 90%;
            margin: auto;
        }}
    </style>
</head>
<body>
    <h1>ivapp C-Ops Token-Level Distribution</h1>
    {ivapp_token_distro_html}  <!-- Display the token-level distribution as a table -->
    <h1>ivapp C-Ops Term Rank Visualization</h1>
    {ivapp_term_rank_html}  <!-- Display the Plotly term rank figure -->
</body>
</html>
"""

# Step 5: Write the final combined HTML content to a file
with open('ivapp_token_distribution_and_term_rank_set_03.html', 'w', encoding='utf-8') as ivapp_file:
    ivapp_file.write(ivapp_html_content)

print("Both visualizations have been saved in 'ivapp_token_distribution_and_term_rank_set_03.html'.")


