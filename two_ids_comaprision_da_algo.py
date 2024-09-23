import pandas as pd
import numpy as np
import time

class ConnectIncidentComparison:
    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2
    
    def connect_time_function(self, func):
        """Helper function to calculate the time taken by each method"""
        start_time = time.time()
        result = func()
        end_time = time.time()
        print(f"Time taken by {func.__name__}: {end_time - start_time:.6f} seconds")
        return result

    # 1. Left Join (merge) Approach
    def connect_left_join_way(self):
        def method():
            diff_df = self.df1.merge(self.df2, on='incident_number', how='left', indicator=True)
            return diff_df[diff_df['_merge'] == 'left_only'][['incident_number']]
        return self.connect_time_function(method)

    # 2. Isin Approach
    def connect_isin_approach(self):
        def method():
            return self.df1[~self.df1['incident_number'].isin(self.df2['incident_number'])][['incident_number']]
        return self.connect_time_function(method)

    # 3. Hashing (Set) Approach
    def connect_hashing_approach(self):
        def method():
            df2_incidents_set = set(self.df2['incident_number'])
            return self.df1[~self.df1['incident_number'].apply(lambda x: x in df2_incidents_set)][['incident_number']]
        return self.connect_time_function(method)

    # 4. Sorting + Binary Search Approach
    def connect_binary_search_approach(self):
        def method():
            df1_sorted = self.df1.sort_values('incident_number')
            df2_sorted = self.df2.sort_values('incident_number')
            df2_incidents_array = df2_sorted['incident_number'].to_numpy()
            return df1_sorted[~df1_sorted['incident_number'].apply(
                lambda x: np.searchsorted(df2_incidents_array, x) < len(df2_incidents_array) and 
                          df2_incidents_array[np.searchsorted(df2_incidents_array, x)] == x)][['incident_number']]
        return self.connect_time_function(method)

    # 5. Trie (Prefix Tree) Approach
    class ConnectTrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False

    class ConnectTrie:
        def __init__(self):
            self.root = ConnectIncidentComparison.ConnectTrieNode()
        
        def connect_insert(self, word):
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = ConnectIncidentComparison.ConnectTrieNode()
                node = node.children[char]
            node.is_end_of_word = True
        
        def connect_search(self, word):
            node = self.root
            for char in word:
                if char not in node.children:
                    return False
                node = node.children[char]
            return node.is_end_of_word

    def connect_trie_approach(self):
        def method():
            trie = self.ConnectTrie()
            for incident in self.df2['incident_number'].astype(str):
                trie.connect_insert(incident)
            return self.df1[~self.df1['incident_number'].astype(str).apply(lambda x: trie.connect_search(x))][['incident_number']]
        return self.connect_time_function(method)

# Sample usage:
df1 = pd.DataFrame({
    'incident_number': ['INC010725559', 'INC010725560', 'INC010725561', 'INC010725562']
})

df2 = pd.DataFrame({
    'incident_number': ['INC010725560', 'INC010725561']
})

# Instantiate the comparison class
comparison = ConnectIncidentComparison(df1, df2)

# Run all methods and check the results
print("Results from Connect Left Join Way:")
print(comparison.connect_left_join_way())

print("\nResults from Connect Isin Approach:")
print(comparison.connect_isin_approach())

print("\nResults from Connect Hashing Approach:")
print(comparison.connect_hashing_approach())

print("\nResults from Connect Binary Search Approach:")
print(comparison.connect_binary_search_approach())

print("\nResults from Connect Trie Approach:")
print(comparison.connect_trie_approach())
