from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import sys
import json
from typing import List, Dict, Any

def calculate_similarity(text1, text2):
    """
    Calculate similarity between two texts using cosine similarity
    """
    # Convert texts to lowercase
    text1 = text1.lower()
    text2 = text2.lower()
    
    # Create vocabulary
    words = set(text1.split() + text2.split())
    
    # Create vector representations
    vec1 = np.zeros(len(words))
    vec2 = np.zeros(len(words))
    
    # Fill vectors
    for i, word in enumerate(words):
        vec1[i] = text1.split().count(word)
        vec2[i] = text2.split().count(word)
    
    # Calculate cosine similarity
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    
    return similarity
