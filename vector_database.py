import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import json
import numpy as np
import faiss
from openai import OpenAI
import pickle
import time
import random

class VectorDatabase:
    def __init__(self, api_key, dimension=1536):
        """
        Initialize the vector database.
        
        Args:
            api_key (str): OpenAI API key
            dimension (int): Dimension of the embeddings
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.documents = []
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        
    def add_documents(self, documents):
        """
        Add documents to the vector database.
        
        Args:
            documents (list): List of text documents to add
        """
        if not documents:
            return
            
        # Clean and preprocess documents
        cleaned_docs = []
        for doc in documents:
            # Remove extra whitespace and normalize
            doc = ' '.join(doc.split())
            # Remove any remaining special characters
            doc = ''.join(char for char in doc if char.isprintable())
            cleaned_docs.append(doc)
            
        # Create embeddings for the documents
        embeddings = self.client.embeddings.create(
            input=cleaned_docs,
            model="text-embedding-3-small"
        ).data
        
        # Convert embeddings to numpy array
        embedding_array = np.array([emb.embedding for emb in embeddings]).astype('float32')
        
        # Add to FAISS index
        self.index.add(embedding_array)
        
        # Store the documents
        self.documents.extend(cleaned_docs)
        
    def search(self, query, k=5):
        """
        Search for similar documents.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            list: List of (document, score) tuples
        """
        if not self.documents:
            return []
            
        # Clean up the query
        query = query.strip()
        if not query.lower().startswith("question:"):
            query = "Question: " + query
            
        # Create embedding for the query
        query_embedding = self.client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        ).data[0].embedding
        
        # Convert to numpy array
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search the index with more results initially
        k_search = min(k * 3, len(self.documents))  # Search for more results than needed
        scores, indices = self.index.search(query_array, k_search)
        
        # Return the documents and their scores
        results = []
        seen_questions = set()  # To avoid duplicate questions
        
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.documents):  # Skip invalid indices
                continue
                
            doc = self.documents[idx]
            
            # Extract question from the document if it's in Q&A format
            question = None
            if "Question:" in doc and "Answer:" in doc:
                question = doc.split("Question:")[1].split("Answer:")[0].strip()
                
            # Skip if we've seen this question before
            if question and question in seen_questions:
                continue
                
            # Add question to seen set if it exists
            if question:
                seen_questions.add(question)
            
            # Improve score for exact matches and Q&A format
            adjusted_score = score
            if "Question:" in doc and "Answer:" in doc:
                adjusted_score *= 0.8  # Boost Q&A format
            if query.lower() in doc.lower():
                adjusted_score *= 0.8  # Boost exact matches
                
            results.append((doc, float(adjusted_score)))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x[1])
        return results[:k]
    
    def save(self, directory):
        """
        Save the vector database to disk.
        
        Args:
            directory (str): Directory to save the database to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save the documents
            with open(os.path.join(directory, 'documents.json'), 'w') as f:
                json.dump(self.documents, f)
            
            # Save the index
            if self.index:
                faiss.write_index(self.index, os.path.join(directory, 'index.faiss'))
            
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def load(self, directory):
        """
        Load the vector database from disk.
        
        Args:
            directory (str): Directory to load the database from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if directory exists
            if not os.path.exists(directory):
                print(f"Directory {directory} does not exist")
                return False
                
            # Check if files exist
            documents_path = os.path.join(directory, 'documents.json')
            index_path = os.path.join(directory, 'index.faiss')
            
            if not (os.path.exists(documents_path) and 
                    os.path.exists(index_path)):
                print("Missing required files in directory")
                return False
            
            # Load the documents
            with open(documents_path, 'r') as f:
                self.documents = json.load(f)
            
            # Load the index
            self.index = faiss.read_index(index_path)
            
            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            return False
    
    def combine_sources(self, pdf_chunks_file, website_chunks_file):
        """
        Combine PDF and website chunks into a single vector database.
        
        Args:
            pdf_chunks_file (str): Path to the PDF chunks JSON file
            website_chunks_file (str): Path to the website chunks JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load PDF chunks
            with open(pdf_chunks_file, 'r') as f:
                pdf_chunks = json.load(f)
            
            # Load website chunks
            with open(website_chunks_file, 'r') as f:
                website_chunks = json.load(f)
            
            # Combine chunks
            all_chunks = pdf_chunks + website_chunks
            
            # Add to the database
            return self.add_documents(all_chunks)
        except Exception as e:
            print(f"Error combining sources: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running the application.")
    
    # Initialize the vector database with mock embeddings for testing
    vector_db = VectorDatabase(api_key, use_mock=True)
    
    # Process the PDF file
    os.system("python3 pdf_processor.py")
    
    # Process the website
    os.system("python3 web_scraper.py")
    
    # Combine the sources
    vector_db.combine_sources(
        "/home/ubuntu/leechy_chatbot/faq_chunks.json",
        "/home/ubuntu/leechy_chatbot/website_chunks.json"
    )
    
    # Save the database
    vector_db.save("/home/ubuntu/leechy_chatbot/vector_db")
    
    # Test the database
    results = vector_db.search("How do I cancel a booking?")
    print("\nSearch results for 'How do I cancel a booking?':")
    for doc, score in results:
        print(f"Score: {score}")
        print(f"Document: {doc[:200]}...")
        print()
