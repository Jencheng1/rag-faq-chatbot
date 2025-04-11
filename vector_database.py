import os
import json
import numpy as np
import faiss
from openai import OpenAI
import pickle
import time
import random

class VectorDatabase:
    def __init__(self, api_key, model="text-embedding-3-small", use_mock=False):
        """
        Initialize the vector database with OpenAI API key.
        
        Args:
            api_key (str): OpenAI API key
            model (str): OpenAI embedding model to use
            use_mock (bool): Whether to use mock embeddings (for testing without API)
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.index = None
        self.documents = []
        self.embeddings = []
        self.use_mock = use_mock
        self.embedding_dimension = 1536  # Default dimension for OpenAI embeddings
        
    def generate_embedding(self, text):
        """
        Generate embedding for a text using OpenAI API or mock embeddings.
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            list: Embedding vector
        """
        if self.use_mock:
            # Generate a deterministic but random-looking embedding based on the text
            # This is only for testing without API access
            random.seed(hash(text) % 10000)
            return [random.uniform(-1, 1) for _ in range(self.embedding_dimension)]
        
        try:
            # Add exponential backoff for rate limiting
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        input=text,
                        model=self.model
                    )
                    return response.data[0].embedding
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        # Rate limit error, wait and retry
                        wait_time = (2 ** attempt) + random.random()  # Exponential backoff with jitter
                        print(f"Rate limit hit, waiting {wait_time:.2f} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        # Other error or max retries reached
                        print(f"Error generating embedding: {e}")
                        return None
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def add_documents(self, documents):
        """
        Add documents to the vector database.
        
        Args:
            documents (list): List of documents to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate embeddings for each document
            new_embeddings = []
            for i, doc in enumerate(documents):
                print(f"Processing document {i+1}/{len(documents)}")
                embedding = self.generate_embedding(doc)
                if embedding:
                    new_embeddings.append(embedding)
                    self.documents.append(doc)
                # Add a small delay to avoid rate limiting
                if not self.use_mock and i % 5 == 0 and i > 0:
                    time.sleep(1)
            
            # Add embeddings to the existing list
            self.embeddings.extend(new_embeddings)
            
            # Create or update the FAISS index
            self._create_or_update_index()
            
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def _create_or_update_index(self):
        """
        Create or update the FAISS index with the current embeddings.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.embeddings:
                print("No embeddings to index")
                return False
                
            # Convert embeddings to numpy array
            embeddings_array = np.array(self.embeddings).astype('float32')
            
            # Get the dimension of the embeddings
            dimension = embeddings_array.shape[1]
            
            # Create a new index
            self.index = faiss.IndexFlatL2(dimension)
            
            # Add the embeddings to the index
            self.index.add(embeddings_array)
            
            return True
        except Exception as e:
            print(f"Error creating/updating index: {e}")
            return False
    
    def search(self, query, k=5):
        """
        Search the vector database for similar documents.
        
        Args:
            query (str): Query text
            k (int): Number of results to return
            
        Returns:
            list: List of (document, score) tuples
        """
        try:
            if not self.index:
                print("No index available for search")
                return []
                
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                return []
            
            # Convert to numpy array
            query_array = np.array([query_embedding]).astype('float32')
            
            # Search the index
            distances, indices = self.index.search(query_array, k)
            
            # Return the results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(distances[0][i])))
            
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
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
            
            # Save the embeddings
            with open(os.path.join(directory, 'embeddings.pkl'), 'wb') as f:
                pickle.dump(self.embeddings, f)
            
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
            embeddings_path = os.path.join(directory, 'embeddings.pkl')
            index_path = os.path.join(directory, 'index.faiss')
            
            if not (os.path.exists(documents_path) and 
                    os.path.exists(embeddings_path) and 
                    os.path.exists(index_path)):
                print("Missing required files in directory")
                return False
            
            # Load the documents
            with open(documents_path, 'r') as f:
                self.documents = json.load(f)
            
            # Load the embeddings
            with open(embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            
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
    # Get API key from environment variable or file
    api_key = os.environ.get("OPENAI_API_KEY", "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")
    
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
