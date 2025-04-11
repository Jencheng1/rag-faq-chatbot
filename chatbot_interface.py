import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from vector_database import VectorDatabase
from openai import OpenAI

class ChatbotInterface:
    def __init__(self, vector_db_path, api_key, model="gpt-3.5-turbo"):
        """
        Initialize the chatbot interface.
        
        Args:
            vector_db_path (str): Path to the vector database
            api_key (str): OpenAI API key
            model (str): OpenAI model to use for chat completion
        """
        self.vector_db_path = vector_db_path
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)
        
        # Initialize vector database
        self.vector_db = VectorDatabase(api_key)
        self.vector_db.load(vector_db_path)
        
    def process_question(self, question, max_results=5):
        """
        Process a question and retrieve relevant context from the vector database.
        
        Args:
            question (str): User's question
            max_results (int): Maximum number of results to retrieve
            
        Returns:
            list: List of relevant documents
        """
        # Search the vector database
        results = self.vector_db.search(question, k=max_results)
        
        # Extract the documents
        documents = [doc for doc, _ in results]
        
        return documents
    
    def generate_answer(self, question, context):
        """
        Generate an answer to the question using the retrieved context.
        
        Args:
            question (str): User's question
            context (list): List of relevant documents
            
        Returns:
            str: Generated answer
        """
        # Combine the context into a single string
        context_text = "\n\n".join(context)
        
        # Create the prompt
        prompt = f"""You are a helpful assistant for Leechy, a rental marketplace app. 
Answer the following question based on the provided context. 
If you don't know the answer or it's not in the context, say so politely and suggest contacting Leechy support.

Context:
{context_text}

Question: {question}

Answer:"""
        
        # Generate the answer
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for Leechy, a rental marketplace app."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I'm sorry, I encountered an error while generating an answer. Please try again later."
    
    def chat(self, question):
        """
        Process a question and generate an answer.
        
        Args:
            question (str): User's question
            
        Returns:
            str: Generated answer
        """
        # Process the question
        context = self.process_question(question)
        
        # Generate the answer
        answer = self.generate_answer(question, context)
        
        return answer

# Flask application for the chatbot
app = Flask(__name__, static_folder='static')

# Initialize the chatbot
api_key = os.environ.get("OPENAI_API_KEY", "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")
chatbot = None

@app.route('/')
def index():
    """Render the chatbot interface."""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API endpoint for chatting with the bot."""
    global chatbot
    
    # Initialize the chatbot if not already done
    if chatbot is None:
        try:
            chatbot = ChatbotInterface('/home/ubuntu/leechy_chatbot/vector_db', api_key)
        except Exception as e:
            return jsonify({"error": f"Failed to initialize chatbot: {str(e)}"}), 500
    
    # Get the question from the request
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Process the question and generate an answer
    try:
        answer = chatbot.chat(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"Error processing question: {str(e)}"}), 500

if __name__ == '__main__':
    # Create the static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)
