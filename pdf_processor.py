import os
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import re

class PDFProcessor:
    def __init__(self, pdf_path):
        """
        Initialize the PDF processor with the path to the PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.raw_text = ""
        self.processed_text = ""
        self.chunks = []
        
    def extract_text(self):
        """
        Extract text from the PDF file.
        
        Returns:
            str: Extracted text from the PDF
        """
        try:
            pdf_reader = PyPDF2.PdfReader(self.pdf_path)
            num_pages = len(pdf_reader.pages)
            
            text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                # Extract text and normalize spaces
                page_text = page.extract_text()
                # Replace multiple spaces and newlines with single space
                page_text = ' '.join(page_text.split())
                text += page_text + " "
                
            self.raw_text = text
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None
    
    def clean_text(self):
        """
        Clean and process the extracted text.
        
        Returns:
            str: Cleaned and processed text
        """
        if not self.raw_text:
            self.extract_text()
            
        # Clean up the text
        text = self.raw_text
        
        # Fix common OCR issues
        replacements = {
            'spor ts': 'sports',
            'machiner y': 'machinery',
            'furnitur e': 'furniture',
            'transpor tation': 'transportation',
            'electr onics': 'electronics',
            'addr ess': 'address',
            'pro\ufb01le': 'profile',
            'arriv e': 'arrive',
            'insur ed': 'insured',
            '\u25cf': '-',  # bullet points
            '\u201c': '"',  # smart quotes
            '\u201d': '"'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Format Q&A pairs consistently
        text = text.replace('Q:', 'Question:')
        text = text.replace('A:', 'Answer:')
        
        # Clean up any remaining whitespace issues
        text = ' '.join(text.split())
        
        self.processed_text = text
        return text
    
    def split_into_chunks(self, chunk_size=500, chunk_overlap=50):
        """
        Split the processed text into chunks for better retrieval.
        
        Args:
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks
            
        Returns:
            list: List of text chunks
        """
        if not self.processed_text:
            self.clean_text()
        
        chunks = []
        
        # First try to extract explicit Q&A pairs
        qa_pattern = r'Question:\s*([^?]+\??)\s*Answer:\s*([^Q]+)'
        matches = re.findall(qa_pattern, self.processed_text, re.IGNORECASE)
        
        for question, answer in matches:
            chunk = f"Question: {question.strip()} Answer: {answer.strip()}"
            chunks.append(chunk)
        
        # If no Q&A pairs found, fall back to regular chunking
        if not chunks:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=[". ", "? ", "! ", ", ", " ", ""]
            )
            chunks = text_splitter.split_text(self.processed_text)
        
        self.chunks = chunks
        return chunks
    
    def extract_qa_pairs(self):
        """
        Attempt to extract question-answer pairs from the FAQ document.
        This is a simple heuristic approach and may need refinement based on the actual PDF structure.
        
        Returns:
            list: List of dictionaries containing question-answer pairs
        """
        if not self.raw_text:
            self.extract_text()
            
        lines = self.raw_text.split('\n')
        qa_pairs = []
        current_section = ""
        current_question = ""
        current_answer = ""
        in_question = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a section header (ends with a colon or is all caps)
            if line.endswith(':') or (line.isupper() and len(line) > 3):
                current_section = line
                continue
                
            # Check if this is a question (starts with a question mark or bullet point)
            if line.startswith('●') or line.startswith('•') or line.startswith('-') or line.startswith('Q:'):
                # If we were processing a question before, save it
                if current_question and current_answer:
                    qa_pairs.append({
                        'section': current_section,
                        'question': current_question,
                        'answer': current_answer
                    })
                
                # Start a new question
                current_question = line.lstrip('●•-Q: ')
                current_answer = ""
                in_question = True
            elif in_question:
                # This is part of an answer
                current_answer += line + " "
        
        # Don't forget the last QA pair
        if current_question and current_answer:
            qa_pairs.append({
                'section': current_section,
                'question': current_question,
                'answer': current_answer
            })
            
        return qa_pairs
    
    def save_chunks(self, output_file):
        """
        Save the text chunks to a JSON file.
        
        Args:
            output_file (str): Path to the output JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.chunks:
            self.split_into_chunks()
            
        try:
            with open(output_file, 'w') as f:
                json.dump(self.chunks, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving chunks to file: {e}")
            return False
    
    def save_qa_pairs(self, output_file):
        """
        Save the extracted QA pairs to a JSON file.
        
        Args:
            output_file (str): Path to the output JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        qa_pairs = self.extract_qa_pairs()
        
        try:
            with open(output_file, 'w') as f:
                json.dump(qa_pairs, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving QA pairs to file: {e}")
            return False
    
    def process_pdf(self, chunks_output=None, qa_output=None):
        """
        Process the PDF file end-to-end: extract text, clean it, split into chunks, and save results.
        
        Args:
            chunks_output (str): Path to save the chunks JSON file
            qa_output (str): Path to save the QA pairs JSON file
            
        Returns:
            dict: Dictionary containing the processing results
        """
        self.extract_text()
        self.clean_text()
        chunks = self.split_into_chunks()
        qa_pairs = self.extract_qa_pairs()
        
        result = {
            "num_chunks": len(chunks),
            "num_qa_pairs": len(qa_pairs),
            "chunks": chunks[:3],  # Preview of first 3 chunks
            "qa_pairs": qa_pairs[:3]  # Preview of first 3 QA pairs
        }
        
        if chunks_output:
            self.save_chunks(chunks_output)
            
        if qa_output:
            self.save_qa_pairs(qa_output)
            
        return result

# Example usage
if __name__ == "__main__":
    pdf_path = "/home/ubuntu/leechy_chatbot/FAQs for Leechy App.pdf"
    processor = PDFProcessor(pdf_path)
    
    # Process the PDF and save results
    result = processor.process_pdf(
        chunks_output="/home/ubuntu/leechy_chatbot/faq_chunks.json",
        qa_output="/home/ubuntu/leechy_chatbot/faq_qa_pairs.json"
    )
    
    print(f"Processed PDF: {pdf_path}")
    print(f"Number of chunks: {result['num_chunks']}")
    print(f"Number of QA pairs: {result['num_qa_pairs']}")
    print("\nChunk preview:")
    for i, chunk in enumerate(result['chunks']):
        print(f"Chunk {i+1}: {chunk[:100]}...")
    
    print("\nQA pairs preview:")
    for i, qa in enumerate(result['qa_pairs']):
        print(f"Q{i+1}: {qa['question']}")
        print(f"A{i+1}: {qa['answer'][:100]}...")
