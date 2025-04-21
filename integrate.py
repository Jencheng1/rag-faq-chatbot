import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import sys
import json
from pdf_processor import PDFProcessor
from web_scraper import WebScraper
from vector_database import VectorDatabase

def main():
    """
    Main integration script to run all components of the Leechy Q&A RAG chatbot.
    """
    print("Starting Leechy Q&A RAG Chatbot Integration...")
    
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "FAQs for Leechy App.pdf")
    vector_db_dir = os.path.join(base_dir, "vector_db")
    
    # Get OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running the application.")
    
    # Initialize vector database
    vector_db = VectorDatabase(api_key)
    
    # Step 1: Process PDF
    print("\n1. Processing PDF...")
    pdf_processor = PDFProcessor(pdf_path)
    pdf_result = pdf_processor.process_pdf(
        chunks_output=os.path.join(base_dir, "faq_chunks.json"),
        qa_output=os.path.join(base_dir, "faq_qa_pairs.json")
    )
    print(f"  - Extracted {pdf_result['num_chunks']} chunks from PDF")
    print(f"  - Identified {pdf_result['num_qa_pairs']} QA pairs from PDF")
    
    # Step 2: Scrape website
    print("\n2. Scraping website...")
    scraper = WebScraper("https://www.leechy.app")
    pages_to_scrape = [
        "https://www.leechy.app/",
        "https://www.leechy.app/terms-of-service",
        "https://www.leechy.app/privacy-policy"
    ]
    scraper.scrape_specific_pages(pages_to_scrape)
    scraper.save_content(os.path.join(base_dir, "website_content.json"))
    
    # Process website content for RAG
    chunks = scraper.process_content_for_rag()
    scraper.save_chunks(chunks, os.path.join(base_dir, "website_chunks.json"))
    print(f"  - Scraped {len(scraper.content)} pages")
    print(f"  - Generated {len(chunks)} chunks for RAG")
    
    # Step 3: Create vector database
    print("\n3. Creating vector database...")
    # Use mock embeddings to avoid API quota issues
    vector_db.combine_sources(
        os.path.join(base_dir, "faq_chunks.json"),
        os.path.join(base_dir, "website_chunks.json")
    )
    
    # Save the database
    os.makedirs(vector_db_dir, exist_ok=True)
    vector_db.save(vector_db_dir)
    print(f"  - Vector database saved to {vector_db_dir}")
    
    # Step 4: Test the database with sample questions
    print("\n4. Testing vector database with sample questions...")
    test_questions = [
        "How do I cancel a booking?",
        "What happens if a renter cancels last-minute?",
        "Will I be charged a fee for canceling a booking?",
        "What if the renter is late returning my item?",
        "How does Leechy handle payments?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        results = vector_db.search(question, k=2)
        print(f"Top 2 results:")
        for i, (doc, score) in enumerate(results):
            print(f"Result {i+1} (score: {score:.4f}):")
            print(f"{doc[:200]}...")
    
    print("\nIntegration completed successfully!")
    print("To run the chatbot interface, execute: python chatbot_interface.py")

if __name__ == "__main__":
    main()
