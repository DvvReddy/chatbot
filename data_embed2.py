import json
import os
import re
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import nltk

# Download all required NLTK data
def download_nltk_data():
    """Download required NLTK data"""
    try:
        print("Downloading NLTK data...")
        # Download all required data
        for package in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
            try:
                nltk.download(package, quiet=True)
                print(f"Downloaded {package}")
            except Exception as e:
                print(f"Error downloading {package}: {e}")
    except Exception as e:
        print(f"Error in download process: {e}")

class EnhancedEmbeddingGenerator:
    def __init__(self,
                 model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 input_dir: str = 'data/processed',
                 output_dir: str = 'data/embeddings'):
        """Initialize with QA-optimized model and enhanced preprocessing"""
        # Download NLTK data first
        download_nltk_data()
        
        print(f"Initializing enhanced embedding generator with {model_name}")
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize stopwords with fallback
        try:
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
        except:
            print("Warning: Using basic stopwords")
            self.stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for'}

        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing pipeline with basic tokenization fallback"""
        if not isinstance(text, str):
            return ""
            
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s?!]', ' ', text)  # Keep question marks and exclamations
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Simple word splitting instead of NLTK tokenization
        words = text.split()
        
        # Remove stopwords and short words
        words = [word for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)

    def format_qa_pair(self, question: str, answer: str) -> str:
        """Format Q&A pair with domain context"""
        clean_q = self.preprocess_text(question)
        clean_a = self.preprocess_text(answer)
        
        # Add domain hints
        domain_keywords = {
            'technical': ['what is', 'how does', 'explain', 'artificial', 'machine', 'learning'],
            'support': ['reset', 'password', 'login', 'account', 'help'],
            'general': ['where', 'when', 'how to', 'what are']
        }
        
        domain = 'general'
        for d, keywords in domain_keywords.items():
            if any(k in clean_q.lower() for k in keywords):
                domain = d
                break
        
        if answer:
            return f"Domain: {domain} Question: {clean_q} Answer: {clean_a}"
        else:
            return f"Domain: {domain} Question: {clean_q}"

    def generate_embeddings(self, data: List[Dict]) -> np.ndarray:
        """Generate normalized embeddings with batch processing"""
        print("Generating enhanced embeddings...")
        
        formatted_texts = [
            self.format_qa_pair(item['question'], item['answer'])
            for item in data
        ]
        
        embeddings = []
        batch_size = 64  # Adjusted for better memory usage
        
        for i in tqdm(range(0, len(formatted_texts), batch_size)):
            batch = formatted_texts[i:i + batch_size]
            embeddings.append(self.model.encode(
                batch,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).cpu().numpy())
            
        return np.vstack(embeddings)

    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create optimized FAISS index"""
        print("Creating FAISS index...")
        
        dimension = embeddings.shape[1]
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Use IVF index for better clustering
        nlist = min(len(embeddings) // 10, 100)  # number of clusters
        quantizer = faiss.IndexFlatIP(dimension)  # Use Inner Product similarity
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train and add vectors
        index.train(embeddings)
        index.add(embeddings)
        
        # Set search parameters
        index.nprobe = 10  # Number of clusters to search
        
        return index

    def process_and_store(self, filename: str = 'processed_conversations.json'):
        """Enhanced processing pipeline with validation"""
        try:
            input_file = os.path.join(self.input_dir, filename)
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Processing {len(data)} Q&A pairs")
            
            # Generate and validate embeddings
            embeddings = self.generate_embeddings(data)
            if len(embeddings) != len(data):
                raise ValueError("Embedding count mismatch with input data")
            
            # Create optimized index
            index = self.create_index(embeddings)
            
            # Save artifacts
            index_path = os.path.join(self.output_dir, 'enhanced_qa_index.faiss')
            data_path = os.path.join(self.output_dir, 'enhanced_qa_data.json')
            model_state_path = os.path.join(self.output_dir, 'model_state.json')
            
            # Save index and data
            faiss.write_index(index, index_path)
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Save model state
            model_state = {
                'model_name': 'sentence-transformers/all-mpnet-base-v2',  # Actual model name
                'embedding_dim': self.embedding_dim,
                'index_path': index_path,
                'data_path': data_path
            }
            with open(model_state_path, 'w') as f:
                json.dump(model_state, f, indent=2)
            
            print(f"Index saved to {index_path}")
            print(f"Data saved to {data_path}")
            print(f"Model state saved to {model_state_path}")
            
            # Enhanced evaluation
            self.evaluate_retrieval_quality(index, data, embeddings)
            
        except Exception as e:
            print(f"Processing error: {e}")
            raise

    def evaluate_retrieval_quality(self, index: faiss.Index, data: List[Dict], embeddings: np.ndarray):
        """Simple evaluation with random test cases and similarity scores"""
        print("\nEvaluating retrieval quality...")
        
        # Generate random test cases from data
        num_test_cases = 10  # Number of random test cases
        test_indices = np.random.choice(len(data), num_test_cases, replace=False)
        test_cases = [data[i]['question'] for i in test_indices]
        
        for query in test_cases:
            formatted_query = self.format_qa_pair(query, "")
            query_embedding = self.model.encode(
                formatted_query,
                convert_to_tensor=True,
                normalize_embeddings=True
            ).cpu().numpy().reshape(1, -1).astype('float32')
            
            similarities, indices = index.search(query_embedding, 5)
            
            print(f"\nQuery: '{query}'")
            print("Top matches:")
            
            for similarity, idx in zip(similarities[0], indices[0]):
                print(f"  Similarity: {similarity:.4f}")
                print(f"  Q: {data[idx]['question']}")
                print(f"  A: {data[idx]['answer']}")
                print("---")
            
            print()  # Empty line for readability

    def query_index(self, index: faiss.Index, data: List[Dict], query: str, top_k: int = 3):
        """Simplified query processing with similarity scores only"""
        formatted_query = self.format_qa_pair(query, "")
        
        query_embedding = self.model.encode(
            formatted_query,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).cpu().numpy().reshape(1, -1).astype('float32')
        
        similarities, indices = index.search(query_embedding, top_k)
        
        return [
            {
                "question": data[idx]['question'],
                "answer": data[idx]['answer'],
                "similarity": float(similarities[0][i])
            }
            for i, idx in enumerate(indices[0])
        ]

    def save_model(self, path: str = 'data/embeddings/model_state.json'):
        """Save model configuration and metadata"""
        model_state = {
            'model_name': self.model.__class__.__name__,
            'embedding_dim': self.embedding_dim,
            'index_path': os.path.join(self.output_dir, 'enhanced_qa_index.faiss'),
            'data_path': os.path.join(self.output_dir, 'enhanced_qa_data.json')
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(model_state, f)

def main():
    """Run the enhanced embedding pipeline"""
    generator = EnhancedEmbeddingGenerator()
    generator.process_and_store()

if __name__ == "__main__":
    main()