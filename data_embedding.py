import json
import os
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

class EmbeddingGenerator:
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 input_dir: str = 'data/processed',
                 output_dir: str = 'data/embeddings'):
        """Initialize the embedding generator"""
        print(f"Initializing embedding generator with {model_name}")
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
    def combine_qa(self, question: str, answer: str) -> str:
        """Combine question and answer into a single text block"""
        return f"Question: {question} Answer: {answer}"
    
    def generate_embeddings(self, data: List[Dict]) -> np.ndarray:
        """Generate embeddings for the Q&A pairs"""
        print("Generating embeddings...")
        
        # Combine Q&A pairs into single texts
        combined_texts = [
            self.combine_qa(item['question'], item['answer'])
            for item in data
        ]
        
        # Generate embeddings with progress bar
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(combined_texts), batch_size)):
            batch = combined_texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, normalize_embeddings=True)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create and populate FAISS index"""
        print("Creating FAISS index...")
        
        nlist = min(len(embeddings) // 10, 100)  # Number of clusters
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        
        # Need to train IVF index
        index.train(embeddings.astype('float32'))
        index.add(embeddings.astype('float32'))
        return index
    
    def process_and_store(self, filename: str = 'processed_conversations.json'):
        """Main method to process data and create index"""
        try:
            # Load processed data
            input_file = os.path.join(self.input_dir, filename)
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Loaded {len(data)} Q&A pairs")
            
            # Generate embeddings
            embeddings = self.generate_embeddings(data)
            
            # Create FAISS index
            index = self.create_faiss_index(embeddings)
            
            # Save the index
            index_path = os.path.join(self.output_dir, 'qa_pairs.index')
            faiss.write_index(index, index_path)
            
            # Save the original data alongside embeddings
            data_path = os.path.join(self.output_dir, 'qa_pairs.json')
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved index to {index_path}")
            print(f"Saved data to {data_path}")
            
            # Test the index
            self.test_index(index, data, embeddings)
            
        except Exception as e:
            print(f"Error in processing and storing: {e}")
            raise
    
    def test_index(self, index: faiss.Index, data: List[Dict], embeddings: np.ndarray):
        """Test the created index with a few queries"""
        print("\nTesting index...")
        
        # Test with a few sample questions
        test_questions = [
            "hi, how are you doing?",  # Greeting
            "what is artificial intelligence?",  # Technical
            "how do I reset my password?",  # Support
            "can you explain machine learning?",  # Educational
            "what are your business hours?"  # Service
        ]
        
        for question in test_questions:
            print(f"\nTest Question: {question}")
            
            # Generate embedding for test question
            question_embedding = self.model.encode([question], normalize_embeddings=True)
            
            # Search the index
            k = 3  # Number of nearest neighbors to retrieve
            D, I = index.search(question_embedding.astype('float32'), k)
            
            print(f"\nTop {k} most similar Q&A pairs:")
            for i, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
                print(f"\n{i}. Distance: {dist:.4f}")
                print(f"Q: {data[idx]['question']}")
                print(f"A: {data[idx]['answer']}")

    def evaluate_index(self, index: faiss.Index, data: List[Dict], embeddings: np.ndarray):
        """Evaluate index quality"""
        sample_size = min(100, len(data))
        samples = np.random.choice(len(data), sample_size, replace=False)
        
        metrics = {
            'exact_match_rate': 0,
            'semantic_match_rate': 0,
            'average_distance': 0
        }
        
        for idx in samples:
            query_embedding = embeddings[idx:idx+1]
            D, I = index.search(query_embedding, 5)
            
            # Check if exact match is found
            metrics['exact_match_rate'] += (idx in I[0])
            
            # Average distance for top matches
            metrics['average_distance'] += D[0].mean()
        
        # Normalize metrics
        metrics = {k: v/sample_size for k, v in metrics.items()}
        return metrics

def main():
    """Main function to run embedding generation and indexing"""
    generator = EmbeddingGenerator()
    generator.process_and_store()

if __name__ == "__main__":
    main() 