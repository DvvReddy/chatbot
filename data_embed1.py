import json
import os
import re
import time
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# -------------------------------
# Enhanced Embedding Generator
# -------------------------------

class EmbeddingGenerator:
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 input_dir: str = 'data/processed',
                 output_dir: str = 'data/embeddings',
                 use_dual_encoding: bool = True):
        """
        Initialize the embedding generator.
        
        Args:
          model_name: Name of the embedding model.
          input_dir: Directory containing processed JSON data.
          output_dir: Directory to store embeddings and FAISS index.
          use_dual_encoding: If True, encode question and answer separately then average.
        """
        print(f"Initializing embedding generator with {model_name}")
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.use_dual_encoding = use_dual_encoding
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text: lowercase, remove extra whitespace, and punctuation."""
        if not text:
            return ""
        text = text.lower().strip()
        # Remove unnecessary punctuation (except basic ones like ? and !)
        text = re.sub(r"[^\w\s\?\!']", " ", text)
        text = " ".join(text.split())
        return text

    def combine_qa(self, question: str, answer: str) -> str:
        """Combine question and answer into a single text block after cleaning."""
        clean_q = self.clean_text(question)
        clean_a = self.clean_text(answer)
        return f"Question: {clean_q}\nAnswer: {clean_a}"

    def dual_encode(self, question: str, answer: str) -> np.ndarray:
        """
        Encode question and answer separately and return the average of the two embeddings.
        This strategy can sometimes improve similarity quality.
        """
        clean_q = self.clean_text(question)
        clean_a = self.clean_text(answer)
        q_emb = self.model.encode([clean_q], normalize_embeddings=True)
        a_emb = self.model.encode([clean_a], normalize_embeddings=True)
        avg_emb = (q_emb + a_emb) / 2
        return avg_emb[0]

    def generate_embeddings(self, data: list) -> np.ndarray:
        """
        Generate embeddings for the Q&A pairs.
        Uses dual encoding if enabled; otherwise, encodes the combined text.
        """
        print("Generating embeddings...")
        embeddings = []
        batch_size = 32
        combined_texts = []  # Used if not dual encoding

        if not self.use_dual_encoding:
            combined_texts = [
                self.combine_qa(item['question'], item['answer'])
                for item in data
            ]
            for i in tqdm(range(0, len(combined_texts), batch_size)):
                batch = combined_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, normalize_embeddings=True)
                embeddings.extend(batch_embeddings)
        else:
            # Dual encoding: process one pair at a time (batching less straightforward here)
            for item in tqdm(data, desc="Dual Encoding"):
                emb = self.dual_encode(item['question'], item['answer'])
                embeddings.append(emb)
                
        return np.array(embeddings)

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create and populate a FAISS index."""
        print("Creating FAISS index...")
        # nlist: number of clusters; adjust based on dataset size for better performance.
        nlist = min(max(len(embeddings) // 20, 10), 100)
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        # Train the index and add embeddings (ensure data type is float32)
        index.train(embeddings.astype('float32'))
        index.add(embeddings.astype('float32'))
        return index

    def process_and_store(self, filename: str = 'processed_conversations.json'):
        """Main method to process data, generate embeddings, create FAISS index, and store results."""
        try:
            # Load processed data from JSON
            input_file = os.path.join(self.input_dir, filename)
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Loaded {len(data)} Q&A pairs")
            # Generate embeddings
            embeddings = self.generate_embeddings(data)
            # Create FAISS index
            index = self.create_faiss_index(embeddings)
            
            # Save FAISS index
            index_path = os.path.join(self.output_dir, 'qa_pairs.index')
            faiss.write_index(index, index_path)
            # Save original Q&A data for retrieval mapping
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

    def test_index(self, index: faiss.Index, data: list, embeddings: np.ndarray):
        """Test the FAISS index with sample queries and print the top results."""
        print("\nTesting index...")
        test_questions = [
            "hi, how are you doing?",                # Greeting
            "what is artificial intelligence?",       # Technical
            "how do I reset my password?",             # Support
            "can you explain machine learning?",       # Educational
            "what are your business hours?"            # Service
        ]
        
        for question in test_questions:
            print(f"\nTest Question: {question}")
            # Generate embedding for test question
            q_emb = self.model.encode([self.clean_text(question)], normalize_embeddings=True)
            k = 3  # Number of nearest neighbors
            D, I = index.search(q_emb.astype('float32'), k)
            
            print(f"\nTop {k} most similar Q&A pairs:")
            for i, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
                print(f"\n{i}. Distance: {dist:.4f}")
                print(f"Q: {data[idx]['question']}")
                print(f"A: {data[idx]['answer']}")
                
    def evaluate_index(self, index: faiss.Index, data: list, embeddings: np.ndarray) -> dict:
        """Evaluate index quality by computing average distances and match rates over a sample."""
        sample_size = min(100, len(data))
        samples = np.random.choice(len(data), sample_size, replace=False)
        metrics = {'exact_match_rate': 0, 'average_distance': 0}
        
        for idx in samples:
            query_embedding = embeddings[idx:idx+1]
            D, I = index.search(query_embedding.astype('float32'), 5)
            metrics['exact_match_rate'] += (idx in I[0])
            metrics['average_distance'] += D[0].mean()
            
        metrics = {k: v / sample_size for k, v in metrics.items()}
        return metrics

def main():
    """Main function to run the enhanced embedding generation and indexing pipeline."""
    generator = EmbeddingGenerator(
        model_name='sentence-transformers/all-mpnet-base-v2',  # Change model if desired
        input_dir='data/processed',
        output_dir='data/embeddings',
        use_dual_encoding=True  # Set to False to use simple combined text method
    )
    generator.process_and_store()

if __name__ == "__main__":
    main()
