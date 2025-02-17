import pandas as pd
import os
from typing import Dict, List
import json

class DataProcessor:
    def __init__(self, input_dir: str = 'data/RAW', output_dir: str = 'data/processed'):
        """Initialize data processor with input and output directories"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
       # os.makedirs(output_dir, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = str(text).strip()
        
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Remove special characters but keep punctuation
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        return text
    
    def process_conversation_data(self, filename: str = 'conversation.csv') -> List[Dict]:
        """Process conversation CSV file"""
        print(f"Processing {filename}...")
        
        try:
            # Read CSV file
            file_path = os.path.join(self.input_dir, filename)
            df = pd.read_csv(file_path)
            
            # Remove index column if it exists
            if 'index' in df.columns:
                df = df.drop('index', axis=1)
            
            # Clean column names
            df.columns = [col.strip().lower() for col in df.columns]
            
            # Ensure required columns exist
            required_columns = {'question', 'answer'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Clean text in both columns
            df['question'] = df['question'].apply(self.clean_text)
            df['answer'] = df['answer'].apply(self.clean_text)
            
            # Remove rows with empty questions or answers
            df = df.dropna(subset=['question', 'answer'])
            df = df[df['question'].str.len() > 0]
            df = df[df['answer'].str.len() > 0]
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['question'])
            
            # Convert to list of dictionaries
            processed_data = []
            for _, row in df.iterrows():
                qa_pair = {
                    'question': row['question'],
                    'answer': row['answer']
                }
                processed_data.append(qa_pair)
            
            # Save processed data
            output_file = os.path.join(self.output_dir, 'processed_conversations.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            print(f"Processed {len(processed_data)} Q&A pairs")
            print(f"Saved to {output_file}")
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing file: {e}")
            return []
    
    def get_data_stats(self, data: List[Dict]) -> Dict:
        """Get statistics about the processed data"""
        if not data:
            return {}
        
        stats = {
            'total_pairs': len(data),
            'avg_question_length': sum(len(d['question']) for d in data) / len(data),
            'avg_answer_length': sum(len(d['answer']) for d in data) / len(data),
            'sample_pairs': data[:3]  # First 3 pairs as samples
        }
        
        return stats

def main():
    """Main function to run data processing"""
    processor = DataProcessor()
    
    # Process conversation data
    processed_data = processor.process_conversation_data()
    
    # Get and display statistics
    if processed_data:
        stats = processor.get_data_stats(processed_data)
        print("\nData Statistics:")
        print(f"Total Q&A pairs: {stats['total_pairs']}")
        print(f"Average question length: {stats['avg_question_length']:.2f} characters")
        print(f"Average answer length: {stats['avg_answer_length']:.2f} characters")
        print("\nSample Q&A pairs:")
        for i, pair in enumerate(stats['sample_pairs'], 1):
            print(f"\n{i}. Q: {pair['question']}")
            print(f"   A: {pair['answer']}")

if __name__ == "__main__":
    main() 