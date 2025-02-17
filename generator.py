import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict
import os

class ResponseGenerator:
    def __init__(self, 
                 model_path: str = 'fine_tuned_gpt2',
                 max_length: int = 100,
                 temperature: float = 0.7,
                 top_k: int = 50,
                 top_p: float = 0.9):
        """Initialize the response generator"""
        print(f"Initializing generator from {model_path}")
        
        try:
            # Load model and tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Generation parameters
            self.max_length = max_length
            self.temperature = temperature
            self.top_k = top_k
            self.top_p = top_p
            
            # Add padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
                self.model.resize_token_embeddings(len(self.tokenizer))
                
            print("Generator initialized successfully")
            
        except Exception as e:
            print(f"Error initializing generator: {e}")
            raise
            
    def format_prompt(self, question: str, context: List[Dict] = None) -> str:
        """Format the input prompt with retrieved context"""
        if context:
            # Format with retrieved similar QA pairs
            context_str = "\n".join([
                f"Q: {qa['question']}\nA: {qa['answer']}"
                for qa in context[:2]  # Use top 2 most relevant pairs
            ])
            prompt = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
        else:
            # Simple question-answer format
            prompt = f"Question: {question}\nAnswer:"
            
        return prompt
    
    @torch.no_grad()
    def generate_response(self, 
                         question: str, 
                         context: List[Dict] = None,
                         max_length: int = None) -> Dict:
        """Generate a response for the given question"""
        try:
            # Format the prompt
            prompt = self.format_prompt(question, context)
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            
            # Generate
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length or self.max_length,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True
            )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the answer part
            answer = generated_text.split("Answer:")[-1].strip()
            
            return {
                "success": True,
                "answer": answer,
                "full_response": generated_text
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def evaluate_response(self, 
                         question: str, 
                         generated_answer: str, 
                         actual_answer: str = None) -> Dict:
        """Evaluate the quality of generated response"""
        evaluation = {
            "length": len(generated_answer.split()),
            "coherent": len(generated_answer) > 10,  # Basic coherence check
        }
        
        if actual_answer:
            # Add comparison metrics if we have ground truth
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, generated_answer.lower(), 
                                       actual_answer.lower()).ratio()
            evaluation["similarity"] = similarity
            
        return evaluation

def main():
    """Test the generator"""
    generator = ResponseGenerator()
    
    test_questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?",
        "Explain deep learning in simple terms."
    ]
    
    print("\nTesting generator...")
    for question in test_questions:
        print(f"\nQ: {question}")
        response = generator.generate_response(question)
        if response["success"]:
            print(f"A: {response['answer']}")
        else:
            print(f"Error: {response['error']}")

if __name__ == "__main__":
    main() 