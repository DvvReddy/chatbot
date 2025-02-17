from src.generator import ResponseGenerator

class QueryService:
    def __init__(self):
        # ... existing initialization ...
        self.generator = ResponseGenerator()
    
    def query(self, question: str, top_k: int = 3):
        try:
            # Get similar QA pairs
            similar_pairs = self.retriever.query_index(self.index, self.data, question, top_k)
            
            # Generate response using context
            response = self.generator.generate_response(question, context=similar_pairs)
            
            return {
                "success": True,
                "generated_answer": response["answer"],
                "similar_pairs": similar_pairs
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 