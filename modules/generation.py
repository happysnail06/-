from transformers import pipeline
import faiss
from embedding import EmbeddingHandler

class AnswerGenerator:
    def __init__(self, model_handler):
        self.model_handler = model_handler
        self.qa_pipeline = pipeline(
            "text-generation",
            model=model_handler.model,
            tokenizer=model_handler.tokenizer,
            max_new_tokens=500,
            do_sample=True,
            top_k=3,
            top_p=0.9,
            temperature=0.3,
            pad_token_id=model_handler.tokenizer.eos_token_id
        )

    def generate_answer_and_collect_results(self, question, data, top_k=3):
        query_vector = EmbeddingHandler().get_embedding(question).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        index = faiss.read_index("faiss_index.bin")
        distances, indices = index.search(query_vector, k=top_k)
        
        contexts = [data[int(i)]["documents"] for i in indices[0]]
        
        flattened_contexts = [
            item if isinstance(item, str) else " ".join(item) 
            for sublist in contexts for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        
        _contexts = " ".join(flattened_contexts)
        prompt = f"""
                You are performing question-answering tasks. Your mission is to answer the given question using the provided context. Use the retrieved context to answer the question.

                Q: ```{question}```
                Context: ```{_contexts}```
                A:
                """
        inputs = self.model_handler.tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt")

        inputs = {k: v.to(self.model_handler.device) for k, v in inputs.items()}
        
        generated_text = self.model_handler.model.generate(
            **inputs,
            max_new_tokens=500
        )
        
        answer = self.model_handler.tokenizer.decode(generated_text[0], skip_special_tokens=True).split("A:")[-1].strip()
        
        return {
            "question": question,
            "answer": answer,
            "contexts": flattened_contexts  # Pass as a list for RAGAS
        }
