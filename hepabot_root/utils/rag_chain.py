from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from transformers import BitsAndBytesConfig

def create_rag_chain(vector_db: Chroma,
                     model_name: str = "Aizelsheikh/llama2-finetuned",
                     top_k: int = 4) -> Any:
    """
    Create a deterministic, context-sensitive RAG chain for medical diagnosis using fine-tuned  model.
    """
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=False
    )

    # Create a text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0,
        top_p=1,
        do_sample=False,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})

    template = """You are a clinical assistant. Use ONLY the context below to provide information.

    DO NOT use any external knowledge not found in the context. Only suggest diagnoses, questions, and tests that are explicitly mentioned in the context.
    
    STRICTLY provide only these three sections:
    
    1. Diagnosis:
    - Provide ONLY diagnoses found in the context with confidence values:
    - Budd Chiari Syndrome (Confidence: value based on context)
    - Cholangiocarcinoma (Confidence: value based on context)
    - Chronic viral hepatitis C (Confidence: value based on context)
    - Hepatic fibrosis (Confidence: value based on context)
    - Hepatocellular Carcinoma (Confidence: value based on context)
    - Hepatitis C (Confidence: value based on context)
    - Chronic hepatic failure (Confidence: value based on context)
    
    Confidence Level: [High if direct match; Medium if suggestive; Low if further evaluation needed]
    
    Only include diagnoses with evidence in the context. Assign confidence (High/Medium/Low) based on how strongly the context supports each diagnosis.

    2. Questions for Doctor:
    List ONLY questions explicitly suggested in the context that a doctor should ask to confirm the diagnosis.
    
    3. Recommended Lab Tests:
    List ONLY lab tests mentioned in the context that would help confirm the diagnosis.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask_question(chain: Any, question: str) -> str:
    """
    Ask a diagnosis question using the RAG chain.
    """
    try:
        response = chain.invoke(question)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"