import os
import csv
from dotenv import load_dotenv
import json
import re
load_dotenv()
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface  import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from rouge_score import rouge_scorer
from typing import Set, List, Dict, Any

def load_reference_answers():
    with open("reference_answers.json", "r") as f:
        return json.load(f)
    
def calculate_rouge_score(reference: str, generated: str):
    """Hitung skor Rouge-1, Rouge-2, lalu simpan ke CSV."""
    # Inisialisasi RougeScorer
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2'], use_stemmer=True
    )
    scores = scorer.score(reference, generated)

    # Ambil hanya nilai fmeasure
    score_data = {
        'rouge1_fmeasure': scores['rouge1'].fmeasure,
        'rouge2_fmeasure': scores['rouge2'].fmeasure
    }
    # Write the scores to CSV
    with open("rouge_score_llama3_baru.csv", mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=score_data.keys())
        
        # Write header only if the file is empty
        file.seek(0, 2)  # Move to the end of the file
        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(score_data)
    return scores

def clean_answer_for_rouge(answer: str) -> str:
    """
    Clean up the generated answer by removing extra information (e.g., documentation references)
    that are not necessary for Rouge score comparison.
    """
    # Remove for deepseek-r1 specific tags (e.g., <think>...</think>)
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE)
    # Remove documentation references (e.g., page numbers or further info about where to find more details)
    answer = re.sub(r"(?i)(Dalam dokumentasi.*?Tahun \d{4}/\d{4}|Untuk mendapatkan informasi lebih lanjut.*)", "", answer)
    # Optionally, remove any other specific extra text if needed (tailor regex to your needs)
    answer = re.sub(r"(?i)([Pp]edoman Akademik UNTAN.*?halaman \d{1,2})", "", answer)
    # Normalize whitespace
    answer = re.sub(r"\s+", " ", answer)
    # Trim and return the cleaned answer
    return answer.strip()

def format_docs(docs):
    result = ""
    for doc in docs:
        result += f"- Title: {doc.metadata.get('title', 'Unknown')}\n"
        result += f"  Page: {doc.metadata.get('page_label', 'Unknown')}\n"
        result += f"  Content: {doc.page_content}\n\n"
    return result
def create_source_string(docs: Set[str]) -> str:
    """Create a numbered string of sources (title + page) from retrieved documents."""
    if not docs:
        return "No sources found."
    
    # Use a set to avoid duplicates
    sources = set()
    
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        page = doc.metadata.get("page_label", "Unknown")
        sources.add(f"{title} (Page {page})")

    # Turn into sorted list
    source_list = list(sources)
    source_list.sort()

    # Build the string
    source_string = "Sources:\n"
    for i, src in enumerate(source_list):
        source_string += f"{i + 1}. {src}\n"

    return source_string
def run_llm(query: str, llm_model_name: str, chat_history: List[Dict[str, Any]] = []) :
    #Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    #Jenis VectorStore
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings,
    )
    
    #Jenis LLM
    llm = ChatOllama(model=llm_model_name, verbose=True)

    #Prompt
    template = """
    You are a Question Answering System for Akademik Universitas Tanjungpura 2025. You will be provided with a question and relevant documents.
    Your task is to answer the question based on the information in the documents context.
    If the Question is not relevant with the context, say "Pertanyaan tidak relevan dengan tugas saya".
    Answer with Indonesian Language.
        <context>
        {context}
        </context>
    Question: {input}

    """
    #Jadikan template sebagai PromptTemplate
    custom_rag_prompt = PromptTemplate.from_template(template = template)

    template_rephrase = """
    You are a helpful assistant. You will be provided with a question and relevant documents.
    
    """
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(query)
    formatted_context = format_docs(retrieved_docs)
    sources_string = create_source_string(retrieved_docs)
    
    final_prompt = custom_rag_prompt.format(context=formatted_context, input=query) 

    answer = llm.invoke(final_prompt)

    #Rouge Score Calculation
    reference_answers = load_reference_answers()
    cleaned_answer = clean_answer_for_rouge(answer.content)  # Clean the answer for Rouge comparison
    # Initialize rouge_scores to None
    rouge_scores = None
    if query in reference_answers:
        reference_answer = reference_answers[query]
        rouge_scores = calculate_rouge_score(reference_answer, cleaned_answer)


    if "Pertanyaan tidak relevan dengan tugas saya" in answer.content:
        sources_string = "-"  # atau "Tidak ada sumber" atau None
    else:
        sources_string = create_source_string(retrieved_docs)
    
    result = {
        "input": query,
        "answer": answer.content,
        "sources": sources_string,
        "rouge_scores": rouge_scores, 
    }
    return result

if __name__ == "__main__":
    query = "Apa visi Universitas Tanjungpura?"
    model_name = "llama3"
    
    result = run_llm(query, model_name)
    print(result)

    