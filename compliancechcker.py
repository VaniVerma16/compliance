# Updated Legal Compliance Checker Using InLegalBERT + Reasoning via OpenRouter (Mistral)

import json
import os
import tempfile
import pdfplumber
import numpy as np
import re
import ast
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import requests
from dotenv import load_dotenv
load_dotenv()



app = Flask(__name__)

analysis_result = {}

# Load InLegalBERT for embeddings
embedding_model_id = "law-ai/InLegalBERT"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
embedding_model = AutoModel.from_pretrained(embedding_model_id)

# OpenRouter API for LLM reasoning (Mistral)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# Legal rules
legal_rules = [
    # Civil Laws
    {"law_id": "CIVIL-001", "category": "Civil Law", "law_text": "Under the Limitation Act, 1963, a person must file a civil suit within the prescribed time limit, failing which the case is dismissed."},
    {"law_id": "CIVIL-002", "category": "Civil Law", "law_text": "Under the Indian Contract Act, 1872, a valid contract requires an offer, acceptance, consideration, and lawful object."},
    {"law_id": "CIVIL-003", "category": "Civil Law", "law_text": "As per the Specific Relief Act, 1963, a party can seek specific performance of a contract if damages are inadequate."},
    {"law_id": "CIVIL-004", "category": "Civil Law", "law_text": "Under the Easements Act, 1882, a person acquires the right to use another's property under specific conditions over a period of time."},
    {"law_id": "CIVIL-005", "category": "Civil Law", "law_text": "The Transfer of Property Act, 1882, governs the transfer of property by sale, mortgage, lease, gift, or exchange."},
    {"law_id": "CIVIL-006", "category": "Civil Law", "law_text": "The Registration Act, 1908, mandates the compulsory registration of certain property documents to prevent fraud."},

    # Common Laws
    {"law_id": "COMMON-001", "category": "Common Law", "law_text": "Under the Indian Penal Code (IPC), 1860, Section 299 defines culpable homicide and its punishments."},
    {"law_id": "COMMON-002", "category": "Common Law", "law_text": "As per the Criminal Procedure Code (CrPC), 1973, an accused person has the right to legal representation and a fair trial."},
    {"law_id": "COMMON-003", "category": "Common Law", "law_text": "Under the Evidence Act, 1872, only admissible evidence can be considered in court proceedings."},
    {"law_id": "COMMON-004", "category": "Common Law", "law_text": "Under the Negotiable Instruments Act, 1881, dishonoring a cheque is a criminal offense."},
    
    # Wage-related Laws
    {"law_id": "WAGE-001", "category": "Wage Law", "law_text": "The Minimum Wages Act, 1948, ensures that workers receive a statutory minimum wage set by the government."},
    {"law_id": "WAGE-002", "category": "Wage Law", "law_text": "The Payment of Wages Act, 1936, mandates timely payment of wages to employees without unauthorized deductions."},
    {"law_id": "WAGE-003", "category": "Wage Law", "law_text": "The Equal Remuneration Act, 1976, ensures that men and women receive equal pay for equal work."},
    {"law_id": "WAGE-004", "category": "Wage Law", "law_text": "The Code on Wages, 2019, consolidates minimum wages, payment of wages, and bonus laws under a single framework."},
    {"law_id": "WAGE-005", "category": "Wage Law", "law_text": "The Payment of Bonus Act, 1965, mandates that eligible employees receive an annual bonus based on company profits."},
    
    # House Agreement Laws
    {"law_id": "HOUSE-001", "category": "Property Law", "law_text": "Under the Transfer of Property Act, 1882, property agreements must be legally registered if they exceed a value threshold."},
    {"law_id": "HOUSE-002", "category": "Property Law", "law_text": "The Registration Act, 1908, mandates that property agreements over 12 months must be registered with the sub-registrar."},
    {"law_id": "HOUSE-003", "category": "Property Law", "law_text": "The Indian Stamp Act, 1899, requires that house agreements be stamped with the appropriate duty as per state laws."},
    {"law_id": "HOUSE-004", "category": "Property Law", "law_text": "Under the Contract Act, 1872, house agreements must include lawful consideration, offer, acceptance, and legal purpose."},

    # Rent Agreement Laws
    {"law_id": "RENT-001", "category": "Rent Law", "law_text": "The Rent Control Act, applicable in various states, regulates rent prices and eviction procedures for tenants."},
    {"law_id": "RENT-002", "category": "Rent Law", "law_text": "The Model Tenancy Act, 2021, provides a legal framework for renting agreements and dispute resolution."},
    {"law_id": "RENT-003", "category": "Rent Law", "law_text": "Under the Maharashtra Rent Control Act, 1999, landlords must not evict tenants without legal grounds."},
    {"law_id": "RENT-004", "category": "Rent Law", "law_text": "Under the Delhi Rent Control Act, 1958, rent hikes cannot exceed a fixed percentage unless mutually agreed upon."},
    {"law_id": "RENT-005", "category": "Rent Law", "law_text": "A rental agreement for over 11 months requires registration under the Registration Act, 1908, to be legally enforceable."},
    
    # Customary Laws
    {"law_id": "CUSTOM-001", "category": "Customary Law", "law_text": "Under the Hindu Succession Act, 1956, ancestral property is inherited by legal heirs, with equal rights to male and female heirs."},
    {"law_id": "CUSTOM-002", "category": "Customary Law", "law_text": "As per the Muslim Personal Law (Shariat) Application Act, 1937, inheritance and marriage are governed by Islamic principles."},
    {"law_id": "CUSTOM-003", "category": "Customary Law", "law_text": "The Parsi Marriage and Divorce Act, 1936, dictates marriage and divorce regulations specific to the Parsi community."},

    # Corporate Laws
    {"law_id": "COMP-001", "category": "Corporate Law", "law_text": "Under the Companies Act, 2013, all companies must register with the Ministry of Corporate Affairs and adhere to governance regulations."},
    {"law_id": "COMP-002", "category": "Corporate Law", "law_text": "As per the SEBI Act, 1992, all listed companies must follow market regulations to prevent insider trading."},
    {"law_id": "COMP-003", "category": "Corporate Law", "law_text": "The LLP Act, 2008, provides a legal framework for limited liability partnerships in India."},
    {"law_id": "COMP-004", "category": "Corporate Law", "law_text": "The Competition Act, 2002, prevents monopolies and promotes fair competition in markets."},

    # IT & Data Privacy Laws
    {"law_id": "IT-001", "category": "IT Act", "law_text": "Under Section 43A of the IT Act, any corporate body handling personal data must ensure data protection. Failure to do so results in liability."},
    {"law_id": "DATA-001", "category": "Data Privacy", "law_text": "The Personal Data Protection Bill requires organizations to obtain explicit consent before collecting personal data."},
    {"law_id": "CYBER-001", "category": "Cyber Law", "law_text": "Under the IT Act, 2000, cyber fraud, hacking, and identity theft are punishable offenses."},
    
    # Labour Laws
    {"law_id": "LABOUR-001", "category": "Labour Law", "law_text": "The Minimum Wages Act, 1948, ensures workers receive a statutory minimum wage."},
    {"law_id": "LABOUR-002", "category": "Labour Law", "law_text": "According to the Industrial Disputes Act, no worker can be terminated without a 30-day notice period."},
    {"law_id": "LABOUR-003", "category": "Labour Law", "law_text": "Under the Factories Act, 1948, factory workers must have safe working conditions, proper sanitation, and regulated working hours."},
    {"law_id": "LABOUR-004", "category": "Labour Law", "law_text": "The Maternity Benefit Act, 1961, provides maternity leave and related benefits to women employees."},
    {"law_id": "LABOUR-005", "category": "Labour Law", "law_text": "The Employees' Provident Fund Act, 1952, ensures retirement benefits for employees through a contributory fund."},

    # Consumer Protection Laws
    {"law_id": "CONSUMER-001", "category": "Consumer Law", "law_text": "The Consumer Protection Act, 2019, provides rights to consumers against unfair trade practices and defective products."},
    {"law_id": "CONSUMER-002", "category": "Consumer Law", "law_text": "Under the Food Safety and Standards Act, 2006, food manufacturers must meet safety and hygiene standards."},

    # Environmental Laws
    {"law_id": "ENV-001", "category": "Environmental Law", "law_text": "The Environment Protection Act, 1986, empowers the central government to protect and improve the environment."},
    {"law_id": "ENV-002", "category": "Environmental Law", "law_text": "Under the Wildlife Protection Act, 1972, the killing and trade of protected animal species are illegal."},
    {"law_id": "ENV-003", "category": "Environmental Law", "law_text": "The Water (Prevention and Control of Pollution) Act, 1974, aims to control water pollution."},
    {"law_id": "ENV-004", "category": "Environmental Law", "law_text": "The Air (Prevention and Control of Pollution) Act, 1981, regulates air pollution control measures."},

    # Banking & Finance Laws
    {"law_id": "BANK-001", "category": "Banking Law", "law_text": "The Banking Regulation Act, 1949, regulates banking operations and ensures compliance with RBI norms."},
    {"law_id": "BANK-002", "category": "Banking Law", "law_text": "The SARFAESI Act, 2002, allows banks to recover bad loans without court intervention."},
    {"law_id": "BANK-003", "category": "Banking Law", "law_text": "The FEMA Act, 1999, governs foreign exchange transactions and regulates cross-border investments."},

    # Real Estate Laws
    {"law_id": "REAL-001", "category": "Real Estate Law", "law_text": "The RERA Act, 2016, regulates the real estate sector and protects homebuyers from fraud."},
    {"law_id": "REAL-002", "category": "Real Estate Law", "law_text": "The Benami Transactions Act, 1988, prohibits the holding of property under a fictitious name to avoid taxes."}
]

# Get rule embeddings
SIMILARITY_THRESHOLD = 0.6

def get_embedding(text):
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.numpy() if embedding.numel() > 0 else None

legal_rules_texts = [rule['law_text'] for rule in legal_rules]
legal_rules_embeddings = [get_embedding(text) for text in legal_rules_texts if get_embedding(text) is not None]
legal_rules_filtered = [rule for i, rule in enumerate(legal_rules) if get_embedding(legal_rules_texts[i]) is not None]

def extract_text(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            pdf_file.save(temp_pdf.name)
            text = ""
            with pdfplumber.open(temp_pdf.name) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages[:10]])
        os.remove(temp_pdf.name)
        if not text.strip():
            return None, "Extracted text is empty. Ensure the PDF is not scanned."
        return text, None
    except Exception as e:
        return None, str(e)

def analyze_contract(contract_text):
    clauses = [cl.strip() for cl in contract_text.split(".") if len(cl.strip()) > 20]
    return {"clauses": clauses}, None

def find_most_relevant_rule(clause):
    clause_embedding = get_embedding(clause)
    if clause_embedding is None or len(clause_embedding.shape) != 1:
        return {"law_text": "No valid embedding generated."}, 0.0

    similarities = cosine_similarity([clause_embedding], legal_rules_embeddings)[0]
    max_index = int(np.argmax(similarities))
    if similarities[max_index] < SIMILARITY_THRESHOLD:
        return {"law_text": "No sufficiently relevant legal rule found."}, similarities[max_index]
    return legal_rules_filtered[max_index], similarities[max_index]

def call_llm(prompt):
    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {"role": "system", "content": "You are a helpful Indian legal assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("LLM API Error:", response.status_code, response.text)
        return None

def extract_json(text):
    try:
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            cleaned = match.group().replace('\n', ' ').replace('\r', '')
            return json.loads(cleaned)
        return ast.literal_eval(text)
    except Exception as e:
        print("JSON parsing error:", e)
        return None

def check_clause_violation(clause):
    matched_rule, similarity = find_most_relevant_rule(clause)
    reasoning_prompt = f"""
    As an Indian legal expert, analyze the following contract clause in relation to the specified legal rule. Determine if the clause violates the rule and explain your reasoning.

    Clause: "{clause}"
    Legal Rule: "{matched_rule['law_text']}"

    Provide your answer in JSON format:
    {{
        "Clause": "{clause}",
        "Legal Rule": "{matched_rule['law_text']}",
        "Violates": "YES or NO",
        "Reason": "<brief reasoning>"
    }}
    """

    response_text = call_llm(reasoning_prompt)
    if not response_text:
        return {
            "Clause": clause,
            "Legal Rule": matched_rule['law_text'],
            "Violates": "UNKNOWN",
            "Reason": "LLM inference failed"
        }

    parsed = extract_json(response_text)
    if parsed:
        return parsed
    else:
        return {
            "Clause": clause,
            "Legal Rule": matched_rule['law_text'],
            "Violates": "UNKNOWN",
            "Reason": "Could not parse LLM response: " + response_text[:200] + ("..." if len(response_text) > 200 else "")
        }

@app.route('/upload', methods=['POST'])
def upload_contract():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    contract_text, error = extract_text(file)
    if error:
        return jsonify({"error": error}), 400

    clauses_json, error = analyze_contract(contract_text)
    if error:
        return jsonify({"error": error}), 500

    results = {}
    for clause in clauses_json.get("clauses", []):
        results[clause] = check_clause_violation(clause)

    return jsonify(results)

@app.route("/check_violation", methods=["POST"])
def check_violation():
    clauses = request.json.get("clauses", [])
    results = {}
    for clause in clauses:
        results[clause] = check_clause_violation(clause)
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5500))
    app.run(host='0.0.0.0', port=port, debug=True)