import json
import os
import tempfile
import pdfplumber
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI

app = Flask(__name__)
analysis_result = {}  # Initialize empty dictionary

# Ensure OpenAI API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Warning: OPENAI_API_KEY is not set. The application will not work properly.")

# Store Indian Legal Rules as a simple list
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

# Convert legal rules to a formatted string for prompts
legal_rules_text = "\n".join([f"{rule['law_id']} ({rule['category']}): {rule['law_text']}" for rule in legal_rules])

# Initialize ChatGPT
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.7, api_key=openai_api_key)

def extract_text(pdf_file):
    """Extract text from a PDF using pdfplumber."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            pdf_file.save(temp_pdf.name)

            text = ""
            with pdfplumber.open(temp_pdf.name) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages[:10]])  # Limit to 10 pages

        os.remove(temp_pdf.name)
        
        if not text.strip():
            return None, "Extracted text is empty. Ensure the PDF is not scanned."

        return text, None
    except Exception as e:
        return None, str(e)

def analyze_contract(contract_text):
    """Extracts key clauses from the contract text using GPT-4 and returns them in JSON format."""
    try:
        extract_clauses_prompt = f"""
        Extract all key legal clauses from the following contract text.
        Return only a JSON object in this format:
        {{ "clauses": ["Clause 1", "Clause 2", "Clause 3"] }}
        
        Contract Text:
        {contract_text}
        """
        response = llm.invoke(extract_clauses_prompt)
        
        # Parse the response content
        response_text = response.content
        clauses_json = json.loads(response_text)
        return clauses_json, None
    except json.JSONDecodeError:
        return None, "Failed to parse AI response. Ensure contract text is valid."
    except Exception as e:
        return None, str(e)

def check_clause_violation(clause):
    """Check if a clause violates any legal rules using direct ChatGPT prompt."""
    prompt = f"""
    Analyze the following contract clause against Indian legal rules and return a JSON response.
    
    Contract Clause: "{clause}"
    
    Relevant Indian Legal Rules:
    {legal_rules_text}
    
    Based on these rules, determine if the clause violates any legal principles.
    Return a JSON object with exactly these fields:
    {{
        "Clause": "{clause}",
        "Legal Rule": "<most relevant legal rule or 'No specific rule found.' if none applies>",
        "Violates": "<YES or NO>",
        "Reason": "<brief explanation>"
    }}
    """
    
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback to ensure we always return the correct format
        return {
            "Clause": clause,
            "Legal Rule": "No specific rule found.",
            "Violates": "UNKNOWN",
            "Reason": "Could not parse AI response. Please review this clause manually."
        }

@app.route('/upload', methods=['POST'])
def upload_contract():
    """API Endpoint to upload and analyze a contract."""
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
    
    # Ensure the output format matches the required JSON structure
    results = {}
    for clause in clauses_json.get("clauses", []):
        results[clause] = check_clause_violation(clause)
    
    return jsonify(results)

@app.route("/check_violation", methods=["POST"])
def check_violation():
    """Endpoint to check violations of contract clauses."""
    clauses = request.json.get("clauses", [])
    
    # Ensure the output format matches the required JSON structure
    results = {}
    for clause in clauses:
        results[clause] = check_clause_violation(clause)
    
    return jsonify(results)
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT from environment, default to 5000
    app.run(host='0.0.0.0', port=port)
