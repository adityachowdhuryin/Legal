import os
import json
import pandas as pd
import torch
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    squad_convert_examples_to_features
)
from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from torch.utils.data import DataLoader, SequentialSampler
import time
import base64
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import logging
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sentence_transformers import SentenceTransformer, util
import numpy as np
import xgboost as xgb
from fuzzywuzzy import fuzz
import fitz  # PyMuPDF for improved PDF extraction
from PIL import Image
import pytesseract
import pdfplumber
import docx2txt
import spacy
from textblob import TextBlob
import tempfile

# Use temporary directory for cloud compatibility
temp_dir = tempfile.gettempdir()
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Setup logging to console and temporary file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(temp_dir, 'app.log')),
        logging.StreamHandler()
    ]
)

# Try importing optional dependencies
try:
    from rouge_score import rouge_scorer
except ImportError:
    logging.warning("The 'rouge-score' package is not installed. ROUGE metrics will be skipped.")

# Setup dependencies
def setup_dependencies():
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        logging.error(f"Error downloading NLTK data: {e}")
        raise
    try:
        if not spacy.util.is_package("en_core_web_sm"):
            spacy.cli.download("en_core_web_sm")
        global nlp
        nlp = spacy.load("en_core_web_sm", disable=['lemmatizer'])
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        logging.error(f"Error loading spaCy model: {e}")
        raise

# Call setup at the start
setup_dependencies()

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    logging.info("GPU not available, using CPU.")
spacy.prefer_gpu()  # Attempt GPU, but spaCy will fallback to CPU if unavailable

# Setups
st.set_page_config(layout="wide")
try:
    image = Image.open('./banner.PNG')
    st.image(image)
except FileNotFoundError:
    st.info("Banner image not found. Skipping banner display.")

# Initialize ground truth
ground_truth = st.session_state.get('ground_truth', [])
if not ground_truth:
    st.warning("No ground truth data provided. Evaluation metrics will not be computed unless a ground truth file is uploaded.")

# Advanced text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'Page \d+ of \d+|CONFIDENTIAL|\[.*?\]|^\s*\d+\s*$', '', text, flags=re.IGNORECASE)
    try:
        blob = TextBlob(text)
        text = str(blob.correct())
    except Exception as e:
        logging.warning(f"Spell checking failed: {e}")
    legal_terms = {
        'noncompete': 'non-compete',
        'non compete': 'non-compete',
        'exclusivity agreement': 'exclusivity',
        'governinglaw': 'governing law'
    }
    for term, normalized in legal_terms.items():
        text = re.sub(rf'\b{term}\b', normalized, text, flags=re.IGNORECASE)
    return text.strip()

# OCR post-processing
def post_process_ocr(text):
    if not text:
        return ''
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    corrected_text = re.sub(r'(\w)([A-Z])', r'\1 \2', corrected_text)
    return clean_text(corrected_text)

# Clause segmentation
def segment_clauses(text):
    doc = nlp(text)
    clauses = []
    current_clause = []
    for sent in doc.sents:
        current_clause.append(sent.text)
        if re.match(r'^(Section|Article|\d+\.)', sent.text, re.IGNORECASE) or sent.text.endswith(':'):
            if current_clause:
                clauses.append(' '.join(current_clause[:-1]))
                current_clause = [sent.text]
    if current_clause:
        clauses.append(' '.join(current_clause))
    return [clean_text(clause) for clause in clauses if clause.strip()]

# Train classifier for risk detection
def train_risk_classifier(classifier_type):
    try:
        logging.info(f"Training {classifier_type} risk classifier...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))
        if 'ground_truth' not in st.session_state or not st.session_state.ground_truth:
            logging.warning("No ground truth data available for training risk classifier.")
            return None, None, None
        annotations = st.session_state.ground_truth.get('annotations', [])
        if not annotations:
            logging.warning("Ground truth JSON does not contain 'annotations' key.")
            return None, None, None
        
        clause_texts = []
        risk_labels = []
        for ann in annotations:
            try:
                clause_text = clean_text(ann['clause']['text'])
                risk_level = ann['risk']['level']
                if not clause_text:
                    logging.warning(f"Invalid clause text in annotation: {ann}")
                    continue
                if risk_level not in ['Low', 'Medium', 'High', 'Unknown']:
                    logging.warning(f"Invalid risk level in annotation: {risk_level}")
                    continue
                clause_texts.append(clause_text)
                risk_labels.append(risk_level)
            except KeyError as e:
                logging.warning(f"Missing key in annotation: {e}, Annotation: {ann}")
                continue
        
        if not clause_texts or len(clause_texts) < 2:
            logging.error("Insufficient valid ground truth data for training.")
            return None, None, None
        
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(risk_labels)
        classes = np.array(label_encoder.classes_)
        logging.debug(f"Classes for label encoder: {classes}")
        
        class_weights = compute_class_weight('balanced', classes=classes, y=risk_labels)
        class_weight_dict = dict(zip(range(len(classes)), class_weights))
        logging.debug(f"Class weights: {class_weights}")
        
        embeddings = embedder.encode(clause_texts)
        logging.debug(f"Generated embeddings shape: {embeddings.shape}")
        
        if classifier_type == "LogisticRegression":
            clf = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                class_weight=class_weight_dict,
                random_state=42
            )
        elif classifier_type == "RandomForest":
            clf = RandomForestClassifier(
                n_estimators=100,
                class_weight=class_weight_dict,
                random_state=42
            )
        elif classifier_type == "XGBoost":
            clf = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=len(classes),
                random_state=42,
                tree_method='gpu_hist' if torch.cuda.is_available() else 'hist'
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
        clf.fit(embeddings, encoded_labels)
        logging.info(f"{classifier_type} risk classifier trained successfully.")
        return clf, embedder, label_encoder
    except Exception as e:
        logging.error(f"Error training {classifier_type} risk classifier: {e}", exc_info=True)
        st.error(f"Error training {classifier_type} risk classifier: {e}")
        return None, None, None

# Predefined questions for clause extraction
questions = [
    'Highlight the parts (if any) of this contract related to "Document Name". Details: The name of the contract',
    'Highlight the parts (if any) of this contract related to "Parties". Details: The two or more parties who signed the contract',
    'Highlight the parts (if any) of this contract related to "Agreement Date". Details: The date of the contract',
    'Highlight the parts (if any) of this contract related to "Effective Date". Details: The date when the contract is effective',
    'Highlight the parts (if any) of this contract related to "Expiration Date". Details: On what date will the contract\'s initial term expire?',
    'Highlight the parts (if any) of this contract related to "Renewal Term". Details: What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.',
    'Highlight the parts (if any) of this contract related to "Notice Period To Terminate Renewal". Details: What is the notice period required to terminate renewal?',
    'Highlight the parts (if any) of this contract related to "Governing Law". Details: Which state/country\'s law governs the interpretation of the contract?',
    'Highlight the parts (if any) of this contract related to "Most Favored Nation". Details: Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?',
    'Highlight the parts (if any) of this contract related to "Non-Compete". Details: Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?',
    'Highlight the parts (if any) of this contract related to "Exclusivity". Details: Is there an exclusive dealing commitment with the counterparty? This includes a commitment to procure all “requirements” from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on collaborating or working with other parties), whether during the contract or after the contract ends (or both).',
    'Highlight the parts (if any) of this contract related to "No-Solicit Of Customers". Details: Is a party restricted from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both)?',
    'Highlight the parts (if any) of this contract related to "Competitive Restriction Exception". Details: This category includes the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers above.',
    'Highlight the parts (if any) of this contract related to "No-Solicit Of Employees". Details: Is there a restriction on a party’s soliciting or hiring employees and/or contractors from the counterparty, whether during the contract or after the contract ends (or both)?',
    'Highlight the parts (if any) of this contract related to "Non-Disparagement". Details: Is there a requirement on a party not to disparage the counterparty?',
    'Highlight the parts (if any) of this contract related to "Termination For Convenience". Details: Can a party terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire)?',
    'Highlight the parts (if any) of this contract related to "Rofr/Rofo/Rofn". Details: Is there a clause granting one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services?',
    'Highlight the parts (if any) of this contract related to "Change Of Control". Details: Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?',
    'Highlight the parts (if any) of this contract related to "Anti-Assignment". Details: Is consent or notice required of a party if the contract is assigned to a third party?',
    'Highlight the parts (if any) of this contract related to "Revenue/Profit Sharing". Details: Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?',
    'Highlight the parts (if any) of this contract related to "Price Restrictions". Details: Is there a restriction on the ability of a party to raise or reduce prices of technology, goods, or services provided?',
    'Highlight the parts (if any) of this contract related to "Minimum Commitment". Details: Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?',
    'Highlight the parts (if any) of this contract related to "Volume Restriction". Details: Is there a fee increase or consent requirement, etc. if one party’s use of the product/services exceeds certain threshold?',
    'Highlight the parts (if any) of this contract related to "Ip Ownership Assignment". Details: Does intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?',
    'Highlight the parts (if any) of this contract related to "Joint Ip Ownership". Details: Is there any clause providing for joint or shared ownership of intellectual property between the parties to the contract?',
    'Highlight the parts (if any) of this contract related to "License Grant". Details: Does the contract contain a license granted by one party to its counterparty?',
    'Highlight the parts (if any) of this contract related to "Non-Transferable License". Details: Does the contract limit the ability of a party to transfer the license being granted to a third party?',
    'Highlight the parts (if any) of this contract related to "Affiliate License-Licensor". Details: Does the contract contain a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor?',
    'Highlight the parts (if any) of this contract related to "Affiliate License-Licensee". Details: Does the contract contain a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor?',
    'Highlight the parts (if any) of this contract related to "Unlimited/All-You-Can-Eat-License". Details: Is there a clause granting one party an “enterprise,” “all you can eat” or unlimited usage license?',
    'Highlight the parts (if any) of this contract related to "Irrevocable Or Perpetual License". Details: Does the contract contain a license grant that is irrevocable or perpetual?',
    'Highlight the parts (if any) of this contract related to "Source Code Escrow". Details: Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy, insolvency, etc.)?',
    'Highlight the parts (if any) of this contract related to "Post-Termination Services". Details: Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?',
    'Highlight the parts (if any) of this contract related to "Audit Rights". Details: Does a party have the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?',
    'Highlight the parts (if any) of this contract related to "Uncapped Liability". Details: Is a party’s liability uncapped upon the breach of its obligation in the contract? This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.',
    'Highlight the parts (if any) of this contract related to "Cap On Liability". Details: Does the contract include a cap on liability upon the breach of a party’s obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.',
    'Highlight the parts (if any) of this contract related to "Liquidated Damages". Details: Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?',
    'Highlight the parts (if any) of this contract related to "Warranty Duration". Details: What is the duration of any warranty against defects or errors in technology, products, or services provided under the contract?',
    'Highlight the parts (if any) of this contract related to "Insurance". Details: Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?',
    'Highlight the parts (if any) of this contract related to "Covenant Not To Sue". Details: Is a party restricted from contesting the validity of the counterparty’s ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?',
    'Highlight the parts (if any) of this contract related to "Third Party Beneficiary". Details: Is there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party?'
]
questions2 = [
    'Document Name:', 'Parties:', 'Agreement Date:', 'Effective Date:', 'Expiration Date:', 'Renewal Term:', 
    'Notice Period To Terminate Renewal:', 'Governing Law:', 'Most Favored Nation:', 'Non-Compete:', 'Exclusivity:', 
    'No-Solicit Of Customers:', 'Competitive Restriction Exception:', 'No-Solicit Of Employees:', 'Non-Disparagement:', 
    'Termination For Convenience:', 'Right of First Refusal, Offer or Negotiation (ROFR/ROFO/ROFN):', 
    'Change Of Control:', 'Anti-Assignment:', 'Revenue/Profit Sharing:', 'Price Restrictions:', 'Minimum Commitment:', 
    'Volume Restriction:', 'Ip Ownership Assignment:', 'Joint Ip Ownership:', 'License Grant:', 
    'Non-Transferable License:', 'Affiliate License-Licensor:', 'Affiliate License-Licensee:', 
    'Unlimited/All-You-Can-Eat-License:', 'Irrevocable Or Perpetual License:', 'Source Code Escrow:', 
    'Post-Termination Services:', 'Audit Rights:', 'Uncapped Liability:', 'Cap On Liability:', 
    'Liquidated Damages:', 'Warranty Duration:', 'Insurance:', 'Covenant Not To Sue:', 'Third Party Beneficiary:'
]

# Clause type mapping
clause_type_mapping = {
    'Non-Disparagement': 'Non-Disparagement Clause',
    'Non-Compete': 'Non-Compete Clause',
    'Exclusivity': 'Exclusivity Clause',
    'No-Solicit Of Customers': 'No-Solicit Clause',
    'No-Solicit Of Employees': 'No-Solicit Clause',
    'Termination For Convenience': 'Termination Clause',
    'Anti-Assignment': 'Assignment Clause',
    'Revenue/Profit Sharing': 'Financial Clause',
    'Price Restrictions': 'Financial Clause',
    'Minimum Commitment': 'Commitment Clause',
    'Volume Restriction': 'Commitment Clause',
    'Ip Ownership Assignment': 'IP Clause',
    'Joint Ip Ownership': 'IP Clause',
    'License Grant': 'License Clause',
    'Non-Transferable License': 'License Clause',
    'Affiliate License-Licensor': 'License Clause',
    'Affiliate License-Licensee': 'License Clause',
    'Unlimited/All-You-Can-Eat-License': 'License Clause',
    'Irrevocable Or Perpetual License': 'License Clause',
    'Source Code Escrow': 'Escrow Clause',
    'Post-Termination Services': 'Termination Clause',
    'Audit Rights': 'Compliance Clause',
    'Uncapped Liability': 'Liability Clause',
    'Cap On Liability': 'Liability Clause',
    'Liquidated Damages': 'Liability Clause',
    'Warranty Duration': 'Warranty Clause',
    'Insurance': 'Insurance Clause',
    'Covenant Not To Sue': 'Legal Clause',
    'Third Party Beneficiary': 'Beneficiary Clause',
    'Governing Law': 'Governing Law Clause',
    'Most Favored Nation': 'MFN Clause',
    'Rofr/Rofo/Rofn': 'ROFR/ROFO/ROFN Clause',
    'Change Of Control': 'Change of Control Clause'
}

# Expand keywords for fallback extraction
def expand_keywords(keywords):
    expanded = {}
    synonyms = {
        "non-compete": ["noncompete", "non compete", "restrict competition"],
        "exclusivity": ["exclusive agreement", "sole provider", "only with"],
        "governing law": ["applicable law", "law of", "jurisdiction"],
        "termination for convenience": ["terminate without cause", "end agreement", "cancellation"],
        "license grant": ["grants license", "licensing", "license to use"]
    }
    for clause, kws in keywords.items():
        expanded_kws = set(kws)
        for kw in kws:
            if kw in synonyms:
                expanded_kws.update(synonyms[kw])
        expanded[clause] = list(expanded_kws)
    return expanded

# Initialize summarization model and tokenizer
def load_summarization_model(model_name):
    try:
        logging.info(f"Loading {model_name} tokenizer...")
        if model_name == "T5":
            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            logging.info(f"Loading {model_name} model...")
            model = T5ForConditionalGeneration.from_pretrained('t5-small')
        elif model_name == "DistilBART":
            tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
            logging.info(f"Loading {model_name} model...")
            model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
        else:
            raise ValueError(f"Unsupported summarization model: {model_name}")
        logging.info(f"Moving {model_name} model to {device}...")
        model = model.to(device)
        logging.info(f"{model_name} model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load {model_name} model: {e}", exc_info=True)
        st.error(f"Failed to load {model_name} model: {e}")
        return None, None

# Risk detection
def detect_risk(clause_text, clause_type, clf, embedder, label_encoder):
    logging.debug(f"Using classifier {type(clf).__name__} for {clause_type}")
    clause_text = clean_text(clause_text)
    if not clause_text:
        logging.warning(f"Invalid clause text for {clause_type}: {clause_text}")
        return {
            "level": "Unknown",
            "issue": "No clause text provided.",
            "explanation": "Unable to assess risk due to missing or invalid clause text."
        }
    if clf is None or embedder is None or label_encoder is None:
        logging.error(f"Classifier, embedder, or label encoder not initialized for {clause_type}.")
        st.warning("Risk classifier not initialized. Using rule-based fallback.")
        clause_text_lower = clause_text.lower()
        high_risk_keywords = ['uncapped liability', 'indemnification', 'non-compete', 'exclusivity', 'liquidated damages']
        medium_risk_keywords = ['termination for convenience', 'change of control', 'anti-assignment', 'audit rights']
        if any(keyword in clause_text_lower for keyword in high_risk_keywords):
            risk_level = "High"
            issue = "Potential high-risk clause detected."
            explanation = "Clause contains terms associated with high risk, such as liability or exclusivity."
        elif any(keyword in clause_text_lower for keyword in medium_risk_keywords):
            risk_level = "Medium"
            issue = "Potential issue requiring review."
            explanation = "Clause contains terms that may pose moderate risk, such as termination or audit rights."
        else:
            risk_level = "Low"
            issue = "No significant issues identified."
            explanation = "Clause appears standard with no immediate red flags."
        logging.debug(f"Rule-based risk for {clause_type}: {risk_level}, Text: {clause_text[:50]}")
        return {
            "level": risk_level,
            "issue": issue,
            "explanation": explanation
        }
    try:
        embedding = embedder.encode([clause_text])[0]
        encoded_prediction = clf.predict([embedding])[0]
        risk_level = label_encoder.inverse_transform([encoded_prediction])[0]
        issue_dict = {
            "Low": "No significant issues identified.",
            "Medium": "Potential issue requiring review.",
            "High": "Significant issue with high exposure.",
            "Unknown": "Unable to assess risk."
        }
        explanation_dict = {
            "Low": "Clause appears standard with no immediate red flags.",
            "Medium": "Review clause for potential risks to flexibility or liability.",
            "High": "Immediate attention required due to significant exposure.",
            "Unknown": "Unable to assess due to invalid input or model error."
        }
        logging.debug(f"Predicted risk for {clause_type}: {risk_level}, Text: {clause_text[:50]}")
        return {
            "level": risk_level,
            "issue": issue_dict.get(risk_level, "Unknown issue."),
            "explanation": explanation_dict.get(risk_level, "Unable to assess.")
        }
    except Exception as e:
        logging.error(f"Error in risk detection for {clause_type}: {e}", exc_info=True)
        return {
            "level": "Unknown",
            "issue": "Error in risk detection.",
            "explanation": f"Failed to predict risk: {str(e)}"
        }

# Summarization function with entity preservation
def summarize_clause(clause_text, clause_type):
    clause_text = clean_text(clause_text)
    if not clause_text:
        logging.warning(f"Invalid or empty clause text for summarization {clause_type}: {clause_text}")
        return {
            "text": "No summary available due to missing clause text.",
            "suggested_action": "Verify that the clause text is correctly extracted."
        }
    try:
        model_name = st.session_state.get('summarization_model', 'T5')
        model, tokenizer = load_summarization_model(model_name)
        if model is None or tokenizer is None:
            logging.error(f"{model_name} model or tokenizer not loaded for {clause_type}")
            return {
                "text": f"Error: {model_name} model or tokenizer not loaded.",
                "suggested_action": "Check model availability and dependencies."
            }
        if len(clause_text) > 1000:
            logging.warning(f"Clause text too long for {clause_type}, truncating to 1000 characters")
            clause_text = clause_text[:1000]
        
        if model_name == "T5":
            input_text = f"summarize: {clause_text}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
            if not inputs['input_ids'].size(1):
                logging.warning(f"Empty input tokens for {clause_type}: {input_text}")
                return {
                    "text": "No summary available due to invalid input tokens.",
                    "suggested_action": "Verify that the clause text is valid and not empty."
                }
            summary_ids = model.generate(
                inputs['input_ids'],
                max_length=50,
                min_length=10,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        else:  # DistilBART
            inputs = tokenizer(clause_text, return_tensors="pt", max_length=512, truncation=True).to(device)
            if not inputs['input_ids'].size(1):
                logging.warning(f"Empty input tokens for {clause_type}: {clause_text}")
                return {
                    "text": "No summary available due to invalid input tokens.",
                    "suggested_action": "Verify that the clause text is valid and not empty."
                }
            summary_ids = model.generate(
                inputs['input_ids'],
                max_length=50,
                min_length=10,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )
            summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        doc = nlp(clause_text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'DATE', 'GPE']:
                entities.append(f"{ent.text} ({ent.label_})")
        if entities:
            summary_text += " Key entities: " + ", ".join(set(entities))
        
        logging.debug(f"Generated summary for {clause_type} with {model_name}: {summary_text}")
        suggested_action = "Review clause for clarity and specificity."
        if clause_type == "Non-Disparagement Clause":
            suggested_action = "Ensure definition of 'disparagement' is clear."
        elif clause_type == "Exclusivity Clause":
            suggested_action = "Verify scope and duration of exclusivity."
        elif clause_type == "License Clause":
            suggested_action = "Confirm license scope, duration, and transferability."
        elif clause_type == "Termination Clause":
            suggested_action = "Check termination triggers and notice periods."
        return {"text": summary_text, "suggested_action": suggested_action}
    except Exception as e:
        logging.error(f"Error in summarization for {clause_type} with {model_name}: {e}", exc_info=True)
        return {
            "text": f"Error generating summary: {str(e)}",
            "suggested_action": "Check input text and model dependencies."
        }

# Fallback clause extraction with expanded keywords
def fallback_clause_extraction(contract, clause_label):
    keywords = {
        "Document Name": ["agreement", "contract title", "this agreement known as"],
        "Parties": ["party of the first part", "between .* and", "parties hereto"],
        "Agreement Date": ["dated as of", "agreement date", "execution date"],
        "Effective Date": ["effective as of", "commencement date"],
        "Expiration Date": ["expires on", "termination date", "end of term"],
        "Renewal Term": ["renewal term", "automatic renewal", "extension period"],
        "Notice Period To Terminate Renewal": ["notice to terminate renewal", "renewal termination notice"],
        "Governing Law": ["governed by", "law of", "applicable law"],
        "Most Favored Nation": ["most favored nation", "mfn clause", "better terms"],
        "Non-Compete": ["non-compete", "restrict.*compete", "competition"],
        "Exclusivity": ["exclusive", "solely", "only with"],
        "No-Solicit Of Customers": ["no-solicit customers", "solicit customers", "customer restriction"],
        "Competitive Restriction Exception": ["exception to non-compete", "carveout exclusivity"],
        "No-Solicit Of Employees": ["no-solicit employees", "hire employees", "employee restriction"],
        "Non-Disparagement": ["non-disparagement", "not disparage"],
        "Termination For Convenience": ["terminate.*convenience", "without cause"],
        "Rofr/Rofo/Rofn": ["right of first refusal", "rofr", "rofo", "rofn"],
        "Change Of Control": ["change of control", "merger.*notice"],
        "Anti-Assignment": ["anti-assignment", "assignment consent", "transfer restriction"],
        "Revenue/Profit Sharing": ["revenue sharing", "profit sharing", "share revenue"],
        "Price Restrictions": ["price restrictions", "pricing controls", "raise prices"],
        "Minimum Commitment": ["minimum commitment", "minimum order", "minimum purchase"],
        "Volume Restriction": ["volume restriction", "exceed threshold", "usage cap"],
        "Ip Ownership Assignment": ["ip ownership assignment", "assign ip", "transfer ip"],
        "Joint Ip Ownership": ["joint ip ownership", "shared ip", "co-own ip"],
        "License Grant": ["grants.*license", "license to"],
        "Non-Transferable License": ["non-transferable license", "license not assignable"],
        "Affiliate License-Licensor": ["affiliate license licensor", "licensor affiliates"],
        "Affiliate License-Licensee": ["affiliate license licensee", "licensee affiliates"],
        "Unlimited/All-You-Can-Eat-License": ["unlimited license", "all you can eat", "enterprise license"],
        "Irrevocable Or Perpetual License": ["irrevocable license", "perpetual license"],
        "Source Code Escrow": ["source code escrow", "deposit source code"],
        "Post-Termination Services": ["post-termination services", "transition services"],
        "Audit Rights": ["audit.*rights", "right to audit"],
        "Uncapped Liability": ["uncapped liability", "unlimited liability"],
        "Cap On Liability": ["cap on liability", "liability limit", "maximum liability"],
        "Liquidated Damages": ["liquidated damages", "termination fee"],
        "Warranty Duration": ["warranty duration", "warranty period", "defects warranty"],
        "Insurance": ["maintain.*insurance", "insurance.*required"],
        "Covenant Not To Sue": ["covenant not to sue", "no contest validity"],
        "Third Party Beneficiary": ["third party beneficiary", "intended beneficiaries", "enforce rights"],
    }
    keywords = expand_keywords(keywords)
    contract = clean_text(contract)
    for keyword in keywords.get(clause_label, []):
        matches = re.findall(rf'.{{0,100}}{keyword}.{{0,100}}', contract, re.IGNORECASE)
        if matches:
            sorted_matches = sorted(matches, key=len, reverse=True)
            return " ".join(sorted_matches[:2])
    return ""

# Evaluation function with semantic matching and semantic summary metric
def evaluate_results(predicted_results, ground_truth, embedder):
    clause_true = []
    clause_pred = []
    risk_true = []
    risk_pred = []
    summary_true = []
    summary_pred = []
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    except NameError:
        logging.warning("rouge-score package missing; skipping ROUGE metrics.")
        scorer = None
    
    for pred in predicted_results:
        try:
            clause_type = pred['clause']['type']
            normalized_clause_type = clause_type.replace(" Clause", "")
            clause_text = clean_text(pred['clause']['text'] if 'text' in pred['clause'] else "")
            if not clause_text:
                logging.warning(f"Empty clause text for predicted clause type: {clause_type}")
                continue
            best_gt = None
            best_ratio = 0
            for gt in ground_truth:
                try:
                    gt_clause_type = gt['clause']['type']
                    gt_embedding = embedder.encode(gt_clause_type)
                    pred_embedding = embedder.encode(clause_type)
                    norm_embedding = embedder.encode(normalized_clause_type)
                    match_ratio1 = util.cos_sim(gt_embedding, pred_embedding)[0][0].item()
                    match_ratio2 = util.cos_sim(gt_embedding, norm_embedding)[0][0].item()
                    match_ratio = max(match_ratio1, match_ratio2)
                    if match_ratio > 0.8:
                        if match_ratio > best_ratio:
                            best_ratio = match_ratio
                            best_gt = gt
                except KeyError as e:
                    logging.warning(f"Missing key in ground truth: {gt}")
                    continue
            if best_gt:
                gt_clause_text = clean_text(best_gt['clause']['text'])
                clause_true.append(gt_clause_text)
                clause_pred.append(clause_text)
                risk_true.append(best_gt['risk']['level'])
                risk_pred.append(pred['risk']['level'])
                summary_true.append(clean_text(best_gt['summary']['text']))
                summary_pred.append(clean_text(pred['summary']['text']))
            else:
                logging.warning(f"No semantic match found for predicted clause type: {clause_type}")
        except KeyError:
            logging.warning(f"Missing key in prediction: {pred}")
            continue
    
    clause_accuracy = sum(1 for t, p in zip(clause_true, clause_pred) if t == p) / len(clause_true) if clause_true else 0
    if clause_true and embedder:
        clause_semantic_similarity = sum(util.cos_sim(embedder.encode([t]), embedder.encode([p]))[0][0].item() for t, p in zip(clause_true, clause_pred)) / len(clause_true)
    else:
        clause_semantic_similarity = 0
    risk_precision, risk_recall, risk_f1, _ = precision_recall_fscore_support(risk_true, risk_pred, average='weighted', zero_division=0)
    risk_accuracy = accuracy_score(risk_true, risk_pred) if risk_true else 0
    
    if scorer:
        rouge1_f1 = sum(scorer.score(t, p)['rouge1'].fmeasure for t, p in zip(summary_true, summary_pred)) / len(summary_true) if summary_true else 0
        rougeL_f1 = sum(scorer.score(t, p)['rougeL'].fmeasure for t, p in zip(summary_true, summary_pred)) / len(summary_true) if summary_true else 0
    else:
        rouge1_f1 = 0
        rougeL_f1 = 0
    
    if summary_true and embedder:
        summary_semantic = sum(util.cos_sim(embedder.encode([t]), embedder.encode([p]))[0][0].item() for t, p in zip(summary_true, summary_pred)) / len(summary_true)
    else:
        summary_semantic = 0
    
    unique_labels = sorted(set(risk_true + risk_pred))
    risk_per_class = precision_recall_fscore_support(risk_true, risk_pred, labels=unique_labels, zero_division=0)
    per_class_metrics = {
        'labels': unique_labels,
        'precision': risk_per_class[0].tolist(),
        'recall': risk_per_class[1].tolist(),
        'f1': risk_per_class[2].tolist()
    }
    
    return {
        'clause_accuracy': clause_accuracy,
        'clause_semantic_similarity': clause_semantic_similarity,
        'risk_precision': risk_precision,
        'risk_recall': risk_recall,
        'risk_f1': risk_f1,
        'risk_accuracy': risk_accuracy,
        'rouge1_f1': rouge1_f1,
        'rougeL_f1': rougeL_f1,
        'summary_semantic': summary_semantic,
        'risk_per_class': per_class_metrics
    }

# Clause details extraction
def extract_clause_details(clause_text, clause_label):
    clause_text = clean_text(clause_text)
    clause_type = clause_type_mapping.get(clause_label, clause_label + " Clause")
    parties = []
    action = ""
    obj = ""
    condition = ""
    entities = []
    
    if not clause_text:
        return {
            "type": clause_type,
            "text": "No clause text extracted.",
            "parties": parties,
            "action": action,
            "object": obj,
            "condition": condition,
            "entities": entities
        }
    
    sentences = sent_tokenize(clause_text)
    for sentence in sentences:
        if "party" in sentence.lower():
            parties.extend(re.findall(r'Party [A-Z]', sentence))
        if clause_label == "Governing Law":
            matches = re.search(r'(?:governed by|law of)\s+([A-Za-z\s,]+)', sentence, re.IGNORECASE)
            if matches:
                obj = matches.group(1).strip()
        if clause_label in ["Non-Compete", "Exclusivity", "No-Solicit Of Customers", "No-Solicit Of Employees"]:
            if "restrict" in sentence.lower() or "prohibit" in sentence.lower():
                action = "restrict"
                obj = re.search(r'(?:restrict|prohibit)\s+([a-z\s]+)', sentence, re.IGNORECASE)
                obj = obj.group(1).strip() if obj else ""
        if "if" in sentence.lower() or "provided that" in sentence.lower():
            condition = sentence
        entities.extend(re.findall(r'[A-Z][a-z]+\s[A-Z][a-z]+', sentence))
    
    return {
        "type": clause_type,
        "text": clause_text,
        "parties": list(set(parties)),
        "action": action,
        "object": obj,
        "condition": condition,
        "entities": list(set(entities))
    }

# Convert structured results to table
def results_to_table(structured_results):
    data = []
    for result in structured_results:
        try:
            clause = result['clause']
            risk = result['risk']
            summary = result['summary']
            data.append({
                'Clause Type': clause['type'],
                'Parties': ', '.join(clause['parties']),
                'Action': clause['action'],
                'Object': clause['object'],
                'Condition': clause['condition'],
                'Entities': ', '.join(clause['entities']),
                'Risk Level': risk['level'],
                'Risk Issue': risk['issue'],
                'Risk Explanation': risk['explanation'],
                'Summary': summary['text'],
                'Suggested Action': summary['suggested_action']
            })
        except KeyError as e:
            logging.error(f"KeyError in results_to_table for result: {result}, error: {e}")
            continue
    return pd.DataFrame(data)

# Prediction function with preprocessing
def run_prediction(question_texts, context_text, model_path):
    max_seq_length = 512
    doc_stride = 256
    n_best_size = 1
    max_query_length = 64
    max_answer_length = 512
    do_lower_case = model_path == "alex-apostolo/legal-bert-base-cuad"
    null_score_diff_threshold = 0.0

    context_text = clean_text(context_text)
    clauses = segment_clauses(context_text)
    context_text = ' '.join(clauses)

    logging.debug(f"Starting prediction with {len(question_texts)} questions and context length {len(context_text)} using model {model_path}")

    def to_list(tensor):
        return tensor.detach().cpu().tolist()

    try:
        config_class, model_class, tokenizer_class = (
            AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer)
        config = config_class.from_pretrained(model_path)
        tokenizer = tokenizer_class.from_pretrained(
            model_path, do_lower_case=do_lower_case, use_fast=False)
        model = model_class.from_pretrained(model_path, config=config)

        model.to(device)
        logging.info(f"Loaded model {model_path} on device {device}")
    except Exception as e:
        logging.error(f"Failed to load model or tokenizer: {e}", exc_info=True)
        raise

    processor = SquadV2Processor()
    examples = []

    for i, question_text in enumerate(question_texts):
        if not isinstance(question_text, str) or not question_text.strip():
            logging.warning(f"Invalid question at index {i}: {question_text}")
            continue
        example = SquadExample(
            qas_id=str(i),
            question_text=question_text,
            context_text=context_text,
            answer_text=None,
            start_position_character=None,
            title="Predict",
            answers=None,
        )
        examples.append(example)
        logging.debug(f"Created example for question {i}: {question_text[:50]}...")

    if not examples:
        logging.error("No valid examples created for prediction")
        return {}

    try:
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
            return_dataset="pt",
            threads=1,
        )
        logging.debug(f"Converted {len(examples)} examples to {len(features)} features")
    except Exception as e:
        logging.error(f"Error converting examples to features: {e}", exc_info=True)
        raise

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

    all_results = []

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            logging.debug(f"Input shapes: input_ids={inputs['input_ids'].shape}, "
                         f"attention_mask={inputs['attention_mask'].shape}, "
                         f"token_type_ids={inputs['token_type_ids'].shape}")

            try:
                outputs = model(**inputs)
            except Exception as e:
                logging.error(f"Error during model inference: {e}", exc_info=True)
                raise

            for i, example_index in enumerate(batch[3]):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs.to_tuple()]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    try:
        final_predictions = compute_predictions_logits(
            all_examples=examples,
            all_features=features,
            all_results=all_results,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            do_lower_case=do_lower_case,
            output_prediction_file=None,
            output_nbest_file=None,
            output_null_log_odds_file=None,
            verbose_logging=False,
            version_2_with_negative=True,
            null_score_diff_threshold=null_score_diff_threshold,
            tokenizer=tokenizer
        )
        logging.debug(f"Generated predictions: {final_predictions}")
    except Exception as e:
        logging.error(f"Error computing predictions: {e}", exc_info=True)
        raise

    cleaned_predictions = {k: v if isinstance(v, str) else "" for k, v in final_predictions.items()}
    return cleaned_predictions

# Initialize session state
if 'contract' not in st.session_state:
    st.session_state['contract'] = ''
    st.session_state['predictions'] = {}
    st.session_state['structured_results'] = []
    st.session_state['docx_file'] = []
    st.session_state['classifier_type'] = 'LogisticRegression'
    st.session_state['summarization_model'] = 'T5'
    st.session_state['extraction_model'] = 'roberta-base'

def read_pdf(file):
    allpages = ''
    try:
        with pdfplumber.open(file) as pdfReader:
            for page in pdfReader.pages:
                text = page.extract_text(layout=True)
                if not text:
                    doc = fitz.open(stream=file.read(), filetype="pdf")
                    page = doc[page._page_number - 1]
                    text = page.get_text("text")
                    if not text.strip():
                        text = ""  # Skip OCR if Tesseract is unavailable
                page_layout = page.extract_words()
                text = ' '.join(word['text'] for word in page_layout if word['top'] > 50 and word['bottom'] < page.height - 50)
                allpages += text + ' '
        return clean_text(allpages)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        logging.error(f"Error reading PDF: {e}")
        return ''

def show_pdf(file):
    try:
        file.seek(0)
        base64_pdf = base64.b64encode(file.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")
        logging.error(f"Error displaying PDF: {e}")

# Main interface with tabs
st.markdown('##### NLP as a service based on **Contract Understanding Atticus Dataset** (CUAD)')
with st.expander("Additional information about the project"):
    st.write("Contract Understanding Atticus Dataset (CUAD) v1 is a corpus of 13,000+ labels in 510 commercial legal contracts that have been manually labeled under the supervision of experienced lawyers to identify 41 types of legal clauses that are considered important in contract review in connection with a corporate transaction, including mergers & acquisitions, etc.")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Upload Document", "Single Clause Query", "Full Analysis", "Visualizations"])

# Tab 1: Upload Document
with tab1:
    st.markdown("##### Upload an external document")
    classifier_type = st.selectbox("Select Classifier", ["LogisticRegression", "RandomForest", "XGBoost"], key="classifier_select")
    st.session_state.classifier_type = classifier_type
    summarization_model = st.selectbox("Select Summarization Model", ["T5", "DistilBART"], key="summarization_model_select")
    st.session_state.summarization_model = summarization_model
    extraction_model = st.selectbox("Select Extraction Model", ["RoBERTa-base", "Legal-BERT-base"], key="extraction_model_select")
    st.session_state.extraction_model = "roberta-base" if extraction_model == "RoBERTa-base" else "alex-apostolo/legal-bert-base-cuad"
    use_synthetic_data = st.checkbox("Use Synthetic Ground Truth for Testing", key="synthetic_data")
    docx_file = st.file_uploader("Upload File", type=['txt', 'docx', 'pdf'], key="uploader")
    ground_truth_file = st.file_uploader("Upload Ground Truth JSON (Optional)", type=['json'], key="ground_truth_uploader")
    Upload_Button = st.button("Upload", key="upload_button")
    if Upload_Button and docx_file is not None:
        with st.spinner("Loading..."):
            try:
                if docx_file.type == "text/plain":
                    raw_text = str(docx_file.read(), "utf-8")
                    st.session_state.contract = clean_text(raw_text)
                    st.session_state.docx_file = None
                    logging.debug(f"Uploaded text file: {raw_text[:100]}...")
                elif docx_file.type == "application/pdf":
                    raw_text = read_pdf(docx_file)
                    st.session_state.contract = raw_text
                    st.session_state.docx_file = docx_file
                    logging.debug(f"Uploaded PDF file, extracted text: {raw_text[:100]}...")
                elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    raw_text = docx2txt.process(docx_file)
                    st.session_state.contract = clean_text(raw_text)
                    st.session_state.docx_file = None
                    logging.debug(f"Uploaded DOCX file, extracted text: {raw_text[:100]}...")
                st.success("File uploaded successfully!")
                
                if use_synthetic_data:
                    st.session_state.ground_truth = {
                        "annotations": [
                            {"clause": {"type": "Non-Compete", "text": "Party A shall not compete with Party B for 2 years."}, "risk": {"level": "High"}, "summary": {"text": "Restricts competition."}},
                            {"clause": {"type": "Exclusivity", "text": "Party A agrees to exclusively deal with Party B."}, "risk": {"level": "Medium"}, "summary": {"text": "Exclusive terms."}},
                            {"clause": {"type": "Governing Law", "text": "This contract is governed by California law."}, "risk": {"level": "Low"}, "summary": {"text": "Specifies governing law."}},
                            {"clause": {"type": "Termination For Convenience", "text": "Either party may terminate with 30 days notice."}, "risk": {"level": "Medium"}, "summary": {"text": "Allows termination."}},
                            {"clause": {"type": "License Grant", "text": "Party B grants Party A a non-exclusive license."}, "risk": {"level": "Low"}, "summary": {"text": "Grants license."}},
                            {"clause": {"type": "Uncapped Liability", "text": "Party A is liable without cap for breaches."}, "risk": {"level": "High"}, "summary": {"text": "Uncapped liability."}},
                            {"clause": {"type": "Audit Rights", "text": "Party B may audit Party A's records."}, "risk": {"level": "Medium"}, "summary": {"text": "Allows audits."}},
                            {"clause": {"type": "Insurance", "text": "Party A must maintain insurance."}, "risk": {"level": "Low"}, "summary": {"text": "Requires insurance."}},
                            {"clause": {"type": "Non-Disparagement", "text": "Parties shall not disparage each other."}, "risk": {"level": "Medium"}, "summary": {"text": "Prohibits disparagement."}},
                            {"clause": {"type": "Change Of Control", "text": "Notice required for change of control."}, "risk": {"level": "Medium"}, "summary": {"text": "Requires notice."}}
                        ]
                    }
                    st.success("Synthetic ground truth loaded successfully!")
                    logging.debug(f"Synthetic ground truth loaded: {json.dumps(st.session_state.ground_truth, indent=2)[:100]}...")
                elif ground_truth_file is not None:
                    try:
                        ground_truth = json.load(ground_truth_file)
                        st.session_state.ground_truth = ground_truth
                        st.success("Ground truth JSON uploaded successfully!")
                        logging.debug(f"Ground truth JSON loaded: {json.dumps(ground_truth, indent=2)[:100]}...")
                    except Exception as e:
                        st.error(f"Error processing ground truth file: {e}")
                        logging.error(f"Error processing ground truth file: {e}")
                
                annotations = st.session_state.ground_truth.get('annotations', [])
                st.write(f"Number of annotations: {len(annotations)}")
                risk_counts = pd.Series([ann['risk']['level'] for ann in annotations]).value_counts()
                st.write("Risk distribution:", risk_counts.to_dict())
                
                clf, embedder, label_encoder = train_risk_classifier(classifier_type)
                if clf is None:
                    st.error("Classifier failed to initialize. Check logs for details.")
                else:
                    st.session_state.clf = clf
                    st.session_state.embedder = embedder
                    st.session_state.label_encoder = label_encoder
                    st.success(f"{classifier_type} risk classifier trained successfully!")
                    logging.debug(f"Stored classifier: {type(clf).__name__}")
            except Exception as e:
                st.error(f"Error processing file: {e}")
                logging.error(f"Error processing file: {e}")

    if st.session_state.get('docx_file') and st.session_state.docx_file.type == "application/pdf":
        with st.expander("Show PDF preview"):
            try:
                docx_file = st.session_state.docx_file
                docx_file.seek(0)
                base64_pdf = base64.b64encode(docx_file.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying PDF preview: {e}")
                logging.error(f"Error displaying PDF preview: {e}")

# Tab 2: Single Clause Query
with tab2:
    st.markdown("##### Choose one of the 41 elements of the contract")
    selected_question = st.selectbox('Query', questions, key="query_select")
    selected_index = questions.index(selected_question)
    question_set = [selected_question]
    Run_Button = st.button("Run the selected query on the contract", key="run_query_button")
    contract = st.session_state.contract
    clf = st.session_state.get('clf')
    embedder = st.session_state.get('embedder')
    label_encoder = st.session_state.get('label_encoder')
    if Run_Button and contract and question_set:
        with st.spinner("Analyzing..."):
            try:
                model_path = st.session_state.extraction_model
                predictions = run_prediction(question_set, contract, model_path)
                logging.debug(f"Single clause predictions: {predictions}")
                clause_text = predictions.get('0', "")
                clause_label = questions2[selected_index].rstrip(':')
                fallback_used = False
                if not isinstance(clause_text, str) or not clean_text(clause_text):
                    clause_text = fallback_clause_extraction(contract, clause_label)
                    if not clause_text:
                        st.warning(f"No valid text extracted for clause: {clause_label}")
                        logging.warning(f"No valid text extracted for clause: {clause_label}")
                    else:
                        fallback_used = True
                        st.info(f"Fallback extraction used for clause: {clause_label} (limited accuracy)")
                        logging.info(f"Used fallback extraction for clause: {clause_label}, Text: {clause_text[:50]}")
                clause_details = extract_clause_details(clause_text, clause_label)
                risk_info = detect_risk(clause_text, clause_label, clf, embedder, label_encoder)
                summary_info = summarize_clause(clause_text, clause_label)
                result = {
                    "clause": clause_details,
                    "risk": risk_info,
                    "summary": summary_info
                }
                st.json(result)
            except Exception as e:
                st.error(f"Error analyzing contract: {e}")
                logging.error(f"Error analyzing contract: {e}")

# Tab 3: Full Analysis
with tab3:
    st.markdown("##### Run a complete analysis and download the results")
    Save_Button = st.button("Run a complete analysis", key="full_analysis_button")
    if Save_Button and st.session_state.contract:
        with st.spinner("Analyzing..."):
            try:
                clf = st.session_state.get('clf')
                embedder = st.session_state.get('embedder')
                label_encoder = st.session_state.get('label_encoder')
                if clf is None or embedder is None or label_encoder is None:
                    st.error("Classifier not initialized. Please upload a ground truth file to train the classifier.")
                    logging.error("Classifier not initialized in full analysis.")
                else:
                    model_path = st.session_state.extraction_model
                    predictions = run_prediction(questions, st.session_state.contract, model_path)
                    logging.debug(f"Full analysis predictions: {predictions}")
                    structured_results = []
                    for i in range(len(questions)):
                        clause_text = predictions.get(str(i), "")
                        clause_label = questions2[i].rstrip(':')
                        fallback_used = False
                        if not isinstance(clause_text, str) or not clean_text(clause_text):
                            clause_text = fallback_clause_extraction(st.session_state.contract, clause_label)
                            if not clause_text:
                                st.warning(f"No valid text extracted for clause: {clause_label}")
                                logging.warning(f"No valid text extracted for clause: {clause_label}")
                                continue
                            else:
                                fallback_used = True
                                st.info(f"Fallback extraction used for clause: {clause_label} (limited accuracy)")
                                logging.info(f"Used fallback extraction for clause: {clause_label}, Text: {clause_text[:50]}")
                        clause_details = extract_clause_details(clause_text, clause_label)
                        risk_info = detect_risk(clause_text, clause_label, clf, embedder, label_encoder)
                        summary_info = summarize_clause(clause_text, clause_label)
                        structured_results.append({
                            "clause": clause_details,
                            "risk": risk_info,
                            "summary": summary_info
                        })
                    st.session_state.structured_results = structured_results
                    results_df = results_to_table(structured_results)
                    results_df.to_csv(os.path.join(temp_dir, f'results_{st.session_state.classifier_type}_{st.session_state.extraction_model.replace("/", "_")}.csv'), index=False)
                    
                    if 'ground_truth' in st.session_state and st.session_state.ground_truth:
                        try:
                            ground_truth_annotations = st.session_state.ground_truth.get('annotations', [])
                            if not ground_truth_annotations:
                                st.warning("Ground truth JSON does not contain 'annotations' key.")
                                logging.warning("Ground truth JSON does not contain 'annotations' key.")
                            else:
                                eval_metrics = evaluate_results(structured_results, ground_truth_annotations, embedder)
                                st.session_state.eval_metrics = eval_metrics
                                st.success("Analysis and evaluation completed!")
                        except Exception as e:
                            st.error(f"Error during evaluation: {e}")
                            logging.error(f"Error during evaluation: {e}, Type: {type(e).__name__}, Line: {e.__traceback__.tb_lineno}")
                    else:
                        st.success("Analysis completed! No ground truth data provided for evaluation.")
                    
                    with st.expander("Debug: Raw Predictions"):
                        st.write(f"Classifier: {st.session_state.classifier_type}, Extraction Model: {st.session_state.extraction_model}")
                        debug_df = pd.DataFrame([
                            {
                                "Clause Type": r["clause"]["type"],
                                "Clause Text": r["clause"]["text"][:50] + "..." if len(r["clause"]["text"]) > 50 else r["clause"]["text"],
                                "Risk Level": r["risk"]["level"]
                            } for r in structured_results
                        ])
                        st.dataframe(debug_df)
            except Exception as e:
                st.error(f"Error running complete analysis: {e}")
                logging.error(f"Error running complete analysis: {e}, Type: {type(e).__name__}, Line: {e.__traceback__.tb_lineno}")

    if st.session_state.structured_results:
        with open(os.path.join(temp_dir, f'results_{st.session_state.classifier_type}_{st.session_state.extraction_model.replace("/", "_")}.csv'), 'rb') as f:
            st.download_button(
                f'Download results as CSV ({st.session_state.classifier_type}, {st.session_state.extraction_model})',
                f,
                file_name=f'results_{st.session_state.classifier_type}_{st.session_state.extraction_model.replace("/", "_")}.csv',
                mime='text/csv'
            )
        with st.expander("Show structured results"):
            st.dataframe(results_to_table(st.session_state.structured_results))
        
        if 'eval_metrics' in st.session_state and st.session_state.eval_metrics:
            st.subheader("Evaluation Metrics")
            metrics = st.session_state.eval_metrics
            st.write(f"**Clause Extraction Accuracy**: {metrics['clause_accuracy']:.2%}")
            st.write(f"**Clause Semantic Similarity**: {metrics['clause_semantic_similarity']:.2%}")
            st.write(f"**Risk Detection Precision**: {metrics['risk_precision']:.2%}")
            st.write(f"**Risk Detection Recall**: {metrics['risk_recall']:.2%}")
            st.write(f"**Risk Detection F1-Score**: {metrics['risk_f1']:.2%}")
            st.write(f"**Risk Detection Accuracy**: {metrics['risk_accuracy']:.2%}")
            st.write(f"**Summary ROUGE-1 F1**: {metrics['rouge1_f1']:.2%}")
            st.write(f"**Summary ROUGE-L F1**: {metrics['rougeL_f1']:.2%}")
            st.write(f"**Summary Semantic Similarity**: {metrics['summary_semantic']:.2%}")
            st.write("**Per-Class Risk Metrics**:")
            per_class = metrics['risk_per_class']
            per_class_df = pd.DataFrame({
                'Label': per_class['labels'],
                'Precision': [f"{x:.2%}" for x in per_class['precision']],
                'Recall': [f"{x:.2%}" for x in per_class['recall']],
                'F1-Score': [f"{x:.2%}" for x in per_class['f1']]
            })
            st.dataframe(per_class_df)

# Tab 4: Visualizations
with tab4:
    st.markdown("##### Contract Clause Visualizations")
    if not st.session_state.structured_results:
        st.warning("No clauses found for visualization, please run a full analysis in the 'Full Analysis' tab.")
    else:
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "Low", "Medium", "High"], key="risk_filter_viz")
        filtered_results = st.session_state.structured_results if risk_filter == "All" else [
            r for r in st.session_state.structured_results if r["risk"]["level"] == risk_filter
        ]

        if not filtered_results:
            st.warning(f"No clauses found with {risk_filter} risk level.")
        else:
            risk_score_map = {"Low": 1, "Medium": 2, "High": 3, "Unknown": 0}
            color_map = {0: "#757575", 1: "#4CAF50", 2: "#FFC107", 3: "#F44336"}
            border_color_map = {0: "#616161", 1: "#388E3C", 2: "#FFA000", 3: "#D32F2F"}
            labels = [result["clause"]["type"] for result in filtered_results]
            risk_scores = [risk_score_map[result["risk"]["level"]] for result in filtered_results]
            background_colors = [color_map[score] for score in risk_scores]
            border_colors = [border_color_map[score] for score in risk_scores]

            st.subheader("Risk Level Bar Chart")
            st.markdown("Visualizing risk levels across contract clauses (Low: Green, Medium: Yellow, High: Red).")
            df = pd.DataFrame({"Clause Type": labels, "Risk Score": risk_scores})
            if df.empty:
                st.warning("No clauses available for the selected risk level.")
            else:
                fig_bar = px.bar(df, x="Clause Type", y="Risk Score", color="Risk Score",
                                 color_continuous_scale=["#757575", "#4CAF50", "#FFC107", "#F44336"],
                                 labels={"Risk Score": "Risk Level"},
                                 height=500)
                fig_bar.update_layout(
                    xaxis_title="Clause Type",
                    yaxis_title="Risk Level",
                    yaxis=dict(
                        tickvals=[0, 1, 2, 3],
                        ticktext=["Unknown", "Low", "Medium", "High"],
                        range=[-0.5, 3.5],
                        tickfont=dict(size=14),
                    ),
                    xaxis=dict(
                        tickangle=45,
                        tickfont=dict(size=10),
                        automargin=True
                    ),
                    showlegend=False,
                    margin=dict(b=150)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("Risk Level Treemap")
            st.markdown("Visualizing clause distribution by risk level, showing hierarchical structure.")
            df_treemap = pd.DataFrame({
                "Clause Type": labels,
                "Risk Level": [result["risk"]["level"] for result in filtered_results],
                "Count": [1] * len(labels)
            })
            fig_treemap = px.treemap(df_treemap, path=["Risk Level", "Clause Type"], values="Count",
                                     color="Risk Level",
                                     color_discrete_map={"Unknown": "#757575", "Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"})
            fig_treemap.update_layout(height=400)
            st.plotly_chart(fig_treemap, use_container_width=True)

            st.subheader("Risk Level Pie Chart")
            st.markdown("Showing proportion of clauses by risk level (Low, Medium, High).")
            risk_counts = pd.Series([result["risk"]["level"] for result in filtered_results]).value_counts()
            df_pie = pd.DataFrame({
                "Risk Level": risk_counts.index,
                "Count": risk_counts.values
            })
            fig_pie = px.pie(df_pie, names="Risk Level", values="Count",
                             color="Risk Level",
                             color_discrete_map={"Unknown": "#757575", "Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"})
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("Risk Level Confusion Matrix")
            st.markdown("Visualizing prediction errors for risk levels.")
            if 'eval_metrics' in st.session_state and st.session_state.eval_metrics and 'ground_truth' in st.session_state:
                ground_truth_annotations = st.session_state.ground_truth.get('annotations', [])
                clause_true = []
                risk_true = []
                risk_pred = []
                for pred in st.session_state.structured_results:
                    try:
                        clause_type = pred['clause']['type']
                        normalized_clause_type = clause_type.replace(" Clause", "")
                        clause_text = clean_text(pred['clause']['text'] if 'text' in pred['clause'] else "")
                        if not clause_text:
                            continue
                        best_gt = None
                        best_ratio = 0
                        for gt in ground_truth_annotations:
                            gt_clause_type = gt['clause']['type']
                            gt_embedding = embedder.encode(gt_clause_type)
                            pred_embedding = embedder.encode(clause_type)
                            norm_embedding = embedder.encode(normalized_clause_type)
                            match_ratio1 = util.cos_sim(gt_embedding, pred_embedding)[0][0].item()
                            match_ratio2 = util.cos_sim(gt_embedding, norm_embedding)[0][0].item()
                            match_ratio = max(match_ratio1, match_ratio2)
                            if match_ratio > 0.8:
                                if match_ratio > best_ratio:
                                    best_ratio = match_ratio
                                    best_gt = gt
                        if best_gt:
                            clause_true.append(clean_text(best_gt['clause']['text']))
                            risk_true.append(best_gt['risk']['level'])
                            risk_pred.append(pred['risk']['level'])
                    except KeyError:
                        continue
                if risk_true and risk_pred:
                    unique_labels = sorted(set(risk_true + risk_pred))
                    cm = confusion_matrix(risk_true, risk_pred, labels=unique_labels)
                    fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="True", color="Count"),
                                       x=unique_labels, y=unique_labels,
                                       color_continuous_scale='Blues')
                    fig_cm.update_layout(height=400)
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    st.warning("No matching clauses for confusion matrix.")

            st.subheader("Clause Details")
            for result in filtered_results:
                st.write(f"**{result['clause']['type']}**: {result['risk']['level']} (Score: {risk_score_map[result['risk']['level']]})")

            if 'eval_metrics' in st.session_state and st.session_state.eval_metrics:
                st.subheader("Evaluation Metrics Bar Chart")
                st.markdown("Visualizing performance metrics for clause extraction, risk detection, and summarization.")
                metrics = st.session_state.eval_metrics
                metric_names = ["Clause Accuracy", "Clause Semantic Similarity", "Risk Precision", "Risk Recall", "Risk F1", "Risk Accuracy", "ROUGE-1 F1", "ROUGE-L F1", "Summary Semantic"]
                metric_values = [
                    metrics['clause_accuracy'],
                    metrics['clause_semantic_similarity'],
                    metrics['risk_precision'],
                    metrics['risk_recall'],
                    metrics['risk_f1'],
                    metrics['risk_accuracy'],
                    metrics['rouge1_f1'],
                    metrics['rougeL_f1'],
                    metrics['summary_semantic']
                ]
                df_metrics = pd.DataFrame({
                    "Metric": metric_names,
                    "Score": metric_values
                })
                fig_metrics = px.bar(df_metrics, x="Metric", y="Score",
                                     color_discrete_sequence=["#4CAF50"] * 6 + ["#FFC107"] * 3,
                                     height=400)
                fig_metrics.update_layout(
                    xaxis_title="Metric",
                    yaxis_title="Score",
                    yaxis=dict(
                        range=[0, 1],
                        tickformat=".0%",
                        tickfont=dict(size=14),
                    ),
                    xaxis=dict(
                        tickangle=45,
                        tickfont=dict(size=10),
                        automargin=True
                    ),
                    showlegend=False,
                    margin=dict(b=150)
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
