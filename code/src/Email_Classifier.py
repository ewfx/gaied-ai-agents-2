# ✅ Install required libraries
!pip install pytesseract pdf2image pdfplumber openai spacy scikit-learn python-docx
!apt-get install -y poppler-utils  # Required for PDF rendering
!pip install PyMuPDF
!apt-get install -y tesseract-ocr
!apt-get install -y poppler-utils  # For PDF rendering
!pip install pytesseract

import pytesseract
from pdf2image import convert_from_path
import pdfplumber
import os
import pandas as pd
from google.colab import files
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document

import email
import base64
import mimetypes

import fitz  # PyMuPDF for PDF link extraction
import requests
from urllib.parse import urlparse, unquote

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ✅ Replace with your Gemini API key
GEMINI_API_KEY = "AIzaSyAMyauJDTCCB9iP_x5oTQd9jK_guT0p1wo"
genai.configure(api_key=GEMINI_API_KEY)


# ✅ Gemini-powered function to detect irrelevant or duplicate emails
def is_duplicate_email(text):
    """
    Uses Gemini to identify irrelevant emails like thank you, good work, etc.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(
            "Analyze the following email and determine if it is irrelevant, such as:\n"
            "- Thank you emails\n"
            "- Good work emails\n"
            "- Acknowledgment or confirmation emails\n"
            "- Auto-replies or generic responses\n\n"
            "Return 'True' if the email is irrelevant, otherwise return 'False'.\n\n"
            f"Email Content:\n{text}"
        ).text.strip()

        # Ensure the response is properly interpreted
        if "true" in response.lower():
            return True
        elif "false" in response.lower():
            return False
        else:
            print(f"⚠️ Unexpected Gemini response: {response}")
            return False

    except Exception as e:
        print(f"❌ Error with Gemini in irrelevant detection: {e}")
        return False


# ✅ Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a text-based PDF."""
    text = ""
    try:
        print(f"pdf has text\n")
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""  # Handle empty pages gracefully
    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path}: {e}")

    return text.strip()


# ✅ Function to extract text from scanned PDF using OCR
def extract_text_from_scanned_pdf(pdf_path):
    """Extracts text from scanned PDF using OCR."""
    text = ""
    try:
        print(f"pdf has scanned images\n")
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
    except Exception as e:
        print(f"❌ Error with OCR on {pdf_path}: {e}")

    return text.strip()


# ✅ Function to extract text from DOCX files
def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"❌ Error extracting text from {docx_path}: {e}")
    return text.strip()



# ✅ Function to extract text from EML files (including attachments)
def extract_text_from_eml(eml_path):
    """Extracts text from EML files, including attachments."""
    text = ""
    try:
        with open(eml_path, "rb") as eml_file:
            msg = email.message_from_binary_file(eml_file)

            # Extract the email body
            for part in msg.walk():
                content_type = part.get_content_type()

                # Extract plain text or HTML email content
                if content_type == "text/plain" or content_type == "text/html":
                    text += part.get_payload(decode=True).decode("utf-8", errors="ignore") + "\n"

                # Handle attachments
                if part.get_filename():
                    attachment_name = part.get_filename()
                    attachment_data = part.get_payload(decode=True)

                    # Save the attachment temporarily
                    ext = os.path.splitext(attachment_name)[1].lower()
                    attachment_path = f"/content/{attachment_name}"

                    with open(attachment_path, "wb") as f:
                        f.write(attachment_data)

                    # Extract text from the attachment
                    if ext == ".pdf":
                        print(f"attachment is pdf\n")
                        text += f"\n[Attachment: {attachment_name}]\n"
                        pdf_text = extract_text_from_pdf(attachment_path) or extract_text_from_scanned_pdf(attachment_path)
                        text += pdf_text if pdf_text else "[Failed to extract text from PDF]"

                    elif ext == ".docx":
                        print(f"attachment is docx\n")
                        text += f"\n[Attachment: {attachment_name}]\n"
                        docx_text = extract_text_from_docx(attachment_path)
                        text += docx_text if docx_text else "[Failed to extract text from DOCX]"

                    elif ext in ['.jpg', '.jpeg', '.png']:
                        print(f"attachment is img\n")
                        text += f"\n[Attachment: {attachment_name}]\n"
                        image_text = pytesseract.image_to_string(attachment_path)
                        text += image_text if image_text else "[Failed to extract text from image]"

                    else:
                        text += f"\n[Attachment: {attachment_name}] - Unsupported format\n"

    except Exception as e:
        print(f"❌ Error extracting text from {eml_path}: {e}")
    return text.strip()


# ✅ Gemini Classification with Confidence Score
def classify_email(text):
    """Identifies the email intent and classifies it into Request Type, Sub Request Type, and Confidence Score."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        response = model.generate_content(
            "Analyze the following email and determine its intent.\n"
            "Then, classify it into one of the given request types and sub-request types strictly from the list below.\n"
            "Also, provide a confidence score (0-100%) indicating how confident you are in the classification.\n\n"
            "**Request Types & Sub Request Types:**\n"
            "- Adjustment: []\n"
            "- AU Transfer: []\n"
            "- Closing Notice: [Reallocation Fees, Amendment Fees, Reallocation Principal]\n"
            "- Commitment Change: [Cashless Roll, Decrease, Increase]\n"
            "- Fee Payment: [Ongoing Fee, Letter of Credit Fee]\n"
            "- Money Movement - Inbound: [Principal, Interest, Principal + Interest, Principal + Interest + Fee]\n"
            "- Money Movement - Outbound: [Timebound, Foreign Currency]\n\n"
            "Format your response as:\n"
            "Intent: <intent summary>\n"
            "Request Type: <matching category>\n"
            "Sub Request Type: <matching subcategory or 'N/A'>\n"
            "Confidence Score: <confidence in %>\n\n"
            f"Email Content:\n{text}"
        ).text.strip()

        return parse_classification(response)

    except Exception as e:
        print(f"❌ Error with Gemini: {e}")
        return "Unknown", "Unknown", "Unknown", "0%"


# ✅ Enhanced parser to extract the confidence score
def parse_classification(response):
    """Extracts intent, request type, sub-request type, and confidence score."""
    
    valid_categories = {
        "Adjustment": [],
        "AU Transfer": [],
        "Closing Notice": ["Reallocation Fees", "Amendment Fees", "Reallocation Principal"],
        "Commitment Change": ["Cashless Roll", "Decrease", "Increase"],
        "Fee Payment": ["Ongoing Fee", "Letter of Credit Fee"],
        "Money Movement - Inbound": ["Principal", "Interest", "Principal + Interest", "Principal + Interest + Fee"],
        "Money Movement - Outbound": ["Timebound", "Foreign Currency"],
    }

    intent, request_type, sub_request_type, confidence = "Unknown", "Unknown", "Unknown", "0%"

    # Extract intent
    if "Intent:" in response:
        intent_start = response.find("Intent:") + len("Intent:")
        intent_end = response.find("Request Type:")
        intent = response[intent_start:intent_end].strip()

    # Extract request type and sub-request type
    for category, subcategories in valid_categories.items():
        if category in response:
            request_type = category
            for sub in subcategories:
                if sub in response:
                    sub_request_type = sub
                    break
            break

    # Extract confidence score
    if "Confidence Score:" in response:
        conf_start = response.find("Confidence Score:") + len("Confidence Score:")
        confidence = response[conf_start:].strip()

    # Handle missing sub-request type
    if request_type != "Unknown" and sub_request_type == "Unknown":
        sub_request_type = "N/A"

    return intent, request_type, sub_request_type, confidence


# ✅ Extract email details using Gemini
def extract_email_details(text):
    """Extract structured email details including classification, key fields, and primary intent."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(
            f"Analyze the following email and extract the following details from this email:\n"
            f"- Sender's Name\n"
            f"- Email Address\n"
            f"- request_type\n"
            f"- sub_request_type\n"
            f"- primary_intent\n"
            f"- deal_name\n"
            f"- amount\n"
            f"- expiration_date\n"
            f"- priority_based_extraction\n"
            f"- duplicate_email_flag\n\n{text}"
          )

        return response.text

    except Exception as e:
        print(f"❌ Error with Gemini: {e}")
        return {"error": "Extraction failed"}


# ✅ Email processing pipeline with confidence score
def process_emails(files):
    results = []

    for file_path in files:
        print(f"\n🔹 Processing: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            email_text = extract_text_from_pdf(file_path)
            if not email_text:
                email_text = extract_text_from_scanned_pdf(file_path)
        elif ext == ".docx":
            email_text = extract_text_from_docx(file_path)
        elif ext == ".eml":
            email_text = extract_text_from_eml(file_path)
        else:
            print(f"⚠️ Unsupported file type: {ext}")
            continue

        if not email_text:
            print("❌ No text extracted.")
            continue

    
    # Duplicate detection
        print(f"start\n")
        print(f"{email_text}")
        duplicate_flag = is_duplicate_email(email_text)

        intent, request_type, sub_request_type, confidence = classify_email(email_text)
        if not duplicate_flag:
            # Classification
            print(f"not duplicate\n")
            extracted_info = extract_email_details(email_text)
        else:
            print(f"duplicate\n")
            request_type, sub_request_type, confidence = "N/A", "N/A", "0%"
            extracted_info = "N/A"



        print(f"📌 Intent: {intent}")
        print(f"📌 Request Type: {request_type}")
        print(f"📌 Sub-Request: {sub_request_type}")
        print(f"📌 Confidence Score: {confidence}")
        print(f"📌 Duplicate Email: {duplicate_flag}")

        #extracted_info = extract_email_details(email_text)
        print("🔍 Extracted Info:")
        print(extracted_info)



        results.append({
            "File": os.path.basename(file_path),
            "Intent": intent,
            "Duplicate Email": duplicate_flag,
            "Request Type": request_type,
            "Sub Request Type": sub_request_type,
            "Confidence Score": confidence,
            "Extracted Text": email_text,
            "Extracted Info": extracted_info
        })

    return results



# ✅ Upload files
uploaded = files.upload()

# ✅ Extract file paths
file_paths = [f"/content/{file}" for file in uploaded.keys()]

# ✅ Process all files
results = process_emails(file_paths)

# ✅ Export results to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("/content/email_extraction_results.csv", index=False)

print("\n✅ Results saved to /content/email_extraction_results.csv")



