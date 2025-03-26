!pip install flask flask-ngrok pdfplumber python-docx pytesseract pillow google-generativeai pandas
!apt-get install -y poppler-utils tesseract-ocr
!pip install pytesseract pdf2image pdfplumber openai spacy scikit-learn python-docx
!apt-get install -y poppler-utils  # Required for PDF rendering
!pip install PyMuPDF
!pip install pyngrok


import os
import json
import pandas as pd
import pdfplumber
from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
from docx import Document
from PIL import Image
import pytesseract
import google.generativeai as genai
from pdf2image import convert_from_path
import email
import traceback


from pyngrok import ngrok

# Ensure ngrok tunnel is connected to the correct port
ngrok.set_auth_token("2urJkErjBMlSyitMdaVLRjGx13G_5bnmGbRwif2f3rkHnAqyt")  # Add your ngrok token if needed
public_url = ngrok.connect(5000).public_url
print(f"🔥 Public URL: {public_url}")

# ✅ Google Gemini API Configuration
GEMINI_API_KEY = "AIzaSyBrCKlN7ijSeOpTX6dbcMRlgYpwfQIhCEA"
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Flask app setup

from flask import Flask, render_template

app = Flask(__name__)
run_with_ngrok(app)





# ✅ Upload folder and allowed file types
UPLOAD_FOLDER = '/content/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'eml'}

# ✅ Duplicate patterns
DUPLICATE_PATTERNS = [
    "thank you", "thanks", "good work", "well done",
    "appreciate it", "noted", "acknowledged", "approved"
]

# ✅ Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ✅ Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path}: {e}")
    return text.strip()

# ✅ Function to extract text from scanned PDF using OCR
def extract_text_from_scanned_pdf(pdf_path):
    text = ""
    try:
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
    except Exception as e:
        print(f"❌ Error with OCR on {pdf_path}: {e}")
    return text.strip()

# ✅ Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"❌ Error extracting text from {docx_path}: {e}")
    return text.strip()

# ✅ Function to extract text from EML files
def extract_text_from_eml(eml_path):
    text = ""
    try:
        with open(eml_path, "rb") as eml_file:
            msg = email.message_from_binary_file(eml_file)
            for part in msg.walk():
                if part.get_content_type() in ["text/plain", "text/html"]:
                    text += part.get_payload(decode=True).decode("utf-8", errors="ignore") + "\n"
    except Exception as e:
        print(f"❌ Error extracting text from {eml_path}: {e}")
    return text.strip()

# ✅ Function to detect duplicate emails

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
# ✅ Gemini classification function
# def classify_email(text):
#     try:
#         model = genai.GenerativeModel('gemini-1.5-pro-latest')
#         response = model.generate_content(
#             "Analyze the following email and determine its intent.\n"
#             "Classify it into request type, sub-request type, and provide a confidence score.\n\n"
#             f"Email Content:\n{text}"
#         ).text.strip()

#         return response
#     except Exception as e:
#         print(f"❌ Error with Gemini: {e}")
#         return "Unknown classification"


def classify_email(text):
    """Identifies the email intent and classifies it into Request Type, Sub Request Type, and Confidence Score."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')

        response = model.generate_content(
            "Analyze the following email and determine its intent.\n"
            "Then, classify it into one of the given request types and sub-request types strictly from the list below, if its confidence score is low then suggest the next best possible Request Types & Sub Request Types.\n"
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
        # response = model.generate_content(
        #     f"Analyze the following email and extract the following details from this email:\n"
        #     f"- Sender's Name\n"
        #     f"- Email Address\n"
        #     f"- request_type\n"
        #     f"- sub_request_type\n"
        #     f"- primary_intent\n"
        #     f"- deal_name\n"
        #     f"- amount\n"
        #     f"- expiration_date\n"
        #     f"- priority_based_extraction\n{text}"
        #     #f"- duplicate_email_flag\n\n{text}"
        #   )
        response = model.generate_content(
            "Extract details from this email and return only a **valid JSON object**, with no extra text or explanations. "
            "Ensure the output follows this exact format:\n\n"
            "Analyze the following email and extract the required details in JSON format:\n"
            "{\n"
            '  "sender_name": "Full name of sender",\n'
            '  "email_address": "Email of sender",\n'
            '  "request_type": "Primary request category based on sender intent",\n'
            '  "sub_request_type": "More specific request category if applicable",\n'
            '  "primary_intent": "The main actionable request in the email",\n'
            '  "key_fields": {\n'
            '    "deal_name": "Extracted deal name if available",\n'
            '    "amount": "Extracted amount if mentioned",\n'
            '    "expiration_date": "Extracted expiration date if present"\n'
            '  },\n'
            "}\n\n"
            f"Email Content:\n{text}"
        )

        return response.text.strip()

    except Exception as e:
        print(f"❌ Error with Gemini: {e}")
        return {"error": "Extraction failed"}

# ✅ Function to process uploaded files
def process_files(files):
    results = []

    for file in files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        ext = os.path.splitext(file.filename)[1].lower()

        # Extract text based on file type
        if ext == '.pdf':
            text = extract_text_from_pdf(file_path) or extract_text_from_scanned_pdf(file_path)
        elif ext == '.docx':
            text = extract_text_from_docx(file_path)
        elif ext == '.eml':
            text = extract_text_from_eml(file_path)
        else:
            continue

        duplicate_flag = is_duplicate_email(text)


        intent, request_type, sub_request_type,confidence = classify_email(text)

        response = extract_email_details(text)
        try:
          response = response.replace("```","") if "```" in response else response
          response = response.replace("json", "") if "json" in response else response
          extraction_details=json.dumps(json.loads(response), indent=2)
        except Exception as e:
          extraction_details = response

        results.append({
            "Filename": file.filename,
            "Duplicate": "Yes" if duplicate_flag else "No",
            "intent": intent,
            "Request_Type": request_type,
            "Sub_Request": sub_request_type,
            "Confidence": confidence,
            "Extracted Text": extraction_details,

        })

    return results

# ✅ Flask routes
@app.route("/", methods=["GET", "POST"])
def upload_files():
    if request.method == "POST":
        if "files[]" not in request.files:
            return redirect(request.url)

        files = request.files.getlist("files[]")

        if not files or not all(allowed_file(file.filename) for file in files):
            return "Invalid file type. Please upload PDF, DOCX, or EML."

        results = process_files(files)

        # Save results to CSV
        results_df = pd.DataFrame(results)
        csv_filename = "email_results.csv"
        csv_path = os.path.join(UPLOAD_FOLDER, csv_filename)
        results_df.to_csv(csv_path, index=False)

        # results_df = pd.DataFrame(results)
        # csv_path = os.path.join(UPLOAD_FOLDER, "email_results.csv")
        # results_df.to_csv(csv_path, index=False)

        return render_template("results.html", results=results, csv_path=csv_path)

    return render_template("upload.html")

@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

# ✅ Run the app
if __name__ == "__main__":
    app.run()
