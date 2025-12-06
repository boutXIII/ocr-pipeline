# import json
# from pathlib import Path
# import numpy as np
# import onnxruntime as ort
# from tokenizers import Tokenizer
# import os

from gliner import GLiNER

def ner_gliner(text: str):
    model = GLiNER.from_pretrained("app\models\gliner\gliner_large-v2.5", load_onnx_model=True, load_tokenizer=True, onnx_model_file="model.onnx")

    if not text or len(text.strip()) == 0:
        return []
    
    labels = [
        "person",
        "date",
        "amount",
        "address",
        "postal code",
        "location",
        "patient",
        "speciality",
        "social security number",
        "adeli",
        "siret",
        "bank account", 
    ]
    personal_labels = [
        "name",                       # Full names
        "first name",                 # First names  
        "last name",                  # Last names
        "name medical professional",  # Healthcare provider names
        "dob",                        # Date of birth
        "age",                        # Age information
        "gender",                     # Gender identifiers
        "marital status"              # Marital status
    ]
    
    contact_labels = [
        "email address",          # Email addresses
        "phone number",           # Phone numbers
        "ip address",             # IP addresses
        "url",                    # URLs
        "location address",       # Street addresses
        "location street",        # Street names
        "location city",          # City names
        "location state",         # State/province names
        "location country",       # Country names
        "location zip"            # ZIP/postal codes
    ]

    id_labels = [
        "passport number",       # Passport numbers
        "driver license",        # Driver's license numbers
        "username",              # Usernames
        "password",              # Passwords
        "vehicle id"             # Vehicle IDs
    ]

    healthcare_labels = [
        "condition",                    # Medical conditions
        "medical process",              # Medical procedures
        "drug",                         # Drugs
        "dose",                         # Dosage information
        "blood type",                   # Blood types
        "injury",                       # Injuries
        "organization medical facility",# Healthcare facility names
        "healthcare number",            # Healthcare numbers
        "medical code"                  # Medical codes
    ]

    financial_labels = [
        "person",                 # Full names
        "account",         # Account numbers
        "bank account",           # Bank account numbers
        "routing number",         # Routing numbers
        "credit card",            # Credit card numbers
        "credit card expiration", # Card expiration dates  
        "cvv",                    # CVV/security codes
        "ssn",                    # Social Security Numbers
        "money"                   # Monetary amounts
    ]

    return model.predict_entities(text, labels, threshold=0.8, flat_ner=False)


