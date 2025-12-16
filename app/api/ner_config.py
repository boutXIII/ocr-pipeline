# ## Personal Information
# - Date of birth
# - Age
# - Gender
# - Last name
# - Occupation
# - Education level
# - Phone number
# - Email
# - Street address
# - City
# - Country
# - Postcode
# - User name
# - Password
# - Tax ID
# - License plate
# - CVV
# - Bank routing number
# - Account number
# - SWIFT BIC
# - Biometric identifier
# - Device identifier
# - Location

# ## Financial Information
# - Account number
# - Bank routing number
# - SWIFT BIC
# - CVV
# - Tax ID
# - API key

# ## Health and Medical Information
# - Blood type
# - Biometric identifier
# - Organ
# - Diseases symptom
# - Diagnostics
# - Preventive medicine
# - Treatment
# - Surgery
# - Drug chemical
# - Medical device technique
# - Personal care

# ## Online and Web-related Information
# - URL
# - IP address
# - Email
# - User name
# - API key

# ## Professional Information
# - Occupation
# - Skill
# - Organization
# - Company name

# ## Location Information
# - City
# - Country
# - Postcode
# - Street address
# - Location

# ## Time-Related Information
# - Date
# - Date time

# ## Miscellaneous
# - Event
# - Miscellaneous

DOCUMENT_CLASSES = {
    "FACT_MEDECINE_DOUCE": [
        "person",
        "date",
        "amount",
        "patient",
        "speciality",
        "social_security_number",
    ],
    "RIB": [
        "account_holder",
        "iban_number",
        "bic_code",
        "bank_account_key",
        "bank_account_number",
        "bank_code",
        "bank_countrer_code",
    ],
    # "ATTESTATION": [
    #     "person",
    #     "date",
    # ],
}
