import os
import streamlit as st
import pandas as pd
import pickle
import bz2
import json
import re
import nltk
import docx
from PIL import Image
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load pre-trained classifier & vectorizer
with bz2.BZ2File("model_resume_classifier.pkl.bz2", "rb") as f:
    model = pickle.load(f)

with open('vectorizer_resume_classifier.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Database CSV
csv_file = "processed_resumes.csv"
try:
    resume_db = pd.read_csv(csv_file)
except FileNotFoundError:
    resume_db = pd.DataFrame(columns=["Name", "Email", "Phone", "Skills", "Category", "Job Description", "Match Score"])

# Streamlit page configuration
st.set_page_config(page_title="Boresha CV", page_icon=":briefcase:", layout="wide")
# Custom CSS for UI styling
st.markdown(
    """
    <style>
    body {
        background-color: white;
        margin: 0;
        padding: 20px;
        border: 10px solid #046307; /* Emerald Green Border */
    }
    .stApp {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Outer container with Emerald Green Background
st.markdown('<div class="outer-container">', unsafe_allow_html=True)

# Main content container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

image_path = os.path.join("images/WhatsApp Image 2025-02-24 at 04.10.26 ")

# Create two columns
col1, col2 = st.columns([1, 2])  # Adjust width ratio as needed

with col1:
    # Display image in first column
    if os.path.exists(image_path):
        header_img = Image.open(image_path)
        st.image(header_img, use_column_width=True)
    
    #header_img = Image.open(image_path)
    new_size = (int(header_img.width * 0.8), int(header_img.height * 0.8))
    header_img_resized = header_img.resize(new_size)
    st.image(header_img_resized)

    else:
        st.error(f"⚠️ Image not found at: {image_path}. Check the filename and path.")

with col2:
    # Display app information in the second column
    st.markdown(
        """
        ## Welcome to the Resume Parser & Job Matcher App! 📄✨  

        🚀 **Enhance Your Resume Experience**  
        - Ensure your CV is **ATS-friendly** and **well-structured**  
        - Extract key information **instantly**  
        - Get insights on **resume improvement**  
        - Check **job match compatibility**  

        🔍 **Want to know if your CV is ATS-compliant?**  
        Upload your resume here and find out! ✅  
        """
    )

# Create Tabs
tab1, tab2 = st.tabs(["📄 Resume Parser & Classifier", "📝 Job Description Matcher"])


# Function to extract text from PDF/DOCX/PNG/JPEG
def extract_text(file):
    text = ""

    if file.name.endswith('.pdf'):
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text() + " "

    elif file.name.endswith('.docx'):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + " "

    elif file.name.endswith(('.png', '.jpg', '.jpeg')):  # OCR for image-based resumes
        image = Image.open(file)
        text = pytesseract.image_to_string(image)

    return ' '.join(text.split())
st.info("Ensure your CV is **ATS-friendly** and **well-structured**")

# Function to extract Name
def extract_name(text):
    words = text.split()
    return words[0] + " " + words[1] if len(words) > 1 else "Not Found"


# Function to extract Email
def extract_email(text):
    match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return match.group(0) if match else "Not Found"


# Function to extract Phone Number
def extract_phone(text):
    match = re.search(r'\+?\d{10,15}', text)
    return match.group(0) if match else "Not Found"


# Function to extract Skills
skills_list = ["Python", "SQL", "Machine Learning", "Deep Learning", "Data Science", "Tableau", "Excel", "Java", "AWS"]


def extract_skills(text):
    found_skills = [skill for skill in skills_list if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE)]
    return list(set(found_skills)) if found_skills else ["Not Found"]


# Function to predict category
def predict_category(text):
    cleaned_text = ' '.join([word for word in word_tokenize(text) if word.lower() not in stopwords.words('english')])
    vectorized_text = vectorizer.transform([cleaned_text])
    return model.predict(vectorized_text)[0]


# Function to calculate job match score
def calculate_match(resume_skills, job_skills):
    matched_skills = set(resume_skills) & set(job_skills)
    match_percentage = (len(matched_skills) / len(job_skills)) * 100 if job_skills else 0
    return matched_skills, round(match_percentage, 2)


# 📄 Resume Parser Tab
with tab1:
    st.title("📄 Resume Parser & Classifier")
    uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, PNG, JPG)", type=["pdf", "docx", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        text = extract_text(uploaded_file)

        if text:
            name = extract_name(text) or "❌ **Not Found** - Try including your full name at the top."
            email = extract_email(text) or "❌ **Not Found** - Ensure a valid email format (e.g., name@example.com)."
            phone = extract_phone(text) or "❌ **Not Found** - Include a valid phone number with country code."
            skills = extract_skills(text) or ["❌ **Not Found** - Add relevant skills to your CV."]
            category = predict_category(text)

            st.subheader("📌 Extracted Details")
            st.write(f"**👤 Name:** {name}")
            st.write(f"**📧 Email:** {email}")
            st.write(f"**📞 Phone:** {phone}")
            st.write(f"**🛠 Skills:** {', '.join(skills)}")
            st.write(f"**📂 Predicted Category:** {category}")

            parsed_data = {"Name": name, "Email": email, "Phone": phone, "Skills": skills, "Category": category}
            json_data = json.dumps(parsed_data, indent=4)
            st.download_button("📥 Download JSON", data=json_data, file_name="parsed_resume.json", mime="application/json")

            st.subheader("📢 Resume Structuring Advice")
            st.markdown("""
            - Use a **professional email** (avoid casual ones like `coolguy123@gmail.com`).  
            - List **technical and soft skills** separately for better visibility.  
            - Clearly define **work experience with dates** in **reverse chronological order**.  
            - Keep it **concise** (preferably **1-2 pages** for clarity).  
            """)

with tab2:
    st.title("📝 Job Description Matcher")
    job_description = st.text_area("Paste Job Description Here", height=200)

    if st.button("Match Resume to Job Description") and uploaded_file:
        job_skills = extract_skills(job_description)
        matched_skills, match_score = calculate_match(skills, job_skills)

        st.subheader("🔍 Job Match Results")
        st.write(f"**🛠 Job Skills Extracted:** {', '.join(job_skills)}")
        st.write(f"**✅ Matched Skills:** {', '.join(matched_skills)}")
        st.write(f"**📊 Match Score:** {match_score}%")

        # Store results in a new row
        new_data = {
            "Name": name,
            "Email": email,
            "Phone": phone,
            "Skills": ', '.join(skills),
            "Experience": "Not Extracted Yet",
            "Certifications": "Not Extracted Yet",
            "Category": category,
            "Job Description": job_description,
            "Match Score": match_score
        }

        # Convert to DataFrame and append
        new_data_df = pd.DataFrame([new_data])
        resume_db = pd.concat([resume_db, new_data_df], ignore_index=True)

        # Save to CSV
        resume_db.to_csv(csv_file, index=False)
        st.success("✅ Resume data successfully saved!")

# Load existing database or create a new one
csv_file = "processed_resumes.csv"
try:
    resume_db = pd.read_csv(csv_file)
except FileNotFoundError:
    resume_db = pd.DataFrame(columns=["Name", "Email", "Phone", "Skills", "Category", "Job Description", "Match Score"])

# Avoid duplicates by checking for existing email IDs before adding
if not resume_db[resume_db['Email'] == email].empty:
    st.warning("⚠ Resume already exists in the database. Skipping duplicate entry.")
else:
    new_data = {
        "Name": name,
        "Email": email,
        "Phone": phone,
        "Skills": ', '.join(skills),
        "Category": category,
        "Job Description": job_description,
        "Match Score": match_score
    }

    new_data_df = pd.DataFrame([new_data])
    resume_db = pd.concat([resume_db, new_data_df], ignore_index=True)

    # Save to CSV for future model improvements
    resume_db.to_csv(csv_file, index=False)
    st.success("✅ Resume data successfully saved!")

# Preprocess the stored resume data for model improvement
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Apply cleaning to all text fields
resume_db['Job Description'] = resume_db['Job Description'].apply(clean_text)
resume_db['Skills'] = resume_db['Skills'].apply(lambda x: ', '.join([skill.lower() for skill in x.split(',')]))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Load updated dataset
df = pd.read_csv("processed_resumes.csv")

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Job Description"])  # Use job descriptions for training
y = df["Category"]  # Labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train new Random Forest model
new_model = RandomForestClassifier(n_estimators=100, random_state=42)
new_model.fit(X_train, y_train)

# Evaluate model performance
y_pred = new_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"New Model Accuracy: {accuracy * 100:.2f}%")

# Save new model & vectorizer
with open('model_resume_classifier.pkl', 'wb') as f:
    pickle.dump(new_model, f)

with open('vectorizer_resume_classifier.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

#if st.button("🔄 Retrain Model with New Data"):
#    st.warning("⚙ Training model... This may take a few minutes.")
#    retrain_model()  # Call retraining function
#    st.success("✅ Model retrained successfully with updated resume data!")

corrected_category = st.selectbox("Is this the correct category?", ["Yes", "No, let me change"])
if corrected_category == "No, let me change":
    new_category = st.selectbox("Select Correct Category", ["Data Science", "Marketing", "Finance", "Engineering"])
    df.loc[df["Email"] == email, "Category"] = new_category
    df.to_csv(csv_file, index=False)
    st.success("✅ Category updated! This helps improve future predictions.")
