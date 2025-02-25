import streamlit as st
import pandas as pd
import pickle
import json
import re
import nltk
import docx
import bz2
from PIL import Image
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load pre-trained classifier & vectorizer
with bz2.BZ2File('model_resume_classifier.pkl.bz2', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer_resume_classifier.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# New CSV database to store all processed resumes
csv_file = "processed_resumes.csv"
try:
    resume_db = pd.read_csv(csv_file)
except FileNotFoundError:
    resume_db = pd.DataFrame(columns=["Name", "Email", "Phone", "Skills", "Experience", "Certifications", "Category", "Job Description", "Match Score"])

# Streamlit page configuration
st.set_page_config(page_title="Resume Parser & Job Matcher", page_icon=":briefcase:", layout="wide")

# Custom CSS for full-page border and layout
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

# Load header image
header_img = Image.open("C:/Users/HP/PycharmProjects/Resume/WhatsApp Image 2025-02-24 at 04.10.26.jpeg")

# Layout for image and introduction
col1, col2 = st.columns([1, 2])
with col1:
    st.image(header_img, width=300)
with col2:
    st.markdown(
        """
        ## Welcome to Resume Parser & Job Matcher 🎯
        Improve your resume's effectiveness and **ATS compatibility** using our tool.
        - Parse **key resume details** (Name, Email, Phone, Skills, Experience, etc.)
        - **Check ATS compliance** and get **structuring tips**
        - **Match your resume** to a job description and see a **compatibility score**
        - **Get insights** on missing details and how to improve them
        """
    )

# Create Tabs
tab1, tab2 = st.tabs(["📄 Resume Parser & Classifier", "📝 Job Description Matcher"])

# Function to extract text from PDF/DOCX
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
    return ' '.join(text.split())

# Function to extract Name
def extract_name(text):
    words = text.split()
    return words[0] + " " + words[1] if len(words) > 1 else "Not Found - Ensure your full name is at the top of your resume."

# Function to extract Email
def extract_email(text):
    match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return match.group(0) if match else "Not Found - Use a professional email format."

# Function to extract Phone Number
def extract_phone(text):
    match = re.search(r'\+?\d{10,15}', text)
    return match.group(0) if match else "Not Found - Ensure your phone number is correctly formatted."

# Function to extract Skills using a predefined list
skills_list = ["Python", "SQL", "Machine Learning", "Deep Learning", "Data Science", "Tableau", "Excel", "Java", "AWS", "TensorFlow", "Kubernetes"]

def extract_skills(text):
    found_skills = [skill for skill in skills_list if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE)]
    return list(set(found_skills)) if found_skills else ["Not Found - Add technical and soft skills as bullet points."]

# Function to predict category
def predict_category(text):
    cleaned_text = ' '.join([word for word in word_tokenize(text) if word.lower() not in stopwords.words('english')])
    vectorized_text = vectorizer.transform([cleaned_text])
    predicted_category = model.predict(vectorized_text)[0]
    return predicted_category

# Function to calculate job match score
def calculate_match(resume_skills, job_skills):
    matched_skills = set(resume_skills) & set(job_skills)
    match_percentage = (len(matched_skills) / len(job_skills)) * 100 if job_skills else 0
    return matched_skills, round(match_percentage, 2)

# 📄 Resume Parser Tab
with tab1:
    st.title("📄 Resume Parser & Classifier")
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

    if uploaded_file is not None:
        text = extract_text(uploaded_file)
        st.subheader("Extracted Resume Text")
        st.text_area("", text, height=200)

        # Extract parsed details
        name = extract_name(text)
        email = extract_email(text)
        phone = extract_phone(text)
        skills = extract_skills(text)
        category = predict_category(text)

        # Save data to CSV
        new_data = pd.DataFrame([[name, email, phone, skills, "", "", category, "", ""]], 
                                columns=resume_db.columns)
        resume_db = pd.concat([resume_db, new_data], ignore_index=True)
        resume_db.to_csv(csv_file, index=False)

        # Display results
        st.subheader("📌 Extracted Details")
        st.write(f"**👤 Name:** {name}")
        st.write(f"**📧 Email:** {email}")
        st.write(f"**📞 Phone:** {phone}")
        st.write(f"**🛠 Skills:** {', '.join(skills)}")
        st.write(f"**📂 Predicted Category:** {category}")

# 📝 Job Description Matcher Tab
with tab2:
    st.title("📝 Job Description Matcher")
    job_description = st.text_area("Paste Job Description Here", height=200)

    if st.button("Match Resume to Job Description"):
        if uploaded_file is None:
            st.warning("⚠️ Please upload a resume first!")
        elif not job_description.strip():
            st.warning("⚠️ Please enter a job description!")
        else:
            job_skills = extract_skills(job_description)
            matched_skills, match_score = calculate_match(skills, job_skills)
            st.subheader("🔍 Job Match Results")
            st.write(f"**📊 Match Score:** {match_score}%")
