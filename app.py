import streamlit as st
import pandas as pd
import pickle
import json
import re
import nltk
import docx
import bz2
import requests
from io import BytesIO
from PIL import Image
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')

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


# Outer container with Emerald Green Background
st.markdown('<div class="outer-container">', unsafe_allow_html=True)

# Main content container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

image_path = "https://github.com/Topester/Boresha-CV/blob/main/WhatsApp%20Image%202025-02-24%20at%2004.10.26.jpeg?raw=true"

# Create two columns
col1, col2 = st.columns([1, 2])  # Adjust width ratio as needed

with col1:
    try:
        # Load image from URL
        response = requests.get(image_path)
        response.raise_for_status()  # Raise error if request fails
        header_img = Image.open(BytesIO(response.content))
        
        # Display original image
        #st.image(header_img, use_column_width=True)

        # Resize image
        new_size = (int(header_img.width * 0.8), int(header_img.height * 0.8))
        header_img_resized = header_img.resize(new_size)
        st.image(header_img_resized)

    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Failed to load image: {e}")
with col2:
    st.markdown(
        """
        ## Welcome to Boresha CV Yako 🎯
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
skills_list = [
    # Programming Languages
    "Python", "SQL", "Java", "C++", "C#", "JavaScript", "R", "Ruby", "Swift", "Kotlin", "Go", "Dart", "Perl", "PHP", 
    "Rust", "Scala", "TypeScript", "Shell Scripting", "Bash", "MATLAB", "Objective-C",

    # Data Science & Machine Learning
    "Machine Learning", "Deep Learning", "Artificial Intelligence", "Data Science", "Natural Language Processing",
    "Computer Vision", "Reinforcement Learning", "Data Analytics", "Big Data", "Predictive Modeling",
    "Statistical Analysis", "Feature Engineering", "MLOps", "Model Deployment", "PyTorch", "TensorFlow",
    "Keras", "Scikit-Learn", "XGBoost", "LightGBM", "OpenCV", "Hugging Face", "Transformers", "AutoML",

    # Database Management
    "SQL", "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Cassandra", "Redis", "DynamoDB", "BigQuery", "Snowflake",
    "Apache Hive", "MariaDB", "GraphQL", "Oracle Database",

    # Cloud & DevOps
    "AWS", "Azure", "Google Cloud", "Kubernetes", "Docker", "Terraform", "Jenkins", "CI/CD", "CloudFormation",
    "Lambda", "EC2", "S3", "Cloud Functions", "Firebase", "Kubernetes", "Helm", "GitOps", "Ansible", "Prometheus",

    # Web Development & Frontend
    "HTML", "CSS", "JavaScript", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "FastAPI", "Bootstrap",
    "Tailwind CSS", "Svelte", "Next.js", "Nuxt.js", "Redux", "GraphQL", "WebSockets",

    # Backend Development
    "REST API", "GraphQL", "Spring Boot", "ASP.NET", "Express.js", "Ruby on Rails", "Flask", "Django", "Go Fiber",
    "NestJS", "FastAPI",

    # Cybersecurity
    "Ethical Hacking", "Penetration Testing", "Cryptography", "Cybersecurity", "Network Security", "SOC",
    "Threat Intelligence", "Cloud Security", "SIEM", "Intrusion Detection", "Identity Management",

    # Software Engineering & Tools
    "Git", "GitHub", "Bitbucket", "Agile", "Scrum", "Kanban", "CI/CD", "JIRA", "Confluence", "Trello", "Test-Driven Development",
    "Code Review", "Microservices", "Serverless Architecture",

    # Business Intelligence & Analytics
    "Tableau", "Power BI", "Looker", "Google Analytics", "Excel", "DAX", "SQL for Data Analysis", "A/B Testing",
    "Marketing Analytics", "Financial Modeling", "ETL", "Data Warehousing", "Dashboarding",

    # Networking & Infrastructure
    "TCP/IP", "DNS", "VPN", "Load Balancing", "Routing", "Firewall", "Network Security", "Wireshark", "Linux Networking",
    "Cloud Networking",

    # Robotics & IoT
    "ROS", "Arduino", "Raspberry Pi", "Edge Computing", "IoT", "Embedded Systems", "Sensors", "Actuators",

    # Finance & Accounting
    "Financial Analysis", "Budgeting", "Forecasting", "Risk Management", "Accounting", "Taxation", "Excel Modeling",
    "Investment Analysis", "Actuarial Science",

    # Soft Skills
    "Communication", "Leadership", "Problem-Solving", "Critical Thinking", "Project Management", "Time Management",
    "Teamwork", "Creativity", "Negotiation", "Emotional Intelligence", "Adaptability", "Decision Making"
]


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
