import os
import streamlit as st
import pandas as pd
import pickle
import bz2
import requests
from io import BytesIO
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
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


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
if os.path.exists(csv_file):
    resume_db = pd.read_csv(csv_file)
else:
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
    return match.group(0) if match else " "


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


# Streamlit UI
st.title("📄 Resume Parser & Job Matcher")

uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, PNG, JPG)", type=["pdf", "docx", "png", "jpg", "jpeg"])

if uploaded_file:
    text = extract_text(uploaded_file)
    if text:
        name = extract_name(text)
        email = extract_email(text)
        phone = extract_phone(text)
        skills = extract_skills(text)
        category = predict_category(text)

        st.subheader("📌 Extracted Details")
        st.write(f"**👤 Name:** {name}")
        st.write(f"**📧 Email:** {email}")
        st.write(f"**📞 Phone:** {phone}")
        st.write(f"**🛠 Skills:** {', '.join(skills)}")
        st.write(f"**📂 Predicted Category:** {category}")

        parsed_data = {
            "Name": name, "Email": email, "Phone": phone, 
            "Skills": ", ".join(skills), "Category": category
        }
        st.download_button("📥 Download JSON", data=json.dumps(parsed_data, indent=4), file_name="parsed_resume.json", mime="application/json")

        # Job Matching
        job_description = st.text_area("Paste Job Description Here", height=150)
        if st.button("Match Resume to Job Description"):
            job_skills = extract_skills(job_description)
            matched_skills, match_score = calculate_match(skills, job_skills)

            st.subheader("🔍 Job Match Results")
            st.write(f"**🛠 Job Skills Extracted:** {', '.join(job_skills)}")
            st.write(f"**✅ Matched Skills:** {', '.join(matched_skills)}")
            st.write(f"**📊 Match Score:** {match_score}%")

            new_data = {
                "Name": name, "Email": email, "Phone": phone, "Skills": ", ".join(skills), 
                "Category": category, "Job Description": job_description, "Match Score": match_score
            }

            new_data_df = pd.DataFrame([new_data])
            resume_db = pd.concat([resume_db, new_data_df], ignore_index=True)
            resume_db.to_csv(csv_file, index=False)
            st.success("✅ Resume data successfully saved!")

        # Allow category correction
        corrected_category = st.selectbox("Is this the correct category?", ["Yes", "No, let me change"])
        if corrected_category == "No, let me change":
            new_category = st.selectbox("Select Correct Category", ["Data Science", "Marketing", "Finance", "Engineering"])
            resume_db.loc[resume_db["Email"] == email, "Category"] = new_category
            resume_db.to_csv(csv_file, index=False)
            st.success("✅ Category updated!")

# Function to retrain model with new data
def retrain_model():
    df = pd.read_csv(csv_file)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["Job Description"])
    y = df["Category"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    new_model = RandomForestClassifier(n_estimators=100, random_state=42)
    new_model.fit(X_train, y_train)
    y_pred = new_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"New Model Accuracy: {accuracy * 100:.2f}%")
    pickle.dump(new_model, open('model_resume_classifier.pkl', 'wb'))
    pickle.dump(vectorizer, open('vectorizer_resume_classifier.pkl', 'wb'))

if st.button("🔄 Retrain Model with New Data"):
    st.warning("⚙ Training model... This may take a few minutes.")
    retrain_model()
    st.success("✅ Model retrained successfully!")


