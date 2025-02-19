import streamlit as st
import PyPDF2
import io
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_file):
    """Extract text content from uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_question_chain():
    """Create LangChain chain for generating questions."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Mistral API key not found in environment variables")
        
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.7,
        max_retries=2,
        mistral_api_key=api_key
    )
    
    template = """
    You are an expert technical recruiter and interviewer. Based on the following resume, generate a set of meaningful interview questions. 
    The questions should cover:
    1. Technical experience and projects mentioned in the resume. Make sure that you ask about any technical terms, algorithms, their preference of a tool, choice of data etcetra mentioned in the resume. 
    2. Domain knowledge relevant to their field. Infer the domain knowledge and ask questions apart from resume related to the domain, Example if the applicant is from AI and has not mentioned anything about CNN or image processing, ask Have you ever worked with images or do you know about CNN" 
    3. Behavioral aspects and soft skills
    4. Problem-solving abilities
    5. Career goals and motivations

    Please organize the questions into these categories and ensure they go beyond surface-level information. 

    Resume content:
    {resume_text}

    Generate 3-4 questions for each category. Format the output with clear category headings and return the questions in markdown format.
    """
    
    prompt = PromptTemplate(
        input_variables=["resume_text"],
        template=template
    )
    
    return LLMChain(llm=llm, prompt=prompt)

def main():
    st.set_page_config(page_title="Help Hire - Interview Question Generator", layout="wide")
    
    # Initialize session state for questions
    if 'questions' not in st.session_state:
        st.session_state['questions'] = None
    
    # Application header
    st.title("🎯 Help Hire")
    st.subheader("Generate Meaningful Interview Questions from Resumes")
    
    # Check for API key in environment variables
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("Mistral API key not found in environment variables. Please check your .env file.")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Resume")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
        
        # Add a generate button
        if uploaded_file:
            if st.button("Generate Questions", key="generate_btn"):
                with st.spinner("Processing resume..."):
                    try:
                        # Extract text from PDF
                        resume_text = extract_text_from_pdf(uploaded_file)
                        
                        # Create and run the question generation chain
                        chain = create_question_chain()
                        questions = chain.run(resume_text)
                        
                        # Store the generated questions in session state
                        st.session_state['questions'] = questions
                        
                        # Show success message
                        st.success("Questions generated successfully!")
                    except Exception as e:
                        st.error(f"Error processing resume: {str(e)}")
    
    with col2:
        st.header("Generated Questions")
        if st.session_state['questions']:
            st.markdown(st.session_state['questions'])
            
            # Add export functionality
            st.download_button(
                label="Download Questions",
                data=st.session_state['questions'],
                file_name="interview_questions.txt",
                mime="text/plain",
                key="download_button"
            )
        else:
            st.info("Upload a resume and click 'Generate Questions' to start")

    # Add information about the application
    with st.sidebar:
        st.header("How it works")
        st.markdown("""
        1. Upload a resume in PDF format
        2. Click 'Generate Questions' button
        3. AI generates tailored interview questions
        4. Questions are categorized by type
        
        ### Features
        - Automatic text extraction
        - AI-powered question generation using Mistral Large
        - Structured output by category
        - Download functionality
        """)

if __name__ == "__main__":
    main()