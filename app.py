# =========================================================
# AI-Powered Resume Screening Dashboard using NLP + RAG
# =========================================================

# Install requirements (for Colab users)
# !pip install sentence-transformers faiss-cpu transformers gradio

import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# Dummy Resume Dataset

resume_texts = [
"""
Name: Ananya Reddy
Education: B.Tech in Computer Science (AIML)
Skills: Python, Machine Learning, NLP, Deep Learning, TensorFlow, Scikit-learn
Projects: Built a chatbot using RAG architecture for college information system.
Experience: AI Intern at TechNova - worked on text classification and embeddings.
""",

"""
Name: Rohit Sharma
Education: B.Tech in Information Technology
Skills: Java, Web Development, SQL, React, Node.js
Projects: Developed an e-commerce website with payment gateway integration.
Experience: Full Stack Intern at WebWorks.
""",

"""
Name: Kavyasree
Education: B.Tech in AIML - Alliance University
Skills: Python, NLP, Deep Learning, Data Science, FAISS, LangChain
Projects: Resume Screening using NLP and RAG, Medical Chatbot using LLM.
Experience: AIML Research Intern - worked on transformer-based models.
"""
]

# Load Embedding Model

model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(resume_texts)

# Build FAISS Vector Database
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load Free HuggingFace LLM

llm = pipeline("text2text-generation", model="google/flan-t5-base")

# Resume Matcher + RAG Function

def resume_matcher(job_desc):
    # Embed Job Description
    query_emb = model.encode([job_desc])

    # Search FAISS
    D, I = index.search(query_emb, k=1)
    best_resume = resume_texts[I[0][0]]

    # RAG Prompt
    prompt = f"""
Job Description:
{job_desc}

Candidate Resume:
{best_resume}

Explain clearly why this candidate is a good match for the job.
"""

    # Generate Explanation
    result = llm(prompt, max_length=200)[0]['generated_text']

    return best_resume, result



# Custom CSS for Dashboard

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0f172a, #1e3a8a, #312e81) !important;
}

h1 {
    text-align: center;
    color: #38bdf8 !important;
    font-size: 36px !important;
    font-weight: bold;
}

h3 {
    text-align: center;
    color: #ffffff !important;
}

p, span, label, div, small {
    color: #ffffff !important;
}

textarea {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #38bdf8 !important;
}

button {
    background: linear-gradient(90deg, #38bdf8, #6366f1) !important;
    color: #ffffff !important;
    font-weight: bold !important;
    border-radius: 8px !important;
}
"""


# Gradio Dashboard UI

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🤖 AI-Powered Resume Screening Dashboard  
        ### NLP + RAG Based Intelligent Candidate Matching  
        Paste a Job Description to find the best candidate with AI-generated explanation.
        ---
        """
    )

    job_input = gr.Textbox(
        lines=6,
        placeholder="Paste Job Description here...",
        label="📄 <span style='color:#ffd700; font-weight:bold'>Job Description</span>"
    )

    submit_btn = gr.Button("🔍 Find Best Candidate")

    best_resume_output = gr.Textbox(
        label="🏆 <span style='color:#00e5ff; font-weight:bold'>Best Matching Resume</span>",
        lines=8
    )

    explanation_output = gr.Textbox(
        label="💡 <span style='color:#ff69b4; font-weight:bold'>AI Explanation</span>",
        lines=8
    )

    submit_btn.click(
        fn=resume_matcher,
        inputs=job_input,
        outputs=[best_resume_output, explanation_output]
    )

    gr.Markdown(
        """
        ---
        **Project:** AI-Powered Resume Screening using NLP + RAG  
        **Developed by:** Kavyasree – B.Tech AIML  
        """
    )
# Launch App

demo.launch()
