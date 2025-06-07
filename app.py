import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import fitz  # PyMuPDF
from newspaper import Article

# Load mBART model for Hindi summarization
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer.src_lang = "hi_IN"

# Hindi LLM-based summarizer function
def summarize_text(text, max_len=512, summary_len=128):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_len, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=summary_len,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]

# Extract text from PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract article from URL
def extract_text_from_url(url):
    article = Article(url, language='hi')
    article.download()
    article.parse()
    return article.text

# Streamlit UI
st.set_page_config(page_title="Hindi LLM Summarizer", layout="wide")
st.title("🧠📜 Hindi LLM Summarizer")

option = st.radio("Choose input type:", ["📄 Upload PDF", "🔗 Enter Article URL"])

text_data = ""

if option == "📄 Upload PDF":
    uploaded_file = st.file_uploader("Upload a Hindi PDF file", type=["pdf"])
    if uploaded_file:
        st.success("✅ PDF uploaded!")
        text_data = extract_text_from_pdf(uploaded_file)

elif option == "🔗 Enter Article URL":
    url = st.text_input("Paste the article URL:")
    if url:
        try:
            with st.spinner("🔍 Extracting article..."):
                text_data = extract_text_from_url(url)
            st.success("✅ Article extracted!")
        except:
            st.error("❌ Failed to extract article. Check the URL.")

if text_data:
    st.subheader("📚 Extracted Text:")
    st.text_area("Text Preview", value=text_data[:1500], height=200)

    if st.button("🧠 Generate Summary"):
        with st.spinner("Summarizing using LLM..."):
            summary = summarize_text(text_data)
        st.subheader("✅ Hindi Summary:")
        st.write(summary)
