# app.py
import gradio as gr
from src.rag_pipeline import answer_question  # your Task 3 module

# Function to process a user question
def chat_rag(user_question):
    """
    Takes a user's question, queries the RAG pipeline,
    and returns the generated answer along with sources.
    """
    if not user_question.strip():
        return "Please type a question.", ""

    # Call Task 3 RAG pipeline
    answer, retrieved_chunks = answer_question(user_question, top_k=5)
    
    # Prepare sources string for display (top 3 chunks)
    sources_display = "\n\n".join(
        [f"- {c['text']}" for c in retrieved_chunks[:3]]
    )
    
    return answer, sources_display

# Gradio UI layout
with gr.Blocks(title="CrediTrust RAG Chatbot") as demo:

    gr.Markdown("## 💬 CrediTrust Complaint Analysis Chatbot")
    gr.Markdown("Ask questions about customer complaints. Answers are generated using retrieved complaint narratives.")

    with gr.Row():
        user_input = gr.Textbox(label="Your Question", placeholder="Type your question here...", lines=2)
        submit_btn = gr.Button("Ask")

    output_answer = gr.Textbox(label="AI Answer", placeholder="Answer will appear here...", lines=6)
    output_sources = gr.Textbox(label="Top Sources from Complaints", placeholder="Sources will appear here...", lines=10)

    clear_btn = gr.Button("Clear")

    # Button click events
    submit_btn.click(chat_rag, inputs=user_input, outputs=[output_answer, output_sources])
    clear_btn.click(lambda: ("", ""), inputs=None, outputs=[output_answer, output_sources, user_input])

# Launch app
demo.launch(share=True, debug=True)
