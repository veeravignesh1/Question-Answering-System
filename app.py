# Imports
import os
import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# os.environ["TRANSFORMERS_CACHE"] = "./cache"

# Caching Model
@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
def load_model(model_name):
    return (
        AutoModelForQuestionAnswering.from_pretrained(
            model_name, cache_dir="./.cache/"
        ),
        AutoTokenizer.from_pretrained(model_name, cache_dir="./.cache/"),
    )


# Main
def main():
    st.title("Question Answering System Demo")
    models = {"distilbert-base-uncased-distilled-squad": "Distilbert"}
    model_name = st.sidebar.selectbox(
        "Choose a model",
        list(models.keys()),
        help="Select from various Predefined Models",
    )
    model, tokenizer = load_model(model_name)
    context = st.text_area(
        "Context",
        height=200,
        help="Enter the paragraph from which the question needs to be asked",
    )
    question = st.text_area(
        "Question", height=30, help="Enter the Question from the pasted paragraph"
    )

    if st.button("Find Answer"):
        st.text("Answer")
        with st.spinner("Finding Answer (This may take some time)"):
            # st.write(context)
            # st.write(question)
            # st.write(model_name)
            question_answering = pipeline(
                task="question-answering", model=model, tokenizer=tokenizer
            )
            result = question_answering(question=question, context=context)
            st.write(result["answer"])
            st.write(result["score"])


if __name__ == "__main__":
    main()
