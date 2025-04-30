import ollama
import re
from typing import List, Dict, Generator
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def ollama_generator(model_name: str, messages: List[Dict]) -> Generator:
    """
    Generate streaming responses from Ollama.
    
    Parameters:
        model_name (str): Name of the Ollama model to use
        messages (List[Dict]): List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        Generator: A generator yielding response chunks
    """
    stream = ollama.chat(model=model_name, messages=messages, stream=True)
    response_text = ""  # Collect response chunks
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            response_text += chunk["message"]["content"] + " "  # Concatenate chunks
            yield chunk["message"]["content"]
        else:
            yield "⚠️ No 'message' key found. Check API output."

    return response_text  # Return full response at the end

def summarize_failures(failure_descriptions, llm):
    """
    Summarize failure descriptions using Ollama.
    
    Parameters:
        failure_descriptions (str): Text containing failure descriptions
        llm: The language model to use for summarization
        
    Returns:
        str: The summarized failure descriptions
    """
    if not failure_descriptions or failure_descriptions.strip() == "❌ No relevant failures found!":
        return "❌ No relevant failure descriptions found for summarization."

    prompt_template = PromptTemplate(
        input_variables=["failure_descriptions"],
        template=(
            "You are an AI expert in failure analysis. The following are extracted details:\n\n"
            "**1️⃣ Failure Descriptions:**\n{failure_descriptions}\n\n"
            "Summarize the common failure trends based on the information provided. "
            "Focus on identifying patterns, recurring issues, and potential root causes. "
            "Organize your summary into clear sections with bullet points where appropriate."
        )
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    summary = chain.run(
        failure_descriptions=failure_descriptions
    )

    # Remove AI-generated thoughts like <think>...</think>
    cleaned_summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

    return cleaned_summary

def get_available_models():
    """
    Get a list of available Ollama models.
    
    Returns:
        list: List of available model names
    """
    try:
        response = ollama.list()
        if "models" in response and isinstance(response["models"], list):
            return [model.model for model in response["models"]]
        else:
            return []
    except Exception as e:
        return []

def enhanced_summarize_failures(failure_descriptions, dtc_list, charging_issues, past_repairs, key_failures, llm):
    """
    Enhanced version of summarize_failures that includes additional context.
    
    Parameters:
        failure_descriptions (str): Text containing failure descriptions
        dtc_list (str): List of DTC codes
        charging_issues (str): Information about charging issues
        past_repairs (str): Information about past repairs
        key_failures (str): Information about key failure events
        llm: The language model to use for summarization
        
    Returns:
        str: The summarized failure information
    """
    if not failure_descriptions or failure_descriptions.strip() == "❌ No relevant failures found!":
        return "❌ No relevant failure descriptions found for summarization."

    prompt_template = PromptTemplate(
        input_variables=["failure_descriptions", "dtc_list", "charging_issues", "past_repairs", "key_failures"],
        template=(
            "You are an AI expert in failure analysis. The following are extracted details:\n\n"
            "**1️⃣ Failure Descriptions:**\n{failure_descriptions}\n\n"
            "**2️⃣ Extracted DTC Codes (along with how many times it has occurred):**\n{dtc_list}\n\n"
            "**3️⃣ AC vs. DC Charging Issues:**\n{charging_issues}\n\n"
            "**4️⃣ Past Repairs and Replacement Plans:**\n{past_repairs}\n\n"
            "**5️⃣ Key Failure Events:**\n{key_failures}\n\n"
            "Summarize the common failure trends based on the information provided. "
            "Focus on identifying patterns, recurring issues, and potential root causes. "
            "Organize your summary into clear sections with bullet points where appropriate."
        )
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    summary = chain.run(
        failure_descriptions=failure_descriptions,
        dtc_list=dtc_list,
        charging_issues=charging_issues,
        past_repairs=past_repairs,
        key_failures=key_failures
    )

    # Remove AI-generated thoughts like <think>...</think>
    cleaned_summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

    return cleaned_summary 
