from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(api_key='type your key here')

# Function to generate a summary using OpenAI API
def generate_summary(text):
    prompt = (
        f"Given the following text, extract and list the necessary details in few lines:\n\n"
        f"{text}\n\n"
        f"amke it tabulate and under 50 words:\n"
       
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].message.content

# Function to answer a custom question using OpenAI API
def answer_question(text, question, history):
    messages = [{"role": "system", "content": "You are a helpful assistant, answer the quesion using the context. Keep the answer to the point and under 100 words."}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": f"Context: {text}\n\nQuestion: {question}"})
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].message.content
