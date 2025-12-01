from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
import dotenv
dotenv.load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

def load_system_prompt() -> str:
    template_path = Path(__file__).parent / "prompt_templates" / "system_prompt.jinja2"
    return template_path.read_text(encoding="utf-8")

app = Flask(__name__)

@app.route('/', methods=['POST'])
def chat():
    data = request.json
    input = data.get('input')
    system_prompt = load_system_prompt()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", input),
    ])
    chain = prompt | model
    result = chain.invoke({"input": input})
    print(result.content)
    return jsonify({"response": result.content})

if __name__ == '__main__':
    app.run(debug=True)