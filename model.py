import os
from github import Github
import base64
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from flask import Flask, request, jsonify, render_template
import cohere
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)

load_dotenv(".env")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

g = Github(os.getenv("GITHUB_TOKEN"))

llm = ChatCohere(model="command-r-plus", cohere_api_key=os.getenv("COHERE_API_KEY"))
parser = StrOutputParser()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index and file tracking
dimension = 384
index = faiss.IndexFlatL2(dimension)
file_paths = []
repo_files = {}

system_prompt = """
You are an expert software engineer and code reviewer. Your goal is to provide detailed, constructive, and context-aware feedback on code, focusing on best practices, readability, optimization, and maintainability. When reviewing code, consider:

- Code Structure: Assess the organization, logic flow, and modularity of the code.
- Readability and Documentation: Comment on naming conventions, inline comments, and overall clarity of the code.
- Efficiency and Optimization: Identify any areas where performance can be improved, including unnecessary computations, memory usage, or any potential bottlenecks.
- Error Handling and Robustness: Ensure the code includes error handling and is resilient to edge cases.
- Security: Point out any security vulnerabilities or risks, such as exposure to injections, unsafe data handling, or other potential security flaws.
- Adherence to Language or Framework Best Practices: Make sure the code follows best practices specific to the language or framework being used.

When providing feedback, prioritize constructive and actionable advice. Offer clear examples or alternatives for any suggested changes. Avoid generic comments and be as specific as possible to help the author understand areas for improvement.
"""

prompt_template = PromptTemplate(
    input_variables=["system_prompt", "context", "query"],
    template="{system_prompt}\n\nHere are some relevant files: {context}\n\nUser's question: {query}\nAnswer:"
)

def fetch_github_repo_files(repo_name):
    repo = g.get_repo(repo_name)
    contents = repo.get_contents("")
    files = {}

    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            file_data = base64.b64decode(file_content.content).decode("utf-8")
            files[file_content.path] = file_data
    return files

def update_rag_index(repo_name):
    global index, file_paths, repo_files
    index.reset()
    file_paths.clear()
    repo_files = fetch_github_repo_files(repo_name)

    for path, content in repo_files.items():
        embedding = embed_text_with_transformers(content)
        index.add(np.array([embedding], dtype=np.float32))
        file_paths.append(path)

    print(f"Re-indexed {len(file_paths)} files from {repo_name}.")

def embed_text_with_transformers(text):
    return embedder.encode(text)

def search_faiss(query, top_k=3):
    query_embedding = embed_text_with_transformers(query)
    query_embedding = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)
    relevant_files = [file_paths[i] for i in indices[0] if i < len(file_paths)]
    return relevant_files

def fetch_file_content(file_paths, repo_files):
    return "\n\n".join([repo_files[file] for file in file_paths])

def generate_response(query, context):
    prompt = prompt_template.format(system_prompt=system_prompt, context=context, query=query)
    chain = llm | parser 
    response = chain.invoke(prompt)
    return response

# Route to serve HTML form
@app.route('/')
def home():
    return '''
    <!doctype html>
    <html lang="en">
      <head>
        <title>Code Review Query</title>
      </head>
      <body>
        <h1>Enter your code review query</h1>
        <form action="/get_response" method="post">
          <label for="query">Query:</label><br><br>
          <input type="text" id="query" name="query" required><br><br>
          <label for="query">Repo:</label><br><br>
          <input type="text" id="repo" name="repo" required><br><br>
          <input type="submit" value="Submit">
        </form>
      </body>
    </html>
    '''

# Route to handle form submission and display response
@app.route('/get_response', methods=['POST'])
def get_response():
    user_query = request.form['query']
    repo_name =  request.form['repo']

    # Ensure the index is up to date
    update_rag_index(repo_name)

    # Retrieve relevant files and their content
    relevant_files = search_faiss(user_query, top_k=3)
    relevant_files_content = fetch_file_content(relevant_files, repo_files)

    # Generate response from the model
    response = generate_response(user_query, relevant_files_content)

    return f'''
    <!doctype html>
    <html lang="en">
      <head>
        <title>Code Review Response</title>
      </head>
      <body>
        <h1>Response to your query</h1>
        <p><strong>Query:</strong> {user_query}</p>
        <p><strong>Response:</strong> {response}</p>
        <a href="/">Back to Query</a>
      </body>
    </html>
    '''

# Webhook route to update index on push event
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if data["ref"].startswith("refs/heads/"):
        repo_name = data["repository"]["full_name"]
        print(f"Push detected in repository {repo_name}. Updating RAG index.")
        update_rag_index(repo_name)
        return jsonify({"status": "Index updated due to push event"}), 200
    return jsonify({"status": "Ignored"}), 200

if __name__ == '__main__':
    repo_name = 'Spirizeon/claxvim'
    update_rag_index(repo_name)
    app.run(port=5000, debug=True)
