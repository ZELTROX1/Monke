import os
from github import Github
import base64
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from flask import Flask, request, jsonify
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import time

# Initialize Flask app
app = Flask(__name__)
load_dotenv(".env")

# Initialize GitHub, LangChain, and other services
g = Github(os.getenv("GITHUB_TOKEN"))
llm = ChatCohere(model="command-r-plus", cohere_api_key=os.getenv("COHERE_API_KEY"))
parser = StrOutputParser()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
file_paths = []
repo_files = {}

# Set up system prompt
system_prompt = """
You are an expert software engineer and code reviewer. Your goal is to provide detailed, constructive, and context-aware feedback on code, focusing on best practices, readability, optimization, and maintainability. When reviewing code, consider:

- Code Structure: Assess the organization, logic flow, and modularity of the code.
- Readability and Documentation: Comment on naming conventions, inline comments, and overall clarity of the code.
- Efficiency and Optimization: Identify any areas where performance can be improved, including unnecessary computations, memory usage, or any potential bottlenecks.
- Error Handling and Robustness: Ensure the code includes error handling and is resilient to edge cases.
- Security: Point out any security vulnerabilities or risks, such as exposure to injections, unsafe data handling, or other potential security flaws.
- Adherence to Language or Framework Best Practices: Make sure the code follows best practices specific to the language or framework being used.

When providing feedback, prioritize constructive and actionable advice. Offer clear examples or alternatives for any suggested changes. Avoid generic comments and be as specific as possible to help the author understand areas for improvement.

restrict your feedback to 50 words, make it super crisp

"""

prompt_template = PromptTemplate(
    input_variables=["system_prompt", "context", "query"],
    template="{system_prompt}\n\nHere are some relevant files: {context}\n\nUser's question: {query}\nAnswer:"
)

# Function to fetch files from GitHub repo
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

# Function to update RAG index
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

# Function to embed text with transformer model
def embed_text_with_transformers(text):
    return embedder.encode(text)

# Function to search FAISS index for relevant files
def search_faiss(query, top_k=3):
    query_embedding = embed_text_with_transformers(query)
    query_embedding = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)
    relevant_files = [file_paths[i] for i in indices[0] if i < len(file_paths)]
    return relevant_files

# Function to get file content
def fetch_file_content(file_paths, repo_files):
    return "\n\n".join([repo_files[file] for file in file_paths])

# Generate RAG response
def generate_response(query, context):
    prompt = prompt_template.format(system_prompt=system_prompt, context=context, query=query)
    chain = llm | parser
    response = chain.invoke(prompt)
    return response

def review_open_prs(repo_name):
    repo = g.get_repo(repo_name)
    open_prs = repo.get_pulls(state='open')
    update_rag_index(repo_name)  # Ensure the index is updated for the latest repo state
    
    for pr in open_prs:
        pr_files = pr.get_files()
        latest_commit_sha = pr.head.sha  # Get the latest commit SHA for the PR
        
        for file in pr_files:
            file_path = file.filename
            if file_path in repo_files:
                # Generate the embedding-based feedback for each file
                relevant_files_content = fetch_file_content([file_path], repo_files)
                review_comment = generate_response(
                    f"Review the file: {file_path}",
                    relevant_files_content
                )
                
                # Post an inline comment on the file in the PR
                try:
                    pr.create_review_comment(
                        body=review_comment,
                        commit_id=latest_commit_sha,
                        path=file_path,
                        position=1  # Adjust this position as needed
                    )
                    print(f"Added review comment on {file_path} in PR #{pr.number}.")
                except Exception as e:
                    print(f"Error creating review comment for {file_path}: {e}")
            else:
                print(f"File {file_path} not found in repo_files. Skipping...")
                
    print("Completed review of all open PRs.")

"""
# Function to review PRs and add comments
def review_open_prs(repo_name):
    repo = g.get_repo(repo_name)
    open_prs = repo.get_pulls(state='open')
    update_rag_index(repo_name)  # Ensure the index is updated for the latest repo state
    for pr in open_prs:
        pr_files = [f.filename for f in pr.get_files()]
        relevant_files_content = fetch_file_content(pr_files, repo_files)
        response = generate_response("Review this pull request.", relevant_files_content)
        
        # Comment on the PR
        pr.create_issue_comment(response)
        print(f"Reviewed PR #{pr.number} and posted comment.")
"""

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    
    # Process only if the action is to open or mark as ready for review a PR
    if data.get("action") in ["opened", "ready_for_review"]:
        repo_name = data["repository"]["full_name"]
        pr_number = data["pull_request"]["number"]
        pull_request = g.get_repo(repo_name).get_pull(pr_number)
        
        print(f"Processing Pull Request #{pr_number} in repository {repo_name}.")
        
        # Refresh the RAG index to include all repo content
        update_rag_index(repo_name)
        
        # Prepare overall feedback for files found in repo_files
        comments = []
        for file in pull_request.get_files():
            file_path = file.filename
            
            # Check if the file exists in repo_files
            if file_path in repo_files:
                # Get content for the specific file for review feedback
                relevant_files_content = fetch_file_content([file_path], repo_files)
                
                # Generate feedback for the file
                review_comment = generate_response(
                    f"Review the file: {file_path}",
                    relevant_files_content
                )
                
                # Append file-specific feedback to comments
                comments.append(f"**{file_path}**\n{review_comment}")
            else:
                print(f"File {file_path} not found in repo_files. Skipping...")

        # Combine comments into a single review body
        review_body = "\n\n".join(comments)
        
        # Add a general review comment to the pull request
        try:    
            pull_request.create_review(body=review_body if review_body else "Automated code review feedback",event="COMMENT")
        except Exception as e:    
            print(f"Error creating review: {e}")
        
        return jsonify({"status": "Review comments added"}), 200

    return jsonify({"status": "Event ignored"}), 200


# Flask entry point
if __name__ == '__main__':
    repo_name = 'Spirizeon/moonphase'
    review_open_prs(repo_name)  # Initial review when app starts
    app.run(port=5000, debug=True)

