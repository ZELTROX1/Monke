![image](https://github.com/user-attachments/assets/962c9b4e-e7c1-4328-8765-828cee0b1a1a)
Automated GitHub Pull Request Reviewer
A Flask-based web application that automates pull request (PR) reviews on GitHub repositories. Using machine learning, FAISS indexing, and LangChain with Cohere, this tool provides insightful feedback on code changes, emphasizing best practices, readability, optimization, and maintainability.

📝 Features
Automatic Code Review: Reviews open PRs and provides constructive feedback on code structure, readability, efficiency, error handling, and security.
Embedding-Based Feedback: Utilizes FAISS indexing with sentence embeddings to identify relevant files and review code changes.
Concise and Contextual Suggestions: Generates short, actionable feedback for each file.
Seamless GitHub Integration: Supports webhook-based integration with GitHub to automate review comments on newly opened or updated PRs.
🛠️ Tech Stack
Flask: Backend framework for handling web requests.
GitHub API: Interface for fetching PR information and posting feedback.
Cohere Language Model: Integrated through LangChain for generating language-based feedback.
Sentence Transformers: Embeddings for analyzing code similarities.
FAISS: Fast indexing and searching of embeddings.
Dotenv: Environment variable management.y.
📝 Usage
Initial Review: Automatically reviews all open PRs in the specified repository on startup.
Webhook-Triggered Reviews: When a new PR is opened or updated, the application:
Updates the embedding index with the latest repo content.
Fetches file changes in the PR.
Generates concise, constructive feedback and posts it as a review comment.
📁 Project Structure
plaintext
Copy code
├── app.py                   # Main application script
├── requirements.txt         # Required dependencies
└── .env                     # API keys (not included in version control)
🤖 System Prompt
The application uses the following prompt to generate feedback:

plaintext
Copy code
You are an expert software engineer and code reviewer. Your goal is to provide detailed, constructive, and context-aware feedback on code, focusing on best practices, readability, optimization, and maintainability.
...
🤝 Contributing
Fork the repository.
Create a new branch for your changes.
Submit a pull request with a description of your changes.
📜 License
This project is licensed under the MIT License.
