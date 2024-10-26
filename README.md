![image](https://github.com/user-attachments/assets/962c9b4e-e7c1-4328-8765-828cee0b1a1a)
<div align="center">
<div align="center">
  
# 🤖 Automated GitHub Pull Request Reviewer
  
![automated-pr-reviewer](https://github.com/user-attachments/assets/sample-banner.png)
  
</div>

### 🌐 Real-time, AI-driven PR feedback for code quality, readability, and optimization.

---

## 🚀 Project Inspiration

With the increasing demand for high-quality code reviews, maintaining consistency and depth in reviewing becomes challenging. This project automates PR reviews using AI to provide constructive feedback, covering essential aspects of code quality, structure, and maintainability—helping developers save time while ensuring code integrity.

---

## 🔍 Key Features

- **Intelligent PR Feedback**: Provides automated feedback on:
  - Code structure, readability, and optimization
  - Robust error handling and security vulnerabilities
  - Best practices tailored to the language or framework in use
- **Machine Learning-Powered Analysis**: Embeds file content using Sentence Transformers and indexes them with FAISS for efficient, content-based retrieval.
- **Real-time Suggestions**: Generates and posts feedback within seconds, offering constructive advice in concise comments.
- **GitHub Integration**: Leverages GitHub API to directly comment on PRs and handle webhooks for seamless updates.

---

## 🛠️ Tech Stack

[![My Skills](https://skillicons.dev/icons?i=python,flask,github,faiss,tensorflow,cohere,dotenv)](https://skillicons.dev)

- **Frontend**: Commenting directly on GitHub PRs.
- **Backend**: Flask app for managing webhook events and API interactions.
- **NLP Model**: Sentence Transformers for embedding code context.
- **Indexing**: FAISS for vector search and retrieval.
- **Language Processing**: Cohere’s LLM via LangChain for advanced text analysis.
- **Environment Management**: Dotenv for sensitive environment variables.

---

## 🧭 Workflow

1. **PR Trigger**: When a PR is opened, ready for review, or updated, GitHub sends a webhook to the app.
2. **Embedding and Indexing**: The code embeds the file content, indexes it, and identifies related files.
3. **AI-Powered Feedback**: The app generates structured feedback focused on code readability, performance, and best practices.
4. **Automated Commenting**: Posts feedback as inline comments on the PR, making review insights accessible.

---

## 🖥️ Screenshots

<div align="center">
  <img src="https://github.com/user-attachments/assets/pr-review-screenshot1.png" width="45%" alt="PR Review Screenshot 1">
  <img src="https://github.com/user-attachments/assets/pr-review-screenshot2.png" width="45%" alt="PR Review Screenshot 2">
</div>

---

## 💡 How to Use

### 1. Clone and Install Dependencies

\`\`\`bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
\`\`\`

### 2. Configure `.env` Variables

Create a `.env` file in the root directory with:

\`\`\`plaintext
GITHUB_TOKEN=your_github_personal_access_token
COHERE_API_KEY=your_cohere_api_key
\`\`\`

### 3. Run the Application

\`\`\`bash
python app.py
\`\`\`

### 4. Set Up GitHub Webhook

Add a webhook in your GitHub repository with the following details:

- **Payload URL**: `http://your-server-domain.com/webhook`
- **Content Type**: `application/json`
- **Events**: Select `Pull requests` to trigger the app when a PR is created or updated.

---

## 🏗️ Project Structure

- **`app.py`**: Main logic for handling GitHub webhooks, embedding text, and generating PR comments.
- **`fetch_github_repo_files`**: Fetches repository files for indexing and embedding.
- **`update_rag_index`**: Updates the FAISS index to stay current with repo changes.
- **`generate_response`**: Uses LangChain with Cohere to generate actionable feedback.

---

## 🚀 Future Enhancements

- **Advanced Analysis**: Incorporate more complex review criteria for enhanced security and performance insights.
- **Multi-repo Support**: Scale to manage PR reviews across multiple repositories.
- **Feedback Customization**: Allow custom feedback templates to accommodate different code standards and practices.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.



