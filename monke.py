import os
import asyncio
import sys
import logging
from typing import List, Dict, Any, Optional

from github import Github
import base64
import requests
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedLLMPipeline:
    def __init__(self, api_key=None):
        """
        Initialize the advanced LLM pipeline with multiple models and configurations
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        # Define specialized models for different tasks
        self.models = {
            "code_review": ChatGroq(
                model="deepseek-r1-distill-llama-70b",
                temperature=0.5,
                groq_api_key=self.api_key
            ),
            "summarization": ChatGroq(
                model="gemma2-9b-it",
                temperature=0.3,
                groq_api_key=self.api_key
            ),
            "detail_extraction": ChatGroq(
                model="deepseek-r1-distill-llama-70b",
                temperature=0.2,
                groq_api_key=self.api_key
            )
        }
        
        # Define prompt templates for different stages
        self.prompt_templates = {
            "initial_analysis": PromptTemplate(
                input_variables=["content"],
                template="""Perform an initial high-level analysis of the following code:
                Code: {content}
                
                Provide a concise overview focusing on:
                1. Overall code structure
                2. Potential architectural patterns
                3. Initial observations
                """
            ),
            "detailed_review": PromptTemplate(
                input_variables=["initial_analysis"],
                template="""Based on the initial analysis:
                {initial_analysis}
                
                Conduct an in-depth review with specific focus on:
                1. Detailed code quality assessment
                2. Potential optimization opportunities
                3. Security and performance considerations
                """
            )
        }

    async def route_task(self, input_content: str) -> str:
        """
        Dynamic task routing based on input characteristics
        """
        if 'import' in input_content or 'def ' in input_content or 'class ' in input_content:
            return 'code_review'
        elif len(input_content) > 500:
            return 'summarization'
        else:
            return 'detail_extraction'

    async def parallel_process(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """
        Parallelize independent subtasks across multiple LLMs
        """
        async def process_task(task):
            model_name = await self.route_task(task['content'])
            model = self.models[model_name]
            
            # Prompt-chaining: Multi-stage processing
            stage1_prompt = self.prompt_templates['initial_analysis']
            stage2_prompt = self.prompt_templates['detailed_review']
            
            # Initial analysis
            initial_analysis = model.invoke(
                stage1_prompt.format(content=task['content'])
            ).content
            
            # Detailed review based on initial analysis
            detailed_review = model.invoke(
                stage2_prompt.format(initial_analysis=initial_analysis)
            ).content
            
            return detailed_review

        # Use asyncio to process tasks concurrently
        return await asyncio.gather(*[process_task(task) for task in tasks])

    def process_github_files(self, files: Dict[str, str]) -> Dict[str, str]:
        """
        Process multiple GitHub files with advanced LLM pipeline
        """
        # Convert files to task format
        tasks = [
            {'path': path, 'content': content} 
            for path, content in files.items()
        ]
        
        # Run async processing
        results = asyncio.run(
            self.parallel_process(tasks)
        )
        
        # Map results back to file paths
        return {
            task['path']: result 
            for task, result in zip(tasks, results)
        }

class GitHubCodeReviewSentinel:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Validate environment variables
        self._validate_environment()
        
        # Initialize GitHub client
        self.github_client = Github(os.getenv("GITHUB_TOKEN"))
        
        # Initialize embedding and indexing
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.file_paths = []
        self.repo_files = {}
        
        # Initialize LLM Pipeline
        self.llm_pipeline = AdvancedLLMPipeline()

    def _validate_environment(self):
        """Validate that all required environment variables are set."""
        required_vars = [
            "GITHUB_TOKEN", 
            "GROQ_API_KEY", 
            "LANGCHAIN_API_KEY"
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def fetch_github_repo_files(self, repo_name):
        """Fetch all files from a GitHub repository"""
        repo = self.github_client.get_repo(repo_name)
        contents = repo.get_contents("")
        files = {}

        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            else:
                try:
                    file_data = base64.b64decode(file_content.content).decode("utf-8")
                    files[file_content.path] = file_data
                except (UnicodeDecodeError, base64.binascii.Error):
                    print(f"Skipping binary file: {file_content.path}")
                    continue
        return files

    def update_rag_index(self, repo_name):
        """Update RAG index for the repository"""
        self.index.reset()
        self.file_paths.clear()
        self.repo_files = self.fetch_github_repo_files(repo_name)

        for path, content in self.repo_files.items():
            embedding = self.embed_text_with_transformers(content)
            self.index.add(np.array([embedding], dtype=np.float32))
            self.file_paths.append(path)

        logger.info(f"Indexed {len(self.file_paths)} files from {repo_name}")

    def embed_text_with_transformers(self, text):
        """Embed text using Sentence Transformers"""
        return self.embedder.encode(text)

    def search_faiss(self, query, top_k=3):
        """Search FAISS index for relevant files"""
        query_embedding = self.embed_text_with_transformers(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        relevant_files = [self.file_paths[i] for i in indices[0] if i < len(self.file_paths)]
        return relevant_files

    def get_open_pr_numbers(self, owner, repo):
        """Fetch open pull request numbers"""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        headers = {
            "Authorization": f"token {os.getenv('GITHUB_TOKEN')}"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            pr_list = response.json()
            return [pr['number'] for pr in pr_list]
        else:
            logger.error(f"Error fetching PRs: {response.status_code}, {response.json()}")
            return []

    def process_pull_request(self, repo_name, pr_number):
        """Process a pull request with advanced review"""
        # Update the RAG index with the repository content
        self.update_rag_index(repo_name)

        # Get the pull request object
        repo = self.github_client.get_repo(repo_name)
        pull_request = repo.get_pull(pr_number)

        # Process files with advanced pipeline
        file_reviews = self.llm_pipeline.process_github_files(self.repo_files)
        
        # Generate comprehensive review
        comprehensive_review = "\n\n".join([
            f"**{path}**\n{review}" 
            for path, review in file_reviews.items()
        ])

        try:
            pull_request.create_review(
                body=comprehensive_review if comprehensive_review else "Automated code review feedback",
                event="COMMENT"
            )
            return {
                "status": "Comprehensive Review Generated", 
                "review": comprehensive_review
            }
        except Exception as e:
            logger.error(f"Error creating review: {e}")
            return {
                "status": "Error creating review", 
                "error": str(e)
            }

def main():
    """Main execution function"""
    try:
        # Initialize the sentinel
        sentinel = GitHubCodeReviewSentinel()
        
        # Default repository
        repo_name = 'Spirizeon/claxvim'
        
        # Get open PRs
        open_prs = sentinel.get_open_pr_numbers("spirizeon", "claxvim")
        logger.info(f"Open PRs: {open_prs}")
        
        if not open_prs:
            logger.info("No open PRs to review!")
            return
        
        # Process each open PR
        for pr_number in open_prs:
            pr_result = sentinel.process_pull_request(repo_name, pr_number)
            logger.info(f"PR {pr_number} Review Status: {pr_result['status']}")
            
            if 'review' in pr_result:
                logger.info(f"Review Details: {pr_result['review']}")
    
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()