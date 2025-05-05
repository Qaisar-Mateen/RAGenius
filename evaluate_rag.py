"""
RAG Evaluation Framework for NLP Project

This script provides a comprehensive evaluation framework for testing the RAG (Retrieval-Augmented Generation)
pipeline implemented in rag_pipeline.py. It includes:

1. Relevance metrics - How well the system retrieves relevant information
2. Factual consistency - Whether the generated answers are factually accurate
3. Answer helpfulness - Qualitative assessment of answer utility
4. Response time analysis - Performance benchmarking
5. Visualizations for inclusion in reports
"""

import os
import time
import json
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle  # Added pickle import here
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Import the RAG pipeline
from rag_pipeline import RAGPipeline
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGEvaluator:
    """
    Evaluation framework for RAG systems.
    """
    
    def __init__(self,
        rag_pipeline: RAGPipeline = None,
        groq_api_key: str = None,
        evaluation_model: str = "llama-3.1-8b-instant",  # Changed to faster model by default
        fallback_model: str = "llama-3-8b-8192",  # Added fallback model
        results_dir: str = "./storage/evaluation",
        test_queries_path: str = None,
        max_queries: int = None,  # Added option to limit number of queries
        rate_limit_delay: int = 60  # Added delay for rate limits
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            rag_pipeline: An initialized RAG pipeline to evaluate
            groq_api_key: API key for Groq (for evaluation model)
            evaluation_model: Model to use for qualitative evaluations
            fallback_model: Fallback model to use if primary model hits rate limits
            results_dir: Directory to store evaluation results
            test_queries_path: Path to test queries JSON file
            max_queries: Maximum number of queries to evaluate (limits for faster results)
            rate_limit_delay: Seconds to wait when rate limit is hit
        """
        self.rag_pipeline = rag_pipeline
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.evaluation_model = evaluation_model
        self.fallback_model = fallback_model
        self.results_dir = results_dir
        self.test_queries_path = test_queries_path
        self.max_queries = max_queries
        self.rate_limit_delay = rate_limit_delay
        self.using_fallback = False
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize LLM for evaluations
        self.evaluator_llm = Groq(api_key=self.groq_api_key)
        
        # Load or create test queries
        self.test_queries = self._load_test_queries()
        
        # Limit number of queries if specified
        if self.max_queries and len(self.test_queries) > self.max_queries:
            self.test_queries = self.test_queries[:self.max_queries]
            logger.info(f"Limited to {self.max_queries} queries for faster evaluation")
        
        # Store results
        self.evaluation_results = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": self.evaluation_model,
                "num_queries": len(self.test_queries)
            },
            "query_results": [],
            "summary_metrics": {}
        }
    
    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """
        Load test queries from a JSON file or use default queries.
        
        Returns:
            List of test query objects
        """
        if self.test_queries_path and os.path.exists(self.test_queries_path):
            try:
                with open(self.test_queries_path, 'r') as f:
                    queries = json.load(f)
                logger.info(f"Loaded {len(queries)} test queries from {self.test_queries_path}")
                return queries
            except Exception as e:
                logger.error(f"Error loading test queries: {e}")
        
        # Default test queries if no file provided or loading failed
        # These are generic questions that would work with most academic documents
        default_queries = [
            {"query": "What are the main findings or conclusions of this document?", "category": "summary"},
            {"query": "What methodology was used in this research?", "category": "methodology"},
            {"query": "What are the key challenges mentioned in the document?", "category": "challenges"},
            {"query": "What future work is suggested in the document?", "category": "future_work"},
            # Reduced from 10 to 4 default queries for faster evaluation
        ]
        
        logger.info(f"Using {len(default_queries)} default test queries")
        return default_queries
    
    def create_custom_test_queries(self, output_path: str = "./test_queries.json") -> None:
        """
        Create custom test queries based on the content of the documents.
        This generates test queries that are more likely to have answers in the documents.
        
        Args:
            output_path: Path to save the generated test queries
        """
        if not self.rag_pipeline or not self.rag_pipeline.index:
            logger.error("RAG pipeline not initialized or no documents indexed")
            return
        
        # Get a sample of document chunks
        node_ids = self.rag_pipeline.index.docstore.get_all_document_hashes()
        sample_size = min(len(node_ids), 5)  # Take up to 5 chunks as samples
        sample_node_ids = np.random.choice(node_ids, size=sample_size, replace=False)
        
        sample_chunks = []
        for node_id in sample_node_ids:
            node = self.rag_pipeline.index.docstore.get_document(node_id)
            sample_chunks.append(node.text)
        
        # Generate questions using the LLM based on content
        prompt = f"""Given the following excerpts from documents, create 10 meaningful questions that can be answered using this content.
Make the questions diverse, including factual questions, conceptual questions, comparative questions, etc.
Use the exact terminology used in the passages to ensure the questions are answerable.

Document excerpts:
{chr(10).join([f"Excerpt {i+1}:{chr(10)}{chunk[:500]}..." for i, chunk in enumerate(sample_chunks)])}

Generate 10 questions in JSON format:
[
    {{"query": "Question 1 about the content?", "category": "categorization"}},
    {{"query": "Question 2 about the content?", "category": "categorization"}},
    ...
]

Use appropriate category labels like: factual, conceptual, summary, methodology, etc.
"""
        
        try:
            response = self.evaluator_llm.chat.completions.create(
                model=self.evaluation_model,  # Add the model parameter
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1500,
                stream=False
            )
            
            response_content = response.choices[0].message.content
            
            # Extract JSON part
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                queries = json.loads(json_str)
                
                # Save to file
                with open(output_path, 'w') as f:
                    json.dump(queries, f, indent=2)
                
                logger.info(f"Created {len(queries)} custom test queries saved to {output_path}")
                self.test_queries = queries
                return queries
            else:
                logger.error("Failed to extract JSON from LLM response")
                return None
                
        except Exception as e:
            logger.error(f"Error creating custom test queries: {e}")
            return None
    
    def run_evaluation(self, top_k_values: List[int] = [3, 5]) -> Dict[str, Any]:
        """
        Run a comprehensive evaluation of the RAG pipeline using multiple metrics.
        
        Args:
            top_k_values: List of top_k values to test for retrieval (reduced default from [3,5,7,10] to just [3,5])
            
        Returns:
            Evaluation results dictionary
        """
        if not self.rag_pipeline or not self.rag_pipeline.index:
            logger.error("RAG pipeline not initialized or no documents indexed")
            return None
        
        logger.info(f"Starting evaluation with {len(self.test_queries)} queries and {len(top_k_values)} top-k values")
        
        all_results = []
        
        # Test each query with different top_k values
        for top_k in top_k_values:
            logger.info(f"Evaluating with top_k = {top_k}")
            
            for query_obj in tqdm(self.test_queries, desc=f"Queries (top_k={top_k})"):
                query = query_obj["query"]
                category = query_obj.get("category", "general")
                
                # Run the query
                start_time = time.time()
                response = self.rag_pipeline.query(query, similarity_top_k=top_k)
                query_time = time.time() - start_time
                
                # Evaluate the response with LLM
                quality_metrics = self._evaluate_response_quality(query, response)
                
                # Collect results
                result = {
                    "query": query,
                    "category": category,
                    "top_k": top_k,
                    "answer": response.get("answer", ""),
                    "sources": response.get("sources", []),
                    "query_time": query_time,
                    "num_sources_returned": len(response.get("sources", [])),
                    "quality_metrics": quality_metrics
                }
                
                all_results.append(result)
        
        # Calculate summary metrics
        summary = self._calculate_summary_metrics(all_results)
        
        # Store the results
        self.evaluation_results["query_results"] = all_results
        self.evaluation_results["summary_metrics"] = summary
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {results_path}")
        
        # Generate visualizations
        self._generate_visualizations(all_results, summary, timestamp)
        
        return self.evaluation_results
    
    def _evaluate_response_quality(self, query: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of a response using the LLM.
        
        Args:
            query: The query that was asked
            response: The response from the RAG pipeline
            
        Returns:
            Quality metrics dictionary
        """
        answer = response.get("answer", "")
        sources = response.get("sources", [])
        
        # Prepare source text for evaluation
        source_text = "\n\n".join([
            f"Source {i+1}: {source.get('text', '')[:300]}..." 
            for i, source in enumerate(sources[:3])  # Only use first 3 sources with truncated text
        ])
        
        # Construct a shorter evaluation prompt
        prompt = f"""Evaluate this RAG response:

Query: {query}

Response: {answer}

Retrieved Sources:
{source_text}

Rate (1-10):
1. Relevance: How relevant is the response?
2. Factual Accuracy: Is it accurate based on sources?
3. Completeness: How completely does it answer?
4. Hallucination: Are there unsupported statements? (10=no hallucinations)
5. Overall score

JSON format only:
{{"relevance": 8, "factual_accuracy": 7, "completeness": 6, "hallucination_assessment": 9, "overall_score": 7.5}}"""

        try:
            # Get model to use based on fallback status
            current_model = self.fallback_model if self.using_fallback else self.evaluation_model
            
            # Handle rate limits with retry logic
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Get evaluation from LLM
                    eval_response = self.evaluator_llm.chat.completions.create(
                        model=current_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=500,  # Reduced from 800
                        stream=False
                    )
                    
                    evaluation_text = eval_response.choices[0].message.content
                    
                    # Parse the JSON response
                    import re
                    json_match = re.search(r'\{.*\}', evaluation_text, re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(0)
                        evaluation = json.loads(json_str)
                        return evaluation
                    else:
                        logger.warning("Failed to extract JSON from LLM evaluation")
                        # Return simplified evaluation on error
                        return {
                            "relevance": 5, 
                            "factual_accuracy": 5, 
                            "completeness": 5,
                            "hallucination_assessment": 5,
                            "overall_score": 5,
                            "error": "Failed to parse evaluation"
                        }
                
                except Exception as e:
                    retry_count += 1
                    error_message = str(e)
                    
                    # Check if this is a rate limit error
                    if is_rate_limit_error(error_message) and retry_count < max_retries:
                        logger.warning(f"Rate limit hit during evaluation. Attempt {retry_count}/{max_retries}")
                        
                        # If we're not using fallback model yet and hit rate limits, switch
                        if not self.using_fallback:
                            logger.info(f"Switching to fallback model: {self.fallback_model}")
                            self.using_fallback = True
                            current_model = self.fallback_model
                            # Reset retry count with new model
                            retry_count = 0
                            time.sleep(1)
                        else:
                            # Wait with exponential backoff
                            wait_time = min(self.rate_limit_delay * (2 ** (retry_count - 1)), 300)
                            logger.info(f"Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                    else:
                        logger.error(f"Error evaluating response on attempt {retry_count}: {e}")
                        if retry_count >= max_retries:
                            break
                        time.sleep(2)  # Brief wait before retry
            
            # If we've exhausted retries, return basic scores to continue evaluation
            return {
                "relevance": 5, 
                "factual_accuracy": 5, 
                "completeness": 5,
                "hallucination_assessment": 5,
                "overall_score": 5,
                "error": f"Error after {max_retries} retries: {str(e)}"
            }
                
        except Exception as e:
            logger.error(f"Unhandled error evaluating response quality: {e}")
            return {
                "relevance": 5, 
                "factual_accuracy": 5, 
                "completeness": 5,
                "hallucination_assessment": 5,
                "overall_score": 5,
                "error": str(e)
            }
    
    def _calculate_summary_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate summary metrics from individual query results.
        
        Args:
            results: List of individual query results
            
        Returns:
            Summary metrics dictionary
        """
        # Create a DataFrame for easier analysis
        df = pd.DataFrame()
        
        # Extract and flatten the nested data
        for result in results:
            row = {
                "query": result["query"],
                "category": result["category"],
                "top_k": result["top_k"],
                "query_time": result["query_time"],
                "num_sources": result["num_sources_returned"]
            }
            
            # Add quality metrics if available
            quality = result.get("quality_metrics", {})
            if "error" not in quality:
                for metric, value in quality.items():
                    if isinstance(value, (int, float)):
                        row[f"quality_{metric}"] = value
            
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        # Calculate summary statistics
        summary = {
            "avg_query_time": df["query_time"].mean(),
            "median_query_time": df["query_time"].median(),
            "avg_num_sources": df["num_sources"].mean(),
            "avg_score_by_top_k": {},
            "avg_score_by_category": {}
        }
        
        # Quality score columns
        quality_cols = [col for col in df.columns if col.startswith("quality_") and col != "quality_reasoning"]
        
        # Calculate average scores by top_k
        for top_k in df["top_k"].unique():
            top_k_df = df[df["top_k"] == top_k]
            top_k_scores = {}
            
            for col in quality_cols:
                metric_name = col.replace("quality_", "")
                top_k_scores[metric_name] = top_k_df[col].mean()
            
            summary["avg_score_by_top_k"][str(top_k)] = top_k_scores
        
        # Calculate average scores by category
        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            cat_scores = {}
            
            for col in quality_cols:
                metric_name = col.replace("quality_", "")
                cat_scores[metric_name] = cat_df[col].mean()
            
            summary["avg_score_by_category"][category] = cat_scores
        
        # Find optimal top_k based on overall score
        if "quality_overall_score" in df.columns:
            avg_by_top_k = df.groupby("top_k")["quality_overall_score"].mean()
            best_top_k = avg_by_top_k.idxmax()
            summary["optimal_top_k"] = int(best_top_k)
            summary["optimal_top_k_score"] = float(avg_by_top_k.max())
        
        return summary
    
    def _generate_visualizations(self, results: List[Dict[str, Any]], summary: Dict[str, Any], timestamp: str):
        """
        Generate visualizations from evaluation results.
        
        Args:
            results: List of individual query results
            summary: Summary metrics dictionary
            timestamp: Timestamp for filename
        """
        # Create a DataFrame for visualization
        df = pd.DataFrame()
        
        # Extract and flatten the data
        for result in results:
            row = {
                "query": result["query"],
                "category": result["category"],
                "top_k": result["top_k"],
                "query_time": result["query_time"],
                "num_sources": result["num_sources_returned"]
            }
            
            # Add quality metrics if available
            quality = result.get("quality_metrics", {})
            if "error" not in quality:
                for metric, value in quality.items():
                    if isinstance(value, (int, float)):
                        row[f"quality_{metric}"] = value
            
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Top-K vs. Quality Metrics
        quality_metrics = [col for col in df.columns if col.startswith("quality_") and col != "quality_reasoning"]
        
        plt.figure(figsize=(14, 8))
        for metric in quality_metrics:
            metric_name = metric.replace("quality_", "").replace("_", " ").title()
            avg_by_top_k = df.groupby("top_k")[metric].mean()
            plt.plot(avg_by_top_k.index, avg_by_top_k.values, marker='o', linewidth=2, label=metric_name)
        
        plt.xlabel('Top-K (Number of Retrieved Documents)', fontsize=12)
        plt.ylabel('Average Score (1-10)', fontsize=12)
        plt.title('Effect of Top-K on Response Quality Metrics', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.results_dir, f"topk_vs_quality_{timestamp}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        # 2. Query Time vs. Top-K
        plt.figure(figsize=(12, 6))
        avg_time_by_top_k = df.groupby("top_k")["query_time"].mean()
        plt.bar(avg_time_by_top_k.index.astype(str), avg_time_by_top_k.values, color='teal', alpha=0.7)
        
        plt.xlabel('Top-K (Number of Retrieved Documents)', fontsize=12)
        plt.ylabel('Average Query Time (seconds)', fontsize=12)
        plt.title('Query Performance by Top-K', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.results_dir, f"query_time_{timestamp}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        # 3. Performance by Category
        if "quality_overall_score" in df.columns:
            plt.figure(figsize=(14, 7))
            sns.boxplot(x="category", y="quality_overall_score", data=df)
            
            plt.xlabel('Query Category', fontsize=12)
            plt.ylabel('Overall Score (1-10)', fontsize=12)
            plt.title('Performance by Query Category', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the figure
            fig_path = os.path.join(self.results_dir, f"category_performance_{timestamp}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        # 4. Heatmap of Quality Metrics
        if quality_metrics:
            # Calculate correlation between metrics
            corr_df = df[quality_metrics].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
            
            plt.title('Correlation Between Quality Metrics', fontsize=14)
            plt.tight_layout()
            
            # Save the figure
            fig_path = os.path.join(self.results_dir, f"metric_correlation_{timestamp}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        logger.info(f"Generated visualizations saved to {self.results_dir}")
    
    def create_report(self, output_path: str = None) -> str:
        """
        Create a markdown report of the evaluation results.
        
        Args:
            output_path: Path to save the markdown report
            
        Returns:
            Path to the saved report
        """
        if not self.evaluation_results or not self.evaluation_results.get("query_results"):
            logger.error("No evaluation results available")
            return None
        
        # Format timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Build report
        report = f"""# RAG System Evaluation Report
*Generated on: {timestamp}*

## Overview
- **Model:** {self.evaluation_results["metadata"]["model"]}
- **Queries Tested:** {self.evaluation_results["metadata"]["num_queries"]}
- **Top-K Values Tested:** {", ".join([str(x["top_k"]) for x in self.evaluation_results["query_results"][:10]]) + "..."}

## Summary Metrics

### Performance
- **Average Query Time:** {self.evaluation_results["summary_metrics"]["avg_query_time"]:.3f} seconds
- **Median Query Time:** {self.evaluation_results["summary_metrics"]["median_query_time"]:.3f} seconds
- **Average Number of Sources Retrieved:** {self.evaluation_results["summary_metrics"]["avg_num_sources"]:.2f}

### Optimal Configuration
"""

        # Add optimal configuration if available
        if "optimal_top_k" in self.evaluation_results["summary_metrics"]:
            report += f"""- **Optimal Top-K Value:** {self.evaluation_results["summary_metrics"]["optimal_top_k"]}
- **Optimal Top-K Overall Score:** {self.evaluation_results["summary_metrics"]["optimal_top_k_score"]:.2f}/10
"""

        # Add scores by top-k
        report += "\n### Quality Scores by Top-K\n\n"
        report += "| Top-K | Relevance | Factual Accuracy | Completeness | Hallucination Assessment | Coherence | Overall |\n"
        report += "|-------|-----------|------------------|-------------|--------------------------|-----------|--------|\n"
        
        for top_k, scores in self.evaluation_results["summary_metrics"]["avg_score_by_top_k"].items():
            # Format values, handling 'N/A' strings properly
            relevance = f"{scores.get('relevance', 'N/A'):.2f}" if isinstance(scores.get('relevance', 'N/A'), (int, float)) else 'N/A'
            factual = f"{scores.get('factual_accuracy', 'N/A'):.2f}" if isinstance(scores.get('factual_accuracy', 'N/A'), (int, float)) else 'N/A'
            completeness = f"{scores.get('completeness', 'N/A'):.2f}" if isinstance(scores.get('completeness', 'N/A'), (int, float)) else 'N/A'
            hallucination = f"{scores.get('hallucination_assessment', 'N/A'):.2f}" if isinstance(scores.get('hallucination_assessment', 'N/A'), (int, float)) else 'N/A'
            coherence = f"{scores.get('coherence', 'N/A'):.2f}" if isinstance(scores.get('coherence', 'N/A'), (int, float)) else 'N/A'
            overall = f"{scores.get('overall_score', 'N/A'):.2f}" if isinstance(scores.get('overall_score', 'N/A'), (int, float)) else 'N/A'
            
            report += f"| {top_k} | {relevance} | {factual} | {completeness} | {hallucination} | {coherence} | {overall} |\n"
        
        # Add scores by category
        report += "\n### Quality Scores by Query Category\n\n"
        report += "| Category | Relevance | Factual Accuracy | Completeness | Hallucination Assessment | Coherence | Overall |\n"
        report += "|----------|-----------|------------------|-------------|--------------------------|-----------|--------|\n"
        
        for category, scores in self.evaluation_results["summary_metrics"]["avg_score_by_category"].items():
            # Format values, handling 'N/A' strings properly
            relevance = f"{scores.get('relevance', 'N/A'):.2f}" if isinstance(scores.get('relevance', 'N/A'), (int, float)) else 'N/A'
            factual = f"{scores.get('factual_accuracy', 'N/A'):.2f}" if isinstance(scores.get('factual_accuracy', 'N/A'), (int, float)) else 'N/A'
            completeness = f"{scores.get('completeness', 'N/A'):.2f}" if isinstance(scores.get('completeness', 'N/A'), (int, float)) else 'N/A'
            hallucination = f"{scores.get('hallucination_assessment', 'N/A'):.2f}" if isinstance(scores.get('hallucination_assessment', 'N/A'), (int, float)) else 'N/A'
            coherence = f"{scores.get('coherence', 'N/A'):.2f}" if isinstance(scores.get('coherence', 'N/A'), (int, float)) else 'N/A'
            overall = f"{scores.get('overall_score', 'N/A'):.2f}" if isinstance(scores.get('overall_score', 'N/A'), (int, float)) else 'N/A'
            
            report += f"| {category} | {relevance} | {factual} | {completeness} | {hallucination} | {coherence} | {overall} |\n"
        
        # Add sample query results
        report += "\n## Sample Query Results\n\n"
        
        # Use optimal top-k if available, otherwise use the first one
        optimal_top_k = self.evaluation_results["summary_metrics"].get("optimal_top_k")
        
        sample_results = []
        if optimal_top_k:
            # Get results with optimal top-k
            sample_results = [r for r in self.evaluation_results["query_results"] if r["top_k"] == optimal_top_k][:5]
        else:
            # Just get some sample results
            sample_results = self.evaluation_results["query_results"][:5]
        
        for i, result in enumerate(sample_results):
            report += f"### Query {i+1}: {result['query']}\n"
            report += f"*Category: {result['category']}*\n\n"
            report += f"**Answer:**\n{result['answer']}\n\n"
            
            # Add quality metrics if available
            if "quality_metrics" in result and "error" not in result["quality_metrics"]:
                report += "**Quality Metrics:**\n"
                for metric, value in result["quality_metrics"].items():
                    if isinstance(value, (int, float)):
                        report += f"- {metric.replace('_', ' ').title()}: {value}\n"
                
                if "reasoning" in result["quality_metrics"]:
                    report += f"\n**Evaluator's Reasoning:**\n{result['quality_metrics']['reasoning']}\n"
            
            report += "\n---\n\n"
        
        # Add recommendations
        report += "\n## Recommendations\n\n"
        
        # Find optimal top-k
        if "optimal_top_k" in self.evaluation_results["summary_metrics"]:
            report += f"1. **Optimal Retrieval Setting:** Use top-k = {self.evaluation_results['summary_metrics']['optimal_top_k']} for the best balance of quality and performance.\n\n"
        
        # Identify strengths and weaknesses
        if "avg_score_by_category" in self.evaluation_results["summary_metrics"]:
            # Identify best and worst categories
            categories = list(self.evaluation_results["summary_metrics"]["avg_score_by_category"].keys())
            if categories:
                best_category = max(categories, key=lambda c: self.evaluation_results["summary_metrics"]["avg_score_by_category"][c].get("overall_score", 0))
                worst_category = min(categories, key=lambda c: self.evaluation_results["summary_metrics"]["avg_score_by_category"][c].get("overall_score", 0))
                
                report += f"2. **Strengths:** The system performs best on '{best_category}' queries.\n\n"
                report += f"3. **Areas for Improvement:** Consider enhancing the system's ability to handle '{worst_category}' queries.\n\n"
        
        # Add appendix with all metrics
        report += "\n## Appendix: Raw Evaluation Metrics\n\n"
        report += "This section contains detailed metrics for all evaluated queries.\n\n"
        
        # Determine output path if not provided
        if not output_path:
            timestamp_file = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.results_dir, f"evaluation_report_{timestamp_file}.md")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report generated and saved to {output_path}")
        return output_path
        

def main():
    """
    Main function to run the RAG evaluator.
    """
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--create-queries", action="store_true", help="Create custom test queries based on document content")
    parser.add_argument("--test-queries", type=str, help="Path to test queries JSON file")
    parser.add_argument("--output-dir", type=str, default="./storage/evaluation", help="Directory to store results")
    parser.add_argument("--top-k", type=str, default="3,5", help="Comma-separated list of top-k values to test")
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instant", help="Model to use for evaluation")
    parser.add_argument("--fallback-model", type=str, default="llama-3-8b-8192", help="Fallback model to use if primary hits rate limits")
    parser.add_argument("--max-queries", type=int, help="Maximum number of queries to evaluate (for faster results)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory containing documents to index")
    
    args = parser.parse_args()
    
    # Parse top-k values
    top_k_values = [int(k) for k in args.top_k.split(",")]
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        llamaparse_api_key=os.getenv("PARSE_API_KEY"),
        storage_dir="./storage"
    )
    
    # Track whether any documents were successfully indexed
    indexed_documents_count = 0
    
    # Load documents if needed
    # Check if the index exists and has documents
    if not hasattr(rag_pipeline, 'index') or not rag_pipeline.index:
        logger.info("No index found. Loading documents from data directory...")
        # Check if there are parsed documents (.pkl files) in the data directory
        import glob
        parsed_files = glob.glob(os.path.join(args.data_dir, "*.pkl"))
        
        if parsed_files:
            logger.info(f"Found {len(parsed_files)} parsed document files. Loading them...")
            for file_path in parsed_files:
                try:
                    # Load the parsed document
                    with open(file_path, 'rb') as f:
                        documents = pickle.load(f)
                    
                    # Check what was loaded
                    logger.info(f"Loaded data type from {file_path}: {type(documents)}")
                    
                    # Handle different document formats
                    if isinstance(documents, list):
                        doc_count = len(documents)
                        if doc_count > 0:
                            # Try to extract Document objects if they're in a nested structure
                            try:
                                # If this is a list of dictionaries with a 'documents' key, extract those
                                if isinstance(documents[0], dict) and 'documents' in documents[0]:
                                    flat_docs = []
                                    for item in documents:
                                        if isinstance(item['documents'], list):
                                            flat_docs.extend(item['documents'])
                                        else:
                                            flat_docs.append(item['documents'])
                                    documents = flat_docs
                                    doc_count = len(documents)
                                
                                # Now index the documents
                                rag_pipeline.index_documents(documents)
                                indexed_documents_count += doc_count
                                logger.info(f"Successfully indexed {doc_count} documents from {file_path}")
                            except Exception as ex:
                                logger.error(f"Error during document extraction from list: {ex}")
                                # Try direct indexing as fallback
                                try:
                                    rag_pipeline.index_documents(documents)
                                    indexed_documents_count += doc_count
                                    logger.info(f"Successfully indexed {doc_count} documents with direct approach from {file_path}")
                                except Exception as direct_ex:
                                    logger.error(f"Direct indexing also failed: {direct_ex}")
                        else:
                            logger.info(f"No documents found in {file_path}")
                    elif isinstance(documents, dict):
                        # Try to extract documents from the dictionary
                        try:
                            if 'documents' in documents and isinstance(documents['documents'], list):
                                doc_count = len(documents['documents'])
                                rag_pipeline.index_documents(documents['documents'])
                                indexed_documents_count += doc_count
                                logger.info(f"Successfully indexed {doc_count} documents from dictionary in {file_path}")
                            else:
                                # Try to convert the dict to a Document if it has the right keys
                                rag_pipeline.index_documents([documents])
                                indexed_documents_count += 1
                                logger.info(f"Successfully indexed 1 document (dictionary) from {file_path}")
                        except Exception as dict_ex:
                            logger.error(f"Error indexing dictionary from {file_path}: {dict_ex}")
                    else:
                        # If it's not a list or dict, try to index it directly
                        try:
                            rag_pipeline.index_documents([documents])
                            indexed_documents_count += 1
                            logger.info(f"Successfully indexed 1 document (unknown type) from {file_path}")
                        except Exception as sub_e:
                            logger.error(f"Error indexing document from {file_path}: {sub_e}")
                            
                    # Verify indexing actually worked by checking the index
                    if hasattr(rag_pipeline, 'index') and rag_pipeline.index.docstore:
                        if hasattr(rag_pipeline.index.docstore, 'get_all_document_hashes'):
                            current_docs = len(rag_pipeline.index.docstore.get_all_document_hashes())
                            logger.info(f"Current total documents in index: {current_docs}")
                            
                except Exception as e:
                    logger.error(f"Error loading or indexing documents from {file_path}: {e}")
                    # Print the full error for debugging
                    import traceback
                    logger.error(traceback.format_exc())
                    
        else:
            # Check for PDF files
            pdf_files = glob.glob(os.path.join(args.data_dir, "*.pdf"))
            if pdf_files:
                logger.info(f"Found {len(pdf_files)} PDF files. Parsing and indexing them...")
                for pdf_path in pdf_files:
                    try:
                        logger.info(f"Processing {pdf_path}...")
                        docs = rag_pipeline.ingest_document(pdf_path)
                        if docs:
                            indexed_documents_count += len(docs)
                            logger.info(f"Successfully indexed {len(docs)} documents from {pdf_path}")
                    except Exception as e:
                        logger.error(f"Error processing or indexing {pdf_path}: {e}")
    else:
        # If the index already exists, count the documents
        try:
            if hasattr(rag_pipeline.index.docstore, 'get_all_document_hashes'):
                indexed_documents_count = len(rag_pipeline.index.docstore.get_all_document_hashes())
                logger.info(f"Found existing index with {indexed_documents_count} documents")
            else:
                # Assume there are documents if the index exists but can't count them
                indexed_documents_count = 1
                logger.info("Found existing index but couldn't count documents")
        except Exception as e:
            logger.error(f"Error checking existing index: {e}")
            # Assume index is valid
            indexed_documents_count = 1
    
    # Check if documents are indexed before proceeding
    if not hasattr(rag_pipeline, 'index') or not rag_pipeline.index or indexed_documents_count == 0:
        logger.error("No documents indexed. Please add documents to the data directory.")
        return
    
    # Create evaluator
    evaluator = RAGEvaluator(
        rag_pipeline=rag_pipeline,
        evaluation_model=args.model,
        results_dir=args.output_dir,
        test_queries_path=args.test_queries
    )
    
    # Create custom test queries if requested
    if args.create_queries:
        evaluator.create_custom_test_queries()
    
    # Run evaluation
    results = evaluator.run_evaluation(top_k_values=top_k_values)
    
    # Generate report
    if results:
        report_path = evaluator.create_report()
        print(f"Evaluation completed. Report saved to {report_path}")
    else:
        print("Evaluation could not be completed. Check the logs for details.")


if __name__ == "__main__":
    main()