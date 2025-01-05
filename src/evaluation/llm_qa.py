import argparse
import re
import time

import mlflow
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from src.llm.deployments import AvailableChatModels, get_chat_model


class LLMJudge:
    def __init__(self, model_name, embedding_model_name="all-MiniLM-L6-v2"):
        self.llm = get_chat_model(model_name=model_name, temperature=0)
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def generate_response(self, input_text):
        """Generate a response from the LLM."""
        messages = [("user", input_text)]
        return self.llm.invoke(messages).content

    def evaluate_response(self, input_text, ground_truth, model_response):
        """Evaluate the response using LLM-as-a-judge."""
        judging_prompt = f"""
        Evaluate the following model's response based on its relevance, accuracy, match with ground truth, and completeness:

        Input: {input_text}

        Model's Response: {model_response}

        Ground Truth: {ground_truth}

        The evaluation should be based on the following criteria:
        1. Relevance: How relevant is the response to the input question?
        2. Accuracy: How accurate is the information provided in the response?
        3. Completeness: How complete is the information provided in the response?
        4. Ground Truth Match: How well does the response match the ground truth? (not the other way around)

        Please provide your evaluation in the following structured format:
        1. Overall Score: X/10
        2. Sub-scores:
           - Relevance: X/10
           - Accuracy: X/10
           - Completeness: X/10
           - Ground Truth Match: X/10
        3. Justification: Write a short explanation of your scores.

        Ensure the output strictly follows this structure.
        """  # noqa
        response = self.llm.invoke([("user", judging_prompt)])
        return response.content

    @staticmethod
    def parse_llm_response(response):
        """Parse the structured LLM response to extract scores and justification."""
        # Extract scores
        overall_score_match = re.search(r"Overall Score:\s*([\d.]+)/10", response, re.IGNORECASE)
        relevance_match = re.search(r"Relevance:\s*([\d.]+)/10", response, re.IGNORECASE)
        accuracy_match = re.search(r"Accuracy:\s*([\d.]+)/10", response, re.IGNORECASE)
        completeness_match = re.search(r"Completeness:\s*([\d.]+)/10", response, re.IGNORECASE)
        ground_truth_match_match = re.search(r"Ground Truth Match:\s*([\d.]+)/10", response, re.IGNORECASE)

        # Parse scores or assign NaN if missing
        overall_score = float(overall_score_match.group(1)) if overall_score_match else np.nan
        relevance = float(relevance_match.group(1)) if relevance_match else np.nan
        accuracy = float(accuracy_match.group(1)) if accuracy_match else np.nan
        completeness = float(completeness_match.group(1)) if completeness_match else np.nan
        ground_truth_match = float(ground_truth_match_match.group(1)) if ground_truth_match_match else np.nan

        # Extract justification
        justification_match = re.search(r"Justification:\s*(.+)", response, re.DOTALL)
        justification = justification_match.group(1).strip() if justification_match else ""

        return {
            "overall_score": overall_score,
            "relevance": relevance,
            "accuracy": accuracy,
            "completeness": completeness,
            "ground_truth_match": ground_truth_match,
            "justification": justification,
        }

    def calculate_match_score(self, ground_truth, model_response):
        """Calculate semantic similarity between ground truth and model response."""
        embeddings = self.embedding_model.encode([ground_truth, model_response], convert_to_tensor=True)
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate LLM responses.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=AvailableChatModels.LLAMA_3_2.name,
        help="""Name of the model to use for evaluation (default: LLAMA_3_2,
          available: LLAMA_3_2, LLAMA_3_1, GEMMA_2)""",
    )
    args = parser.parse_args()
    model_name = AvailableChatModels[args.model_name]
    print(f"Evaluating model: {model_name}")
    # Initialize the judge
    judge = LLMJudge(model_name=model_name)

    # Evaluation dataset
    eval_data = pd.DataFrame(
        {
            "inputs": [
                "How does useEffect() work?",
                "What does the static keyword in a function mean?",
                "What does the 'finally' block in Python do?",
                "What is the difference between multiprocessing and multithreading?",
                "Who is Michael Scott?",  # Incorrect ground truth for testing
            ],
            "ground_truth": [
                "The useEffect() hook tells React that your component needs to do something after render...",
                "Static members belong to the class, rather than a specific instance...",
                "'Finally' defines a block of code to run when the try... except...else block is final...",
                "Multithreading refers to the ability of a processor to execute multiple threads...",
                "He is actually Dwight Schrute.",
            ],
        }
    )

    # Initialize metrics storage
    evaluation_records = []

    # Start MLflow run
    mlflow.set_experiment("Basic QA Eval")
    with mlflow.start_run():
        # log the model name
        mlflow.log_param("model_name", model_name)
        # set model name as tag
        mlflow.set_tag("model_name", model_name)
        # log all params of the llm model
        for key, value in dict(judge.llm._get_ls_params()).items():
            mlflow.log_param(f"llm_{key}", value)

        for _, row in eval_data.iterrows():
            input_text = row["inputs"]
            ground_truth = row["ground_truth"]

            # Generate model response
            start_time = time.time()
            model_response = judge.generate_response(input_text)
            end_time = time.time()
            response_time = end_time - start_time
            # Add response time to the evaluation records
            evaluation_records.append({"response_time": response_time})

            # Calculate embedding-based match score
            embedding_match_score = judge.calculate_match_score(ground_truth, model_response)

            # Evaluate using LLM-as-a-judge
            evaluation_response = judge.evaluate_response(input_text, ground_truth, model_response)

            # Parse LLM-as-a-judge response
            parsed_response = judge.parse_llm_response(evaluation_response)

            # Append results
            evaluation_records.append(
                {
                    "input": input_text,
                    "ground_truth": ground_truth,
                    "model_response": model_response,
                    "embedding_match_score": float(embedding_match_score),
                    "ground_truth_match_score": float(parsed_response["ground_truth_match"]),
                    "overall_llm_score": float(parsed_response["overall_score"]),
                    "relevance": float(parsed_response["relevance"]),
                    "accuracy": float(parsed_response["accuracy"]),
                    "completeness": float(parsed_response["completeness"]),
                    "justification": parsed_response["justification"],
                }
            )

        # Log metrics and artifacts
        evaluation_df = pd.DataFrame(evaluation_records)
        # calculate the average score of each llm metric
        llm_metrics = ["ground_truth_match_score", "relevance", "accuracy", "completeness"]
        evaluation_df["overall_avg_score"] = evaluation_df[llm_metrics].mean(axis=1)
        evaluation_df.to_csv("evaluation_records_with_match.csv", index=False)

        mlflow.log_metric("avg_embedding_match_score", np.nanmean(evaluation_df["embedding_match_score"]))
        mlflow.log_metric("avg_ground_truth_match_score", np.nanmean(evaluation_df["ground_truth_match_score"]))
        mlflow.log_metric("avg_overall_llm_score", np.nanmean(evaluation_df["overall_llm_score"]))
        mlflow.log_metric("avg_overall_score", np.nanmean(evaluation_df["overall_llm_score"]))
        mlflow.log_metric("avg_relevance_score", np.nanmean(evaluation_df["relevance"]))
        mlflow.log_metric("avg_accuracy_score", np.nanmean(evaluation_df["accuracy"]))
        mlflow.log_metric("avg_completeness_score", np.nanmean(evaluation_df["completeness"]))
        mlflow.log_metric("avg_response_time", np.nanmean(evaluation_df["response_time"]))
        mlflow.log_artifact("evaluation_records_with_match.csv")


if __name__ == "__main__":
    main()
