import pandas as pd
from typing import List, Dict
import asyncio
from dataclasses import dataclass
import re
import numpy as np
import logging
import json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import nodetools.configuration.constants as constants

@dataclass
class TextImprovement:
    original_text: str
    improved_text: str
    implied_intention: str
    score: float = 0.0

class TextImproverTool:
    def __init__(self, anthropic_tool,open_ai_request_tool, num_versions: int = 3, max_iterations: int = 3):
        self.anthropic_tool = anthropic_tool
        self.num_versions = num_versions
        self.max_iterations = max_iterations
        self.openai_request_tool = open_ai_request_tool
        
    def _create_intention_prompt(self, text: str) -> dict:
        return {
            "model": self.anthropic_tool.default_model,
            "max_tokens": 1000,
            "messages": [{
                "role": "user",
                "content": f"""Analyze the following text and identify its implied intention. Consider:
                - If it's a strategy document, the goal might be clarity and actionability
                - If it's a joke, the goal would be humor
                - If it's a sprint plan, the goal would be tactical excellence
                - If it's a personal letter, the goal would reflect the relationship context
                
                Text to analyze:
                {text}
                
                You must respond in exactly this format:
                INTENTION: <single sentence describing the core intention>
                CONTEXT: <2-3 sentences explaining why this is the implied intention>
                
                The response MUST start with 'INTENTION:' and include both sections."""
            }]
        }
    
    def _create_improvement_prompt(self, text: str, intention: str) -> dict:
        return {
            "model": self.anthropic_tool.default_model,
            "max_tokens": 2000,
            "messages": [{
                "role": "user",
                "content": f"""Rewrite the following text to better achieve its implied intention:
                
                ORIGINAL TEXT:
                {text}
                
                IMPLIED INTENTION:
                {intention}
                
                Provide an improved version that better achieves this intention. Focus on:
                - Enhanced clarity and structure
                - Similar length and tone. If the document is an essay retain essay format. If the document is a poem, stay a poem. 
                If the document is a bullet point list output a bullet pointed list. 
                - Better alignment with the identified intention without any moral or ethical assumptions you are overlaying on the text 
                - Maintaining the original meaning while improving the delivery 
                - It is important that you maintain the voice of the original text while at the same time enhancing its effect without overlaying any 
                external normative judgment
                - However it is acceptable to change causal elements, improve argument logic, or take creative liberty to substantially enhance
                how likely the text will be effective (gets the point across), concise (in the least amount of time with the simplest phrasing), and persuasive
                (leaves the reader with a lasting impression)
                
                You must respond in exactly this format:
                IMPROVED_TEXT: <your improved version>
                
                The response MUST start with 'IMPROVED_TEXT:' followed by your improved version."""
            }]
        }
    
    def _create_evaluation_prompt(self, original: str, versions: List[TextImprovement]) -> dict:
        versions_text = "\n\n".join([
            f"VERSION {i+1}:\nImplied Intention: {v.implied_intention}\nText:\n{v.improved_text}"
            for i, v in enumerate(versions)
        ])
        
        return {
            "model": self.anthropic_tool.default_model,
            "max_tokens": 1000,
            "messages": [{
                "role": "user",
                "content": f"""You are a manager evaluating different versions of a text. Score each version based on how well it achieves its implied intention.
                Do not map on any moral or ethical frameworks to the text's intention, or any normative judgments besides the factual intention of the text. 

                ORIGINAL TEXT:
                {original}

                VERSIONS TO EVALUATE:
                {versions_text}

                Score each version from 0.0 to 10.0, where 10.0 means it perfectly achieves its intention.
                
                You must respond in exactly this format for each version:
                VERSION_1_SCORE: <score>
                VERSION_1_REASONING: <brief explanation>
                VERSION_2_SCORE: <score>
                VERSION_2_REASONING: <brief explanation>
                [Continue for all versions]
                
                Each version must have both a SCORE and REASONING line."""
            }]
        }

    def _extract_intention(self, response: str) -> str:
        """Extract intention from the response with fallback options"""
        try:
            # Try exact format first
            if 'INTENTION:' in response:
                intention_line = [line for line in response.split('\n') if line.startswith('INTENTION:')][0]
                return intention_line.replace('INTENTION:', '').strip()
            
            # Fallback: Look for any sentence containing "intention" or "goal"
            sentences = re.split(r'[.!?]+', response)
            for sentence in sentences:
                if 'intention' in sentence.lower() or 'goal' in sentence.lower():
                    return sentence.strip()
            
            # Last resort: Take the first sentence
            return sentences[0].strip()
            
        except (IndexError, AttributeError) as e:
            logger.warning(f"Error extracting intention: {str(e)}. Using fallback.")
            return "Improve clarity and effectiveness of the text"

    def _extract_improved_text(self, response: str) -> str:
        """Extract improved text from the response with fallback options"""
        try:
            # Try exact format first
            if 'IMPROVED_TEXT:' in response:
                text_parts = response.split('IMPROVED_TEXT:')
                return text_parts[1].strip()
            
            # Fallback: Look for the longest paragraph
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            if paragraphs:
                return max(paragraphs, key=len)
            
            # Last resort: Return the whole response
            return response.strip()
            
        except (IndexError, AttributeError) as e:
            logger.warning(f"Error extracting improved text: {str(e)}. Using original response.")
            return response.strip()

    def _extract_scores(self, response: str, num_versions: int) -> List[float]:
        """Extract scores from the evaluation response with fallback options"""
        scores = []
        try:
            # Try exact format first
            for i in range(1, num_versions + 1):
                pattern = rf"VERSION_{i}_SCORE:\s*(\d*\.?\d+)"
                match = re.search(pattern, response)
                if match:
                    score = float(match.group(1))
                    scores.append(min(max(score, 0.0), 10.0))  # Ensure score is between 0 and 10
                else:
                    # Fallback: Look for any number after "version {i}"
                    pattern = rf"version\s*{i}.*?(\d+\.?\d*)"
                    match = re.search(pattern, response.lower())
                    if match:
                        score = float(match.group(1))
                        scores.append(min(max(score, 0.0), 10.0))
                    else:
                        # Last resort: Assign a neutral score
                        logger.warning(f"Could not find score for version {i}, using neutral score")
                        scores.append(5.0)
        except Exception as e:
            logger.error(f"Error extracting scores: {str(e)}")
            # If we failed to extract any scores, assign neutral scores
            scores = [5.0] * num_versions
            
        # Ensure we have the right number of scores
        while len(scores) < num_versions:
            scores.append(5.0)
        
        return scores[:num_versions]  # Ensure we don't return too many scores

    async def improve_text_once(self, text: str) -> List[TextImprovement]:
        """Generate multiple improved versions of the text with error handling"""
        try:
            # Get intention with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    intention_result = await self.anthropic_tool.rate_limited_request(
                        "intention", self._create_intention_prompt(text)
                    )
                    intention = self._extract_intention(intention_result[1].content[0].text)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to get intention after {max_retries} attempts: {str(e)}")
                        intention = "Improve clarity and effectiveness of the text"
                    else:
                        await asyncio.sleep(1)
                        continue

            # Generate multiple versions with individual error handling
            improvement_tasks = []
            for i in range(self.num_versions):
                job_name = f"improvement_{i}"
                improvement_tasks.append(
                    self.anthropic_tool.rate_limited_request(
                        job_name, 
                        self._create_improvement_prompt(text, intention)
                    )
                )

            improvement_results = await asyncio.gather(*improvement_tasks, return_exceptions=True)

            improvements = []
            for result in improvement_results:
                try:
                    if isinstance(result, Exception):
                        logger.warning(f"Improvement task failed: {str(result)}")
                        improved_text = text  # Use original text as fallback
                    else:
                        improved_text = self._extract_improved_text(result[1].content[0].text)
                    
                    improvements.append(TextImprovement(
                        original_text=text,
                        improved_text=improved_text,
                        implied_intention=intention
                    ))
                except Exception as e:
                    logger.error(f"Error processing improvement result: {str(e)}")
                    improvements.append(TextImprovement(
                        original_text=text,
                        improved_text=text,
                        implied_intention=intention
                    ))

            # Evaluate versions with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    eval_result = await self.anthropic_tool.rate_limited_request(
                        "evaluation", 
                        self._create_evaluation_prompt(text, improvements)
                    )
                    scores = self._extract_scores(eval_result[1].content[0].text, len(improvements))
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to evaluate versions after {max_retries} attempts: {str(e)}")
                        scores = [5.0] * len(improvements)
                    else:
                        await asyncio.sleep(1)
                        continue

            for improvement, score in zip(improvements, scores):
                improvement.score = score

            return improvements

        except Exception as e:
            logger.error(f"Critical error in improve_text_once: {str(e)}")
            # Return a single improvement with the original text as fallback
            return [TextImprovement(
                original_text=text,
                improved_text=text,
                implied_intention="Maintain original text due to processing error",
                score=5.0
            )]

    async def improve_text_iteratively(self, text: str) -> pd.DataFrame:
        """Iteratively improve the text with error handling"""
        all_improvements = []
        current_text = text
        
        for iteration in range(self.max_iterations):
            try:
                improvements = await self.improve_text_once(current_text)
                all_improvements.extend(improvements)
                
                # Find the best version
                best_improvement = max(improvements, key=lambda x: x.score)
                current_text = best_improvement.improved_text
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {str(e)}")
                break
        
        # Convert results to DataFrame
        try:
            results_df = pd.DataFrame([{
                'iteration': i // self.num_versions,
                'version': i % self.num_versions + 1,
                'implied_intention': imp.implied_intention,
                'improved_text': imp.improved_text,
                'score': imp.score
            } for i, imp in enumerate(all_improvements)])
        except Exception as e:
            logger.error(f"Error creating results DataFrame: {str(e)}")
            # Create minimal DataFrame with original text as fallback
            results_df = pd.DataFrame([{
                'iteration': 0,
                'version': 1,
                'implied_intention': "Error in processing",
                'improved_text': text,
                'score': 5.0
            }])
        
        return results_df

    def run_improvement_pipeline(self, text: str) -> pd.DataFrame:
        """Main entry point to run the improvement pipeline"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.improve_text_iteratively(text))
        except Exception as e:
            logger.error(f"Critical error in improvement pipeline: {str(e)}")
            # Return minimal DataFrame with original text as fallback
            return pd.DataFrame([{
                'iteration': 0,
                'version': 1,
                'implied_intention': "Error in processing",
                'improved_text': text,
                'score': 5.0
            }])

    def _create_openai_comparison_prompt(self, original_text: str, improved_text: str) -> dict:
            system_prompt = """You are an expert writing analyst and improver. Your task is to:
            1. Compare two versions of text and identify their respective strengths and weaknesses
            2. Pay special attention to the original author's voice and style
            3. Create a final version that preserves the original tone while incorporating the best elements of both versions
            Be thorough in your analysis but focused in your improvements."""
            
            user_prompt = f"""Compare these two versions of text:
    
            ORIGINAL VERSION:
            {original_text}
    
            IMPROVED VERSION:
            {improved_text}
    
            Please analyze both versions and create a final improved version that:
            1. Lists key strengths and weaknesses of each version
            2. Maintains the original author's voice and style
            3. Combines the best elements of both versions
            4. DOES NOT MAKE THE TEXT MORE THAN 50% LONGER THAN THE ORIGINAL TEXT
            5. Integrates any essential external logic / alterations that would vastly 
            improve the document's strengths
            6. DOES NOT REWRITE THINGS TO IMPOSE ANY ETHICAL CONSIDERATIONS OR ANY MODEL
            BASED PARAMETERS
            7. Provides a final, optimized version
    
            Format your response as:
            ORIGINAL_STRENGTHS: <list key strengths>
            ORIGINAL_WEAKNESSES: <list key weaknesses>
            IMPROVED_STRENGTHS: <list key strengths>
            IMPROVED_WEAKNESSES: <list key weaknesses>
            FINAL_VERSION: <your optimized version>"""
            
            return system_prompt, user_prompt

    def _extract_final_version(self, response_text: str) -> str:
        """Extract the final version from OpenAI's response"""
        try:
            if 'FINAL_VERSION:' in response_text:
                final_version = response_text.split('FINAL_VERSION:')[1].strip()
                return final_version
            return response_text  # Return full response if we can't extract final version
        except Exception as e:
            logger.error(f"Error extracting final version: {str(e)}")
            return response_text

    async def improve_text_with_openai_analysis(self, text: str) -> dict:
        """Improve text using both Anthropic and OpenAI tools"""
        try:
            # First, get the best version from Anthropic
            results_df = await self.improve_text_iteratively(text)
            best_version = results_df.sort_values('score', ascending=False).iloc[0]
            
            # Create prompts for OpenAI analysis
            system_prompt, user_prompt = self._create_openai_comparison_prompt(
                original_text=text,
                improved_text=best_version['improved_text']
            )
            
            # Get OpenAI's analysis and final version
            openai_response = self.openai_request_tool.o1_preview_simulated_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            final_version = self._extract_final_version(
                openai_response.choices[0].message.content
            )
            
            return {
                'original_text': text,
                'best_anthropic_version': best_version['improved_text'],
                'anthropic_intention': best_version['implied_intention'],
                'anthropic_score': best_version['score'],
                'final_version': final_version,
                'full_analysis': openai_response.choices[0].message.content
            }

        except Exception as e:
            logger.error(f"Error in improve_text_with_openai_analysis: {str(e)}")
            return {
                'original_text': text,
                'best_anthropic_version': text,
                'anthropic_intention': "Error in processing",
                'anthropic_score': 5.0,
                'final_version': text,
                'full_analysis': str(e)
            }

    def run_hybrid_improvement_pipeline(self, text: str) -> dict:
        """Main entry point for the hybrid improvement pipeline"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.improve_text_with_openai_analysis(text))
        except Exception as e:
            logger.error(f"Critical error in hybrid improvement pipeline: {str(e)}")
            return {
                'original_text': text,
                'best_anthropic_version': text,
                'anthropic_intention': "Error in processing",
                'anthropic_score': 5.0,
                'final_version': text,
                'full_analysis': str(e)
            }
##PINFIND
class TextScoringSystem:
    def __init__(self, open_ai_request_tool, batch_size: int = 40):
        self.open_ai_request_tool = open_ai_request_tool
        self.batch_size = 40
        self.model = constants.DEFAULT_OPEN_AI_MODEL
    
    def create_scoring_api_prompt(self, original_text: str, final_text: str):
        api_args = {
            "model": self.model,
            "messages": [{
                "role": "system",
                "content": "You are an expert writing analyst focused on objective scoring."
            }, {
                "role": "user",
                "content": f"""Score these two text versions:

ORIGINAL VERSION:
{original_text}

FINAL VERSION:
{final_text}

Score both versions (0-10) on:
1. Effectiveness (intention achievement)
2. Clarity (logical consistency)
3. Conciseness (minimal fluff)
4. Persuasion (motivation impact)

Respond EXACTLY in this format without any extra text:
{{"original": {{"effectiveness": N, "clarity": N, "conciseness": N, "persuasion": N}}, "final": {{"effectiveness": N, "clarity": N, "conciseness": N, "persuasion": N}}}}"""
            }],
            "temperature": 0
        }
        return api_args

    def create_numbered_copies(self, df, n):
        """
        Create n copies of the input dataframe and add a numbered index.
        
        Args:
        df (pd.DataFrame): Input dataframe
        n (int): Number of copies to create
        
        Returns:
        pd.DataFrame: Resulting dataframe with n copies and numbered index
        """
        # Create n copies of the dataframe
        df_copies = pd.concat([df] * n, ignore_index=True)
        
        # Add a numbered index
        df_copies['numbered_index'] = range(1, len(df_copies) + 1)
        
        # Set the 'numbered_index' as the index
        df_copies.set_index('numbered_index', inplace=True)
        
        return df_copies

    def create_x_iterative_api_arg_df(self, nruns: int, original_text: str, final_text: str):
        """
        Create a DataFrame of API arguments for batch processing.
        
        Args:
        nruns (int): Number of runs to create
        original_text (str): Original text to score
        final_text (str): Final text to score
        
        Returns:
        pd.DataFrame: DataFrame ready for async completion
        """
        api_args = self.create_scoring_api_prompt(original_text, final_text)
        df_constructor = pd.DataFrame([[api_args]])
        api_args_df = self.create_numbered_copies(df_constructor, nruns)
        api_args_df.columns = ['prompt']
        
        full_df = self.open_ai_request_tool.create_writable_df_for_async_chat_completion(
            arg_async_map=api_args_df['prompt'].to_dict()
        )
        full_df['count'] = 1
        return full_df

    def create_full_dataframe_for_n_runs(self, original_text: str, final_text: str, nruns: int = None):  # Made nruns optional
        """
        Create full dataframe for exactly 40 runs.
        """
        # Single batch of exactly 40 runs
        df = self.create_x_iterative_api_arg_df(
            nruns=40,  # Always 40 runs
            original_text=original_text,
            final_text=final_text
        )
        print(f"Generated {len(df)} scoring runs")
        return df


    def process_scoring_results(self, scoring_df):
        """
        Process the scoring results with better error handling.
        """
        try:
            # Create distribution map of raw responses
            dist_df = scoring_df[['choices__message__content', 'count']].copy()
            dist_df = dist_df.fillna('')  # Handle NaN values
            distribution_map = dist_df.groupby('choices__message__content').sum()
            distribution_map = distribution_map.sort_values('count', ascending=False)
            
            # Process scores with better error handling
            valid_scores = []
            for _, row in scoring_df.iterrows():
                try:
                    content = row['choices__message__content']
                    if pd.isna(content) or not content:
                        continue
                        
                    scores = json.loads(content)
                    # Validate score structure
                    if not ('original' in scores and 'final' in scores):
                        continue
                        
                    valid_scores.append(scores)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Error processing row: {str(e)}")
                    continue
            
            if not valid_scores:
                logger.error("No valid scores found in results")
                return {
                    'error': 'No valid scores found',
                    'raw_data': scoring_df.to_dict('records')
                }
                    
            # Calculate statistics
            metrics = ['effectiveness', 'clarity', 'conciseness', 'persuasion']
            versions = ['original', 'final']
            
            results = {
                'raw_scores': valid_scores,
                'completion_data': scoring_df.to_dict('records'),
                'distribution_map': distribution_map.to_dict()
            }
            
            # Calculate per-version statistics
            for version in versions:
                results[f'{version}_version'] = {}
                for metric in metrics:
                    values = [float(score[version][metric]) for score in valid_scores]
                    results[f'{version}_version'][metric] = {
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values))
                    }
                
                # Calculate overall scores
                all_values = [float(score[version][m]) for score in valid_scores for m in metrics]
                results[f'{version}_version']['overall'] = {
                    'mean': float(np.mean(all_values)),
                    'median': float(np.median(all_values)),
                    'std': float(np.std(all_values))
                }
            
            # Calculate improvement percentage
            improvement_pct = (
                (results['final_version']['overall']['mean'] - 
                 results['original_version']['overall']['mean']) / 
                results['original_version']['overall']['mean'] * 100
            )
            
            results['improvement_percentage'] = float(improvement_pct)
            results['num_samples'] = len(valid_scores)
            results['total_runs'] = len(scoring_df)
            results['valid_run_percentage'] = (len(valid_scores) / len(scoring_df)) * 100
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing scoring results: {str(e)}")
            return {
                'error': str(e),
                'raw_data': scoring_df.to_dict('records')
            }

    async def score_texts(self, original_text: str, final_text: str) -> dict:
        """Main entry point for scoring texts"""
        try:
            scoring_df = self.create_full_dataframe_for_n_runs(
                original_text=original_text,
                final_text=final_text
            )
            return self.process_scoring_results(scoring_df)
        except Exception as e:
            logger.error(f"Error in score_texts: {str(e)}")
            return None

class EnhancedTextImproverTool(TextImproverTool):
    def __init__(self, anthropic_tool, open_ai_request_tool, num_versions: int = 3, max_iterations: int = 3, enable_scoring: bool = False):
        super().__init__(anthropic_tool, open_ai_request_tool, num_versions, max_iterations)
        self.enable_scoring = enable_scoring
        if enable_scoring:
            self.scoring_system = TextScoringSystem(open_ai_request_tool)
    
    async def run_enhanced_improvement_pipeline(self, text: str) -> Dict:
        try:
            # First get improved version using the existing hybrid pipeline
            improvement_result = await self.improve_text_with_openai_analysis(text)
            
            # Only run scoring if enabled
            if self.enable_scoring:
                try:
                    # Then score the versions - create the dataframe first
                    scoring_df = self.scoring_system.create_full_dataframe_for_n_runs(
                        original_text=improvement_result['original_text'],
                        final_text=improvement_result['final_version']
                    )
                    
                    # Process results immediately after getting them
                    scores = self.scoring_system.process_scoring_results(scoring_df)
                    
                    return {
                        **improvement_result,
                        'scoring_analysis': scores,
                        'raw_scoring_data': scoring_df.to_dict('records')
                    }
                except Exception as e:
                    logger.error(f"Scoring failed but continuing with improvement results: {str(e)}")
                    return improvement_result
            else:
                # Return just the improvement results without scoring
                return improvement_result
            
        except Exception as e:
            logger.error(f"Error in enhanced improvement pipeline: {str(e)}")
            return {
                'original_text': text,
                'best_anthropic_version': text,
                'anthropic_intention': "Error in processing",
                'anthropic_score': 5.0,
                'final_version': text,
                'full_analysis': str(e),
                'scoring_analysis': None if self.enable_scoring else None,
                'raw_scoring_data': None if self.enable_scoring else None
            }

    def run_pipeline(self, text: str) -> Dict:
        """Non-async entry point for running the pipeline"""
        try:
            # Create new event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run pipeline and ensure we wait for completion
            result = loop.run_until_complete(self.run_enhanced_improvement_pipeline(text))
            
            # Return results immediately
            return result
            
        except Exception as e:
            logger.error(f"Critical error in pipeline: {str(e)}")
            return {
                'original_text': text,
                'best_anthropic_version': text,
                'anthropic_intention': "Error in processing",
                'anthropic_score': 5.0,
                'final_version': text,
                'full_analysis': str(e),
                'scoring_analysis': None if self.enable_scoring else None,
                'raw_scoring_data': None if self.enable_scoring else None
            }