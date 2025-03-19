import os
import random
import hashlib
import time
import re
from typing import List, Dict, Any
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex
from rate_limiter import RateLimiter  # Import the rate limiter

class QuestionGenerator:
    def __init__(
        self, 
        index: VectorStoreIndex,
        api_key: str,
        llm_model: str = "llama3-70b-8192",
        seed_questions_file: str = "prompt.txt",
        max_tokens_per_question: int = 40  # New parameter for token limit
    ):
        self.index = index
        self.api_key = api_key
        self.llm_model = llm_model
        self.max_tokens = max_tokens_per_question
        
        # Set up LLM with error handling
        try:
            self.llm = Groq(model=llm_model, api_key=api_key)
        except Exception as e:
            print(f"Warning: Error initializing Groq LLM: {e}")
            print("Will attempt to recover when generating questions")
            self.llm = None
        
        # Load seed questions
        self.seed_questions = self.load_seed_questions(seed_questions_file)
        
        # Initialize tracking dictionaries BEFORE calling setup_question_types
        # For deduplication
        self.question_hashes = set()
        
        # Track distributions
        self.generated_by_type = {}
        self.generated_by_complexity = {}
        self.generated_by_subtopic = {}
        
        # Configure question types and complexities
        self.setup_question_types()
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_minute=30,
            requests_per_day=14400,
            tokens_per_minute=6000,
            tokens_per_day=500000
        )
    
    def load_seed_questions(self, filepath: str) -> List[str]:
        """Load seed questions from a file"""
        if not os.path.exists(filepath):
            print(f"Warning: Seed questions file {filepath} not found")
            return []
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(questions)} seed questions from {filepath}")
            return questions
        except Exception as e:
            print(f"Error loading seed questions: {str(e)}")
            return []
    
    def setup_question_types(self):
        """Set up question types, complexities, and their distributions"""
        # Question types with weights for distribution
        self.question_types = {
            "factual": {
                "description": "Direct questions about facts, definitions, or specific information",
                "weight": 0.15,
                "examples": [
                    "What are the key elements of mutual fund selection described by Bogle?",
                    "How is the transformer-based deep learning model structured for stock price prediction?"
                ]
            },
            "conceptual": {
                "description": "Questions about theories, principles, or abstract ideas",
                "weight": 0.15,
                "examples": [
                    "How does Bogle's concept of cost efficiency contribute to long-term wealth creation?",
                    "What is the role of financial engineering in modern capital markets?"
                ]
            },
            "relationship": {
                "description": "Questions exploring connections between different concepts",
                "weight": 0.20,
                "examples": [
                    "How does the integration of historical data with transformer models relate to prediction accuracy?",
                    "What is the relationship between capital budgeting decisions and shareholder wealth?"
                ]
            },
            "application": {
                "description": "Questions about applying concepts to real-world scenarios",
                "weight": 0.20,
                "examples": [
                    "How can agentic AI systems be applied to improve model risk management in financial institutions?",
                    "How can blockchain technology be implemented to address regulatory challenges in finance?"
                ]
            },
            "evaluative": {
                "description": "Questions requiring assessment of effectiveness, quality, or value",
                "weight": 0.15,
                "examples": [
                    "How effective are transformer models compared to traditional neural networks for stock prediction?",
                    "What are the strengths and limitations of using random forest models for credit risk assessment?"
                ]
            },
            "multi_part": {
                "description": "Complex questions with multiple related components",
                "weight": 0.15,
                "examples": [
                    "What are the primary differences between traditional and AI-based credit risk scoring methods, and how do these differences impact SMEs' access to credit?",
                    "How does corporate finance explain the agency problem between managers and shareholders, and what strategies are recommended for mitigating these conflicts?"
                ]
            }
        }
        
        # Complexity levels with target word counts
        self.complexity_levels = {
            "basic": {"max_words": 20, "weight": 0.30},
            "intermediate": {"max_words": 35, "weight": 0.35},
            "advanced": {"max_words": 50, "weight": 0.25},
            "complex": {"max_words": 75, "weight": 0.10}
        }
        
        # Financial subtopics to ensure domain coverage
        self.financial_subtopics = [
            "investment_management",
            "mutual_funds",
            "stock_prediction",
            "algorithmic_trading",
            "blockchain_finance",
            "credit_risk",
            "corporate_finance",
            "quantitative_finance",
            "financial_engineering",
            "personal_finance",
            "fintech",
            "financial_modeling",
            "ai_in_finance",
            "capital_budgeting",
            "portfolio_theory"
        ]
        
        # Initialize tracking counters
        for qtype in self.question_types:
            self.generated_by_type[qtype] = 0
            
        for level in self.complexity_levels:
            self.generated_by_complexity[level] = 0
            
        for subtopic in self.financial_subtopics + ["general_finance"]:
            self.generated_by_subtopic[subtopic] = 0
    
    def enforce_rate_limit(self, prompt_text: str):
        """
        Advanced rate limiting with token awareness
        
        Args:
            prompt_text: The text of the prompt being sent
        """
        # Estimate tokens for prompt and completion
        estimated_prompt_tokens = self.rate_limiter.estimate_tokens(prompt_text)
        expected_completion_tokens = 200  # Typical completion size for question generation
        total_tokens = estimated_prompt_tokens + expected_completion_tokens
        
        # Wait if needed - now handling tuple return
        result = self.rate_limiter.wait_if_needed(total_tokens)
        # Check if this is the new tuple format
        if isinstance(result, tuple):
            wait_time, api_key = result
            # Update the API key
            self.api_key = api_key
        else:
            # Old format for backward compatibility
            wait_time = result
        
        if wait_time > 0:
            print(f"Rate limit applied: waited {wait_time:.2f}s")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 chars per token for English
        return max(1, len(text) // 4)
    
    def enforce_token_limit(self, question: str, max_tokens: int = None) -> str:
        """
        Ensure a question stays within the token limit
        
        Args:
            question: Original question text
            max_tokens: Maximum tokens allowed (default: self.max_tokens)
            
        Returns:
            Truncated question if needed
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        estimated_tokens = self.estimate_tokens(question)
        
        if estimated_tokens <= max_tokens:
            return question
        
        # If over the limit, truncate with a smart approach
        # First try to keep complete sentences
        sentences = re.split(r'(?<=[.!?])\s+', question)
        result = ""
        
        for sentence in sentences:
            if self.estimate_tokens(result + sentence) <= max_tokens - 3:  # Leave room for "..."
                result += sentence + " "
            else:
                break
        
        # If result is still empty (first sentence was too long), truncate the words
        if not result:
            words = question.split()
            for word in words:
                if self.estimate_tokens(result + word) <= max_tokens - 3:
                    result += word + " "
                else:
                    break
        
        return result.strip() + "..."
    
    def create_question_prompt(self, 
                               question_type: str, 
                               complexity_focus: List[str], 
                               subtopic_focus: List[str]) -> str:
        """
        Create a prompt for generating questions of specific type, complexity, and subtopics
        """
        # Select examples based on question type
        type_examples = self.question_types[question_type]["examples"]
        
        # Select relevant seed questions that match the subtopics
        relevant_seeds = []
        for question in self.seed_questions:
            for subtopic in subtopic_focus:
                # Convert subtopic to search terms
                search_terms = subtopic.replace('_', ' ').split()
                if any(term.lower() in question.lower() for term in search_terms):
                    relevant_seeds.append(question)
                    break
        
        # If we don't have enough relevant seeds, add some random ones
        if len(relevant_seeds) < 5:
            random_seeds = random.sample(
                [q for q in self.seed_questions if q not in relevant_seeds],
                min(5 - len(relevant_seeds), len(self.seed_questions))
            )
            relevant_seeds.extend(random_seeds)
        
        # Limit to 5 seed questions for focus
        relevant_seeds = relevant_seeds[:5]
        
        # Complexity guidance
        complexity_guidance = "\n".join([
            f"- {level.upper()}: {self.complexity_levels[level]['max_words']} words maximum" 
            for level in complexity_focus
        ])
        
        # Type-specific guidance
        type_guidance = ""
        if question_type == "relationship":
            type_guidance = """
            RELATIONSHIP QUESTION PATTERNS:
            - How does X relate to Y in the context of Z?
            - What connections exist between X and Y in financial markets?
            - How do changes in X influence Y in the financial world?
            
            Focus on exploring meaningful connections between financial concepts.
            """
        elif question_type == "application":
            type_guidance = """
            APPLICATION QUESTION PATTERNS:
            - How can X be applied to solve Y problem in financial contexts?
            - What practical implementations of X could address Y challenge?
            - In what ways can X be utilized to enhance Y outcomes in finance?
            
            Focus on real-world applications of financial concepts.
            """
        elif question_type == "multi_part":
            type_guidance = """
            MULTI-PART QUESTION GUIDANCE:
            - Include 2-3 related components that build on each other
            - Connect different aspects of the same topic
            - Use connectors like "and," "also," or transitions between parts
            """
        
        # Token limit guidance - NEW
        token_guidance = f"""
        TOKEN LIMIT REQUIREMENT:
        - Each question must be no longer than {self.max_tokens} tokens (approximately {self.max_tokens * 4} characters)
        - Questions that are too long will be truncated
        - Focus on brevity while maintaining clarity and specificity
        """
        
        # Build prompt
        subtopic_str = ", ".join([s.replace("_", " ") for s in subtopic_focus])
        complexity_str = ", ".join(complexity_focus)
        
        prompt = f"""
        You are a financial domain expert. Generate 15 diverse and high-quality {question_type.upper()} questions about finance.

        FOCUS AREAS:
        - Question Type: {question_type.upper()}
        - Complexity Levels: {complexity_str}
        - Financial Subtopics: {subtopic_str}

        SEED QUESTIONS FOR REFERENCE:
        {chr(10).join(f"- {q}" for q in relevant_seeds)}

        QUESTION TYPE DESCRIPTION:
        {self.question_types[question_type]["description"]}
        
        {type_guidance}

        COMPLEXITY GUIDANCE:
        {complexity_guidance}
        
        {token_guidance}

        REQUIREMENTS:
        1. Create questions specifically focused on {subtopic_str}
        2. Ensure questions have appropriate complexity based on the focus levels
        3. Vary question structures and formats to maintain diversity
        4. Make questions detailed and specific rather than general
        5. Avoid phrases like "According to the text" or "Based on the knowledge base"
        6. Ensure each question ends with a question mark
        7. Create questions that would require domain expertise to answer properly
        8. For relationship questions, explicitly explore connections between concepts
        9. For application questions, focus on real-world applications
        10. For multi-part questions, include related components separated by commas or "and"
        11. KEEP ALL QUESTIONS UNDER {self.max_tokens} TOKENS
        
        Return ONLY the questions, one per line, no numbering or additional text.
        """
        
        return prompt
    
    def classify_subtopic(self, question: str) -> str:
        """Classify the financial subtopic of a question with improved accuracy"""
        question_lower = question.lower()
        
        # Define keywords for each subtopic
        subtopic_keywords = {
            "investment_management": [
                "investment", "portfolio", "asset", "allocation", "diversification",
                "investor", "invest", "asset management", "fund manager", "robo-advisor"
            ],
            "mutual_funds": [
                "mutual fund", "etf", "index fund", "bogle", "prospectus", 
                "expense ratio", "nav", "net asset value", "fund manager"
            ],
            "stock_prediction": [
                "predict", "forecast", "stock price", "market prediction", "trend",
                "price movement", "technical analysis", "chart pattern", "time series"
            ],
            "algorithmic_trading": [
                "algorithmic", "algorithm", "trading", "strategy", "execution",
                "high frequency", "automated trading", "trade execution", "backtest"
            ],
            "blockchain_finance": [
                "blockchain", "distributed ledger", "token", "tokenization", 
                "smart contract", "decentralized", "consensus", "byzantine", "node"
            ],
            "crypto_blockchain": [
                "crypto", "bitcoin", "ethereum", "ledger", "wallet", "mining",
                "hash", "block", "cryptocurrency", "coin", "satoshi", "bitcoin"
            ],
            "credit_risk": [
                "credit", "risk", "default", "loan", "rating", "score", "lending",
                "borrower", "creditor", "debt", "interest rate", "collateral"
            ],
            "corporate_finance": [
                "corporate", "capital structure", "dividend", "merger", "acquisition",
                "ipo", "equity", "debt", "cfo", "earnings", "financial statement"
            ],
            "quantitative_finance": [
                "quant", "mathematical", "stochastic", "model", "derivative",
                "option pricing", "monte carlo", "black-scholes", "volatility"
            ],
            "financial_engineering": [
                "engineering", "derivative", "option", "swap", "structured product",
                "hedging", "pricing model", "binomial", "exotic", "instrument"
            ],
            "personal_finance": [
                "personal", "budget", "retirement", "savings", "individual",
                "401k", "ira", "tax", "mortgage", "loan", "credit score"
            ],
            "fintech": [
                "fintech", "technology", "digital", "mobile", "app",
                "online banking", "digital payment", "open banking", "regtech"
            ],
            "financial_modeling": [
                "model", "modeling", "excel", "forecast", "projection",
                "simulation", "scenario", "sensitivity", "valuation", "dcf"
            ],
            "ai_in_finance": [
                "ai", "artificial intelligence", "machine learning", "neural", "transformer",
                "deep learning", "algorithm", "predictive", "data science", "chatbot"
            ],
            "capital_budgeting": [
                "capital", "budget", "npv", "irr", "investment decision",
                "project finance", "payback period", "cost of capital", "hurdle rate"
            ],
            "portfolio_theory": [
                "portfolio", "efficient frontier", "modern portfolio", "sharpe", "capm",
                "diversification", "asset allocation", "risk-return", "beta", "alpha"
            ],
            "banking": [
                "bank", "banking", "deposit", "withdrawal", "interest rate", 
                "reserve", "central bank", "commercial bank", "retail banking"
            ],
            "risk_management": [
                "risk management", "hedging", "insurance", "exposure", "var",
                "stress test", "scenario analysis", "counterparty risk", "operational risk"
            ]
        }
        
        # Enhanced scoring with phrase weighting and multi-word matching
        subtopic_scores = {}
        
        for subtopic, keywords in subtopic_keywords.items():
            score = 0
            for keyword in keywords:
                # Single word keywords
                if " " not in keyword and keyword in question_lower:
                    score += 1
                
                # Multi-word phrases (exact matches get higher weight)
                elif " " in keyword and keyword in question_lower:
                    score += 3
                
                # Check for semantic variants (partial matches of multi-word phrases)
                elif " " in keyword:
                    parts = keyword.split()
                    if all(part in question_lower for part in parts):
                        score += 2
            
            subtopic_scores[subtopic] = score
        
        # Special case handling for common overlaps
        if "blockchain" in question_lower:
            if any(term in question_lower for term in ["bitcoin", "crypto", "wallet", "mining"]):
                subtopic_scores["crypto_blockchain"] += 2
            else:
                subtopic_scores["blockchain_finance"] += 2
        
        if "ai" in question_lower or "machine learning" in question_lower:
            subtopic_scores["ai_in_finance"] += 2
        
        # Check specific question context clues
        finance_context_clues = {
            "investment_management": ["portfolio optimization", "asset allocation", "investment strategy"],
            "risk_management": ["mitigate risk", "risk assessment", "manage risk"],
            "financial_modeling": ["forecasting", "predict", "projection", "model"],
            "corporate_finance": ["company", "firm", "corporation", "business"]
        }
        
        for subtopic, clues in finance_context_clues.items():
            for clue in clues:
                if clue in question_lower:
                    subtopic_scores[subtopic] += 1
        
        # Get the highest scoring subtopic
        max_score = max(subtopic_scores.values())
        if max_score > 0:
            best_subtopics = [s for s, score in subtopic_scores.items() if score == max_score]
            return random.choice(best_subtopics)
        
        # Default if no clear match
        return "general_finance"
    
    def determine_complexity(self, question: str) -> str:
        """Determine the complexity level of a question based on word count and structure"""
        word_count = len(question.split())
        
        # Basic classification by word count
        if word_count <= self.complexity_levels["basic"]["max_words"]:
            complexity = "basic"
        elif word_count <= self.complexity_levels["intermediate"]["max_words"]:
            complexity = "intermediate"
        elif word_count <= self.complexity_levels["advanced"]["max_words"]:
            complexity = "advanced"
        else:
            complexity = "complex"
        
        # Check for multi-part or relationship indicators
        indicators = {
            "complex": ["compare", "contrast", "and how", ", and", "; also", "what are the differences", 
                       "advantages and disadvantages"],
            "advanced": ["relationship between", "connection between", "affect", "influence", 
                        "implications of", "impact of"]
        }
        
        # Update complexity based on indicators
        for level, phrases in indicators.items():
            if any(phrase in question.lower() for phrase in phrases):
                if self.complexity_levels[level]["max_words"] >= word_count:
                    complexity = level
                break
        
        return complexity
    
    def generate_questions(self, num_questions: int) -> List[Dict]:
        """
        Generate diverse questions with controlled distribution
        
        Args:
            num_questions: Number of questions to generate
            
        Returns:
            List of question dictionaries with metadata
        """
        generated_questions = []
        
        # Calculate target distribution by type
        type_targets = {}
        remaining = num_questions
        
        # Allocate questions by type based on weights
        for qtype, info in self.question_types.items():
            count = int(num_questions * info["weight"])
            type_targets[qtype] = count
            remaining -= count
        
        # Distribute any remainder
        for qtype in type_targets:
            if remaining <= 0:
                break
            type_targets[qtype] += 1
            remaining -= 1
        
        # Generate questions for each type
        for qtype, target in type_targets.items():
            print(f"Generating {target} {qtype} questions...")
            
            batch_num = 1
            qtype_questions = []
            
            while len(qtype_questions) < target:
                # Select complexity levels to focus on (weighted selection)
                complexity_focus = random.choices(
                    list(self.complexity_levels.keys()),
                    weights=[self.complexity_levels[c]["weight"] for c in self.complexity_levels],
                    k=2  # Focus on 2 complexity levels per batch
                )
                
                # Select subtopics to focus on
                subtopic_focus = random.sample(self.financial_subtopics, k=3)
                
                # Create prompt
                prompt = self.create_question_prompt(qtype, complexity_focus, subtopic_focus)
                
                # Apply rate limiting
                self.enforce_rate_limit(prompt)
                
                try:
                    # Make sure LLM is initialized
                    if self.llm is None:
                        self.llm = Groq(model=self.llm_model, api_key=self.api_key)
                    
                    # Generate questions
                    response = self.llm.complete(prompt)
                    questions_text = str(response).strip()
                    
                    # Process each generated question
                    raw_questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
                    
                    for question in raw_questions:
                        # Skip if we already have enough questions of this type
                        if len(qtype_questions) >= target:
                            break
                            
                        # Skip if not a proper question
                        if not question.endswith('?'):
                            question += '?'
                            
                        # Hash for deduplication
                        q_hash = hashlib.md5(question.lower().encode()).hexdigest()
                        
                        if q_hash in self.question_hashes:
                            continue
                        
                        # Determine complexity and subtopic
                        complexity = self.determine_complexity(question)
                        subtopic = self.classify_subtopic(question)
                        
                        # Enforce token limit - NEW
                        question = self.enforce_token_limit(question, self.max_tokens)
                        
                        # Create question item
                        question_item = {
                            "question": question,
                            "type": qtype,
                            "complexity": complexity,
                            "subtopic": subtopic,
                            "batch": batch_num,
                            "token_count": self.estimate_tokens(question)  # Store token count
                        }
                        
                        # Add to results
                        qtype_questions.append(question_item)
                        self.question_hashes.add(q_hash)
                        
                        # Update distribution counters
                        self.generated_by_type[qtype] = self.generated_by_type.get(qtype, 0) + 1
                        self.generated_by_complexity[complexity] = self.generated_by_complexity.get(complexity, 0) + 1
                        self.generated_by_subtopic[subtopic] = self.generated_by_subtopic.get(subtopic, 0) + 1
                
                except Exception as e:
                    print(f"Error generating questions: {str(e)}")
                    time.sleep(5)  # Wait before retrying
                
                # Increment batch counter
                batch_num += 1
                print(f"Generated {len(qtype_questions)}/{target} {qtype} questions")
                
                # Add delay between batches to avoid rate limits
                time.sleep(3)
                
                # Avoid infinite loops
                if batch_num > 10:
                    print(f"Reached batch limit for {qtype}, moving on...")
                    break
            
            # Add questions to overall list
            generated_questions.extend(qtype_questions)
        
        # Print distribution statistics
        print("\nQuestion Distribution:")
        print("By Type:", self.generated_by_type)
        print("By Complexity:", self.generated_by_complexity)
        print("By Subtopic:", self.generated_by_subtopic)
        
        # Print token statistics
        token_counts = [item["token_count"] for item in generated_questions]
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            max_tokens = max(token_counts)
            print(f"Token Statistics: Avg={avg_tokens:.1f}, Max={max_tokens}")
        
        return generated_questions