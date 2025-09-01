"""
Metrics implementation for PQS, SCCS, IIR, and CEM.
These metrics are calculated from strategy results.
"""

from typing import Dict, Any, List
import re

class MetricsCalculator:
    """Calculates all evaluation metrics."""
    
    @staticmethod
    def calculate_pqs(response: str, scenario: Dict[str, Any], strategy: str) -> float:
        """
        Plan Quality Score (PQS): 0-100
        Measures the quality and completeness of the generated plan.
        """
        base_score = 40.0
        
        # Completeness: Check if required outputs are addressed
        completeness_score = MetricsCalculator._calculate_completeness(response, scenario)
        
        # Structure: Check for proper organization
        structure_score = MetricsCalculator._calculate_structure(response, strategy)
        
        # Detail: Assess depth and detail level
        detail_score = MetricsCalculator._calculate_detail(response)
        
        # Domain expertise: Check for domain-specific elements
        domain_score = MetricsCalculator._calculate_domain_expertise(response, scenario)
        
        # Complexity adjustment
        complexity_multiplier = {"low": 0.9, "medium": 1.0, "high": 1.1}
        multiplier = complexity_multiplier.get(scenario.get('complexity', 'medium'), 1.0)
        
        total_score = (base_score + completeness_score + structure_score + detail_score + domain_score) * multiplier
        return min(100.0, max(0.0, total_score))
    
    @staticmethod
    def calculate_sccs(response: str, strategy: str) -> float:
        """
        Self-Check Confidence Score (SCCS): 0-100
        Measures confidence indicators in the response.
        """
        base_score = 50.0
        
        # Confidence indicators
        confidence_keywords = ["confident", "certain", "sure", "validated", "verified", "confirmed", "assured"]
        uncertainty_keywords = ["uncertain", "unclear", "might", "maybe", "possibly", "unsure", "doubt"]
        
        confidence_count = sum(1 for keyword in confidence_keywords 
                             if keyword in response.lower())
        uncertainty_count = sum(1 for keyword in uncertainty_keywords 
                              if keyword in response.lower())
        
        confidence_score = confidence_count * 8 - uncertainty_count * 5
        
        # Strategy-specific adjustments
        strategy_adjustments = {
            "srlp": 10,   # SRLP has self-assessment built in
            "tot": 8,     # ToT has evaluation phases
            "react": 5,   # ReAct has observation feedback
            "cot": 0      # CoT is basic reasoning
        }
        
        strategy_bonus = strategy_adjustments.get(strategy, 0)
        
        # Check for explicit confidence statements
        confidence_patterns = [
            r"confidence.*(?:high|strong|certain)",
            r"(?:very|highly|extremely)\s+confident",
            r"validated.*(?:successful|effective|good)",
            r"quality.*(?:high|excellent|good)"
        ]
        
        pattern_bonus = sum(5 for pattern in confidence_patterns 
                          if re.search(pattern, response.lower()))
        
        total_score = base_score + confidence_score + strategy_bonus + pattern_bonus
        return min(100.0, max(0.0, total_score))
    
    @staticmethod
    def calculate_iir(response: str, strategy: str) -> float:
        """
        Iteration Improvement Rate (IIR): 0-100
        Measures evidence of iterative improvement in the response.
        """
        base_score = 30.0
        
        # Improvement keywords
        improvement_keywords = ["improve", "enhance", "refine", "optimize", "better", "upgrade", 
                              "revise", "adjust", "modify", "update", "iterate"]
        
        improvement_count = sum(1 for keyword in improvement_keywords 
                              if keyword in response.lower())
        improvement_score = min(25, improvement_count * 3)
        
        # Strategy-specific iteration indicators
        strategy_patterns = {
            "srlp": [r"stage \d+", r"refine", r"self-assessment", r"quality assurance"],
            "tot": [r"thought \d+", r"evaluation", r"compare", r"synthesiz"],
            "react": [r"thought \d+", r"action \d+", r"observation", r"cycle"],
            "cot": [r"step \d+", r"therefore", r"next", r"then"]
        }
        
        patterns = strategy_patterns.get(strategy, [])
        pattern_score = sum(5 for pattern in patterns 
                          if re.search(pattern, response.lower()))
        pattern_score = min(20, pattern_score)
        
        # Check for explicit iteration language
        iteration_phrases = ["based on", "building on", "following from", "as a result", 
                           "consequently", "therefore", "thus leading to"]
        iteration_score = sum(3 for phrase in iteration_phrases 
                            if phrase in response.lower())
        iteration_score = min(15, iteration_score)
        
        # Strategy multipliers based on inherent iteration capability
        strategy_multipliers = {"srlp": 1.2, "tot": 1.1, "react": 1.1, "cot": 0.9}
        multiplier = strategy_multipliers.get(strategy, 1.0)
        
        total_score = (base_score + improvement_score + pattern_score + iteration_score) * multiplier
        return min(100.0, max(0.0, total_score))
    
    @staticmethod
    def calculate_cem(response: str, scenario: Dict[str, Any], strategy: str) -> float:
        """
        Cost Efficiency Metric (CEM): 0-100
        Measures cost-effectiveness and resource optimization.
        """
        base_score = 50.0
        
        # Efficiency keywords
        efficiency_keywords = ["efficient", "cost-effective", "optimize", "streamline", "minimal", 
                             "economical", "budget-friendly", "resource-aware", "lean"]
        
        efficiency_count = sum(1 for keyword in efficiency_keywords 
                             if keyword in response.lower())
        efficiency_score = min(20, efficiency_count * 4)
        
        # Resource optimization indicators
        resource_keywords = ["resource", "time", "cost", "budget", "effort", "minimize", "reduce"]
        resource_count = sum(1 for keyword in resource_keywords 
                           if keyword in response.lower())
        resource_score = min(15, resource_count * 2)
        
        # Complexity adjustment (more complex problems have lower expected efficiency)
        complexity_adjustments = {"low": 15, "medium": 10, "high": 5}
        complexity_bonus = complexity_adjustments.get(scenario.get('complexity', 'medium'), 10)
        
        # Strategy efficiency characteristics
        strategy_efficiencies = {
            "srlp": 0,    # SRLP is thorough but not necessarily efficient
            "cot": 10,    # CoT is relatively efficient
            "tot": -5,    # ToT is thorough but resource-intensive
            "react": 5    # ReAct has action-based efficiency
        }
        strategy_adjustment = strategy_efficiencies.get(strategy, 0)
        
        # Check for specific efficiency mentions
        efficiency_patterns = [
            r"(?:save|reduce|minimize).*(?:time|cost|effort)",
            r"(?:quick|fast|rapid|swift).*(?:solution|approach)",
            r"(?:low|minimal|reduced).*(?:cost|effort|resource)"
        ]
        
        pattern_bonus = sum(5 for pattern in efficiency_patterns 
                          if re.search(pattern, response.lower()))
        
        total_score = (base_score + efficiency_score + resource_score + 
                      complexity_bonus + strategy_adjustment + pattern_bonus)
        return min(100.0, max(0.0, total_score))
    
    @staticmethod
    def _calculate_completeness(response: str, scenario: Dict[str, Any]) -> float:
        """Calculate completeness score based on expected outputs coverage."""
        expected_outputs = scenario.get('expected_outputs', [])
        if not expected_outputs:
            return 20.0  # Full points if no specific outputs expected
        
        coverage_count = 0
        for output in expected_outputs:
            # Check if output keywords appear in response
            keywords = output.lower().split()
            if any(keyword in response.lower() for keyword in keywords if len(keyword) > 3):
                coverage_count += 1
        
        coverage_ratio = coverage_count / len(expected_outputs)
        return coverage_ratio * 20.0  # Max 20 points for completeness
    
    @staticmethod
    def _calculate_structure(response: str, strategy: str) -> float:
        """Calculate structure score based on strategy-specific organization."""
        structure_indicators = {
            "srlp": ["stage", "plan generation", "self-assessment", "refinement", "quality"],
            "cot": ["step", "first", "then", "next", "finally", "therefore"],
            "tot": ["thought", "branch", "evaluation", "synthesis", "compare"],
            "react": ["thought", "action", "observation", "cycle"]
        }
        
        indicators = structure_indicators.get(strategy, ["step", "plan", "analysis"])
        
        structure_count = sum(1 for indicator in indicators 
                            if indicator in response.lower())
        
        # Check for numbered lists or clear sections
        numbered_sections = len(re.findall(r'\d+\.|\d+\)', response))
        bullet_points = len(re.findall(r'[-•*]\s', response))
        
        organization_score = min(5, (numbered_sections + bullet_points) * 0.5)
        indicator_score = min(10, structure_count * 2)
        
        return organization_score + indicator_score  # Max 15 points
    
    @staticmethod
    def _calculate_detail(response: str) -> float:
        """Calculate detail score based on response depth."""
        word_count = len(response.split())
        sentence_count = len(re.findall(r'[.!?]+', response))
        
        # Optimal word count range: 300-800 words
        if 300 <= word_count <= 800:
            word_score = 10
        elif word_count < 300:
            word_score = (word_count / 300) * 10
        else:
            word_score = max(5, 10 - (word_count - 800) / 200)
        
        # Average sentence length (complexity indicator)
        avg_sentence_length = word_count / max(sentence_count, 1)
        if 12 <= avg_sentence_length <= 20:
            complexity_score = 5
        else:
            complexity_score = max(0, 5 - abs(avg_sentence_length - 16) / 4)
        
        return min(15.0, word_score + complexity_score)  # Max 15 points
    
    @staticmethod
    def _calculate_domain_expertise(response: str, scenario: Dict[str, Any]) -> float:
        """Calculate domain expertise score."""
        domain = scenario.get('domain', '')
        
        domain_keywords = {
            "travel_planning": ["itinerary", "accommodation", "transport", "visa", "booking", "destination"],
            "software_project": ["architecture", "development", "testing", "deployment", "requirements", "coding"],
            "event_organization": ["venue", "catering", "logistics", "marketing", "registration", "vendor"],
            "research_study": ["methodology", "participants", "data", "analysis", "ethics", "publication"],
            "business_launch": ["market", "strategy", "funding", "operations", "revenue", "competition"]
        }
        
        keywords = domain_keywords.get(domain, [])
        keyword_count = sum(1 for keyword in keywords if keyword in response.lower())
        
        # Domain-specific expertise points
        expertise_score = min(10, keyword_count * 2)
        
        return expertise_score  # Max 10 points

class MetricsAggregator:
    """Aggregates metrics across multiple results."""
    
    @staticmethod
    def aggregate_by_strategy(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics by strategy."""
        strategy_metrics = {}
        
        for result in results:
            strategy = result.get('strategy')
            if not strategy:
                continue
                
            if strategy not in strategy_metrics:
                strategy_metrics[strategy] = {
                    'pqs': [], 'sccs': [], 'iir': [], 'cem': [],
                    'execution_time': [], 'cost_usd': [], 'tokens_used': []
                }
            
            metrics = result.get('metrics', {})
            strategy_metrics[strategy]['pqs'].append(metrics.get('pqs', 0))
            strategy_metrics[strategy]['sccs'].append(metrics.get('sccs', 0))
            strategy_metrics[strategy]['iir'].append(metrics.get('iir', 0))
            strategy_metrics[strategy]['cem'].append(metrics.get('cem', 0))
            strategy_metrics[strategy]['execution_time'].append(result.get('execution_time', 0))
            strategy_metrics[strategy]['cost_usd'].append(result.get('cost_usd', 0))
            strategy_metrics[strategy]['tokens_used'].append(result.get('tokens_used', 0))
        
        # Calculate averages
        aggregated = {}
        for strategy, metrics in strategy_metrics.items():
            aggregated[strategy] = {}
            for metric, values in metrics.items():
                if values:
                    aggregated[strategy][f'{metric}_mean'] = sum(values) / len(values)
                    aggregated[strategy][f'{metric}_std'] = (
                        sum((x - aggregated[strategy][f'{metric}_mean']) ** 2 for x in values) / len(values)
                    ) ** 0.5 if len(values) > 1 else 0
                else:
                    aggregated[strategy][f'{metric}_mean'] = 0
                    aggregated[strategy][f'{metric}_std'] = 0
        
        return aggregated
    
    @staticmethod
    def aggregate_by_provider(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics by provider."""
        provider_metrics = {}
        
        for result in results:
            provider = result.get('provider')
            if not provider:
                continue
                
            if provider not in provider_metrics:
                provider_metrics[provider] = {
                    'execution_time': [], 'cost_usd': [], 'tokens_used': [],
                    'success_count': 0, 'total_count': 0
                }
            
            provider_metrics[provider]['total_count'] += 1
            if result.get('success', False):
                provider_metrics[provider]['success_count'] += 1
                provider_metrics[provider]['execution_time'].append(result.get('execution_time', 0))
                provider_metrics[provider]['cost_usd'].append(result.get('cost_usd', 0))
                provider_metrics[provider]['tokens_used'].append(result.get('tokens_used', 0))
        
        # Calculate averages and success rates
        aggregated = {}
        for provider, metrics in provider_metrics.items():
            aggregated[provider] = {
                'success_rate': metrics['success_count'] / max(metrics['total_count'], 1),
                'total_requests': metrics['total_count']
            }
            
            for metric in ['execution_time', 'cost_usd', 'tokens_used']:
                values = metrics[metric]
                if values:
                    aggregated[provider][f'{metric}_mean'] = sum(values) / len(values)
                    aggregated[provider][f'{metric}_total'] = sum(values)
                else:
                    aggregated[provider][f'{metric}_mean'] = 0
                    aggregated[provider][f'{metric}_total'] = 0
        
        return aggregated
