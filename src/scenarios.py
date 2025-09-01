"""
Scenario generation for 450 scenarios across 5 domains (90 each).
Each scenario includes context, complexity level, and expected outputs.
"""

import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
from pathlib import Path

@dataclass
class Scenario:
    """Individual scenario data structure."""
    id: str
    domain: str
    title: str
    description: str
    complexity: str  # low, medium, high
    context: Dict[str, Any]
    expected_outputs: List[str]

class ScenarioGenerator:
    """Generates exactly 450 scenarios (90 per domain)."""
    
    def __init__(self, domains: List[str], scenarios_per_domain: int = 90, seed: int = 42):
        self.domains = domains
        self.scenarios_per_domain = scenarios_per_domain
        self.seed = seed
        self.scenarios: Dict[str, List[Scenario]] = {}
        self.dropped_scenario_ids: List[str] = []
        
        random.seed(seed)
    
    def generate_all_scenarios(self) -> None:
        """Generate scenarios for all domains."""
        for domain in self.domains:
            domain_scenarios = self._generate_domain_scenarios(domain)
            self.scenarios[domain] = domain_scenarios
    
    def _generate_domain_scenarios(self, domain: str) -> List[Scenario]:
        """Generate scenarios for a specific domain."""
        scenarios = []
        
        # Generate extra scenarios then trim to exact count for determinism
        buffer_count = 20
        total_to_generate = self.scenarios_per_domain + buffer_count
        
        for i in range(total_to_generate):
            scenario = self._create_scenario(domain, i)
            scenarios.append(scenario)
        
        # Sort deterministically and take exactly the required number
        scenarios = sorted(scenarios, key=lambda x: x.id)
        final_scenarios = scenarios[:self.scenarios_per_domain]
        
        # Track dropped scenarios
        for scenario in scenarios[self.scenarios_per_domain:]:
            self.dropped_scenario_ids.append(scenario.id)
        
        return final_scenarios
    
    def _create_scenario(self, domain: str, index: int) -> Scenario:
        """Create a single scenario based on domain."""
        scenario_id = f"{domain}_{index:03d}"
        
        domain_generators = {
            "travel_planning": self._create_travel_scenario,
            "software_project": self._create_software_scenario,
            "event_organization": self._create_event_scenario,
            "research_study": self._create_research_scenario,
            "business_launch": self._create_business_scenario
        }
        
        generator = domain_generators.get(domain)
        if not generator:
            raise ValueError(f"Unknown domain: {domain}")
        
        return generator(scenario_id, index)
    
    def _create_travel_scenario(self, scenario_id: str, index: int) -> Scenario:
        """Create travel planning scenario."""
        destinations = ["Tokyo", "Paris", "New York", "Sydney", "Cairo", "London", "Rome", "Bangkok", 
                       "Barcelona", "Amsterdam", "Vienna", "Prague", "Istanbul", "Dubai", "Singapore"]
        durations = ["3 days", "1 week", "10 days", "2 weeks", "3 weeks", "1 month"]
        budget_types = ["budget", "mid-range", "luxury", "ultra-luxury"]
        travel_styles = ["cultural", "adventure", "relaxation", "business", "family", "solo", "romantic"]
        
        destination = destinations[index % len(destinations)]
        duration = durations[index % len(durations)]
        budget_type = budget_types[index % len(budget_types)]
        travel_style = travel_styles[index % len(travel_styles)]
        complexity = ["low", "medium", "high"][index % 3]
        
        travelers = random.randint(1, 6)
        budget_amount = random.randint(1000, 10000) * (1 if budget_type == "budget" else 2 if budget_type == "mid-range" else 4)
        
        return Scenario(
            id=scenario_id,
            domain="travel_planning",
            title=f"{budget_type.title()} {duration} {travel_style} trip to {destination}",
            description=f"Plan a comprehensive {budget_type} {duration} {travel_style} trip to {destination} for {travelers} traveler(s).",
            complexity=complexity,
            context={
                "destination": destination,
                "duration": duration,
                "budget_type": budget_type,
                "budget_amount": budget_amount,
                "travelers": travelers,
                "travel_style": travel_style,
                "season": random.choice(["spring", "summer", "fall", "winter"]),
                "special_requirements": random.choice(["none", "accessibility", "dietary restrictions", "visa requirements"])
            },
            expected_outputs=[
                "Detailed daily itinerary",
                "Accommodation recommendations with booking details",
                "Transportation plan (flights, local transport)",
                "Budget breakdown by category",
                "Activity and attraction suggestions",
                "Dining recommendations",
                "Travel insurance and safety considerations",
                "Packing list and travel tips"
            ]
        )
    
    def _create_software_scenario(self, scenario_id: str, index: int) -> Scenario:
        """Create software project scenario."""
        project_types = ["web application", "mobile app", "desktop software", "API service", "game", 
                        "ML/AI system", "blockchain app", "IoT system", "microservices", "cloud platform"]
        technologies = ["React/Node.js", "Python/Django", "Java/Spring", "Swift/iOS", "Kotlin/Android",
                       "Vue.js/Express", "Angular/.NET", "Python/Flask", "Ruby/Rails", "Go/Gin"]
        industries = ["healthcare", "finance", "education", "e-commerce", "entertainment", "logistics", 
                     "social media", "productivity", "gaming", "real estate"]
        
        project_type = project_types[index % len(project_types)]
        technology = technologies[index % len(technologies)]
        industry = industries[index % len(industries)]
        complexity = ["low", "medium", "high"][index % 3]
        
        team_size = random.randint(2, 12)
        timeline_months = random.randint(3, 18)
        budget = random.randint(50000, 500000)
        
        return Scenario(
            id=scenario_id,
            domain="software_project",
            title=f"{industry.title()} {project_type} using {technology}",
            description=f"Develop a {project_type} for the {industry} industry using {technology} stack.",
            complexity=complexity,
            context={
                "project_type": project_type,
                "technology_stack": technology,
                "industry": industry,
                "team_size": team_size,
                "timeline_months": timeline_months,
                "budget": budget,
                "target_users": random.randint(1000, 100000),
                "platform": random.choice(["web", "mobile", "desktop", "cross-platform"]),
                "compliance_requirements": random.choice(["GDPR", "HIPAA", "SOX", "PCI-DSS", "none"])
            },
            expected_outputs=[
                "Project architecture and technical design",
                "Development timeline and milestones",
                "Resource allocation and team structure",
                "Technology stack justification",
                "User stories and requirements specification",
                "Testing strategy and quality assurance plan",
                "Deployment and DevOps strategy",
                "Risk assessment and mitigation plan",
                "Budget breakdown and cost estimation",
                "Performance and scalability considerations"
            ]
        )
    
    def _create_event_scenario(self, scenario_id: str, index: int) -> Scenario:
        """Create event organization scenario."""
        event_types = ["conference", "wedding", "corporate meeting", "trade show", "concert", "workshop",
                      "festival", "product launch", "gala dinner", "seminar", "exhibition", "networking event"]
        venues = ["hotel ballroom", "convention center", "outdoor park", "beach resort", "corporate office",
                 "university campus", "cultural center", "restaurant", "museum", "rooftop venue"]
        
        event_type = event_types[index % len(event_types)]
        venue_type = venues[index % len(venues)]
        complexity = ["low", "medium", "high"][index % 3]
        
        attendees = random.randint(50, 2000)
        duration_days = random.randint(1, 5)
        budget = random.randint(10000, 200000)
        
        return Scenario(
            id=scenario_id,
            domain="event_organization",
            title=f"{event_type.title()} at {venue_type}",
            description=f"Organize a {event_type} for {attendees} attendees at a {venue_type} venue.",
            complexity=complexity,
            context={
                "event_type": event_type,
                "venue_type": venue_type,
                "attendees": attendees,
                "duration_days": duration_days,
                "budget": budget,
                "date_flexibility": random.choice(["fixed", "flexible", "seasonal"]),
                "catering_required": random.choice([True, False]),
                "av_requirements": random.choice(["basic", "advanced", "minimal"]),
                "target_audience": random.choice(["professionals", "general public", "students", "executives"])
            },
            expected_outputs=[
                "Event timeline and schedule",
                "Venue selection and booking strategy",
                "Vendor coordination plan",
                "Budget allocation and cost control",
                "Marketing and promotion strategy",
                "Registration and attendee management",
                "Logistics and operations plan",
                "Risk management and contingency planning",
                "Catering and hospitality arrangements",
                "Post-event evaluation framework"
            ]
        )
    
    def _create_research_scenario(self, scenario_id: str, index: int) -> Scenario:
        """Create research study scenario."""
        fields = ["psychology", "medicine", "computer science", "economics", "sociology", "engineering",
                 "education", "environmental science", "neuroscience", "data science", "public health"]
        methods = ["experimental", "observational", "survey-based", "case study", "meta-analysis",
                  "longitudinal", "cross-sectional", "mixed-methods", "qualitative", "quantitative"]
        
        field = fields[index % len(fields)]
        method = methods[index % len(methods)]
        complexity = ["low", "medium", "high"][index % 3]
        
        participants = random.randint(50, 1000)
        duration_months = random.randint(6, 36)
        funding = random.randint(25000, 1000000)
        
        return Scenario(
            id=scenario_id,
            domain="research_study",
            title=f"{field.title()} {method} research study",
            description=f"Design and conduct a {method} research study in {field}.",
            complexity=complexity,
            context={
                "research_field": field,
                "methodology": method,
                "participants": participants,
                "duration_months": duration_months,
                "funding_amount": funding,
                "ethics_approval_needed": random.choice([True, False]),
                "data_type": random.choice(["quantitative", "qualitative", "mixed"]),
                "collaboration": random.choice(["single institution", "multi-institutional", "international"]),
                "publication_target": random.choice(["peer-reviewed journal", "conference", "report"])
            },
            expected_outputs=[
                "Research design and methodology",
                "Literature review and background analysis",
                "Participant recruitment strategy",
                "Data collection plan and instruments",
                "Statistical analysis framework",
                "Ethics and compliance documentation",
                "Timeline and milestone schedule",
                "Budget and resource allocation",
                "Risk assessment and quality control",
                "Dissemination and publication plan"
            ]
        )
    
    def _create_business_scenario(self, scenario_id: str, index: int) -> Scenario:
        """Create business launch scenario."""
        business_types = ["tech startup", "restaurant", "e-commerce store", "consulting firm", "SaaS platform",
                         "retail store", "service business", "manufacturing company", "franchise", "online marketplace"]
        industries = ["technology", "food & beverage", "healthcare", "education", "finance", "retail",
                     "entertainment", "logistics", "real estate", "professional services"]
        
        business_type = business_types[index % len(business_types)]
        industry = industries[index % len(industries)]
        complexity = ["low", "medium", "high"][index % 3]
        
        initial_investment = random.randint(25000, 1000000)
        target_revenue_y1 = random.randint(100000, 2000000)
        employees = random.randint(1, 25)
        
        return Scenario(
            id=scenario_id,
            domain="business_launch",
            title=f"{industry.title()} {business_type} launch",
            description=f"Launch a {business_type} in the {industry} industry.",
            complexity=complexity,
            context={
                "business_type": business_type,
                "industry": industry,
                "initial_investment": initial_investment,
                "target_revenue_y1": target_revenue_y1,
                "initial_employees": employees,
                "target_market": random.choice(["B2B", "B2C", "B2B2C"]),
                "geographic_scope": random.choice(["local", "regional", "national", "international"]),
                "business_model": random.choice(["subscription", "one-time purchase", "freemium", "marketplace"]),
                "timeline_to_launch": random.randint(3, 12)
            },
            expected_outputs=[
                "Business plan and strategy",
                "Market analysis and competitive landscape",
                "Financial projections and funding strategy",
                "Product/service development plan",
                "Marketing and customer acquisition strategy",
                "Operations and organizational structure",
                "Legal and regulatory compliance",
                "Technology and infrastructure requirements",
                "Risk analysis and mitigation strategies",
                "Launch timeline and milestone tracking"
            ]
        )
    
    def validate_scenarios(self) -> bool:
        """Validate that exactly 450 scenarios were generated."""
        total_scenarios = sum(len(scenarios) for scenarios in self.scenarios.values())
        
        if total_scenarios != self.scenarios_per_domain * len(self.domains):
            raise ValueError(f"Expected {self.scenarios_per_domain * len(self.domains)} scenarios, got {total_scenarios}")
        
        for domain in self.domains:
            if len(self.scenarios[domain]) != self.scenarios_per_domain:
                raise ValueError(f"Domain {domain} has {len(self.scenarios[domain])} scenarios, expected {self.scenarios_per_domain}")
        
        return True
    
    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        """Get all scenarios as dictionaries."""
        all_scenarios = []
        for domain_scenarios in self.scenarios.values():
            for scenario in domain_scenarios:
                all_scenarios.append(asdict(scenario))
        
        # Sort for deterministic order
        return sorted(all_scenarios, key=lambda x: (x['domain'], x['id']))
    
    def save_scenarios(self, output_dir: Path) -> None:
        """Save scenarios to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scenarios_data = {
            "metadata": {
                "total_scenarios": sum(len(scenarios) for scenarios in self.scenarios.values()),
                "scenarios_per_domain": self.scenarios_per_domain,
                "domains": self.domains,
                "dropped_scenario_ids": self.dropped_scenario_ids,
                "seed": self.seed
            },
            "scenarios": {
                domain: [asdict(scenario) for scenario in scenarios]
                for domain, scenarios in self.scenarios.items()
            }
        }
        
        scenarios_file = output_dir / "scenarios.json"
        with open(scenarios_file, 'w') as f:
            json.dump(scenarios_data, f, indent=2)
    
    def print_summary(self) -> None:
        """Print scenario generation summary."""
        total_scenarios = sum(len(scenarios) for scenarios in self.scenarios.values())
        domain_counts = [len(self.scenarios[domain]) for domain in self.domains]
        
        print(f"\nScenario Generation Summary:")
        print(f"Total scenarios: {total_scenarios}")
        print(f"Scenarios per domain: {','.join(map(str, domain_counts))}")
        print(f"Dropped scenarios: {len(self.dropped_scenario_ids)}")
        if self.dropped_scenario_ids:
            print(f"Sample dropped IDs: {', '.join(self.dropped_scenario_ids[:5])}{'...' if len(self.dropped_scenario_ids) > 5 else ''}")
        print(f"Domains: {', '.join(self.domains)}")
        
        # Complexity distribution
        complexity_dist = {"low": 0, "medium": 0, "high": 0}
        for scenarios in self.scenarios.values():
            for scenario in scenarios:
                complexity_dist[scenario.complexity] += 1
        
        print(f"Complexity distribution: {complexity_dist}")
