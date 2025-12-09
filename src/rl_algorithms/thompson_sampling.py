"""
Thompson Sampling Agent for Message Style Selection

PROBLEM: Which message style gets best response rate?

CONTEXT:
- contact_title: str ('recruiter', 'manager', 'director', 'vp')
- company_culture: str ('casual', 'formal', 'mixed')
- has_connection: bool (True/False)

ARMS (Actions):
- 'formal': Professional, structured messaging
- 'casual': Friendly, conversational messaging  
- 'connection_focused': Emphasize mutual connections

REWARDS:
+1 if got response
+0 if no response

THOMPSON SAMPLING:
Each arm has Beta(α, β) distribution
α = successes + 1
β = failures + 1

To choose arm:
1. Sample from each Beta distribution
2. Pick arm with highest sample
3. Update based on outcome
"""

import numpy as np
import json
from typing import Dict, Tuple, Optional
from datetime import datetime


class ThompsonSamplingMessenger:
    """Thompson Sampling agent for message style selection"""
    
    def __init__(self, load_from_dict: Optional[Dict] = None):
        """
        Initialize Thompson Sampling agent
        
        Args:
            load_from_dict: Optional dict to load existing distributions
        """
        
        # Arms (message styles)
        self.arms = ['formal', 'casual', 'connection_focused']
        
        # Beta distributions per context
        # Format: {context_key: {arm: {'alpha': a, 'beta': b}}}
        self.distributions = {}
        
        # Statistics
        self.total_selections = 0
        self.total_successes = 0
        self.arm_history = []  # Track which arms were selected
        
        # Load existing distributions if provided
        if load_from_dict:
            self.load_from_dict(load_from_dict)
        
        print(f"✅ Thompson Sampling Agent initialized")
        print(f"   Arms: {self.arms}")
    
    def _get_context_key(
        self,
        contact_title: str,
        company_culture: str,
        has_connection: bool
    ) -> str:
        """
        Convert context tuple to string key
        
        Args:
            contact_title: 'recruiter', 'manager', 'director', 'vp', etc.
            company_culture: 'casual', 'formal', 'mixed'
            has_connection: True/False
        
        Returns:
            Context key string (e.g., "recruiter_casual_True")
        """
        # Simplify title categories
        title_lower = contact_title.lower()
        
        if any(word in title_lower for word in ['recruit', 'talent', 'hr']):
            title_category = 'recruiter'
        elif any(word in title_lower for word in ['vp', 'vice president', 'chief', 'head']):
            title_category = 'executive'
        elif any(word in title_lower for word in ['director', 'lead']):
            title_category = 'director'
        else:
            title_category = 'manager'
        
        return f"{title_category}_{company_culture}_{has_connection}"
    
    def select_arm(
        self,
        contact_title: str,
        company_culture: str,
        has_connection: bool
    ) -> str:
        """
        Select message style using Thompson Sampling
        
        Args:
            contact_title: Contact's job title
            company_culture: Company culture type
            has_connection: Whether you have a connection
        
        Returns:
            Selected arm/style ('formal', 'casual', 'connection_focused')
        
        Example:
            style = agent.select_arm('Recruiting Manager', 'casual', True)
            # Returns: 'connection_focused' (probabilistically)
        """
        context_key = self._get_context_key(contact_title, company_culture, has_connection)
        
        # Initialize context if not seen before
        if context_key not in self.distributions:
            self.distributions[context_key] = {
                arm: {'alpha': 1.0, 'beta': 1.0}  # Uniform prior
                for arm in self.arms
            }
        
        # Thompson Sampling: Sample from each Beta distribution
        samples = {}
        for arm in self.arms:
            alpha = self.distributions[context_key][arm]['alpha']
            beta = self.distributions[context_key][arm]['beta']
            
            # Sample from Beta(α, β)
            sample = np.random.beta(alpha, beta)
            samples[arm] = sample
        
        # Select arm with highest sample
        selected_arm = max(samples, key=samples.get)
        
        # Track selection
        self.total_selections += 1
        self.arm_history.append({
            'context': context_key,
            'arm': selected_arm,
            'samples': samples
        })
        
        return selected_arm
    
    def update(
        self,
        contact_title: str,
        company_culture: str,
        has_connection: bool,
        arm: str,
        got_response: bool
    ):
        """
        Update Beta distribution based on outcome
        
        Args:
            contact_title: Contact's job title
            company_culture: Company culture type
            has_connection: Whether you have a connection
            arm: Arm that was selected
            got_response: True if got response, False otherwise
        
        Example:
            # You used 'casual' style and got a response
            agent.update('Recruiter', 'casual', True, 'casual', got_response=True)
            # Updates: casual arm's alpha += 1 (success!)
            
            # You used 'formal' style and got no response
            agent.update('VP Data', 'formal', False, 'formal', got_response=False)
            # Updates: formal arm's beta += 1 (failure)
        """
        context_key = self._get_context_key(contact_title, company_culture, has_connection)
        
        # Initialize context if not seen before
        if context_key not in self.distributions:
            self.distributions[context_key] = {
                a: {'alpha': 1.0, 'beta': 1.0}
                for a in self.arms
            }
        
        # Update Beta distribution
        if got_response:
            # Success: Increment alpha
            self.distributions[context_key][arm]['alpha'] += 1
            self.total_successes += 1
        else:
            # Failure: Increment beta
            self.distributions[context_key][arm]['beta'] += 1
    
    def get_arm_probabilities(
        self,
        contact_title: str,
        company_culture: str,
        has_connection: bool
    ) -> Dict[str, float]:
        """
        Get current probability estimates for each arm
        
        Returns the mean of each Beta distribution:
        P(success) = α / (α + β)
        
        Args:
            contact_title: Contact's job title
            company_culture: Company culture
            has_connection: Connection status
        
        Returns:
            Dict of {arm: probability}
        
        Example:
            probs = agent.get_arm_probabilities('Recruiter', 'casual', True)
            # Returns: {'formal': 0.45, 'casual': 0.65, 'connection_focused': 0.72}
        """
        context_key = self._get_context_key(contact_title, company_culture, has_connection)
        
        if context_key not in self.distributions:
            # Uniform prior
            return {arm: 0.5 for arm in self.arms}
        
        probabilities = {}
        for arm in self.arms:
            alpha = self.distributions[context_key][arm]['alpha']
            beta = self.distributions[context_key][arm]['beta']
            
            # Mean of Beta distribution
            prob = alpha / (alpha + beta)
            probabilities[arm] = prob
        
        return probabilities
    
    def get_best_arm(
        self,
        contact_title: str,
        company_culture: str,
        has_connection: bool
    ) -> Tuple[str, float]:
        """
        Get best arm based on current knowledge (pure exploitation)
        
        Args:
            contact_title: Contact's job title
            company_culture: Company culture
            has_connection: Connection status
        
        Returns:
            (best_arm, probability)
        
        Example:
            arm, prob = agent.get_best_arm('Recruiter', 'casual', True)
            print(f"Best: {arm} with {prob:.1%} success rate")
        """
        probs = self.get_arm_probabilities(contact_title, company_culture, has_connection)
        best_arm = max(probs, key=probs.get)
        best_prob = probs[best_arm]
        
        return best_arm, best_prob
    
    def get_statistics(self) -> Dict:
        """
        Get summary statistics
        
        Returns:
            Dict with agent statistics
        """
        total_contexts = len(self.distributions)
        
        # Calculate overall success rate
        success_rate = self.total_successes / max(self.total_selections, 1)
        
        # Get arm selection counts
        arm_counts = {arm: 0 for arm in self.arms}
        for entry in self.arm_history:
            arm_counts[entry['arm']] += 1
        
        return {
            'total_contexts': total_contexts,
            'total_selections': self.total_selections,
            'total_successes': self.total_successes,
            'success_rate': success_rate,
            'arm_counts': arm_counts
        }
    
    def to_dict(self) -> Dict:
        """
        Serialize distributions to dictionary (for database storage)
        
        Returns:
            Dict that can be JSON serialized
        """
        return {
            'distributions': self.distributions,
            'arms': self.arms,
            'total_selections': self.total_selections,
            'total_successes': self.total_successes,
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def load_from_dict(self, data: Dict):
        """
        Load distributions from dictionary
        
        Args:
            data: Dict from to_dict()
        """
        self.distributions = data.get('distributions', {})
        self.arms = data.get('arms', self.arms)
        self.total_selections = data.get('total_selections', 0)
        self.total_successes = data.get('total_successes', 0)
        
        print(f"✅ Loaded distributions for {len(self.distributions)} contexts")


# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Thompson Sampling Messenger")
    print("=" * 70)
    print()
    
    # Initialize agent
    agent = ThompsonSamplingMessenger()
    
    # Test 1: Select arm for new context
    print("TEST 1: Select arm for recruiter at casual startup")
    print("-" * 70)
    
    arm = agent.select_arm(
        contact_title='Recruiting Manager',
        company_culture='casual',
        has_connection=True
    )
    print(f"Selected arm: {arm}")
    print()
    
    # Test 2: Simulate learning episodes
    print("TEST 2: Simulate 200 learning episodes")
    print("-" * 70)
    
    # Simulate different scenarios with different true success rates
    scenarios = [
        {
            'title': 'Recruiting Manager',
            'culture': 'casual',
            'connection': True,
            'true_rates': {'formal': 0.3, 'casual': 0.6, 'connection_focused': 0.7}  # Connection-focused best!
        },
        {
            'title': 'VP of Engineering',
            'culture': 'formal',
            'connection': False,
            'true_rates': {'formal': 0.5, 'casual': 0.2, 'connection_focused': 0.1}  # Formal best!
        },
        {
            'title': 'Director of Data',
            'culture': 'mixed',
            'connection': True,
            'true_rates': {'formal': 0.4, 'casual': 0.4, 'connection_focused': 0.6}  # Connection-focused best!
        }
    ]
    
    for episode in range(200):
        # Pick random scenario
        scenario = scenarios[episode % len(scenarios)]
        
        # Agent selects arm
        arm = agent.select_arm(
            scenario['title'],
            scenario['culture'],
            scenario['connection']
        )
        
        # Simulate outcome based on true success rate
        success_prob = scenario['true_rates'][arm]
        got_response = np.random.random() < success_prob
        
        # Update agent
        agent.update(
            scenario['title'],
            scenario['culture'],
            scenario['connection'],
            arm,
            got_response
        )
    
    print(f"Completed 200 training episodes")
    print()
    
    # Test 3: Check learned preferences
    print("TEST 3: Learned preferences")
    print("-" * 70)
    
    for scenario in scenarios:
        print(f"\nContext: {scenario['title']}, {scenario['culture']}, connection={scenario['connection']}")
        print(f"True best: {max(scenario['true_rates'], key=scenario['true_rates'].get)} "
              f"(rate={max(scenario['true_rates'].values()):.1%})")
        
        probs = agent.get_arm_probabilities(
            scenario['title'],
            scenario['culture'],
            scenario['connection']
        )
        
        print(f"Learned probabilities:")
        for arm, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {arm:20s}: {prob:.1%}")
        
        best_arm, best_prob = agent.get_best_arm(
            scenario['title'],
            scenario['culture'],
            scenario['connection']
        )
        print(f"Agent's choice: {best_arm} (estimated {best_prob:.1%})")
    
    # Test 4: Statistics
    print("\n\nTEST 4: Agent statistics")
    print("-" * 70)
    
    stats = agent.get_statistics()
    print(f"Total contexts explored: {stats['total_contexts']}")
    print(f"Total selections: {stats['total_selections']}")
    print(f"Total successes: {stats['total_successes']}")
    print(f"Overall success rate: {stats['success_rate']:.1%}")
    print(f"\nArm selection counts:")
    for arm, count in stats['arm_counts'].items():
        print(f"  {arm:20s}: {count} times")
    print()
    
    # Test 5: Serialization
    print("TEST 5: Save/load distributions")
    print("-" * 70)
    
    # Save
    saved_data = agent.to_dict()
    print(f"Saved distributions: {len(saved_data['distributions'])} contexts")
    
    # Load into new agent
    new_agent = ThompsonSamplingMessenger(load_from_dict=saved_data)
    print(f"Loaded into new agent successfully")
    
    print("\n" + "=" * 70)
    print("✅ Thompson Sampling Agent Test Complete!")
    print("=" * 70)
