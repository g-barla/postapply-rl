"""
Q-Learning Agent for Follow-Up Timing Optimization

PROBLEM: When to follow up after applying to a job?

STATE SPACE:
- days_since_applied: int (0-30)
- company_type: str ('startup', 'midsize', 'enterprise')
- has_connection: bool (True/False)

ACTION SPACE:
- wait_1d, wait_3d, wait_5d, wait_7d, wait_10d, wait_14d

REWARD FUNCTION:
+20 if got response
+50 if got interview
-2 per day waited (encourages faster follow-up)

Q-LEARNING UPDATE:
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
"""

import numpy as np
import json
import random
from typing import Dict, Tuple, List, Optional
from datetime import datetime


class QLearningScheduler:
    """Q-Learning agent for optimal follow-up timing"""
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        load_from_dict: Optional[Dict] = None
    ):
        """
        Initialize Q-Learning agent
        
        Args:
            learning_rate: α, how much to update Q-values (0-1)
            discount_factor: γ, importance of future rewards (0-1)
            epsilon: Exploration rate for ε-greedy (0-1)
            load_from_dict: Optional dict to load existing Q-table
        """
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Action space: How many days to wait before following up
        self.actions = ['wait_1d', 'wait_3d', 'wait_5d', 'wait_7d', 'wait_10d', 'wait_14d']
        
        # Q-table: {state_key: {action: q_value}}
        # State key format: "days_company_connection"
        # Example: "3_startup_True" means 3 days, startup, has connection
        self.q_table = {}
        
        # Statistics
        self.total_updates = 0
        self.episode_rewards = []
        
        # Load existing Q-table if provided
        if load_from_dict:
            self.load_from_dict(load_from_dict)
        
        print(f"✅ Q-Learning Agent initialized")
        print(f"   α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
        print(f"   Actions: {self.actions}")
    
    def _get_state_key(
        self,
        days_since_applied: int,
        company_type: str,
        has_connection: bool
    ) -> str:
        """
        Convert state tuple to string key for Q-table
        
        Args:
            days_since_applied: Days since application
            company_type: 'startup', 'midsize', 'enterprise'
            has_connection: True/False
        
        Returns:
            State key string (e.g., "3_startup_True")
        """
        # Discretize days into buckets to reduce state space
        if days_since_applied <= 2:
            day_bucket = "0-2"
        elif days_since_applied <= 5:
            day_bucket = "3-5"
        elif days_since_applied <= 10:
            day_bucket = "6-10"
        else:
            day_bucket = "11+"
        
        return f"{day_bucket}_{company_type}_{has_connection}"
    
    def get_action(
        self,
        days_since_applied: int,
        company_type: str,
        has_connection: bool,
        explore: bool = True
    ) -> str:
        """
        Choose action using ε-greedy policy
        
        Args:
            days_since_applied: Days since application
            company_type: 'startup', 'midsize', 'enterprise'
            has_connection: True/False
            explore: If True, use ε-greedy. If False, always exploit (greedy)
        
        Returns:
            Action string (e.g., 'wait_5d')
        
        Example:
            action = agent.get_action(3, 'startup', True)
            # Returns: 'wait_3d' or 'wait_5d'
        """
        state_key = self._get_state_key(days_since_applied, company_type, has_connection)
        
        # Initialize state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}
        
        # ε-greedy: Explore vs Exploit
        if explore and random.random() < self.epsilon:
            # EXPLORE: Random action
            action = random.choice(self.actions)
        else:
            # EXPLOIT: Best action based on Q-values
            q_values = self.q_table[state_key]
            max_q = max(q_values.values())
            
            # Get all actions with max Q-value (handle ties randomly)
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)
        
        return action
    
    def update(
        self,
        state: Tuple[int, str, bool],
        action: str,
        reward: float,
        next_state: Optional[Tuple[int, str, bool]] = None
    ):
        """
        Update Q-value using Q-Learning update rule
        
        Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        
        Args:
            state: (days_since_applied, company_type, has_connection)
            action: Action taken (e.g., 'wait_5d')
            reward: Reward received
            next_state: Next state (None if terminal)
        
        Example:
            # You waited 5 days, then followed up and got response (+20)
            agent.update(
                state=(3, 'startup', True),
                action='wait_5d',
                reward=20 - 2*5,  # +20 for response, -10 for waiting 5 days
                next_state=None   # Terminal state
            )
        """
        state_key = self._get_state_key(*state)
        
        # Initialize state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Calculate max Q-value for next state (0 if terminal)
        if next_state is None:
            max_next_q = 0.0
        else:
            next_state_key = self._get_state_key(*next_state)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = {a: 0.0 for a in self.actions}
            max_next_q = max(self.q_table[next_state_key].values())
        
        # Q-Learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[state_key][action] = new_q
        
        # Statistics
        self.total_updates += 1
    
    def get_best_action(
        self,
        days_since_applied: int,
        company_type: str,
        has_connection: bool
    ) -> Tuple[str, float]:
        """
        Get best action and its Q-value (pure exploitation)
        
        Args:
            days_since_applied: Days since application
            company_type: 'startup', 'midsize', 'enterprise'
            has_connection: True/False
        
        Returns:
            (best_action, q_value)
        
        Example:
            action, confidence = agent.get_best_action(3, 'startup', True)
            print(f"Best: {action} (Q={confidence:.2f})")
        """
        state_key = self._get_state_key(days_since_applied, company_type, has_connection)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}
        
        q_values = self.q_table[state_key]
        best_action = max(q_values, key=q_values.get)
        best_q = q_values[best_action]
        
        return best_action, best_q
    
    def get_q_table_summary(self) -> Dict:
        """
        Get summary statistics of Q-table
        
        Returns:
            Dict with stats about learned Q-table
        """
        if not self.q_table:
            return {
                'total_states': 0,
                'total_updates': self.total_updates,
                'avg_q_value': 0.0
            }
        
        all_q_values = []
        for state_actions in self.q_table.values():
            all_q_values.extend(state_actions.values())
        
        return {
            'total_states': len(self.q_table),
            'total_updates': self.total_updates,
            'avg_q_value': np.mean(all_q_values) if all_q_values else 0.0,
            'max_q_value': np.max(all_q_values) if all_q_values else 0.0,
            'min_q_value': np.min(all_q_values) if all_q_values else 0.0
        }
    
    def to_dict(self) -> Dict:
        """
        Serialize Q-table to dictionary (for database storage)
        
        Returns:
            Dict that can be JSON serialized
        """
        return {
            'q_table': self.q_table,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'actions': self.actions,
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def load_from_dict(self, data: Dict):
        """
        Load Q-table from dictionary
        
        Args:
            data: Dict from to_dict()
        """
        self.q_table = data.get('q_table', {})
        self.alpha = data.get('alpha', self.alpha)
        self.gamma = data.get('gamma', self.gamma)
        self.epsilon = data.get('epsilon', self.epsilon)
        self.total_updates = data.get('total_updates', 0)
        self.actions = data.get('actions', self.actions)
        
        print(f"✅ Loaded Q-table with {len(self.q_table)} states")


# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Q-Learning Scheduler")
    print("=" * 70)
    print()
    
    # Initialize agent
    agent = QLearningScheduler(
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1
    )
    
    # Test 1: Get action for new state
    print("TEST 1: Get action for startup, 3 days, has connection")
    print("-" * 70)
    
    action = agent.get_action(
        days_since_applied=3,
        company_type='startup',
        has_connection=True
    )
    print(f"Selected action: {action}")
    print()
    
    # Test 2: Simulate learning from experience
    print("TEST 2: Simulate learning episodes")
    print("-" * 70)
    
    # Simulate 100 episodes
    for episode in range(100):
        # Random initial state
        days = random.randint(0, 10)
        company = random.choice(['startup', 'midsize', 'enterprise'])
        connection = random.choice([True, False])
        
        # Get action
        action = agent.get_action(days, company, connection)
        
        # Simulate outcome
        # Shorter waits for startups, longer for enterprise
        # Better outcomes with connections
        wait_days = int(action.split('_')[1].replace('d', ''))
        
        # Calculate reward
        base_reward = 0
        if wait_days <= 3 and company == 'startup':
            base_reward = 20  # Good timing for startup
        elif wait_days >= 7 and company == 'enterprise':
            base_reward = 20  # Enterprise needs more time
        
        if connection:
            base_reward += 10  # Connections help
        
        # Add random noise
        response = random.random() < 0.3  # 30% response rate
        if response:
            base_reward += 20
        
        # Subtract waiting penalty
        reward = base_reward - (2 * wait_days)
        
        # Update Q-table
        agent.update(
            state=(days, company, connection),
            action=action,
            reward=reward,
            next_state=None
        )
    
    print(f"Completed 100 training episodes")
    print()
    
    # Test 3: Check learned policy
    print("TEST 3: Learned policy (best actions)")
    print("-" * 70)
    
    test_states = [
        (3, 'startup', True),
        (3, 'startup', False),
        (7, 'enterprise', True),
        (7, 'enterprise', False),
    ]
    
    for state in test_states:
        days, company, connection = state
        action, q_value = agent.get_best_action(days, company, connection)
        
        print(f"State: {days} days, {company}, connection={connection}")
        print(f"  → Best action: {action} (Q={q_value:.2f})")
        print()
    
    # Test 4: Get summary stats
    print("TEST 4: Q-table statistics")
    print("-" * 70)
    
    stats = agent.get_q_table_summary()
    print(f"Total states explored: {stats['total_states']}")
    print(f"Total updates: {stats['total_updates']}")
    print(f"Average Q-value: {stats['avg_q_value']:.2f}")
    print(f"Max Q-value: {stats['max_q_value']:.2f}")
    print(f"Min Q-value: {stats['min_q_value']:.2f}")
    print()
    
    # Test 5: Serialization
    print("TEST 5: Save/load Q-table")
    print("-" * 70)
    
    # Save
    saved_data = agent.to_dict()
    print(f"Saved Q-table: {len(saved_data['q_table'])} states")
    
    # Load into new agent
    new_agent = QLearningScheduler(load_from_dict=saved_data)
    print(f"Loaded into new agent successfully")
    
    print("\n" + "=" * 70)
    print("✅ Q-Learning Agent Test Complete!")
    print("=" * 70)
