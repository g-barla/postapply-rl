"""
Simulation Framework for RL Performance Evaluation

Generates 100 synthetic applications and compares:
- Baseline (random policy)
- RL System (Q-Learning + Thompson Sampling)
"""
import sys
sys.path.append('src')

import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

from rl_algorithms.q_learning import QLearningScheduler
from rl_algorithms.thompson_sampling import ThompsonSamplingMessenger


class JobSearchSimulator:
    """
    Simulates realistic job search scenarios
    
    Generates synthetic applications with realistic outcome probabilities
    based on timing, message style, company type, and connections
    """
    
    def __init__(self, seed: int = 42):
        """Initialize simulator"""
        random.seed(seed)
        np.random.seed(seed)
        
        # Application templates
        self.companies = [
            ('Google', 'enterprise', 'mixed'),
            ('Meta', 'enterprise', 'casual'),
            ('Amazon', 'enterprise', 'mixed'),
            ('Microsoft', 'enterprise', 'formal'),
            ('Apple', 'enterprise', 'formal'),
            ('Stripe', 'midsize', 'casual'),
            ('Databricks', 'midsize', 'casual'),
            ('Snowflake', 'midsize', 'mixed'),
            ('Airbnb', 'midsize', 'casual'),
            ('Uber', 'midsize', 'mixed'),
            ('TechStartup Inc', 'startup', 'casual'),
            ('AI Ventures', 'startup', 'casual'),
            ('DataFlow Co', 'startup', 'casual'),
            ('CloudNine', 'startup', 'casual'),
            ('NextGen Analytics', 'startup', 'mixed'),
        ]
        
        self.roles = [
            ('Data Analyst', 'mid'),
            ('Business Intelligence Analyst', 'mid'),
            ('Data Scientist', 'mid'),
            ('Junior Data Analyst', 'junior'),
            ('Senior Data Analyst', 'senior'),
        ]
        
        self.contact_types = [
            ('recruiter', 'Recruiting Manager'),
            ('manager', 'Hiring Manager'),
            ('director', 'Director of Analytics'),
            ('executive', 'VP of Data'),
        ]
        
        print("✅ Simulator initialized")
    
    def generate_application(self, app_number: int) -> Dict:
        """Generate one synthetic application"""
        
        # Random company and role
        company, company_type, culture = random.choice(self.companies)
        role, seniority = random.choice(self.roles)
        
        # Connection probability based on company type
        connection_probs = {
            'startup': 0.2,   # 20% have connection at startups
            'midsize': 0.15,  # 15% at midsize
            'enterprise': 0.1  # 10% at big companies
        }
        has_connection = random.random() < connection_probs[company_type]
        
        # Contact type
        contact_category, contact_title = random.choice(self.contact_types)
        
        return {
            'app_number': app_number,
            'company': company,
            'role': role,
            'company_type': company_type,
            'company_culture': culture,
            'seniority': seniority,
            'has_connection': has_connection,
            'contact_category': contact_category,
            'contact_title': contact_title
        }
    
    def simulate_outcome(
        self,
        app: Dict,
        follow_up_days: int,
        message_style: str
    ) -> Tuple[bool, bool]:
        """
        Simulate realistic outcome based on actions
        
        Returns:
            (got_response, got_interview)
        """
        
        # BASE SUCCESS RATES (ground truth)
        base_response_rate = 0.25  # 25% baseline
        
        # TIMING FACTORS
        timing_multiplier = self._calculate_timing_multiplier(
            app['company_type'],
            follow_up_days,
            app['has_connection']
        )
        
        # STYLE FACTORS
        style_multiplier = self._calculate_style_multiplier(
            app['contact_category'],
            app['company_culture'],
            message_style,
            app['has_connection']
        )
        
        # CONNECTION BOOST
        connection_boost = 1.3 if app['has_connection'] else 1.0
        
        # FINAL RESPONSE PROBABILITY
        response_prob = min(0.8, base_response_rate * timing_multiplier * style_multiplier * connection_boost)
        
        got_response = random.random() < response_prob
        
        # INTERVIEW PROBABILITY (if got response)
        if got_response:
            interview_prob = 0.3  # 30% of responses lead to interviews
            got_interview = random.random() < interview_prob
        else:
            got_interview = False
        
        return got_response, got_interview
    
    def _calculate_timing_multiplier(
        self,
        company_type: str,
        days: int,
        has_connection: bool
    ) -> float:
        """Calculate timing quality multiplier"""
        
        # OPTIMAL TIMING (ground truth)
        optimal_timing = {
            'startup': (1, 5),      # Best: 1-5 days
            'midsize': (3, 7),      # Best: 3-7 days
            'enterprise': (5, 10)   # Best: 5-10 days
        }
        
        opt_min, opt_max = optimal_timing[company_type]
        
        # Connection gives more flexibility
        if has_connection:
            opt_min = max(1, opt_min - 2)
            opt_max = opt_max + 3
        
        # Calculate multiplier
        if opt_min <= days <= opt_max:
            return 1.5  # Optimal timing = 50% boost
        elif days < opt_min:
            # Too early
            penalty = (opt_min - days) * 0.1
            return max(0.5, 1.0 - penalty)
        else:
            # Too late
            penalty = (days - opt_max) * 0.05
            return max(0.5, 1.0 - penalty)
    
    def _calculate_style_multiplier(
        self,
        contact_category: str,
        culture: str,
        style: str,
        has_connection: bool
    ) -> float:
        """Calculate message style quality multiplier"""
        
        # OPTIMAL STYLES (ground truth)
        style_scores = {
            ('recruiter', 'casual', 'casual'): 1.4,
            ('recruiter', 'casual', 'connection_focused'): 1.5,
            ('recruiter', 'formal', 'formal'): 1.3,
            ('recruiter', 'mixed', 'casual'): 1.2,
            
            ('manager', 'casual', 'casual'): 1.3,
            ('manager', 'formal', 'formal'): 1.3,
            ('manager', 'mixed', 'connection_focused'): 1.4,
            
            ('director', 'casual', 'casual'): 1.2,
            ('director', 'formal', 'formal'): 1.4,
            ('director', 'mixed', 'formal'): 1.3,
            
            ('executive', 'casual', 'connection_focused'): 1.3,
            ('executive', 'formal', 'formal'): 1.5,
            ('executive', 'mixed', 'formal'): 1.4,
        }
        
        # Get score for this combination
        key = (contact_category, culture, style)
        multiplier = style_scores.get(key, 1.0)
        
        # Connection-focused always gets boost with connections
        if style == 'connection_focused' and has_connection:
            multiplier *= 1.2
        
        return multiplier


class ExperimentRunner:
    """Runs baseline vs RL experiments"""
    
    def __init__(self):
        self.simulator = JobSearchSimulator()
        print("✅ Experiment Runner initialized")
    
    def run_baseline(self, num_episodes: int = 100) -> Dict:
        """
        Run baseline experiment (random policy)
        
        Returns:
            Results dict with metrics
        """
        print(f"\n{'='*70}")
        print(f"BASELINE EXPERIMENT (Random Policy)")
        print(f"{'='*70}")
        
        results = {
            'episodes': [],
            'response_rate_history': [],
            'interview_rate_history': [],
            'total_responses': 0,
            'total_interviews': 0
        }
        
        for i in range(num_episodes):
            # Generate application
            app = self.simulator.generate_application(i + 1)
            
            # RANDOM POLICY
            random_timing = random.choice([1, 3, 5, 7, 10, 14])
            random_style = random.choice(['formal', 'casual', 'connection_focused'])
            
            # Simulate outcome
            got_response, got_interview = self.simulator.simulate_outcome(
                app, random_timing, random_style
            )
            
            # Track results
            if got_response:
                results['total_responses'] += 1
            if got_interview:
                results['total_interviews'] += 1
            
            # Rolling response rate
            response_rate = results['total_responses'] / (i + 1)
            interview_rate = results['total_interviews'] / (i + 1)
            
            results['response_rate_history'].append(response_rate)
            results['interview_rate_history'].append(interview_rate)
            
            results['episodes'].append({
                'app_number': i + 1,
                'company': app['company'],
                'timing': random_timing,
                'style': random_style,
                'got_response': got_response,
                'got_interview': got_interview
            })
            
            # Progress
            if (i + 1) % 25 == 0:
                print(f"  Episode {i+1}: Response rate = {response_rate:.1%}, Interview rate = {interview_rate:.1%}")
        
        final_response_rate = results['total_responses'] / num_episodes
        final_interview_rate = results['total_interviews'] / num_episodes
        
        print(f"\n✅ BASELINE RESULTS:")
        print(f"   Total responses: {results['total_responses']}/{num_episodes} ({final_response_rate:.1%})")
        print(f"   Total interviews: {results['total_interviews']}/{num_episodes} ({final_interview_rate:.1%})")
        
        return results
    
    def run_rl_experiment(self, num_episodes: int = 100) -> Dict:
        """
        Run RL experiment (Q-Learning + Thompson Sampling)
        
        Returns:
            Results dict with metrics
        """
        print(f"\n{'='*70}")
        print(f"RL EXPERIMENT (Q-Learning + Thompson Sampling)")
        print(f"{'='*70}")
        
        # Initialize RL agents
        ql_agent = QLearningScheduler(learning_rate=0.1, discount_factor=0.9, epsilon=0.15)
        ts_agent = ThompsonSamplingMessenger()
        
        results = {
            'episodes': [],
            'response_rate_history': [],
            'interview_rate_history': [],
            'total_responses': 0,
            'total_interviews': 0,
            'q_values_history': [],
            'ts_confidence_history': []
        }
        
        for i in range(num_episodes):
            # Generate application
            app = self.simulator.generate_application(i + 1)
            
            # RL POLICY
            # Q-Learning selects timing
            days_since = random.randint(0, 5)  # Simulate some days passing
            ql_action = ql_agent.get_action(
                days_since_applied=days_since,
                company_type=app['company_type'],
                has_connection=app['has_connection']
            )
            rl_timing = int(ql_action.split('_')[1].replace('d', ''))
            
            # Thompson Sampling selects style
            rl_style = ts_agent.select_arm(
                contact_title=app['contact_title'],
                company_culture=app['company_culture'],
                has_connection=app['has_connection']
            )
            
            # Simulate outcome
            got_response, got_interview = self.simulator.simulate_outcome(
                app, rl_timing, rl_style
            )
            
            # UPDATE RL AGENTS (LEARNING HAPPENS HERE!)
            # Q-Learning update
            reward = 0
            if got_interview:
                reward = 50
            elif got_response:
                reward = 20
            reward -= (2 * rl_timing)  # Penalty for waiting
            
            ql_agent.update(
                state=(days_since, app['company_type'], app['has_connection']),
                action=ql_action,
                reward=reward,
                next_state=None
            )
            
            # Thompson Sampling update
            ts_agent.update(
                contact_title=app['contact_title'],
                company_culture=app['company_culture'],
                has_connection=app['has_connection'],
                arm=rl_style,
                got_response=got_response
            )
            
            # Track results
            if got_response:
                results['total_responses'] += 1
            if got_interview:
                results['total_interviews'] += 1
            
            # Rolling rates
            response_rate = results['total_responses'] / (i + 1)
            interview_rate = results['total_interviews'] / (i + 1)
            
            results['response_rate_history'].append(response_rate)
            results['interview_rate_history'].append(interview_rate)
            
            # Track learning metrics
            avg_q = ql_agent.get_q_table_summary()['avg_q_value']
            results['q_values_history'].append(avg_q)
            
            ts_stats = ts_agent.get_statistics()
            results['ts_confidence_history'].append(ts_stats['success_rate'])
            
            results['episodes'].append({
                'app_number': i + 1,
                'company': app['company'],
                'timing': rl_timing,
                'style': rl_style,
                'got_response': got_response,
                'got_interview': got_interview,
                'reward': reward
            })
            
            # Progress
            if (i + 1) % 25 == 0:
                print(f"  Episode {i+1}: Response rate = {response_rate:.1%}, Interview rate = {interview_rate:.1%}, Avg Q = {avg_q:.2f}")
        
        final_response_rate = results['total_responses'] / num_episodes
        final_interview_rate = results['total_interviews'] / num_episodes
        
        print(f"\n✅ RL RESULTS:")
        print(f"   Total responses: {results['total_responses']}/{num_episodes} ({final_response_rate:.1%})")
        print(f"   Total interviews: {results['total_interviews']}/{num_episodes} ({final_interview_rate:.1%})")
        print(f"   Final Avg Q-value: {avg_q:.2f}")
        
        return results, ql_agent, ts_agent


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("=" * 70)
    print("POSTAPPLY RL SYSTEM - SIMULATION EXPERIMENTS")
    print("=" * 70)
    
    runner = ExperimentRunner()
    
    # Run experiments
    print("\n🎯 Running experiments with 100 applications each...\n")
    
    baseline_results = runner.run_baseline(num_episodes=500)
    rl_results, ql_agent, ts_agent = runner.run_rl_experiment(num_episodes=500)
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    
    baseline_response = baseline_results['total_responses'] / 100
    rl_response = rl_results['total_responses'] / 100
    improvement = ((rl_response - baseline_response) / baseline_response) * 100
    
    print(f"\nResponse Rates:")
    print(f"  Baseline: {baseline_response:.1%}")
    print(f"  RL System: {rl_response:.1%}")
    print(f"  Improvement: {improvement:+.1f}%")
    
    baseline_interview = baseline_results['total_interviews'] / 100
    rl_interview = rl_results['total_interviews'] / 100
    interview_improvement = ((rl_interview - baseline_interview) / max(baseline_interview, 0.01)) * 100
    
    print(f"\nInterview Rates:")
    print(f"  Baseline: {baseline_interview:.1%}")
    print(f"  RL System: {rl_interview:.1%}")
    print(f"  Improvement: {interview_improvement:+.1f}%")
    
    # Save results
    results_data = {
        'baseline': baseline_results,
        'rl': rl_results,
        'summary': {
            'baseline_response_rate': baseline_response,
            'rl_response_rate': rl_response,
            'improvement_pct': improvement,
            'baseline_interview_rate': baseline_interview,
            'rl_interview_rate': rl_interview,
            'interview_improvement_pct': interview_improvement
        }
    }
    
    with open('simulation_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\n💾 Results saved to simulation_results.json")
    print("\n✅ Simulation Complete!")
    print("=" * 70)
