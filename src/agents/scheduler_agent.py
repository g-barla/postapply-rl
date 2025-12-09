"""
Scheduler Agent: Recommends follow-up timing using Q-Learning
"""
import sys
sys.path.append('src')

from rl_algorithms.q_learning import QLearningScheduler
from database import SessionLocal, Application, RLState
from datetime import datetime
from typing import Dict
import json


class SchedulerAgent:
    """Agent for follow-up timing optimization using Q-Learning"""
    
    def __init__(self):
        self.ql_agent = QLearningScheduler(
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.1
        )
        self._load_from_database()
        print("✅ Scheduler Agent initialized")
    
    def _load_from_database(self):
        """Load Q-table from database"""
        db = SessionLocal()
        try:
            rl_state = db.query(RLState).filter(
                RLState.agent_type == 'q_learning'
            ).first()
            
            if rl_state and rl_state.q_table:
                self.ql_agent.load_from_dict({
                    'q_table': rl_state.q_table,
                    'alpha': rl_state.learning_rate,
                    'gamma': rl_state.discount_factor,
                    'epsilon': rl_state.epsilon,
                    'total_updates': rl_state.total_updates
                })
                print(f"   📊 Loaded Q-table with {len(rl_state.q_table)} states")
        except Exception as e:
            print(f"   ⚠️  No existing Q-table found")
        finally:
            db.close()
    
    def save_to_database(self):
        """Save Q-table to database"""
        db = SessionLocal()
        try:
            ql_data = self.ql_agent.to_dict()
            
            rl_state = db.query(RLState).filter(
                RLState.agent_type == 'q_learning'
            ).first()
            
            if rl_state:
                rl_state.q_table = ql_data['q_table']
                rl_state.total_updates = ql_data['total_updates']
                rl_state.last_updated = datetime.utcnow()
            else:
                rl_state = RLState(
                    agent_type='q_learning',
                    q_table=ql_data['q_table'],
                    learning_rate=ql_data['alpha'],
                    discount_factor=ql_data['gamma'],
                    epsilon=ql_data['epsilon'],
                    total_updates=ql_data['total_updates']
                )
                db.add(rl_state)
            
            db.commit()
            print("   💾 Q-table saved")
        except Exception as e:
            db.rollback()
            print(f"   ❌ Save error: {e}")
        finally:
            db.close()
    
    def get_recommendation(self, application_id: int) -> Dict:
        """Get follow-up timing recommendation"""
        
        print(f"\n⏰ Recommendation for app {application_id}")
        
        db = SessionLocal()
        try:
            app = db.query(Application).filter(Application.id == application_id).first()
            
            if not app:
                return {'status': 'error', 'error': 'Application not found'}
            
            days_since = (datetime.utcnow() - app.applied_date).days
            
            print(f"   📅 {days_since} days since applied")
            print(f"   🏢 {app.company_type}, connection={app.has_connection}")
            
            # Get recommendation
            action, q_value = self.ql_agent.get_best_action(
                days_since_applied=days_since,
                company_type=app.company_type or 'midsize',
                has_connection=app.has_connection or False
            )
            
            wait_days = int(action.split('_')[1].replace('d', ''))
            
            print(f"   ✅ Wait {wait_days} more days (Q={q_value:.2f})")
            
            return {
                'status': 'success',
                'action': action,
                'wait_days': wait_days,
                'q_value': q_value,
                'total_days': days_since + wait_days
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'status': 'error', 'error': str(e)}
        finally:
            db.close()
    
    def update_from_outcome(
        self,
        application_id: int,
        action_taken: str,
        got_response: bool,
        got_interview: bool = False
    ):
        """Update Q-Learning from real outcome"""
        
        db = SessionLocal()
        try:
            app = db.query(Application).filter(Application.id == application_id).first()
            
            if not app:
                return
            
            days_since = (datetime.utcnow() - app.applied_date).days
            wait_days = int(action_taken.split('_')[1].replace('d', ''))
            
            # Calculate reward
            reward = 0
            if got_interview:
                reward += 50
            elif got_response:
                reward += 20
            
            reward -= (2 * wait_days)  # Penalty for waiting
            
            # Update Q-Learning
            self.ql_agent.update(
                state=(days_since, app.company_type or 'midsize', app.has_connection or False),
                action=action_taken,
                reward=reward,
                next_state=None
            )
            
            self.save_to_database()
            print(f"   📈 Updated Q-table (reward={reward})")
            
        finally:
            db.close()


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Scheduler Agent")
    print("=" * 70)
    
    agent = SchedulerAgent()
    
    # Simulate: Get recommendation for application ID 1
    rec = agent.get_recommendation(application_id=1)
    
    if rec['status'] == 'success':
        print(f"\n✅ Recommendation: Wait {rec['wait_days']} days")
    
    print("\n✅ Scheduler Agent works!")
