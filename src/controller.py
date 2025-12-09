"""
Multi-Agent Controller
Orchestrates Tracker, Scheduler, and Message agents
"""
import sys
sys.path.append('src')

from agents.tracker_agent import TrackerAgent
from agents.scheduler_agent import SchedulerAgent
from agents.message_agent import MessageAgent
from database import SessionLocal, Application
from datetime import datetime
from typing import Dict, Optional, List


class PostApplyController:
    """
    Main controller for PostApply RL System
    
    Orchestrates all agents and provides unified interface
    """
    
    def __init__(self):
        """Initialize all agents"""
        print("🚀 Initializing PostApply Controller...")
        
        self.tracker = TrackerAgent()
        self.scheduler = SchedulerAgent()
        self.message_agent = MessageAgent()
        
        print("✅ Controller ready!\n")
    
    def add_application(
        self,
        company: str,
        role: str,
        description: str,
        job_url: Optional[str] = None,
        posted_date: Optional[datetime] = None,
        closing_date: Optional[datetime] = None,
        applied_date: Optional[datetime] = None
    ) -> Dict:
        """
        Add new job application
        
        Coordinates: Tracker Agent
        
        Returns:
            Dict with application details and recommendations
        """
        print(f"📝 Adding application: {role} at {company}")
        print("="*70)
        
        # Track application
        result = self.tracker.track_application(
            company=company,
            role=role,
            description=description,
            job_url=job_url,
            posted_date=posted_date,
            closing_date=closing_date,
            applied_date=applied_date
        )
        
        if result['status'] == 'success':
            app_id = result['application_id']
            
            print(f"\n✅ Application tracked successfully!")
            print(f"   Application ID: {app_id}")
            
            return {
                'status': 'success',
                'application_id': app_id,
                'data': result
            }
        else:
            return result
    
    def get_recommendations(self, application_id: int) -> Dict:
        """
        Get all recommendations for an application
        
        Coordinates: Scheduler + Message Agent
        
        Returns:
            Dict with timing and style recommendations
        """
        print(f"\n🎯 Getting recommendations for application {application_id}")
        print("="*70)
        
        # Get timing recommendation
        timing = self.scheduler.get_recommendation(application_id)
        
        # Get style recommendation
        style = self.message_agent.get_style_recommendation(application_id)
        
        return {
            'status': 'success',
            'timing': timing,
            'style': style
        }
    
    def score_message(
        self,
        application_id: int,
        message: str,
        subject: Optional[str] = None
    ) -> Dict:
        """
        Score a draft message
        
        Coordinates: Message Agent
        """
        return self.message_agent.score_message(
            application_id=application_id,
            message_text=message,
            subject=subject
        )
    
    def record_outcome(
        self,
        application_id: int,
        action_taken: str,
        message_style: str,
        got_response: bool,
        got_interview: bool = False
    ):
        """
        Record outcome and update RL agents
        
        Coordinates: Scheduler + Message Agent
        
        Args:
            application_id: Application ID
            action_taken: Timing action used (e.g., 'wait_5d')
            message_style: Style used ('formal', 'casual', 'connection_focused')
            got_response: Whether you got a response
            got_interview: Whether you got an interview
        """
        print(f"\n📈 Recording outcome for application {application_id}")
        print("="*70)
        
        # Update Scheduler (Q-Learning)
        self.scheduler.update_from_outcome(
            application_id=application_id,
            action_taken=action_taken,
            got_response=got_response,
            got_interview=got_interview
        )
        
        # Update Message Agent (Thompson Sampling)
        self.message_agent.update_from_outcome(
            application_id=application_id,
            style_used=message_style,
            got_response=got_response
        )
        
        print("✅ RL agents updated with outcome!")
    
    def get_all_applications(self) -> List[Dict]:
        """Get all applications from database"""
        db = SessionLocal()
        try:
            apps = db.query(Application).order_by(Application.applied_date.desc()).all()
            
            results = []
            for app in apps:
                days_since = (datetime.utcnow() - app.applied_date).days
                
                results.append({
                    'id': app.id,
                    'company': app.company,
                    'role': app.role,
                    'status': app.status,
                    'days_since_applied': days_since,
                    'company_type': app.company_type,
                    'seniority': app.seniority
                })
            
            return results
        finally:
            db.close()
    
    def get_rl_statistics(self) -> Dict:
        """Get statistics from RL agents"""
        
        # Q-Learning stats
        ql_stats = self.scheduler.ql_agent.get_q_table_summary()
        
        # Thompson Sampling stats
        ts_stats = self.message_agent.ts_agent.get_statistics()
        
        return {
            'q_learning': ql_stats,
            'thompson_sampling': ts_stats
        }


# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing PostApply Controller")
    print("=" * 70)
    print()
    
    # Initialize controller
    controller = PostApplyController()
    
    # Test 1: Add application
    print("\nTEST 1: Add Application")
    print("-" * 70)
    
    result = controller.add_application(
        company="Tesla",
        role="Data Analyst",
        description="Join our data team analyzing vehicle analytics and manufacturing data.",
        applied_date=datetime.now()
    )
    
    if result['status'] == 'success':
        app_id = result['application_id']
        print(f"✅ Added application ID: {app_id}")
        
        # Test 2: Get recommendations
        print("\n\nTEST 2: Get Recommendations")
        print("-" * 70)
        
        recs = controller.get_recommendations(app_id)
        
        if recs['timing']['status'] == 'success':
            print(f"⏰ Timing: Wait {recs['timing']['wait_days']} days")
        
        if recs['style']['status'] == 'success':
            print(f"💬 Style: {recs['style']['style']} ({recs['style']['confidence']:.1%})")
        
        # Test 3: Score a message
        print("\n\nTEST 3: Score Message")
        print("-" * 70)
        
        test_message = """Hi,

I recently applied for the Data Analyst position at Tesla. I'm excited about Tesla's mission and have 3 years of experience with Python and SQL.

Would you be open to a brief call?

Thanks,
Geetika"""
        
        score = controller.score_message(app_id, test_message)
        
        if score['status'] == 'success':
            print(f"📊 Score: {score['overall_score']}/100")
        
        # Test 4: Record outcome
        print("\n\nTEST 4: Record Outcome")
        print("-" * 70)
        
        controller.record_outcome(
            application_id=app_id,
            action_taken='wait_5d',
            message_style='casual',
            got_response=True,
            got_interview=False
        )
        
        # Test 5: Get all applications
        print("\n\nTEST 5: View All Applications")
        print("-" * 70)
        
        all_apps = controller.get_all_applications()
        print(f"Total applications: {len(all_apps)}")
        for app in all_apps[:3]:  # Show first 3
            print(f"  • {app['company']} - {app['role']} ({app['days_since_applied']} days ago)")
        
        # Test 6: RL Statistics
        print("\n\nTEST 6: RL Statistics")
        print("-" * 70)
        
        stats = controller.get_rl_statistics()
        print(f"Q-Learning states: {stats['q_learning']['total_states']}")
        print(f"Q-Learning updates: {stats['q_learning']['total_updates']}")
        print(f"Thompson Sampling contexts: {stats['thompson_sampling']['total_contexts']}")
        print(f"Thompson Sampling selections: {stats['thompson_sampling']['total_selections']}")
    
    print("\n" + "=" * 70)
    print("✅ Controller Test Complete!")
    print("=" * 70)
    print("\nController provides unified API for entire system!")
