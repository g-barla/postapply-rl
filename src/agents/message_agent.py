"""
Message Agent: Selects message style using Thompson Sampling
"""
import sys
sys.path.append('src')

from rl_algorithms.thompson_sampling import ThompsonSamplingMessenger
from tools.message_scorer import MessageQualityScorer
from database import SessionLocal, Application, Contact, RLState, Message
from datetime import datetime
from typing import Dict, Optional


class MessageAgent:
    """Agent for message style selection using Thompson Sampling"""
    
    def __init__(self):
        self.ts_agent = ThompsonSamplingMessenger()
        self.scorer = MessageQualityScorer(use_ai=False)  # Rule-based for speed
        self._load_from_database()
        print("✅ Message Agent initialized")
    
    def _load_from_database(self):
        """Load Thompson Sampling distributions from database"""
        db = SessionLocal()
        try:
            rl_state = db.query(RLState).filter(
                RLState.agent_type == 'thompson_sampling'
            ).first()
            
            if rl_state and rl_state.thompson_params:
                self.ts_agent.load_from_dict({
                    'distributions': rl_state.thompson_params,
                    'total_selections': rl_state.total_updates
                })
                print(f"   📊 Loaded distributions for {len(rl_state.thompson_params)} contexts")
        except Exception as e:
            print(f"   ⚠️  No existing distributions found")
        finally:
            db.close()
    
    def save_to_database(self):
        """Save Thompson Sampling distributions to database"""
        db = SessionLocal()
        try:
            ts_data = self.ts_agent.to_dict()
            
            rl_state = db.query(RLState).filter(
                RLState.agent_type == 'thompson_sampling'
            ).first()
            
            if rl_state:
                rl_state.thompson_params = ts_data['distributions']
                rl_state.total_updates = ts_data['total_selections']
                rl_state.last_updated = datetime.utcnow()
            else:
                rl_state = RLState(
                    agent_type='thompson_sampling',
                    thompson_params=ts_data['distributions'],
                    total_updates=ts_data['total_selections']
                )
                db.add(rl_state)
            
            db.commit()
            print("   💾 Distributions saved")
        except Exception as e:
            db.rollback()
            print(f"   ❌ Save error: {e}")
        finally:
            db.close()
    
    def get_style_recommendation(self, application_id: int) -> Dict:
        """Get message style recommendation"""
        
        print(f"\n💬 Style recommendation for app {application_id}")
        
        db = SessionLocal()
        try:
            app = db.query(Application).filter(Application.id == application_id).first()
            
            if not app:
                return {'status': 'error', 'error': 'Application not found'}
            
            # Get top contact
            contact = db.query(Contact).filter(
                Contact.application_id == application_id
            ).order_by(Contact.relevance_score.desc()).first()
            
            if not contact:
                contact_title = 'manager'
            else:
                contact_title = contact.title
            
            print(f"   👤 Contact: {contact_title}")
            print(f"   🏢 Culture: {app.company_culture}")
            
            # Get style recommendation
            style = self.ts_agent.select_arm(
                contact_title=contact_title,
                company_culture=app.company_culture or 'mixed',
                has_connection=app.has_connection or False
            )
            
            # Get probabilities
            probs = self.ts_agent.get_arm_probabilities(
                contact_title=contact_title,
                company_culture=app.company_culture or 'mixed',
                has_connection=app.has_connection or False
            )
            
            print(f"   ✅ Recommended style: {style}")
            print(f"   📊 Confidence: {probs[style]:.1%}")
            
            return {
                'status': 'success',
                'style': style,
                'confidence': probs[style],
                'all_probabilities': probs
            }
            
        finally:
            db.close()
    
    def score_message(
        self,
        application_id: int,
        message_text: str,
        subject: Optional[str] = None
    ) -> Dict:
        """Score a draft message"""
        
        print(f"\n📝 Scoring message for app {application_id}")
        
        db = SessionLocal()
        try:
            app = db.query(Application).filter(Application.id == application_id).first()
            
            if not app:
                return {'status': 'error', 'error': 'Application not found'}
            
            context = {
                'company': app.company,
                'role': app.role,
                'contact_name': '',
                'contact_title': ''
            }
            
            # Get top contact for context
            contact = db.query(Contact).filter(
                Contact.application_id == application_id
            ).order_by(Contact.relevance_score.desc()).first()
            
            if contact:
                context['contact_name'] = contact.name
                context['contact_title'] = contact.title
            
            # Score message
            result = self.scorer.score_message(
                message=message_text,
                subject=subject,
                context=context
            )
            
            print(f"   ✅ Overall score: {result['overall_score']}/100")
            
            return {
                'status': 'success',
                **result
            }
            
        finally:
            db.close()
    
    def update_from_outcome(
        self,
        application_id: int,
        style_used: str,
        got_response: bool
    ):
        """Update Thompson Sampling from outcome"""
        
        db = SessionLocal()
        try:
            app = db.query(Application).filter(Application.id == application_id).first()
            
            if not app:
                return
            
            contact = db.query(Contact).filter(
                Contact.application_id == application_id
            ).order_by(Contact.relevance_score.desc()).first()
            
            contact_title = contact.title if contact else 'manager'
            
            # Update Thompson Sampling
            self.ts_agent.update(
                contact_title=contact_title,
                company_culture=app.company_culture or 'mixed',
                has_connection=app.has_connection or False,
                arm=style_used,
                got_response=got_response
            )
            
            self.save_to_database()
            print(f"   📈 Updated distributions (response={got_response})")
            
        finally:
            db.close()


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Message Agent")
    print("=" * 70)
    
    agent = MessageAgent()
    
    # Get style recommendation
    rec = agent.get_style_recommendation(application_id=1)
    
    if rec['status'] == 'success':
        print(f"\n✅ Recommended: {rec['style']} ({rec['confidence']:.1%})")
    
    print("\n✅ Message Agent works!")
