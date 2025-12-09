"""
Tracker Agent: Logs applications and extracts data
"""
import sys
sys.path.append('src')

from tools.job_parser import JobDataExtractor
from tools.contact_finder import ContactFinder
from database import SessionLocal, Application, Contact
from datetime import datetime
from typing import Dict, List, Optional


class TrackerAgent:
    """Agent responsible for tracking new job applications"""
    
    def __init__(self):
        self.job_parser = JobDataExtractor()
        self.contact_finder = ContactFinder(use_real_apis=False)
        print("✅ Tracker Agent initialized")
    
    def track_application(
        self,
        company: str,
        role: str,
        description: str,
        job_url: Optional[str] = None,
        posted_date: Optional[datetime] = None,
        closing_date: Optional[datetime] = None,
        applied_date: Optional[datetime] = None
    ) -> Dict:
        """Track a new job application"""
        
        print(f"\n📝 Tracking: {role} at {company}")
        
        # Extract job data
        print("   🔍 Extracting job data...")
        job_data = self.job_parser.extract_from_manual_input(
            company=company,
            role=role,
            description=description,
            job_url=job_url,
            posted_date=posted_date,
            closing_date=closing_date
        )
        
        print(f"   ✅ Type: {job_data['company_type']}, Urgency: {job_data['urgency_score']:.0f}/100")
        
        # Find contacts
        print("   👥 Finding contacts...")
        contacts = self.contact_finder.find_contacts(company, role, max_results=3)
        print(f"   ✅ Found {len(contacts)} contacts")
        
        # Save to database
        print("   💾 Saving to database...")
        db = SessionLocal()
        
        try:
            application = Application(
                company=company,
                role=role,
                description=description,
                job_url=job_url,
                applied_date=applied_date or datetime.utcnow(),
                company_type=job_data['company_type'],
                seniority=job_data['seniority'],
                company_culture=job_data['company_culture'],
                has_connection=False,  # Default, can be updated
                status='applied'
            )
            
            db.add(application)
            db.commit()
            db.refresh(application)
            
            app_id = application.id
            
            # Save top 3 contacts
            for contact in contacts[:3]:
                contact_record = Contact(
                    application_id=app_id,
                    name=contact['name'],
                    email=contact['email'],
                    title=contact['title'],
                    linkedin_url=contact.get('linkedin', ''),
                    relevance_score=contact['relevance_score'],
                    connection_strength=contact['connection_strength']
                )
                db.add(contact_record)
            
            db.commit()
            print(f"   ✅ Saved (ID: {app_id})")
            
            return {
                'status': 'success',
                'application_id': app_id,
                'job_data': job_data,
                'contacts': contacts[:3]
            }
            
        except Exception as e:
            db.rollback()
            print(f"   ❌ Error: {e}")
            return {'status': 'error', 'error': str(e)}
        finally:
            db.close()


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Tracker Agent")
    print("=" * 70)
    
    agent = TrackerAgent()
    
    result = agent.track_application(
        company="Snowflake",
        role="Data Analyst",
        description="Join our analytics team working with SQL and Python.",
        posted_date=datetime(2024, 12, 1),
        applied_date=datetime(2024, 12, 7)
    )
    
    print(f"\n{'='*70}")
    print(f"Result: {result['status']}")
    if result['status'] == 'success':
        print(f"Application ID: {result['application_id']}")
        print(f"Urgency: {result['job_data']['urgency_score']:.0f}/100")
    print("✅ Tracker Agent works!")
