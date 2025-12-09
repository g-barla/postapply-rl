"""
Test full workflow with YOUR job description
"""
import sys
sys.path.append('src')

from agents.tracker_agent import TrackerAgent
from agents.scheduler_agent import SchedulerAgent
from agents.message_agent import MessageAgent
from datetime import datetime, timedelta

print("=" * 70)
print("FULL WORKFLOW TEST WITH REAL JD")
print("=" * 70)

# Initialize agents
tracker = TrackerAgent()
scheduler = SchedulerAgent()
message_agent = MessageAgent()

# YOUR JOB DESCRIPTION HERE (paste any real JD!)
company = "Microsoft"
role = "Business Intelligence Analyst"
description = """
We are seeking a talented Business Intelligence Analyst to join our data team. 
You will work with SQL, Power BI, and Azure to analyze business metrics and 
create dashboards for stakeholders. 

Requirements:
- 2+ years experience with SQL and data visualization
- Strong analytical and communication skills
- Experience with Power BI or Tableau
- Bachelor's degree in relevant field

We offer competitive compensation and great benefits!
"""

# STEP 1: TRACKER AGENT
print("\n" + "="*70)
print("STEP 1: Tracking Application")
print("="*70)

result = tracker.track_application(
    company=company,
    role=role,
    description=description,
    job_url="https://microsoft.com/careers/123",
    posted_date=datetime.now() - timedelta(days=5),  # Posted 5 days ago
    applied_date=datetime.now() - timedelta(days=2)  # Applied 2 days ago
)

if result['status'] == 'success':
    app_id = result['application_id']
    print(f"\n✅ APPLICATION TRACKED!")
    print(f"   ID: {app_id}")
    print(f"   Company Type: {result['job_data']['company_type']}")
    print(f"   Seniority: {result['job_data']['seniority']}")
    print(f"   Urgency: {result['job_data']['urgency_score']}/100")
    print(f"   Top Contact: {result['contacts'][0]['name']} - {result['contacts'][0]['title']}")
    
    # STEP 2: SCHEDULER AGENT
    print("\n" + "="*70)
    print("STEP 2: Getting Follow-Up Timing Recommendation")
    print("="*70)
    
    timing = scheduler.get_recommendation(app_id)
    
    if timing['status'] == 'success':
        print(f"\n✅ TIMING RECOMMENDATION:")
        print(f"   Action: {timing['action']}")
        print(f"   Wait: {timing['wait_days']} more days")
        print(f"   Total days since applied: {timing['total_days']}")
        print(f"   Confidence (Q-value): {timing['q_value']:.2f}")
    
    # STEP 3: MESSAGE AGENT
    print("\n" + "="*70)
    print("STEP 3: Getting Message Style Recommendation")
    print("="*70)
    
    style = message_agent.get_style_recommendation(app_id)
    
    if style['status'] == 'success':
        print(f"\n✅ STYLE RECOMMENDATION:")
        print(f"   Recommended: {style['style']}")
        print(f"   Confidence: {style['confidence']:.1%}")
        print(f"   All probabilities:")
        for arm, prob in style['all_probabilities'].items():
            print(f"      {arm}: {prob:.1%}")
    
    # STEP 4: SCORE A DRAFT MESSAGE
    print("\n" + "="*70)
    print("STEP 4: Scoring Draft Message")
    print("="*70)
    
    draft_message = f"""Hi {result['contacts'][0]['name'].split()[0]},

I recently applied for the {role} position at {company} and wanted to reach out directly. 
I'm particularly interested in {company}'s work on cloud analytics and have 3 years of 
experience with SQL, Power BI, and Azure.

Would you be open to a brief 15-minute call to discuss the role and team?

Thank you for your time!

Best regards,
Geetika"""
    
    score = message_agent.score_message(
        app_id,
        draft_message,
        subject=f"Following up on {role} application"
    )
    
    if score['status'] == 'success':
        print(f"\n✅ MESSAGE SCORE:")
        print(f"   Overall: {score['overall_score']}/100")
        print(f"   Personalization: {score['personalization_score']}/100")
        print(f"   Clarity: {score['clarity_score']}/100")
        print(f"   Professionalism: {score['professionalism_score']}/100")
        print(f"   Length: {score['length_score']}/100")
        print(f"\n   Feedback:")
        for item in score['feedback']:
            print(f"      • {item}")
        print(f"\n   💡 Suggestion: {score['suggested_improvements']}")

print("\n" + "="*70)
print("✅ FULL WORKFLOW COMPLETE!")
print("="*70)
print("\nYou can now:")
print("1. Revise your message based on feedback")
print("2. Wait for the recommended time to follow up")
print("3. Use the recommended message style")
