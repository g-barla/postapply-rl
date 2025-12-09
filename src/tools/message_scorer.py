"""
Custom Tool #3: Message Quality Scorer
Uses OpenAI GPT-4 to score follow-up messages and provide feedback

HOW IT WORKS:
1. Takes a message + context (company, role, contact info)
2. Primary: Sends to GPT-4 for intelligent analysis
3. Fallback: Uses rule-based scoring if AI fails
4. Returns detailed scores + feedback + suggestions
5. Thompson Sampling agent uses scores to learn optimal message styles
"""

from typing import Dict, Optional
import os
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()


class MessageQualityScorer:
    """Score job application follow-up messages using AI"""
    
    def __init__(self, use_ai: bool = True):
        """
        Initialize message scorer
        
        Args:
            use_ai: If True, use OpenAI API. If False, use rule-based scoring
        """
        self.use_ai = use_ai
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if self.use_ai and self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
            print("✅ OpenAI API loaded")
        else:
            print("⚠️  Using rule-based scoring (no OpenAI API)")
            self.use_ai = False
    
    def score_message(
        self,
        message: str,
        subject: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Score a follow-up message
        
        Args:
            message: The message body text
            subject: Optional email subject line
            context: Optional context dict with:
                - company: Company name
                - role: Job role
                - contact_name: Person's name
                - contact_title: Person's job title
                - days_since_applied: Days since application
        
        Returns:
            Dict with:
            {
                'overall_score': 85.0,
                'personalization_score': 90.0,
                'clarity_score': 85.0,
                'professionalism_score': 80.0,
                'length_score': 85.0,
                'feedback': ['Point 1', 'Point 2', ...],
                'suggested_improvements': 'Specific suggestion...',
                'word_count': 175,
                'scoring_method': 'ai' or 'rule-based'
            }
        
        Example:
            scorer = MessageQualityScorer()
            
            message = "Hi Sarah, I recently applied for..."
            
            context = {
                'company': 'Snowflake',
                'role': 'Data Analyst',
                'contact_name': 'Sarah',
                'contact_title': 'Recruiting Manager'
            }
            
            result = scorer.score_message(message, context=context)
            print(f"Score: {result['overall_score']}/100")
        """
        
        if self.use_ai:
            return self._score_with_ai(message, subject, context)
        else:
            return self._score_with_rules(message, subject, context)
    
    def _score_with_ai(
        self,
        message: str,
        subject: Optional[str],
        context: Optional[Dict]
    ) -> Dict:
        """
        Score message using OpenAI GPT-4
        
        Process:
        1. Build context string from provided info
        2. Create detailed prompt for GPT-4
        3. Call OpenAI API
        4. Parse JSON response
        5. Return structured scores
        """
        
        # Build context string for GPT-4
        context_str = ""
        if context:
            context_str = f"""
Context about the application:
- Company: {context.get('company', 'Unknown')}
- Role: {context.get('role', 'Unknown')}
- Contact: {context.get('contact_name', 'Unknown')}
- Contact Title: {context.get('contact_title', 'Unknown')}
- Days since applied: {context.get('days_since_applied', 'Unknown')}
"""
        
        # Create prompt for GPT-4 (expert career coach persona)
        prompt = f"""You are an expert career coach analyzing a job application follow-up message.

{context_str}

Subject Line: {subject or 'N/A'}

Message Body:
{message}

Please analyze this follow-up message and provide:

1. **Personalization Score (0-100)**: Does it mention specific details about the company, role, or connection? Is it tailored or generic?

2. **Clarity Score (0-100)**: Is the ask clear? Is it easy to understand? Well-structured?

3. **Professionalism Score (0-100)**: Appropriate tone? No typos? Respectful?

4. **Length Score (0-100)**: Right length (150-300 words ideal)? Not too short/long?

5. **Overall Score (0-100)**: Weighted average (Personalization 35%, Clarity 30%, Professional 20%, Length 15%)

6. **Feedback**: 3-5 bullet points of specific feedback (what's good, what needs work)

7. **Suggested Improvements**: One specific actionable improvement to make.

Respond in this EXACT JSON format (no markdown, no extra text):
{{
  "personalization_score": 85,
  "clarity_score": 90,
  "professionalism_score": 88,
  "length_score": 80,
  "overall_score": 86,
  "feedback": [
    "Strong opening that mentions specific project",
    "Clear ask for informational interview",
    "Could add more about why you're interested"
  ],
  "suggested_improvements": "Add 1-2 sentences about what specifically excites you about their work on X project"
}}"""

        try:
            print("🤖 Analyzing message with GPT-4...")
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper, faster, sufficient for this task
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert career coach who analyzes job application messages. Always respond with valid JSON only, no markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower = more consistent scoring
                max_tokens=1000
            )
            
            # Extract response text
            ai_response = response.choices[0].message.content.strip()
            
            # Clean response (remove markdown code blocks if present)
            ai_response = ai_response.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            result = json.loads(ai_response)
            
            # Add metadata
            result['word_count'] = len(message.split())
            result['scoring_method'] = 'ai'
            
            print(f"✅ AI scoring complete - Overall: {result['overall_score']}/100")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse AI response as JSON: {e}")
            print(f"Raw response: {ai_response[:200]}")
            print("Falling back to rule-based scoring...")
            return self._score_with_rules(message, subject, context)
            
        except Exception as e:
            print(f"❌ OpenAI API error: {str(e)}")
            print("Falling back to rule-based scoring...")
            return self._score_with_rules(message, subject, context)
    
    def _score_with_rules(
        self,
        message: str,
        subject: Optional[str],
        context: Optional[Dict]
    ) -> Dict:
        """
        Fallback: Rule-based scoring using simple keyword matching
        
        This is less sophisticated than AI but works without API calls
        Good for testing and when OpenAI API is unavailable
        """
        
        word_count = len(message.split())
        char_count = len(message)
        
        # 1. PERSONALIZATION SCORE (0-100)
        personalization_score = 50  # Start with base score
        
        if context:
            # +20 if mentions company name
            if context.get('company', '').lower() in message.lower():
                personalization_score += 20
            
            # +15 if mentions role
            if context.get('role', '').lower() in message.lower():
                personalization_score += 15
            
            # +15 if mentions contact name
            if context.get('contact_name', '').lower() in message.lower():
                personalization_score += 15
        
        # Check for research indicators
        research_keywords = ['impressed by', 'excited about', 'following', 'admire', 'noticed']
        if any(keyword in message.lower() for keyword in research_keywords):
            personalization_score = min(100, personalization_score + 10)
        
        personalization_score = min(100, personalization_score)
        
        # 2. CLARITY SCORE (0-100)
        clarity_score = 60  # Base score
        
        # +20 if has clear ask
        clear_asks = ['would love to', 'could we', 'would you be open to', 'hoping to', 'wondering if']
        if any(ask in message.lower() for ask in clear_asks):
            clarity_score += 20
        
        # +10 if has greeting
        if any(greeting in message.lower() for greeting in ['hi', 'hello', 'dear', 'good morning']):
            clarity_score += 10
        
        # +10 if has proper closing
        if any(closing in message.lower() for closing in ['best', 'thanks', 'sincerely', 'regards', 'thank you']):
            clarity_score += 10
        
        clarity_score = min(100, clarity_score)
        
        # 3. PROFESSIONALISM SCORE (0-100)
        professionalism_score = 80  # Start high, deduct for issues
        
        # Deduct for unprofessional elements
        if '!!' in message or '???' in message:
            professionalism_score -= 10  # Too many exclamation marks
        
        if any(word in message.lower() for word in ['hey', 'yo', 'sup', 'gonna', 'wanna', 'kinda']):
            professionalism_score -= 15  # Too casual
        
        if message.isupper():
            professionalism_score -= 30  # All caps
        
        if not any(c.isupper() for c in message):
            professionalism_score -= 20  # No capital letters at all
        
        professionalism_score = max(0, min(100, professionalism_score))
        
        # 4. LENGTH SCORE (0-100)
        # Ideal: 100-250 words (roughly 500-1500 characters)
        if word_count < 50:
            length_score = 40  # Way too short
        elif word_count < 100:
            length_score = 70  # A bit short
        elif 100 <= word_count <= 250:
            length_score = 100  # Perfect!
        elif 250 < word_count <= 400:
            length_score = 80  # Getting long
        else:
            length_score = 50  # Too long, people won't read
        
        # 5. OVERALL SCORE (weighted average)
        overall_score = (
            personalization_score * 0.35 +  # Most important
            clarity_score * 0.30 +
            professionalism_score * 0.20 +
            length_score * 0.15
        )
        
        # 6. GENERATE FEEDBACK
        feedback = []
        
        if personalization_score >= 80:
            feedback.append("✅ Great personalization - mentions specific details")
        elif personalization_score < 50:
            feedback.append("⚠️ Message feels generic - add company/role specifics")
        else:
            feedback.append("⚠️ Could be more personalized")
        
        if clarity_score >= 80:
            feedback.append("✅ Clear ask and well-structured")
        else:
            feedback.append("⚠️ Make your ask more explicit and clear")
        
        if professionalism_score >= 80:
            feedback.append("✅ Professional and respectful tone")
        else:
            feedback.append("⚠️ Tone could be more professional")
        
        if length_score >= 80:
            feedback.append("✅ Good length - concise and complete")
        elif word_count < 100:
            feedback.append("⚠️ Too short - add more context")
        else:
            feedback.append("⚠️ Too long - be more concise")
        
        # 7. SUGGESTED IMPROVEMENT (most critical issue)
        if personalization_score < 70:
            improvement = "Add 1-2 specific details about why you're interested in this company/role (mention a project, value, or achievement)"
        elif clarity_score < 70:
            improvement = "Make your ask more explicit: 'Would you be open to a 15-minute call next week?'"
        elif word_count < 100:
            improvement = "Expand to 150-200 words with more context about your background and interest"
        elif word_count > 300:
            improvement = "Cut to 200-250 words - focus on 2-3 key points only"
        else:
            improvement = "Message looks solid! Consider adding one specific example of your relevant experience"
        
        return {
            'overall_score': round(overall_score, 1),
            'personalization_score': round(personalization_score, 1),
            'clarity_score': round(clarity_score, 1),
            'professionalism_score': round(professionalism_score, 1),
            'length_score': round(length_score, 1),
            'feedback': feedback,
            'suggested_improvements': improvement,
            'word_count': word_count,
            'scoring_method': 'rule-based'
        }


# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Message Quality Scorer")
    print("=" * 70)
    print()
    
    # Initialize scorer (will try to use OpenAI API)
    scorer = MessageQualityScorer(use_ai=True)
    
    # TEST 1: Well-crafted personalized message
    print("TEST 1: Well-crafted personalized message")
    print("=" * 70)
    
    message1 = """Hi Sarah,

I recently applied for the Data Analyst position at Snowflake and wanted to reach out directly. I'm particularly excited about Snowflake's work on real-time data sharing and have been following your team's innovations in this space.

With my background in SQL, Python, and Tableau, I believe I could contribute meaningfully to your analytics team. Would you be open to a brief 15-minute call to discuss the role and team?

Thank you for your time!

Best regards,
Geetika"""

    context1 = {
        'company': 'Snowflake',
        'role': 'Data Analyst',
        'contact_name': 'Sarah',
        'contact_title': 'Recruiting Manager',
        'days_since_applied': 3
    }
    
    result1 = scorer.score_message(
        message=message1,
        subject="Following up on Data Analyst application",
        context=context1
    )
    
    print(f"\n📊 SCORES:")
    print(f"   Overall: {result1['overall_score']}/100")
    print(f"   Personalization: {result1['personalization_score']}/100")
    print(f"   Clarity: {result1['clarity_score']}/100")
    print(f"   Professionalism: {result1['professionalism_score']}/100")
    print(f"   Length: {result1['length_score']}/100")
    print(f"   Word Count: {result1['word_count']}")
    print(f"   Method: {result1['scoring_method']}")
    
    print(f"\n💡 FEEDBACK:")
    for item in result1['feedback']:
        print(f"   • {item}")
    
    print(f"\n🎯 SUGGESTED IMPROVEMENT:")
    print(f"   {result1['suggested_improvements']}")
    
    # TEST 2: Generic poor message
    print("\n\n" + "=" * 70)
    print("TEST 2: Generic message (should score lower)")
    print("=" * 70)
    
    message2 = """Hey,

I applied to your company. Can we talk?

Thanks"""

    result2 = scorer.score_message(
        message=message2,
        subject="Job",
        context=context1
    )
    
    print(f"\n📊 SCORES:")
    print(f"   Overall: {result2['overall_score']}/100")
    print(f"   Personalization: {result2['personalization_score']}/100")
    print(f"   Clarity: {result2['clarity_score']}/100")
    print(f"   Professionalism: {result2['professionalism_score']}/100")
    print(f"   Length: {result2['length_score']}/100")
    print(f"   Word Count: {result2['word_count']}")
    print(f"   Method: {result2['scoring_method']}")
    
    print(f"\n💡 FEEDBACK:")
    for item in result2['feedback']:
        print(f"   • {item}")
    
    print(f"\n🎯 SUGGESTED IMPROVEMENT:")
    print(f"   {result2['suggested_improvements']}")
    
    print("\n" + "=" * 70)
    print("✅ Message Scorer Test Complete!")
    print("=" * 70)
