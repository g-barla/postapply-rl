"""
Custom Tool #1: Job Data Extractor (IMPROVED)
Extracts and scores job posting data using REAL dates
"""
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
import re


class JobDataExtractor:
    """Extract and score job posting information"""
    
    def __init__(self):
        self.company_types = {
            'startup': ['startup', 'early stage', 'series a', 'series b', 'seed'],
            'midsize': ['growth', 'scale-up', 'series c', 'series d'],
            'enterprise': ['fortune', 'enterprise', 'global', 'multinational', 'public company']
        }
        
        self.seniority_keywords = {
            'junior': ['junior', 'entry', 'associate', 'graduate', '0-2 years', 'intern'],
            'mid': ['mid-level', 'intermediate', '2-5 years', 'experienced'],
            'senior': ['senior', 'lead', 'principal', 'staff', '5+ years', 'expert']
        }
        
        self.culture_keywords = {
            'casual': ['casual', 'startup culture', 'flexible', 'remote-first', 'async'],
            'formal': ['corporate', 'professional', 'traditional', 'enterprise'],
            'mixed': ['hybrid', 'balanced', 'modern']
        }
    
    def extract_from_url(self, url: str) -> Dict:
        """
        Extract job data from URL
        Note: Real scraping would need BeautifulSoup/Selenium
        For now, this is a placeholder that returns manual input structure
        """
        print(f"⚠️  URL parsing not implemented yet. Please use manual entry.")
        return self._create_empty_template()
    
    def extract_from_manual_input(
        self,
        company: str,
        role: str,
        description: str,
        job_url: Optional[str] = None,
        posted_date: Optional[datetime] = None,  # NEW
        closing_date: Optional[datetime] = None,  # NEW
        internship_start: Optional[datetime] = None,  # NEW (for internships)
    ) -> Dict:
        """
        Extract job data from manual input with REAL date-based urgency
        """
        
        # Classify company type
        company_type = self._classify_company_type(description)
        
        # Classify seniority
        seniority = self._classify_seniority(role, description)
        
        # Classify culture
        culture = self._classify_culture(description)
        
        # Calculate urgency score using REAL dates
        urgency = self._calculate_urgency_from_dates(
            posted_date=posted_date,
            closing_date=closing_date,
            internship_start=internship_start,
            fallback_description=description  # Use keywords if no dates provided
        )
        
        return {
            'company': company,
            'role': role,
            'description': description,
            'job_url': job_url,
            'posted_date': posted_date.isoformat() if posted_date else None,
            'closing_date': closing_date.isoformat() if closing_date else None,
            'internship_start': internship_start.isoformat() if internship_start else None,
            'company_type': company_type,
            'seniority': seniority,
            'company_culture': culture,
            'urgency_score': urgency,
            'extracted_at': datetime.utcnow().isoformat()
        }
    
    def _classify_company_type(self, text: str) -> str:
        """Classify company as startup, midsize, or enterprise"""
        text_lower = text.lower()
        
        scores = {
            'startup': 0,
            'midsize': 0,
            'enterprise': 0
        }
        
        for company_type, keywords in self.company_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[company_type] += 1
        
        max_type = max(scores, key=scores.get)
        return max_type if scores[max_type] > 0 else 'midsize'
    
    def _classify_seniority(self, role: str, description: str) -> str:
        """Classify job seniority level"""
        text = (role + " " + description).lower()
        
        scores = {
            'junior': 0,
            'mid': 0,
            'senior': 0
        }
        
        for level, keywords in self.seniority_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    scores[level] += 1
        
        max_level = max(scores, key=scores.get)
        return max_level if scores[max_level] > 0 else 'mid'
    
    def _classify_culture(self, text: str) -> str:
        """Classify company culture"""
        text_lower = text.lower()
        
        scores = {
            'casual': 0,
            'formal': 0,
            'mixed': 0
        }
        
        for culture, keywords in self.culture_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[culture] += 1
        
        max_culture = max(scores, key=scores.get)
        return max_culture if scores[max_culture] > 0 else 'mixed'
    
    def _calculate_urgency_from_dates(
        self,
        posted_date: Optional[datetime],
        closing_date: Optional[datetime],
        internship_start: Optional[datetime],
        fallback_description: str
    ) -> float:
        """
        Calculate urgency score (0-100) based on REAL dates
        
        Priority logic:
        1. If closing_date exists → use days until deadline
        2. If posted_date exists → use days since posting
        3. If internship_start exists → use days until start
        4. Fallback → keyword-based (old method)
        """
        now = datetime.utcnow()
        
        # PRIORITY 1: Application deadline (most important!)
        if closing_date:
            days_until_close = (closing_date - now).days
            
            if days_until_close < 0:
                return 0.0  # Already closed!
            elif days_until_close <= 3:
                return 95.0  # URGENT - closes in 3 days
            elif days_until_close <= 7:
                return 85.0  # High urgency - 1 week
            elif days_until_close <= 14:
                return 70.0  # Medium-high - 2 weeks
            elif days_until_close <= 30:
                return 55.0  # Medium - 1 month
            else:
                return 40.0  # Low urgency - plenty of time
        
        # PRIORITY 2: Days since posted
        if posted_date:
            days_since_posted = (now - posted_date).days
            
            if days_since_posted <= 3:
                return 90.0  # Fresh posting! Apply now
            elif days_since_posted <= 7:
                return 75.0  # Posted this week
            elif days_since_posted <= 14:
                return 60.0  # Posted in last 2 weeks
            elif days_since_posted <= 30:
                return 45.0  # Posted this month
            else:
                return 25.0  # Old posting - may be filled
        
        # PRIORITY 3: Internship start date (for internships)
        if internship_start:
            days_until_start = (internship_start - now).days
            
            # Assume applications open 3-6 months before start
            if days_until_start <= 60:  # 2 months out
                return 90.0  # Very urgent!
            elif days_until_start <= 90:  # 3 months out
                return 75.0  # High urgency
            elif days_until_start <= 120:  # 4 months out
                return 60.0  # Good timing
            elif days_until_start <= 180:  # 6 months out
                return 45.0  # Early but okay
            else:
                return 30.0  # Too early
        
        # FALLBACK: Use keyword-based scoring (old method)
        return self._calculate_urgency_from_keywords(fallback_description)
    
    def _calculate_urgency_from_keywords(self, description: str) -> float:
        """
        Fallback method: keyword-based urgency (less reliable)
        Only used when no dates provided
        """
        urgency_indicators = {
            'high': ['urgent', 'immediate', 'asap', 'closing soon', 'apply now'],
            'medium': ['hiring', 'open position', 'actively recruiting'],
            'low': ['future', 'planning', 'potential']
        }
        
        text_lower = description.lower()
        score = 50.0  # Base score
        
        for keyword in urgency_indicators['high']:
            if keyword in text_lower:
                score += 10.0
        
        for keyword in urgency_indicators['medium']:
            if keyword in text_lower:
                score += 5.0
        
        for keyword in urgency_indicators['low']:
            if keyword in text_lower:
                score -= 10.0
        
        return max(0.0, min(100.0, score))
    
    def _create_empty_template(self) -> Dict:
        """Return empty template for manual filling"""
        return {
            'company': '',
            'role': '',
            'description': '',
            'job_url': '',
            'posted_date': None,
            'closing_date': None,
            'internship_start': None,
            'company_type': 'midsize',
            'seniority': 'mid',
            'company_culture': 'mixed',
            'urgency_score': 50.0,
            'extracted_at': datetime.utcnow().isoformat()
        }


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing Job Data Extractor with REAL dates...\n")
    
    extractor = JobDataExtractor()
    now = datetime.utcnow()
    
    # Test 1: Job posted 2 days ago, closes in 5 days (URGENT!)
    print("=" * 60)
    print("TEST 1: Fresh posting, closing soon")
    print("=" * 60)
    
    result1 = extractor.extract_from_manual_input(
        company="TechCorp",
        role="Data Analyst",
        description="Join our data team analyzing customer behavior.",
        job_url="https://linkedin.com/jobs/123",
        posted_date=now - timedelta(days=2),
        closing_date=now + timedelta(days=5)
    )
    
    print(f"Company: {result1['company']}")
    print(f"Posted: 2 days ago")
    print(f"Closes: In 5 days")
    print(f"Urgency Score: {result1['urgency_score']}/100 (Should be ~85-95)")
    print()
    
    # Test 2: Job posted 25 days ago, no deadline (OLD)
    print("=" * 60)
    print("TEST 2: Old posting, no deadline")
    print("=" * 60)
    
    result2 = extractor.extract_from_manual_input(
        company="BigCorp",
        role="Senior Data Analyst",
        description="Experienced analyst needed for enterprise data warehouse.",
        job_url="https://indeed.com/jobs/456",
        posted_date=now - timedelta(days=25)
    )
    
    print(f"Company: {result2['company']}")
    print(f"Posted: 25 days ago")
    print(f"Closes: No deadline")
    print(f"Urgency Score: {result2['urgency_score']}/100 (Should be ~45)")
    print()
    
    # Test 3: Summer 2025 internship (starting June 1, 2025)
    print("=" * 60)
    print("TEST 3: Summer internship application")
    print("=" * 60)
    
    summer_start = datetime(2025, 6, 1)
    days_until_summer = (summer_start - now).days
    
    result3 = extractor.extract_from_manual_input(
        company="StartupCo",
        role="Data Science Intern",
        description="Summer 2025 internship program for data science students.",
        job_url="https://startupco.com/careers",
        posted_date=now - timedelta(days=5),
        internship_start=summer_start
    )
    
    print(f"Company: {result3['company']}")
    print(f"Internship starts: June 1, 2025 ({days_until_summer} days)")
    print(f"Urgency Score: {result3['urgency_score']}/100")
    print()
    
    # Test 4: No dates provided - fallback to keywords
    print("=" * 60)
    print("TEST 4: No dates - keyword fallback")
    print("=" * 60)
    
    result4 = extractor.extract_from_manual_input(
        company="MysteryCompany",
        role="Data Analyst",
        description="Hiring urgently! Apply now for this open position."
    )
    
    print(f"Company: {result4['company']}")
    print(f"No dates provided - using keywords")
    print(f"Urgency Score: {result4['urgency_score']}/100 (Keyword-based)")
    
    print("\n✅ Job Data Extractor (with dates) is working!")
