"""
Custom Tool #2: Contact Finder (BULLETPROOF VERSION)
Multi-layer contact finding with detailed debugging
"""

from typing import List, Dict, Optional
from datetime import datetime
import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()


class ContactFinder:
    """Find and rank relevant contacts with bulletproof fallback"""
    
    def __init__(self, use_real_apis: bool = True, debug: bool = True):
        self.use_real_apis = use_real_apis
        self.debug = debug  # Show detailed logs
        
        # API Keys
        self.hunter_api_key = os.getenv('HUNTER_API_KEY')
        self.apollo_api_key = os.getenv('APOLLO_API_KEY')
        
        # API endpoints
        self.hunter_domain_search = "https://api.hunter.io/v2/domain-search"
        self.apollo_people_search = "https://api.apollo.io/v1/mixed_people/search"
        
        # Scoring weights
        self.title_weights = {
            'hiring manager': 10,
            'recruiter': 9,
            'talent acquisition': 9,
            'talent': 8,
            'head of': 8,
            'director': 7,
            'vp': 6,
            'vice president': 6,
            'manager': 5,
            'lead': 4,
            'senior': 3
        }
        
        if self.debug:
            print(f"🔑 Hunter.io API: {'✅ Loaded' if self.hunter_api_key else '❌ Missing'}")
            print(f"🔑 Apollo.io API: {'✅ Loaded' if self.apollo_api_key else '❌ Missing'}")
    
    def find_contacts(
        self,
        company: str,
        role: str,
        max_results: int = 5,
        manual_contacts: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Find contacts with multi-layer fallback
        
        Args:
            company: Company name
            role: Job role
            max_results: Max contacts to return
            manual_contacts: Optional list of manually entered contacts
        
        Returns:
            List of scored and ranked contacts
        """
        
        all_contacts = []
        
        # LAYER 1: Try Hunter.io
        if self.use_real_apis and self.hunter_api_key:
            domain = self._get_company_domain(company)
            if self.debug:
                print(f"\n📍 LAYER 1: Hunter.io")
                print(f"   Domain: {domain}")
            
            hunter_contacts = self._find_contacts_hunter(domain, role, max_results)
            all_contacts.extend(hunter_contacts)
            
            if self.debug:
                print(f"   Result: {len(hunter_contacts)} contacts found")
        
        # LAYER 2: Try Apollo.io if we have <3 contacts
        if len(all_contacts) < 3 and self.apollo_api_key:
            if self.debug:
                print(f"\n📍 LAYER 2: Apollo.io (need more contacts)")
            
            apollo_contacts = self._find_contacts_apollo(company, role, max_results - len(all_contacts))
            all_contacts.extend(apollo_contacts)
            
            if self.debug:
                print(f"   Result: {len(apollo_contacts)} contacts found")
        
        # LAYER 3: Use manual contacts if provided
        if manual_contacts and len(all_contacts) < 3:
            if self.debug:
                print(f"\n📍 LAYER 3: Manual contacts")
                print(f"   Result: {len(manual_contacts)} contacts added")
            all_contacts.extend(manual_contacts)
        
        # LAYER 4: Fallback to realistic mock data
        if len(all_contacts) == 0:
            if self.debug:
                print(f"\n📍 LAYER 4: Mock data fallback")
            all_contacts = self._generate_mock_contacts(company, role, max_results)
        
        # Score and rank all contacts
        scored_contacts = self._score_and_rank(all_contacts, role)
        
        if self.debug:
            print(f"\n✅ FINAL: Returning {len(scored_contacts[:max_results])} contacts")
        
        return scored_contacts[:max_results]
    
    def _find_contacts_hunter(self, domain: str, role: str, max_results: int) -> List[Dict]:
        """Find contacts via Hunter.io with detailed error handling"""
        
        try:
            params = {
                'domain': domain,
                'api_key': self.hunter_api_key,
                'limit': min(max_results * 2, 10)
            }
            
            response = requests.get(self.hunter_domain_search, params=params, timeout=10)
            
            if self.debug:
                print(f"   API Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Debug: Show full response structure
                if self.debug:
                    print(f"   Response keys: {list(data.keys())}")
                    if 'data' in data:
                        print(f"   Data keys: {list(data['data'].keys())}")
                        if 'emails' in data['data']:
                            print(f"   Emails found: {len(data['data']['emails'])}")
                
                if 'data' not in data or 'emails' not in data['data']:
                    return []
                
                contacts = []
                for email_data in data['data']['emails']:
                    first = email_data.get('first_name', '')
                    last = email_data.get('last_name', '')
                    name = f"{first} {last}".strip()
                    
                    if not name:
                        continue
                    
                    contact = {
                        'name': name,
                        'email': email_data.get('value', ''),
                        'title': email_data.get('position', 'Unknown'),
                        'linkedin': email_data.get('linkedin', ''),
                        'company': domain.replace('.com', '').title(),
                        'source': 'hunter.io',
                        'confidence': email_data.get('confidence', 0)
                    }
                    
                    if contact['title'] and contact['title'] != 'Unknown':
                        contacts.append(contact)
                
                return contacts
            
            elif response.status_code == 401:
                if self.debug:
                    print(f"   ❌ Invalid API key")
                return []
            
            elif response.status_code == 429:
                if self.debug:
                    print(f"   ⚠️  Rate limit (25 searches/month)")
                return []
            
            else:
                if self.debug:
                    print(f"   ⚠️  Unknown error")
                    try:
                        print(f"   Response: {response.json()}")
                    except:
                        print(f"   Response: {response.text[:200]}")
                return []
                
        except Exception as e:
            if self.debug:
                print(f"   ❌ Exception: {str(e)}")
            return []
    
    def _find_contacts_apollo(self, company: str, role: str, max_results: int) -> List[Dict]:
        """Find contacts via Apollo.io"""
        
        if not self.apollo_api_key:
            return []
        
        try:
            search_titles = self._get_relevant_titles(role)
            
            payload = {
                'api_key': self.apollo_api_key,
                'q_organization_name': company,
                'person_titles': search_titles,
                'page': 1,
                'per_page': min(max_results, 10)
            }
            
            headers = {'Content-Type': 'application/json'}
            
            response = requests.post(
                self.apollo_people_search,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if self.debug:
                print(f"   API Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if 'people' not in data:
                    return []
                
                contacts = []
                for person in data['people']:
                    contact = {
                        'name': person.get('name', ''),
                        'email': person.get('email', ''),
                        'title': person.get('title', 'Unknown'),
                        'linkedin': person.get('linkedin_url', ''),
                        'company': company,
                        'source': 'apollo.io'
                    }
                    
                    if contact['name'] and contact['title']:
                        contacts.append(contact)
                
                return contacts
            
            else:
                if self.debug:
                    print(f"   ⚠️  Apollo error")
                return []
                
        except Exception as e:
            if self.debug:
                print(f"   ❌ Exception: {str(e)}")
            return []
    
    def _generate_mock_contacts(self, company: str, role: str, max_results: int) -> List[Dict]:
        """Generate realistic mock contacts"""
        
        company_clean = company.lower().replace(' ', '')
        
        contacts = [
            {
                'name': 'Sarah Chen',
                'title': 'Senior Recruiting Manager',
                'email': f"s.chen@{company_clean}.com",
                'linkedin': 'https://linkedin.com/in/sarahchen',
                'company': company,
                'source': 'mock'
            },
            {
                'name': 'Michael Rodriguez',
                'title': 'Director of Data Analytics',
                'email': f"m.rodriguez@{company_clean}.com",
                'linkedin': 'https://linkedin.com/in/mrodriguez',
                'company': company,
                'source': 'mock'
            },
            {
                'name': 'Emily Johnson',
                'title': 'Talent Acquisition Specialist',
                'email': f"e.johnson@{company_clean}.com",
                'linkedin': 'https://linkedin.com/in/emilyjohnson',
                'company': company,
                'source': 'mock'
            },
            {
                'name': 'David Kim',
                'title': 'VP of Business Intelligence',
                'email': f"d.kim@{company_clean}.com",
                'linkedin': 'https://linkedin.com/in/davidkim',
                'company': company,
                'source': 'mock'
            },
            {
                'name': 'Lisa Martinez',
                'title': 'Head of Data Science',
                'email': f"l.martinez@{company_clean}.com",
                'linkedin': 'https://linkedin.com/in/lisamartinez',
                'company': company,
                'source': 'mock'
            }
        ]
        
        return contacts[:max_results]
    
    def _get_company_domain(self, company: str) -> str:
        """Convert company name to domain"""
        
        domain_map = {
            'google': 'google.com',
            'meta': 'meta.com',
            'facebook': 'meta.com',
            'amazon': 'amazon.com',
            'microsoft': 'microsoft.com',
            'apple': 'apple.com',
            'netflix': 'netflix.com',
            'tesla': 'tesla.com',
            'snowflake': 'snowflake.com',
            'databricks': 'databricks.com',
            'salesforce': 'salesforce.com',
        }
        
        company_lower = company.lower()
        for key, domain in domain_map.items():
            if key in company_lower:
                return domain
        
        company_clean = company.lower().replace(' ', '').replace(',', '')
        return f"{company_clean}.com"
    
    def _get_relevant_titles(self, role: str) -> List[str]:
        """Get relevant job titles to search"""
        
        base = ['Recruiting Manager', 'Talent Acquisition', 'Hiring Manager']
        
        role_lower = role.lower()
        
        if 'data' in role_lower or 'analyst' in role_lower:
            base.extend(['Director of Data', 'Head of Analytics'])
        
        if 'engineer' in role_lower:
            base.extend(['Director of Engineering', 'Engineering Manager'])
        
        return base[:5]
    
    def _score_and_rank(self, contacts: List[Dict], role: str) -> List[Dict]:
        """Score and rank all contacts"""
        
        scored = []
        for contact in contacts:
            score_info = self._score_contact(contact, role)
            
            scored.append({
                **contact,
                'relevance_score': score_info['total_score'],
                'score_breakdown': score_info,
                'connection_strength': 'cold'
            })
        
        scored.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored
    
    def _score_contact(self, contact: Dict, target_role: str) -> Dict:
        """Score contact relevance"""
        
        title = contact.get('title', '').lower()
        target = target_role.lower()
        
        title_score = 0
        for keyword, weight in self.title_weights.items():
            if keyword in title:
                title_score = max(title_score, weight * 5)
        
        seniority_score = 30 if any(w in title for w in ['vp', 'director', 'head']) else 20 if any(w in title for w in ['manager', 'lead']) else 10
        
        department_score = 20 if any(k in target and k in title for k in ['data', 'analytics', 'engineer']) else 15 if any(w in title for w in ['talent', 'recruiting']) else 0
        
        total = min(100, title_score + seniority_score + department_score)
        
        return {
            'total_score': round(total, 1),
            'title_score': title_score,
            'seniority_score': seniority_score,
            'department_score': department_score
        }


# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 70)
    print("BULLETPROOF Contact Finder Test")
    print("=" * 70)
    
    finder = ContactFinder(use_real_apis=True, debug=True)
    
    # Test with real company
    contacts = finder.find_contacts(
        company="Snowflake",
        role="Data Analyst",
        max_results=5
    )
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {len(contacts)} contacts")
    print("=" * 70)
    
    for i, c in enumerate(contacts, 1):
        print(f"\n{i}. {c['name']} ({c['source']})")
        print(f"   {c['title']}")
        print(f"   {c['email']}")
        print(f"   Relevance: {c['relevance_score']}/100")
    
    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)
