"""
Database setup and models for PostApply RL System
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ==================== TABLES ====================

class Application(Base):
    """Track every job application"""
    __tablename__ = "applications"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Job Details
    company = Column(String(200), nullable=False)
    role = Column(String(200), nullable=False)
    job_url = Column(Text)
    description = Column(Text)
    
    # Application Info
    applied_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="applied")
    
    # Context for RL
    company_type = Column(String(50))
    seniority = Column(String(50))
    has_connection = Column(Boolean, default=False)
    company_culture = Column(String(50))
    
    # Outcomes
    response_date = Column(DateTime, nullable=True)
    days_to_response = Column(Integer, nullable=True)
    got_interview = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Contact(Base):
    """Store found contacts for each application"""
    __tablename__ = "contacts"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, nullable=False)
    
    # Contact Info
    name = Column(String(200))
    email = Column(String(200))
    title = Column(String(200))
    linkedin_url = Column(Text)
    
    # Scoring
    relevance_score = Column(Float, default=0.0)
    connection_strength = Column(String(50))
    
    # Status
    contacted = Column(Boolean, default=False)
    contact_date = Column(DateTime, nullable=True)
    responded = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class RLState(Base):
    """Store Q-Learning and Thompson Sampling state"""
    __tablename__ = "rl_state"
    
    id = Column(Integer, primary_key=True, index=True)
    
    agent_type = Column(String(50))
    q_table = Column(JSON, nullable=True)
    thompson_params = Column(JSON, nullable=True)
    
    learning_rate = Column(Float, default=0.1)
    discount_factor = Column(Float, default=0.9)
    epsilon = Column(Float, default=0.1)
    
    total_updates = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Message(Base):
    """Store follow-up messages and their outcomes"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, nullable=False)
    
    message_style = Column(String(50))
    message_text = Column(Text)
    subject_line = Column(String(200))
    
    sent_date = Column(DateTime)
    days_after_application = Column(Integer)
    
    quality_score = Column(Float, default=0.0)
    personalization_score = Column(Float)
    clarity_score = Column(Float)
    length_score = Column(Float)
    
    got_response = Column(Boolean, default=False)
    response_date = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


# ==================== DATABASE FUNCTIONS ====================

def init_database():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created!")


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    print("Testing database connection...")
    init_database()
    print("🎉 Database is ready!")
