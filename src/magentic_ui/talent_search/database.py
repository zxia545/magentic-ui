"""
Database module for Talent Search System
Handles all database operations for storing search results

支持SQLite和PostgreSQL，提供候选人和搜索任务的持久化存储
"""
import sqlite3
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from . import schemas
from . import config
from .utils import normalize_whitespace


class TalentSearchDB:
    """Database manager for talent search system"""
    
    def __init__(self, db_type: str = "sqlite", db_path: str = None, **kwargs):
        """
        Initialize database connection
        
        Args:
            db_type: "sqlite" or "postgresql"
            db_path: Database file path (for SQLite)
            **kwargs: Additional connection parameters for PostgreSQL
        """
        self.db_type = db_type
        
        if db_type == "sqlite":
            if db_path is None:
                db_path = Path(config.SQLITE_DB_PATH)
            
            # Ensure directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            
            print(f"[Database] Connected to SQLite database: {db_path}")
            
        elif db_type == "postgresql":
            try:
                import psycopg2
                import psycopg2.extras
            except ImportError:
                raise ImportError("psycopg2 is required for PostgreSQL support. Install with: pip install psycopg2-binary")
            
            self.conn = psycopg2.connect(
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 5432),
                database=kwargs.get("database", "talent_search"),
                user=kwargs.get("user"),
                password=kwargs.get("password")
            )
            self.conn.autocommit = False
            print(f"[Database] Connected to PostgreSQL database: {kwargs.get('database')}")
            
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        self._init_schema()
    
    def _init_schema(self):
        """Create tables if not exist"""
        cursor = self.conn.cursor()
        
        # SQLite schema
        if self.db_type == "sqlite":
            # 1. Search tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_tasks (
                    task_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    
                    search_query TEXT NOT NULL,
                    keywords TEXT,
                    venues TEXT,
                    years TEXT,
                    degree_levels TEXT,
                    top_n INTEGER,
                    
                    status TEXT NOT NULL,
                    rounds_completed INTEGER DEFAULT 0,
                    total_candidates INTEGER DEFAULT 0,
                    matching_candidates INTEGER DEFAULT 0,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    
                    spec_json TEXT,
                    search_logs TEXT
                )
            """)
            
            # 2. Candidates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    name_normalized TEXT UNIQUE,
                    
                    position TEXT,
                    affiliation TEXT,
                    candidate_category TEXT,
                    career_stage TEXT,
                    
                    email TEXT,
                    homepage TEXT,
                    google_scholar TEXT,
                    github TEXT,
                    linkedin TEXT,
                    openreview TEXT,
                    
                    final_score REAL DEFAULT 0,
                    score_paper REAL DEFAULT 0,
                    total_score INTEGER DEFAULT 0,
                    
                    introduction_json TEXT,
                    research_interests_json TEXT,
                    selected_research_json TEXT,
                    awards_json TEXT,
                    professional_services_json TEXT,
                    education_json TEXT,
                    industrial_experience_json TEXT,
                    teaching_json TEXT,
                    contact_json TEXT,
                    radar_json TEXT,
                    
                    trigger_paper_title TEXT,
                    trigger_paper_url TEXT,
                    trigger_paper_venue TEXT,
                    trigger_paper_score INTEGER,
                    
                    first_discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    discovery_count INTEGER DEFAULT 1,
                    
                    raw_data_json TEXT
                )
            """)
            
            # 3. Task-Candidates association table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS task_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    candidate_id INTEGER NOT NULL,
                    
                    discovered_in_round INTEGER,
                    rank_in_task INTEGER,
                    score_in_task REAL,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (task_id) REFERENCES search_tasks(task_id) ON DELETE CASCADE,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE,
                    
                    UNIQUE(task_id, candidate_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidates_category ON candidates(candidate_category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidates_affiliation ON candidates(affiliation)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidates_final_score ON candidates(final_score DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_candidates_task ON task_candidates(task_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_candidates_candidate ON task_candidates(candidate_id)")
            
        # TODO: Add PostgreSQL schema if needed
        
        self.conn.commit()
        print("[Database] Schema initialized successfully")
    
    def save_search_task(self, task_state: schemas.SearchTaskState, status: str = "completed") -> bool:
        """
        Save or update a search task
        
        Args:
            task_state: SearchTaskState object
            status: Task status (running/completed/cancelled)
            
        Returns:
            True if successful
        """
        cursor = self.conn.cursor()
        
        # Extract search query info
        spec = task_state.spec
        
        # Build search query display
        keywords_list = spec.keywords if spec.keywords else []
        search_query = ", ".join(keywords_list) if keywords_list else "research papers"
        
        # Serialize lists to JSON
        keywords_json = json.dumps(keywords_list)
        venues_json = json.dumps(spec.venues if spec.venues else [])
        years_json = json.dumps(spec.years if spec.years else [])
        degree_levels_json = json.dumps(spec.degree_levels if spec.degree_levels else [])
        spec_json = json.dumps(spec.model_dump())
        
        # Upsert logic for SQLite
        if self.db_type == "sqlite":
            cursor.execute("""
                INSERT INTO search_tasks (
                    task_id, search_query, keywords, venues, years, 
                    degree_levels, top_n, status, rounds_completed,
                    total_candidates, matching_candidates, spec_json, 
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status = excluded.status,
                    rounds_completed = excluded.rounds_completed,
                    total_candidates = excluded.total_candidates,
                    matching_candidates = excluded.matching_candidates,
                    updated_at = excluded.updated_at,
                    completed_at = CASE WHEN excluded.status = 'completed' THEN CURRENT_TIMESTAMP ELSE completed_at END
            """, (
                task_state.task_id,
                search_query,
                keywords_json,
                venues_json,
                years_json,
                degree_levels_json,
                spec.top_n,
                status,
                task_state.rounds_completed,
                len(task_state.candidates_accum),
                len(task_state.candidates_accum),  # Assume all are matching for now
                spec_json,
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
        
        self.conn.commit()
        print(f"[Database] ✅ Saved search task: {task_state.task_id}")
        return True
    
    def save_candidate(self, candidate: schemas.CandidateOverview, task_id: str = None) -> int:
        """
        Save or update a candidate
        
        Args:
            candidate: CandidateOverview object
            task_id: Optional task ID for linking
            
        Returns:
            candidate_id (database primary key)
        """
        cursor = self.conn.cursor()
        
        # Normalize name for deduplication
        name_normalized = self._normalize_name(candidate.name)
        
        # Check if candidate already exists
        cursor.execute("SELECT candidate_id, discovery_count FROM candidates WHERE name_normalized = ?", (name_normalized,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing candidate
            candidate_id = existing[0] if self.db_type == "sqlite" else existing["candidate_id"]
            self._update_candidate(cursor, candidate_id, candidate)
        else:
            # Insert new candidate
            candidate_id = self._insert_candidate(cursor, candidate, name_normalized)
        
        # Link to task if provided
        if task_id:
            self._link_candidate_to_task(cursor, task_id, candidate_id, candidate)
        
        self.conn.commit()
        return candidate_id
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for deduplication (lowercase, trim spaces)"""
        return normalize_whitespace(name).lower().strip()
    
    def _insert_candidate(self, cursor, candidate: schemas.CandidateOverview, name_normalized: str) -> int:
        """Insert new candidate record"""
        # Extract data from candidate
        intro = candidate.introduction
        contact = candidate.contact
        
        # Serialize complex fields to JSON
        introduction_json = json.dumps(intro.model_dump())
        research_interests_json = json.dumps([ri.model_dump() for ri in candidate.research_interests])
        selected_research_json = json.dumps(candidate.selected_research.model_dump())
        awards_json = json.dumps([a.model_dump() for a in candidate.awards])
        professional_services_json = json.dumps([ps.model_dump() for ps in candidate.professional_services])
        career_education_json = json.dumps([ce.model_dump() for ce in candidate.career_education_history])
        industrial_experience_json = json.dumps([ie.model_dump() for ie in candidate.industrial_experience])
        contact_json = json.dumps(contact.model_dump())
        radar_json = json.dumps(candidate.radar)
        raw_data_json = json.dumps(candidate.model_dump(by_alias=True))
        
        cursor.execute("""
            INSERT INTO candidates (
                name, name_normalized, position, affiliation, candidate_category,
                email, homepage, google_scholar, github, linkedin, openreview,
                final_score, score_paper, total_score,
                introduction_json, research_interests_json, selected_research_json,
                awards_json, professional_services_json, career_education_json,
                industrial_experience_json, contact_json, radar_json,
                trigger_paper_title, trigger_paper_url, trigger_paper_venue, trigger_paper_score,
                raw_data_json, first_discovered_at, last_updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            candidate.name,
            name_normalized,
            intro.position,
            intro.affiliation,
            candidate.candidate_category,
            contact.email,
            contact.homepage,
            contact.google_scholar,
            contact.github,
            contact.linkedin,
            contact.openreview,
            candidate.final_score,
            candidate.score_paper,
            candidate.total_score,
            introduction_json,
            research_interests_json,
            selected_research_json,
            awards_json,
            professional_services_json,
            career_education_json,
            industrial_experience_json,
            contact_json,
            radar_json,
            candidate.trigger_paper_title,
            candidate.trigger_paper_url,
            candidate.trigger_paper_venue,
            candidate.trigger_paper_score,
            raw_data_json,
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        
        # Get inserted ID
        if self.db_type == "sqlite":
            candidate_id = cursor.lastrowid
        else:
            candidate_id = cursor.fetchone()[0]
        
        print(f"[Database] 📝 Inserted new candidate: {candidate.name} (ID: {candidate_id})")
        return candidate_id
    
    def _update_candidate(self, cursor, candidate_id: int, candidate: schemas.CandidateOverview):
        """Update existing candidate with new information"""
        intro = candidate.introduction
        raw_data_json = json.dumps(candidate.model_dump(by_alias=True))
        
        cursor.execute("""
            UPDATE candidates SET
                position = ?,
                affiliation = ?,
                final_score = MAX(final_score, ?),
                last_updated_at = ?,
                discovery_count = discovery_count + 1,
                raw_data_json = ?
            WHERE candidate_id = ?
        """, (
            intro.position,
            intro.affiliation,
            candidate.final_score,
            datetime.now().isoformat(),
            raw_data_json,
            candidate_id
        ))
        
        print(f"[Database] 🔄 Updated existing candidate: {candidate.name} (ID: {candidate_id})")
    
    def _link_candidate_to_task(self, cursor, task_id: str, candidate_id: int, candidate: schemas.CandidateOverview):
        """Link candidate to search task"""
        try:
            cursor.execute("""
                INSERT INTO task_candidates (
                    task_id, candidate_id, discovered_in_round, score_in_task
                ) VALUES (?, ?, ?, ?)
            """, (
                task_id,
                candidate_id,
                candidate.discovered_in_round,
                candidate.final_score
            ))
        except sqlite3.IntegrityError:
            # Already linked, skip
            pass
    
    def get_candidate_by_name(self, name: str) -> Optional[schemas.CandidateOverview]:
        """Retrieve candidate by name"""
        cursor = self.conn.cursor()
        name_normalized = self._normalize_name(name)
        
        cursor.execute("SELECT raw_data_json FROM candidates WHERE name_normalized = ?", (name_normalized,))
        row = cursor.fetchone()
        
        if row:
            data = json.loads(row[0] if self.db_type == "sqlite" else row["raw_data_json"])
            return schemas.CandidateOverview(**data)
        return None
    
    def search_candidates(self, 
                         keywords: List[str] = None,
                         category: str = None,
                         affiliation: str = None,
                         min_score: float = None,
                         limit: int = 20) -> List[schemas.CandidateOverview]:
        """
        Search candidates with filters
        
        Args:
            keywords: Research keywords to match (OR logic)
            category: Candidate category filter
            affiliation: Affiliation filter (partial match)
            min_score: Minimum final score
            limit: Maximum results
            
        Returns:
            List of matching candidates
        """
        cursor = self.conn.cursor()
        
        query = "SELECT raw_data_json FROM candidates WHERE 1=1"
        params = []
        
        if category:
            query += " AND candidate_category = ?"
            params.append(category)
        
        if affiliation:
            query += " AND affiliation LIKE ?"
            params.append(f"%{affiliation}%")
        
        if min_score is not None:
            query += " AND final_score >= ?"
            params.append(min_score)
        
        # Simple keyword matching (searches in raw JSON)
        if keywords:
            query += " AND ("
            keyword_conditions = []
            for kw in keywords:
                keyword_conditions.append("raw_data_json LIKE ?")
                params.append(f"%{kw}%")
            query += " OR ".join(keyword_conditions)
            query += ")"
        
        query += " ORDER BY final_score DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            data = json.loads(row[0] if self.db_type == "sqlite" else row["raw_data_json"])
            results.append(schemas.CandidateOverview(**data))
        
        return results
    
    def get_search_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent search tasks"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT task_id, search_query, keywords, venues, years, 
                   status, rounds_completed, total_candidates, created_at
            FROM search_tasks
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            if self.db_type == "sqlite":
                results.append({
                    "task_id": row[0],
                    "search_query": row[1],
                    "keywords": json.loads(row[2]) if row[2] else [],
                    "venues": json.loads(row[3]) if row[3] else [],
                    "years": json.loads(row[4]) if row[4] else [],
                    "status": row[5],
                    "rounds_completed": row[6],
                    "total_candidates": row[7],
                    "created_at": row[8]
                })
            else:
                results.append({
                    "task_id": row["task_id"],
                    "search_query": row["search_query"],
                    "keywords": json.loads(row["keywords"]) if row["keywords"] else [],
                    "venues": json.loads(row["venues"]) if row["venues"] else [],
                    "years": json.loads(row["years"]) if row["years"] else [],
                    "status": row["status"],
                    "rounds_completed": row["rounds_completed"],
                    "total_candidates": row["total_candidates"],
                    "created_at": row["created_at"]
                })
        
        return results
    
    def get_candidates_by_task(self, task_id: str) -> List[schemas.CandidateOverview]:
        """Get all candidates associated with a specific task"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT c.raw_data_json 
            FROM candidates c
            JOIN task_candidates tc ON c.candidate_id = tc.candidate_id
            WHERE tc.task_id = ?
            ORDER BY tc.score_in_task DESC
        """, (task_id,))
        
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            data = json.loads(row[0] if self.db_type == "sqlite" else row["raw_data_json"])
            results.append(schemas.CandidateOverview(**data))
        
        return results
    
    def get_candidate_count(self) -> int:
        """Get total number of candidates in database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM candidates")
        count = cursor.fetchone()[0]
        return count
    
    def get_task_count(self) -> int:
        """Get total number of search tasks in database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM search_tasks")
        count = cursor.fetchone()[0]
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        # Total candidates
        cursor.execute("SELECT COUNT(*) FROM candidates")
        total_candidates = cursor.fetchone()[0]
        
        # Total tasks
        cursor.execute("SELECT COUNT(*) FROM search_tasks")
        total_tasks = cursor.fetchone()[0]
        
        # Candidates by category
        cursor.execute("""
            SELECT candidate_category, COUNT(*) 
            FROM candidates 
            GROUP BY candidate_category
        """)
        category_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Top affiliations
        cursor.execute("""
            SELECT affiliation, COUNT(*) as cnt
            FROM candidates
            WHERE affiliation IS NOT NULL AND affiliation != ''
            GROUP BY affiliation
            ORDER BY cnt DESC
            LIMIT 10
        """)
        top_affiliations = [(row[0], row[1]) for row in cursor.fetchall()]
        
        return {
            "total_candidates": total_candidates,
            "total_tasks": total_tasks,
            "candidates_by_category": category_counts,
            "top_affiliations": top_affiliations
        }
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        print("[Database] Connection closed")


# Global database instance
_db_instance: Optional[TalentSearchDB] = None

def get_db(db_type: str = "sqlite", db_path: str = None, **kwargs) -> TalentSearchDB:
    """
    Get or create global database instance
    
    Args:
        db_type: "sqlite" or "postgresql"
        db_path: Database file path (for SQLite)
        **kwargs: Additional connection parameters for PostgreSQL
        
    Returns:
        TalentSearchDB instance
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = TalentSearchDB(db_type=db_type, db_path=db_path, **kwargs)
    
    return _db_instance


def close_db():
    """Close global database instance"""
    global _db_instance
    
    if _db_instance is not None:
        _db_instance.close()
        _db_instance = None
