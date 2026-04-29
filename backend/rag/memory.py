"""
In-memory conversation memory with per-session history.
"""

import uuid
from typing import Dict, List, Optional
from datetime import datetime

from config import settings


class ConversationMemory:
    """Stores conversation history per session for multi-turn context."""

    def __init__(self):
        # session_id -> list of {"role": str, "content": str, "timestamp": str}
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Return existing session_id or create a new one."""
        if session_id and session_id in self._sessions:
            return session_id
        new_id = session_id or str(uuid.uuid4())
        self._sessions[new_id] = []
        return new_id

    def add_message(self, session_id: str, role: str, content: str):
        """Append a message to the session history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        # Trim to max history
        max_msgs = settings.MAX_CONVERSATION_HISTORY * 2  # user + assistant pairs
        if len(self._sessions[session_id]) > max_msgs:
            self._sessions[session_id] = self._sessions[session_id][-max_msgs:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return the conversation history for a session."""
        return self._sessions.get(session_id, [])

    def clear_session(self, session_id: str):
        """Delete a session's history."""
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        """Return all active session IDs."""
        return list(self._sessions.keys())
