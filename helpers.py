from sqlalchemy import select
from database import Recipe
import uuid
from urllib.parse import urlparse, urlunparse

def normalize_url(url: str) -> str:
    """Normalize URL for consistent comparison."""
    try:
        parsed = urlparse(url.strip())
        # Remove query parameters and fragments for TikTok URLs
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(), 
            parsed.path.rstrip('/') or '/',
            '', '', ''  # Remove params, query, and fragment
        ))
        return normalized
    except Exception:
        return url.strip()

async def is_duplicate_source(user_id: str, source_url: str, db) -> bool:
    """Check if a recipe with the same user_id and source_url already exists."""
    user_uuid = uuid.UUID(user_id)
    normalized_url = normalize_url(source_url)
    
    result = await db.execute(
        select(Recipe).where(
            Recipe.user_id == user_uuid,
            Recipe.source_url == normalized_url
        )
    )
    recipe = result.scalar_one_or_none()
    return recipe is not None