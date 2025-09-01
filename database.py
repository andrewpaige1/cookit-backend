# database.py
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, DateTime, Integer, Text, UUID, JSON, Boolean
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- Database Configuration ---
DATABASE_URL = os.environ.get("DATABASE_URL")
# Format: postgresql+asyncpg://username:password@host:port/database

if not DATABASE_URL:
    # Fallback for development (you can use local postgres if needed)
    DATABASE_URL = "postgresql+asyncpg://localhost:5432/platoai_dev"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Set to False in production
    pool_size=20,
    max_overflow=0,
)

# Create session maker
async_session_maker = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# --- Database Models ---
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True)  # Use Supabase UUID
    # email = Column(String(255), unique=True, nullable=True, index=True)  # Made nullable/removed
    auth_provider = Column(String(50), default="supabase")
    taste_profile = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

from sqlalchemy import UniqueConstraint

class Recipe(Base):
    __tablename__ = "recipes"
    __table_args__ = (
        UniqueConstraint('user_id', 'source_url', name='uix_user_source'),
    )
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)  # Link to user
    title = Column(Text, nullable=False, index=True)  # Searchable title
    source_url = Column(Text)  # Original URL
    # Fast searchable columns
    cuisine_type = Column(String(50), nullable=True, index=True)  # 'Italian', 'Asian', etc.
    meal_type = Column(String(50), nullable=True, index=True)     # 'Breakfast', 'Dinner', etc.
    difficulty = Column(String(20), nullable=True, index=True)    # 'Easy', 'Medium', 'Hard'
    # JSONB hybrid approach
    recipe_data = Column(JSON, nullable=False)  # Full LLM output (ingredients, instructions, metadata)
    searchable_ingredients = Column(JSON)  # Array of ingredient names for fast search
    searchable_tags = Column(JSON)  # Array of tags for fast search
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class UserRecipe(Base):
    __tablename__ = "user_recipes"
    
    user_id = Column(UUID(as_uuid=True), nullable=False, primary_key=True)
    recipe_id = Column(UUID(as_uuid=True), nullable=False, primary_key=True)
    rating = Column(Integer)  # 1-5 stars
    saved_at = Column(DateTime, default=datetime.utcnow)
    last_cooked = Column(DateTime)
    cook_count = Column(Integer, default=0)
    notes = Column(Text)

# --- Database Functions ---
async def get_async_session():
    """Get async database session"""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()

async def create_tables():
    """Create all database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def drop_tables():
    """Drop all database tables (for development)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

# --- User Management Functions ---
async def get_or_create_user(user_data: dict, db: AsyncSession) -> User:
    """Get existing user or create new user from Supabase JWT data"""
    from sqlalchemy import select
    import uuid
    
    user_id = uuid.UUID(user_data["user_id"])
    
    # Try to get existing user
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        # Create new user with Supabase data
        user = User(
            id=user_id,
            #email=user_data["email"],
            auth_provider="supabase",
            taste_profile={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        #print(f"✅ Created new user: {user.email} (ID: {user.id})")
    else:
        # User exists, they're active (no need to update anything for now)
        #print(f"✅ User authenticated: {user.email} (ID: {user.id})")
        print(f"✅ User authenticated: {user.id}")
    return user

# --- Recipe Management Functions ---
async def create_recipe_hybrid(
    recipe_data: dict, 
    user_id: str, 
    source_url: str,
    processing_stats: dict,
    db: AsyncSession
) -> Recipe:
    """Create recipe using hybrid JSONB approach"""
    import uuid
    
    # Extract searchable ingredients from the recipe data
    ingredients_list = recipe_data.get("ingredients", [])
    searchable_ingredients = []
    
    for ingredient in ingredients_list:
        # Extract ingredient name (simple approach - could be enhanced with NLP)
        # Remove quantities and common words
        ingredient_name = ingredient.lower().strip()
        # Basic cleaning - remove numbers, measurements
        import re
        clean_name = re.sub(r'[\d\-\.\s]*(cups?|tbsp|tsp|lbs?|oz|grams?|kg|ml|l)\s*', '', ingredient_name)
        clean_name = re.sub(r'^\d+[\s\-]*', '', clean_name)  # Remove leading numbers
        clean_name = clean_name.strip()
        
        if clean_name and len(clean_name) > 2:  # Only add meaningful ingredient names
            searchable_ingredients.append(clean_name)
    
    # Extract searchable tags (clean and normalize)
    tags_list = recipe_data.get("tags", [])
    searchable_tags = []
    for tag in tags_list:
        if isinstance(tag, str) and len(tag.strip()) > 1:
            clean_tag = tag.lower().strip()
            searchable_tags.append(clean_tag)
    
    # Extract searchable fields for dedicated columns
    cuisine_type = recipe_data.get("cuisine_type")
    meal_type = recipe_data.get("meal_type") 
    difficulty = recipe_data.get("difficulty")
    
    # Normalize extracted values
    if cuisine_type:
        cuisine_type = cuisine_type.strip().title()  # "italian" -> "Italian"
    if meal_type:
        meal_type = meal_type.strip().title()        # "breakfast" -> "Breakfast"
    if difficulty:
        difficulty = difficulty.strip().title()      # "easy" -> "Easy"
    
    # Combine recipe data with processing metadata
    full_recipe_data = {
        **recipe_data,
        "processing_metadata": processing_stats
    }
    
    # Create recipe with hybrid approach
    recipe = Recipe(
        title=recipe_data.get("title", "Untitled Recipe"),
        source_url=source_url,
        user_id=uuid.UUID(user_id),
        cuisine_type=cuisine_type,
        meal_type=meal_type,
        difficulty=difficulty,
        recipe_data=full_recipe_data,  # Full LLM output + metadata
        searchable_ingredients=searchable_ingredients,  # Extracted for fast search
        searchable_tags=searchable_tags  # Extracted for fast search
    )
    
    db.add(recipe)
    await db.commit()
    await db.refresh(recipe)
    
    return recipe

async def search_recipes_by_ingredient(ingredient: str, user_id: str, db: AsyncSession) -> list[Recipe]:
    """Fast ingredient-based recipe search using searchable_ingredients array"""
    from sqlalchemy import select, text
    import uuid
    
    user_uuid = uuid.UUID(user_id)
    
    # Use PostgreSQL array contains operator for fast search
    result = await db.execute(
        select(Recipe).where(
            Recipe.user_id == user_uuid,
            text("searchable_ingredients @> :ingredient")
        ).params(ingredient=f'["{ingredient.lower()}"]')
    )
    
    return list(result.scalars().all())

async def search_recipes_by_cuisine(cuisine_type: str, user_id: str, db: AsyncSession) -> list[Recipe]:
    """Fast cuisine-based recipe search using dedicated indexed column"""
    from sqlalchemy import select
    import uuid
    
    user_uuid = uuid.UUID(user_id)
    
    result = await db.execute(
        select(Recipe).where(
            Recipe.user_id == user_uuid,
            Recipe.cuisine_type.ilike(f"%{cuisine_type}%")
        ).order_by(Recipe.created_at.desc())
    )
    
    return list(result.scalars().all())

async def search_recipes_by_meal_type(meal_type: str, user_id: str, db: AsyncSession) -> list[Recipe]:
    """Fast meal type-based recipe search using dedicated indexed column"""
    from sqlalchemy import select
    import uuid
    
    user_uuid = uuid.UUID(user_id)
    
    result = await db.execute(
        select(Recipe).where(
            Recipe.user_id == user_uuid,
            Recipe.meal_type.ilike(f"%{meal_type}%")
        ).order_by(Recipe.created_at.desc())
    )
    
    return list(result.scalars().all())

async def search_recipes_by_difficulty(difficulty: str, user_id: str, db: AsyncSession) -> list[Recipe]:
    """Fast difficulty-based recipe search using dedicated indexed column"""
    from sqlalchemy import select
    import uuid
    
    user_uuid = uuid.UUID(user_id)
    
    result = await db.execute(
        select(Recipe).where(
            Recipe.user_id == user_uuid,
            Recipe.difficulty.ilike(f"%{difficulty}%")
        ).order_by(Recipe.created_at.desc())
    )
    
    return list(result.scalars().all())

async def search_recipes_by_tag(tag: str, user_id: str, db: AsyncSession) -> list[Recipe]:
    """Fast tag-based recipe search using searchable_tags array"""
    from sqlalchemy import select, text
    import uuid
    
    user_uuid = uuid.UUID(user_id)
    
    # Use PostgreSQL array contains operator for fast search
    result = await db.execute(
        select(Recipe).where(
            Recipe.user_id == user_uuid,
            text("searchable_tags @> :tag")
        ).params(tag=f'["{tag.lower()}"]').order_by(Recipe.created_at.desc())
    )
    
    return list(result.scalars().all())

async def get_recipe_hybrid(recipe_id: str, user_id: str, db: AsyncSession) -> dict | None:
    """Get recipe with hybrid approach - single fast query"""
    from sqlalchemy import select
    import uuid
    
    recipe_uuid = uuid.UUID(recipe_id)
    user_uuid = uuid.UUID(user_id)
    
    result = await db.execute(
        select(Recipe).where(
            Recipe.id == recipe_uuid,
            Recipe.user_id == user_uuid
        )
    )
    
    recipe = result.scalar_one_or_none()
    if not recipe:
        return None
    
    # Safely get the recipe data value  
    recipe_data_value = getattr(recipe, 'recipe_data', None)
    
    # Safely spread recipe data
    result_dict = {
        "id": str(recipe.id),
        "title": recipe.title,
        "source_url": recipe.source_url,
        "created_at": recipe.created_at.isoformat(),
        "searchable_ingredients": recipe.searchable_ingredients,
    }
    
    # Add recipe data fields if it exists and is a dict
    if recipe_data_value and isinstance(recipe_data_value, dict):
        result_dict.update(recipe_data_value)
    
    return result_dict
