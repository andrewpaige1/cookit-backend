import os
import tempfile
import shutil
import base64
import asyncio
import concurrent.futures
import time
import json
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from datetime import datetime
import cv2 # OpenCV for video processing
from fastapi import FastAPI, HTTPException, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import yt_dlp
import openai
from dotenv import load_dotenv

# Import database components
from database import get_async_session, create_tables, Recipe, User, UserRecipe, get_or_create_user, create_recipe_hybrid, get_recipe_hybrid, search_recipes_by_ingredient, search_recipes_by_cuisine, search_recipes_by_meal_type, search_recipes_by_difficulty, search_recipes_by_tag
# Import authentication
from auth import verify_supabase_jwt, optional_auth, is_auth_configured
# Import assistant
from assistant import recipe_assistant

# --- Load Environment Variables ---
load_dotenv()

# --- Database Lifespan Event ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables on startup"""
    try:
        await create_tables()
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("💡 Make sure to update DATABASE_URL in .env with your Azure PostgreSQL details")
    
    yield  # App runs here
    
    # Cleanup code would go here if needed
    print("🔄 Shutting down...")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="PlatoAI Multi-Modal Recipe Parser",
    lifespan=lifespan
)

# --- OpenAI Client Initialization ---
# This will now automatically read the key from your .env file
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_frames_from_video(video_path, max_frames=12, quality_factor=0.7):
    """
    Optimized frame extraction with intelligent sampling and compression.
    - Uses adaptive sampling based on video length
    - Compresses frames to reduce API payload size
    - Limits total frames for cost control
    """
    print(f"Extracting frames from {video_path}...")
    vidcap = cv2.VideoCapture(video_path)
    
    # Get video metadata
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    if fps == 0:
        fps = 30
        
    # Adaptive frame sampling based on video duration
    if duration <= 30:  # Short videos - sample every 3 seconds
        frame_interval = int(fps * 3)
    elif duration <= 120:  # Medium videos - sample every 5 seconds
        frame_interval = int(fps * 5)
    else:  # Long videos - sample every 8 seconds
        frame_interval = int(fps * 8)
    
    # Ensure we don't exceed max_frames
    estimated_frames = total_frames // frame_interval
    if estimated_frames > max_frames:
        frame_interval = total_frames // max_frames
    
    base64_frames = []
    frame_count = 0
    
    print(f"Video info: {duration:.1f}s, {fps:.1f}fps, sampling every {frame_interval} frames")
    
    while vidcap.isOpened() and len(base64_frames) < max_frames:
        success, frame = vidcap.read()
        if not success:
            break
        
        if frame_count % frame_interval == 0:
            # Resize frame to reduce payload size (maintain aspect ratio)
            height, width = frame.shape[:2]
            if width > 640:  # Resize if too large
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Compress with quality factor
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(85 * quality_factor)]
            success_flag, encoded_image = cv2.imencode(".jpg", frame, encode_params)
            
            if success_flag:
                base64_frames.append(base64.b64encode(encoded_image.tobytes()).decode("utf-8"))
        
        frame_count += 1
        
    vidcap.release()
    print(f"Extracted {len(base64_frames)} optimized frames.")
    return base64_frames

async def download_video_optimized(video_url: str, temp_dir: str, max_duration: int = 120) -> str:
    """
    Optimized video download with quality and duration limits.
    - Downloads lower quality for faster processing
    - Limits duration to reduce processing time
    - Uses concurrent execution
    """
    print(f"Downloading optimized video from: {video_url}")
    video_filepath_template = os.path.join(temp_dir, 'video.%(ext)s')

    ydl_opts = {
        # Choose a reasonable quality (not the highest)
        'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
        'outtmpl': video_filepath_template,
        # Limit duration to reduce download time and processing
        'download_archive': None,  # Don't use archive
        'extract_flat': False,
        'writethumbnail': False,
        'writeinfojson': False,
        'ignoreerrors': True,
        # Add postprocessor to trim video if it's too long
        'postprocessor_args': {
            'ffmpeg': ['-t', str(max_duration)]  # Limit to max_duration seconds
        } if max_duration else {},
    }

    def download_task():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            return ydl.prepare_filename(info_dict)

    # Run download in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        downloaded_video_path = await loop.run_in_executor(executor, download_task)

    if not os.path.exists(downloaded_video_path):
        raise HTTPException(status_code=500, detail="Failed to download video.")
    
    return downloaded_video_path

async def transcribe_audio_async(video_path: str, client) -> str:
    """Async audio transcription to run concurrently with frame extraction."""
    print("Transcribing audio...")
    try:
        def transcribe_task():
            with open(video_path, "rb") as video_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=video_file
                )
            return transcription.text

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            transcript_text = await loop.run_in_executor(executor, transcribe_task)
        
        print(f"Transcript received: {transcript_text[:150]}...")
        return transcript_text
    except Exception as e:
        print(f"Could not transcribe audio: {e}")
        return "No speech detected in the video."
# --- API Endpoint for Parsing Video Recipes ---
@app.post("/parse-recipe")
async def parse_recipe(
    url: str = Body(..., embed=True),
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Authenticated video recipe parser with concurrent processing and database storage.
    Requires valid Supabase JWT token in Authorization header.
    - Downloads lower quality video with duration limits
    - Processes frames and audio concurrently
    - Uses intelligent frame sampling
    - Saves recipe to database linked to authenticated user
    """
    start_time = time.time()
    video_url = url
    temp_dir = tempfile.mkdtemp()

    try:
        print(f"🚀 RECIPE PARSE: Starting for URL: {video_url}")
        print(f"🔐 RECIPE PARSE: User data: {user_data}")
        
        # --- Step 0: Get or create user in database ---
        print("� RECIPE PARSE: Getting or creating user...")
        try:
            user = await get_or_create_user(user_data, db)
            print(f"✅ RECIPE PARSE: User ready (ID: {user.id})")
        except Exception as e:
            print(f"❌ RECIPE PARSE: User creation failed: {e}")
            raise HTTPException(status_code=500, detail=f"User creation failed: {e}")

        # --- Step 1: Download optimized video ---
        print("⬇️ RECIPE PARSE: Starting video download...")
        try:
            download_start = time.time()
            downloaded_video_path = await download_video_optimized(
                video_url, temp_dir, max_duration=120
            )
            download_time = time.time() - download_start
            print(f"✅ RECIPE PARSE: Download completed in {download_time:.2f}s")
        except Exception as e:
            print(f"❌ RECIPE PARSE: Video download failed: {e}")
            raise HTTPException(status_code=500, detail=f"Video download failed: {e}")

        # --- Step 2 & 3: Process frames and audio concurrently ---
        print("🎬 RECIPE PARSE: Processing video frames and audio...")
        try:
            processing_start = time.time()
            
            # Run frame extraction and audio transcription in parallel
            frame_task = asyncio.create_task(
                asyncio.to_thread(extract_frames_from_video, downloaded_video_path, max_frames=10)
            )
            audio_task = asyncio.create_task(
                transcribe_audio_async(downloaded_video_path, client)
            )
            
            # Wait for both tasks to complete
            base64_frames, transcript_text = await asyncio.gather(frame_task, audio_task)
            processing_time = time.time() - processing_start
            print(f"✅ RECIPE PARSE: Processing completed in {processing_time:.2f}s")
        except Exception as e:
            print(f"❌ RECIPE PARSE: Video processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")

        # --- Step 4: Extract Recipe using Multi-Modal LLM ---
        llm_start = time.time()
        print("Extracting recipe with multi-modal LLM...")
        system_prompt = """
        You are an expert recipe assistant. Your task is to analyze the provided text transcript and video frames to extract comprehensive recipe information. The video has been optimized to show key moments.

        Rely on both visual cues from the frames and text from the transcript. If the transcript is sparse or silent, prioritize the visual information.

        Return the result as a clean JSON object. Use your culinary knowledge to make reasonable estimates for fields like time and servings, but if you cannot determine a value with high confidence, return `null` for that field.

        Here is an example of the desired output format:
        {
            "title": "Spicy Chicken Stir-Fry",
            "description": "A quick and easy stir-fry with tender chicken and fresh vegetables in a savory sauce.",
            "ingredients": [
                "1 lb chicken breast, cut into bite-sized pieces",
                "1 red bell pepper, sliced",
                "1 head of broccoli, cut into florets",
                "3 cloves garlic, minced",
                "1/4 cup soy sauce"
            ],
            "instructions": [
                "In a large skillet or wok, heat oil over medium-high heat.",
                "Add chicken and cook until browned.",
                "Add bell pepper and broccoli and stir-fry for 3-5 minutes.",
                "Add garlic and cook for another minute until fragrant.",
                "Stir in soy sauce and serve immediately."
            ],
            "prep_time": "15 minutes",
            "cook_time": "10 minutes",
            "servings": 4,
            "difficulty": "Easy",
            "cuisine_type": "Asian",
            "meal_type": "Dinner",
            "tags": ["quick", "weeknight meal", "healthy"]
        }

        Guidelines for each field:
        - title: Always extract or create a descriptive name
        - description: Provide a brief 1-2 sentence description
        - ingredients: List all visible/mentioned ingredients with quantities when possible
        - instructions: Extract step-by-step cooking process
        - prep_time: Estimate if you can see prep work, otherwise null
        - cook_time: Estimate if you can see cooking process, otherwise null
        - servings: Estimate from ingredient quantities or visual portions, otherwise null
        - difficulty: Assess based on technique complexity (Easy/Medium/Hard)
        - cuisine_type: Identify if clearly evident from ingredients/style, otherwise null
        - meal_type: Categorize as Breakfast/Lunch/Dinner/Snack/Dessert based on dish type
        - tags: Include relevant descriptors like "quick", "healthy", "vegetarian", etc.

        Now, analyze the following content and provide the JSON output.
        """
        
        # Optimize message construction
        content_items = [{"type": "text", "text": f"Transcript: {transcript_text}"}]
        
        # Add frames with lower detail for faster processing
        for frame in base64_frames:
            frame_item = {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}", 
                    "detail": "low"  # Use low detail for faster processing
                }
            }
            content_items.append(frame_item)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_items}
        ]

        # Use most cost-effective OpenAI model for recipe extraction
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 60% cheaper than gpt-4o, excellent for structured tasks
            response_format={"type": "json_object"},
            messages=messages,
            max_tokens=800,   # Reduced tokens for recipe extraction
            temperature=0.1   # Lower temperature for more consistent results
        )
        print(response.choices[0].message.content)

        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        
        print(f"LLM processing completed in {llm_time:.2f}s")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Performance breakdown - Download: {download_time:.2f}s, Processing: {processing_time:.2f}s, LLM: {llm_time:.2f}s")
        
        # --- Step 5: Parse and Save Recipe to Database ---
        try:
            response_content = response.choices[0].message.content
            if response_content is None:
                raise ValueError("Empty response from LLM")
                
            recipe_data = json.loads(response_content)
            
            # Create recipe using hybrid approach
            recipe = await create_recipe_hybrid(
                recipe_data={
                    "title": recipe_data.get("title", "Untitled Recipe"),
                    "description": recipe_data.get("description", ""),
                    "ingredients": recipe_data.get("ingredients", []),
                    "instructions": recipe_data.get("instructions", []),
                    "prep_time": recipe_data.get("prep_time"),
                    "cook_time": recipe_data.get("cook_time"),
                    "servings": recipe_data.get("servings"),
                    "difficulty": recipe_data.get("difficulty"),
                    "cuisine_type": recipe_data.get("cuisine_type"),
                    "meal_type": recipe_data.get("meal_type"),
                    "tags": recipe_data.get("tags", [])
                },
                user_id=str(user.id),
                source_url=video_url,
                processing_stats={
                    "processing_time": total_time,
                    "download_time": download_time,
                    "llm_time": llm_time,
                    "frames_extracted": len(base64_frames),
                    "transcript_length": len(transcript_text),
                    #"user_email": user.email
                },
                db=db
            )
            
            print(f"✅ Recipe saved to database with ID: {recipe.id}")
            
            # Return recipe with database ID and user info
            return {
                "id": str(recipe.id),
                "title": recipe.recipe_data.get("title"),
                "description": recipe.recipe_data.get("description"),
                "ingredients": recipe.recipe_data.get("ingredients", []),
                "instructions": recipe.recipe_data.get("instructions", []),
                "prep_time": recipe.recipe_data.get("prep_time"),
                "cook_time": recipe.recipe_data.get("cook_time"),
                "servings": recipe.recipe_data.get("servings"),
                "difficulty": recipe.recipe_data.get("difficulty"),
                "cuisine_type": recipe.recipe_data.get("cuisine_type"),
                "meal_type": recipe.recipe_data.get("meal_type"),
                "tags": recipe.recipe_data.get("tags", []),
                "source_url": recipe.source_url,
                "created_at": recipe.created_at.isoformat(),
                "user_id": str(user.id),
                "processing_stats": recipe.recipe_data.get("metadata", {})
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse recipe JSON: {e}")
            return {"error": "Failed to parse recipe", "raw_response": response_content or "No content"}
        
        return response_content or "No content returned"

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    finally:
        print("Cleaning up temporary files.")
        shutil.rmtree(temp_dir)

@app.get("/recipes")
async def get_recipes(
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """Get all recipes for the authenticated user"""
    from sqlalchemy import select
    import uuid
    
    user_id = uuid.UUID(user_data["user_id"])
    
    result = await db.execute(
        select(Recipe)
        .where(Recipe.user_id == user_id)
        .order_by(Recipe.created_at.desc())
    )
    recipes = result.scalars().all()
    
    return [
        {
            "id": str(recipe.id),
            "title": recipe.recipe_data.get("title"),
            "description": recipe.recipe_data.get("description"),
            "ingredients": recipe.recipe_data.get("ingredients", []),
            "instructions": recipe.recipe_data.get("instructions", []),
            "prep_time": recipe.recipe_data.get("prep_time"),
            "cook_time": recipe.recipe_data.get("cook_time"),
            "servings": recipe.recipe_data.get("servings"),
            "cuisine_type": recipe.cuisine_type,
            "meal_type": recipe.meal_type,
            "difficulty": recipe.difficulty,
            "tags": recipe.searchable_tags,
            "source_url": recipe.source_url,
            "created_at": recipe.created_at.isoformat()
        }
        for recipe in recipes
    ]

@app.get("/recipes/{recipe_id}")
async def get_recipe(
    recipe_id: str, 
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """Get a specific recipe by ID (must belong to authenticated user)"""
    from sqlalchemy import select
    import uuid
    
    try:
        recipe_uuid = uuid.UUID(recipe_id)
        user_uuid = uuid.UUID(user_data["user_id"])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    
    result = await db.execute(
        select(Recipe).where(
            Recipe.id == recipe_uuid,
            Recipe.user_id == user_uuid  # Ensure recipe belongs to user
        )
    )
    recipe = result.scalar_one_or_none()
    
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found or access denied")
    
    return {
        "id": str(recipe.id),
        "title": recipe.recipe_data.get("title"),
        "description": recipe.recipe_data.get("description"),
        "ingredients": recipe.recipe_data.get("ingredients", []),
        "instructions": recipe.recipe_data.get("instructions", []),
        "prep_time": recipe.recipe_data.get("prep_time"),
        "cook_time": recipe.recipe_data.get("cook_time"),
        "servings": recipe.recipe_data.get("servings"),
        "cuisine_type": recipe.cuisine_type,
        "meal_type": recipe.meal_type,
        "difficulty": recipe.difficulty,
        "tags": recipe.searchable_tags,
        "source_url": recipe.source_url,
        "created_at": recipe.created_at.isoformat(),
        "processing_stats": recipe.recipe_data.get("processing_stats", {})
    }

@app.get("/recipes/search/{ingredient}")
async def search_recipes_by_ingredient_endpoint(
    ingredient: str,
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """Search recipes by ingredient name. Returns recipes containing the specified ingredient."""
    import uuid
    try:
        user_uuid = uuid.UUID(user_data["user_id"])
        recipes = await search_recipes_by_ingredient(ingredient, str(user_uuid), db)
        
        return [
            {
                "id": str(recipe.id),
                "title": recipe.recipe_data.get("title"),
                "description": recipe.recipe_data.get("description"),
                "ingredients": recipe.recipe_data.get("ingredients", []),
                "prep_time": recipe.recipe_data.get("prep_time"),
                "cook_time": recipe.recipe_data.get("cook_time"),
                "servings": recipe.recipe_data.get("servings"),
                "cuisine_type": recipe.cuisine_type,
                "meal_type": recipe.meal_type,
                "difficulty": recipe.difficulty,
                "tags": recipe.searchable_tags,
                "source_url": recipe.source_url,
                "created_at": recipe.created_at.isoformat(),
                "matched_ingredients": recipe.searchable_ingredients
            }
            for recipe in recipes
        ]
    except Exception as e:
        print(f"Error searching recipes: {e}")
        raise HTTPException(status_code=500, detail="Failed to search recipes")

@app.get("/recipes/filter/cuisine/{cuisine_type}")
async def get_recipes_by_cuisine(
    cuisine_type: str,
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """Get recipes by cuisine type (e.g., Italian, Asian, Mexican) - Fast indexed search"""
    try:
        recipes = await search_recipes_by_cuisine(cuisine_type, user_data["user_id"], db)
        
        return [
            {
                "id": str(recipe.id),
                "title": recipe.recipe_data.get("title"),
                "description": recipe.recipe_data.get("description"),
                "cuisine_type": recipe.cuisine_type,
                "meal_type": recipe.meal_type,
                "difficulty": recipe.difficulty,
                "prep_time": recipe.recipe_data.get("prep_time"),
                "cook_time": recipe.recipe_data.get("cook_time"),
                "servings": recipe.recipe_data.get("servings"),
                "tags": recipe.searchable_tags,
                "source_url": recipe.source_url,
                "created_at": recipe.created_at.isoformat()
            }
            for recipe in recipes
        ]
    except Exception as e:
        print(f"Error filtering recipes by cuisine: {e}")
        raise HTTPException(status_code=500, detail="Failed to filter recipes")

@app.get("/recipes/filter/meal/{meal_type}")
async def get_recipes_by_meal_type(
    meal_type: str,
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """Get recipes by meal type (Breakfast, Lunch, Dinner, Snack, Dessert) - Fast indexed search"""
    try:
        recipes = await search_recipes_by_meal_type(meal_type, user_data["user_id"], db)
        
        return [
            {
                "id": str(recipe.id),
                "title": recipe.recipe_data.get("title"),
                "description": recipe.recipe_data.get("description"),
                "cuisine_type": recipe.cuisine_type,
                "meal_type": recipe.meal_type,
                "difficulty": recipe.difficulty,
                "prep_time": recipe.recipe_data.get("prep_time"),
                "cook_time": recipe.recipe_data.get("cook_time"),
                "servings": recipe.recipe_data.get("servings"),
                "tags": recipe.searchable_tags,
                "source_url": recipe.source_url,
                "created_at": recipe.created_at.isoformat()
            }
            for recipe in recipes
        ]
    except Exception as e:
        print(f"Error filtering recipes by meal type: {e}")
        raise HTTPException(status_code=500, detail="Failed to filter recipes")

@app.get("/recipes/filter/difficulty/{difficulty}")
async def get_recipes_by_difficulty(
    difficulty: str,
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """Get recipes by difficulty level (Easy, Medium, Hard) - Fast indexed search"""
    try:
        recipes = await search_recipes_by_difficulty(difficulty, user_data["user_id"], db)
        
        return [
            {
                "id": str(recipe.id),
                "title": recipe.recipe_data.get("title"),
                "description": recipe.recipe_data.get("description"),
                "cuisine_type": recipe.cuisine_type,
                "meal_type": recipe.meal_type,
                "difficulty": recipe.difficulty,
                "prep_time": recipe.recipe_data.get("prep_time"),
                "cook_time": recipe.recipe_data.get("cook_time"),
                "servings": recipe.recipe_data.get("servings"),
                "tags": recipe.searchable_tags,
                "source_url": recipe.source_url,
                "created_at": recipe.created_at.isoformat()
            }
            for recipe in recipes
        ]
    except Exception as e:
        print(f"Error filtering recipes by difficulty: {e}")
        raise HTTPException(status_code=500, detail="Failed to filter recipes")

@app.get("/recipes/filter/tag/{tag}")
async def get_recipes_by_tag(
    tag: str,
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """Get recipes by tag (vegetarian, quick, healthy, etc.) - Fast array search"""
    try:
        recipes = await search_recipes_by_tag(tag, user_data["user_id"], db)
        
        return [
            {
                "id": str(recipe.id),
                "title": recipe.recipe_data.get("title"),
                "description": recipe.recipe_data.get("description"),
                "cuisine_type": recipe.cuisine_type,
                "meal_type": recipe.meal_type,
                "difficulty": recipe.difficulty,
                "prep_time": recipe.recipe_data.get("prep_time"),
                "cook_time": recipe.recipe_data.get("cook_time"),
                "servings": recipe.recipe_data.get("servings"),
                "tags": recipe.searchable_tags,
                "source_url": recipe.source_url,
                "created_at": recipe.created_at.isoformat(),
                "matched_tag": tag
            }
            for recipe in recipes
        ]
    except Exception as e:
        print(f"Error filtering recipes by tag: {e}")
        raise HTTPException(status_code=500, detail="Failed to filter recipes")

@app.get("/me")
async def get_user_profile(
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """Get current user's profile and stats"""
    from sqlalchemy import select, func
    import uuid
    
    user_id = uuid.UUID(user_data["user_id"])
    user = await get_or_create_user(user_data, db)
    
    # Get user's recipe count
    recipe_count_result = await db.execute(
        select(func.count(Recipe.id)).where(Recipe.user_id == user_id)
    )
    recipe_count = recipe_count_result.scalar() or 0
    
    return {
        "id": str(user.id),
        #"email": user.email,
        "auth_provider": user.auth_provider,
        "taste_profile": user.taste_profile,
        "created_at": user.created_at.isoformat(),
        "stats": {
            "total_recipes": recipe_count
        }
    }

@app.get("/auth/status")
async def auth_status(user_data: Optional[dict] = Depends(optional_auth)):
    """Check authentication status (optional auth)"""
    if user_data:
        return {
            "authenticated": True,
            "user_id": user_data["user_id"],
            "email": user_data["email"]
        }
    else:
        return {
            "authenticated": False,
            "message": "No valid authentication provided"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {
        "status": "healthy", 
        "message": "PlatoAI Recipe Parser is running",
        "auth_configured": is_auth_configured()
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PlatoAI Multi-Modal Recipe Parser",
        "version": "2.0-authenticated",
        "features": [
            "User-specific recipe storage",
            "Optimized video downloading with quality limits",
            "Concurrent frame extraction and audio transcription", 
            "Intelligent adaptive frame sampling",
            "Compressed frame payloads for faster API calls",
            "Performance monitoring and timing",
            "Contextual AI Recipe Assistant"
        ],
        "endpoints": {
            "/parse-recipe": "POST - Extract recipe from video URL (requires auth)",
            "/recipes": "GET - Get user's saved recipes (requires auth)",
            "/recipes/{id}": "GET - Get specific recipe (requires auth)",
            "/assistant/chat": "POST - Chat with AI assistant about a recipe (requires auth)",
            "/assistant/quick-commands": "GET - Get available quick commands for a recipe",
            "/me": "GET - Get user profile (requires auth)",
            "/auth/status": "GET - Check authentication status (optional auth)",
            "/health": "GET - Health check",
            "/": "GET - API information"
        },
        "authentication": {
            "configured": is_auth_configured(),
            #"type": "Supabase JWT",
            "header": "Authorization: Bearer <jwt_token>"
        }
    }

# MARK: - AI Assistant Endpoints

@app.post("/assistant/chat")
async def chat_with_assistant(
    recipe_id: str = Body(..., embed=True),
    message: str = Body(..., embed=True),
    conversation_history: Optional[List[Dict[str, str]]] = Body(None, embed=True),
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Chat with AI assistant about a specific recipe.
    Provides contextual responses based on recipe ingredients, instructions, and details.
    """
    try:
        # Process message with recipe context
        response_data = await recipe_assistant.process_message(
            message=message,
            recipe_id=recipe_id,
            user_id=user_data["user_id"],
            db=db,
            conversation_history=conversation_history or []
        )
        
        return {
            "success": True,
            "message": response_data["response"],
            "type": response_data["type"],
            "data": response_data.get("data"),
            "intent": response_data.get("intent"),
            "recipe_context": response_data.get("recipe_context"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Assistant chat error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Assistant unavailable: {str(e)}"
        )

@app.get("/assistant/quick-commands/{recipe_id}")
async def get_quick_commands(
    recipe_id: str,
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get contextual quick commands for a specific recipe.
    Returns smart suggestions based on recipe content.
    """
    try:
        # Get recipe context to generate smart commands
        recipe_data = await recipe_assistant.get_recipe_context(recipe_id, user_data["user_id"], db)
        
        # Base commands that work for all recipes
        commands = [
            "Create shopping list",
            "Make it gluten-free", 
            "Substitute eggs",
        ]
        
        # Add scaling commands based on current servings
        current_servings = recipe_data.get("servings")
        if current_servings:
            if current_servings != 2:
                commands.insert(0, "Scale to 2 servings")
            if current_servings != 4:
                commands.insert(0, "Scale to 4 servings")
            if current_servings != 6:
                commands.insert(0, "Scale to 6 servings")
        else:
            commands.insert(0, "Scale to 6 servings")
        
        # Add step navigation if recipe has instructions
        if recipe_data.get("instructions"):
            commands.insert(0, "Read next step")
            commands.append("Walk me through cooking")
        
        # Add cuisine-specific commands
        cuisine = recipe_data.get("cuisine_type", "").lower()
        if "italian" in cuisine:
            commands.append("Wine pairing")
            commands.append("Make it vegetarian")
        elif "asian" in cuisine:
            commands.append("Spice level options")
            commands.append("Rice pairing")
        elif "mexican" in cuisine:
            commands.append("Spice it up")
            commands.append("Side dish ideas")
        
        # Add meal-type specific commands
        meal_type = recipe_data.get("meal_type", "").lower()
        if "dessert" in meal_type:
            commands.append("Make it healthier")
            commands.append("Decoration tips")
        elif "breakfast" in meal_type:
            commands.append("Make ahead tips")
            commands.append("Protein options")
        
        # Add difficulty-based commands
        difficulty = recipe_data.get("difficulty", "").lower()
        if "hard" in difficulty or "medium" in difficulty:
            commands.append("Simplify this recipe")
            commands.append("Technique tips")
        
        return {
            "recipe_id": recipe_id,
            "recipe_title": recipe_data.get("title"),
            "commands": commands[:8],  # Limit to 8 most relevant commands
            "recipe_context": {
                "servings": current_servings,
                "cuisine_type": recipe_data.get("cuisine_type"),
                "meal_type": recipe_data.get("meal_type"),
                "difficulty": recipe_data.get("difficulty"),
                "has_instructions": bool(recipe_data.get("instructions"))
            }
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Quick commands error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate quick commands: {str(e)}"
        )

@app.post("/assistant/voice")
async def process_voice_command(
    recipe_id: str = Body(..., embed=True),
    command: str = Body(..., embed=True),
    current_step: Optional[int] = Body(None, embed=True),
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Process voice commands for hands-free cooking.
    Optimized for voice interaction with short, actionable responses.
    """
    try:
        # Get recipe context
        recipe_data = await recipe_assistant.get_recipe_context(recipe_id, user_data["user_id"], db)
        instructions = recipe_data.get("instructions", [])
        
        command_lower = command.lower().strip()
        
        # Handle navigation commands
        if any(phrase in command_lower for phrase in ["next step", "continue", "what's next"]):
            if current_step is not None and current_step < len(instructions) - 1:
                next_step = current_step + 1
                return {
                    "type": "navigation",
                    "action": "next_step",
                    "step_number": next_step + 1,  # 1-indexed for user
                    "step_content": instructions[next_step],
                    "response": f"Step {next_step + 1}: {instructions[next_step]}",
                    "has_next": next_step < len(instructions) - 1
                }
            else:
                return {
                    "type": "navigation", 
                    "action": "complete",
                    "response": "You've completed all the steps! Your dish should be ready to serve."
                }
        
        elif any(phrase in command_lower for phrase in ["previous step", "go back", "repeat"]):
            if current_step is not None and current_step > 0:
                prev_step = current_step - 1
                return {
                    "type": "navigation",
                    "action": "previous_step", 
                    "step_number": prev_step + 1,
                    "step_content": instructions[prev_step],
                    "response": f"Step {prev_step + 1}: {instructions[prev_step]}",
                    "has_previous": prev_step > 0
                }
            else:
                return {
                    "type": "navigation",
                    "action": "first_step",
                    "response": "You're already at the first step."
                }
        
        elif any(phrase in command_lower for phrase in ["start over", "begin", "first step"]):
            if instructions:
                return {
                    "type": "navigation",
                    "action": "start",
                    "step_number": 1,
                    "step_content": instructions[0],
                    "response": f"Let's start cooking! Step 1: {instructions[0]}",
                    "total_steps": len(instructions)
                }
        
        # Handle timer commands
        elif any(word in command_lower for word in ["timer", "time", "minutes", "seconds"]):
            # Extract time from command
            import re
            time_match = re.search(r'(\d+)\s*(minute|min|second|sec)', command_lower)
            if time_match:
                time_value = int(time_match.group(1))
                time_unit = time_match.group(2)
                return {
                    "type": "timer",
                    "action": "set_timer",
                    "duration": time_value,
                    "unit": time_unit,
                    "response": f"Timer set for {time_value} {time_unit}{'s' if time_value != 1 else ''}."
                }
        
        # For other commands, use the general assistant
        response_data = await recipe_assistant.process_message(
            message=command,
            recipe_id=recipe_id,
            user_id=user_data["user_id"],
            db=db
        )
        
        # Optimize response for voice - keep it concise
        response_text = response_data["response"]
        if len(response_text) > 200:
            # Truncate for voice but keep important info
            response_text = response_text[:200] + "... Ask me for more details if needed."
        
        return {
            "type": response_data["type"],
            "response": response_text,
            "original_length": len(response_data["response"]),
            "data": response_data.get("data"),
            "optimized_for_voice": True
        }
        
    except Exception as e:
        print(f"Voice command error: {e}")
        return {
            "type": "error",
            "response": "I didn't catch that. Could you try again?",
            "error": str(e)
        }

@app.get("/assistant/step/{recipe_id}/{step_number}")
async def get_recipe_step(
    recipe_id: str,
    step_number: int,
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get a specific recipe step with context.
    Useful for voice navigation and follow mode.
    """
    try:
        recipe_data = await recipe_assistant.get_recipe_context(recipe_id, user_data["user_id"], db)
        instructions = recipe_data.get("instructions", [])
        
        # Convert to 0-indexed
        step_index = step_number - 1
        
        if step_index < 0 or step_index >= len(instructions):
            raise HTTPException(status_code=404, detail="Step not found")
        
        # Get prep time context if available
        prep_context = ""
        if step_index == 0 and recipe_data.get("prep_time"):
            prep_context = f"Prep time: {recipe_data['prep_time']}. "
        
        # Get cook time context for cooking steps
        cook_context = ""
        if "cook" in instructions[step_index].lower() and recipe_data.get("cook_time"):
            cook_context = f"Total cook time: {recipe_data['cook_time']}. "
        
        return {
            "recipe_id": recipe_id,
            "recipe_title": recipe_data.get("title"),
            "step_number": step_number,
            "step_content": instructions[step_index],
            "context": prep_context + cook_context,
            "total_steps": len(instructions),
            "has_previous": step_index > 0,
            "has_next": step_index < len(instructions) - 1,
            "progress_percentage": round((step_number / len(instructions)) * 100)
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Get step error: {e}")
        raise HTTPException(status_code=500, detail="Could not get recipe step")

@app.get("/assistant/nutrition/{recipe_id}")
async def get_nutrition_analysis(
    recipe_id: str,
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get detailed nutritional analysis for a recipe.
    Includes calories, macronutrients, allergens, and health score.
    """
    try:
        recipe_data = await recipe_assistant.get_recipe_context(recipe_id, user_data["user_id"], db)
        nutrition_data = await recipe_assistant.analyze_nutrition(recipe_data)
        
        return {
            "recipe_id": recipe_id,
            "recipe_title": recipe_data.get("title"),
            "servings": recipe_data.get("servings"),
            "nutrition": nutrition_data,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Nutrition analysis error: {e}")
        raise HTTPException(status_code=500, detail="Could not analyze nutrition")

@app.get("/assistant/pairings/{recipe_id}")
async def get_wine_pairings(
    recipe_id: str,
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get wine and beverage pairing suggestions for a recipe.
    Tailored to cuisine type and flavor profile.
    """
    try:
        recipe_data = await recipe_assistant.get_recipe_context(recipe_id, user_data["user_id"], db)
        pairing_data = await recipe_assistant.suggest_wine_pairing(recipe_data)
        
        return {
            "recipe_id": recipe_id,
            "recipe_title": recipe_data.get("title"),
            "cuisine_type": recipe_data.get("cuisine_type"),
            "meal_type": recipe_data.get("meal_type"),
            "pairings": pairing_data,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Wine pairing error: {e}")
        raise HTTPException(status_code=500, detail="Could not generate pairings")

@app.post("/assistant/scale")
async def scale_recipe(
    recipe_id: str = Body(..., embed=True),
    target_servings: int = Body(..., embed=True),
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Scale a recipe to different serving sizes.
    Returns proportionally adjusted ingredient quantities.
    """
    try:
        recipe_data = await recipe_assistant.get_recipe_context(recipe_id, user_data["user_id"], db)
        current_servings = recipe_data.get("servings", 4)
        
        scaled_ingredients = recipe_assistant.scale_ingredients(
            recipe_data.get("ingredients", []),
            current_servings,
            target_servings
        )
        
        return {
            "recipe_id": recipe_id,
            "recipe_title": recipe_data.get("title"),
            "scaling": {
                "original_servings": current_servings,
                "target_servings": target_servings,
                "scale_factor": target_servings / current_servings
            },
            "scaled_ingredients": scaled_ingredients,
            "original_ingredients": recipe_data.get("ingredients", []),
            "instructions": recipe_data.get("instructions", [])  # Instructions stay the same
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Recipe scaling error: {e}")
        raise HTTPException(status_code=500, detail="Could not scale recipe")

@app.get("/assistant/shopping-list/{recipe_id}")
async def generate_shopping_list(
    recipe_id: str,
    servings: Optional[int] = None,
    user_data: dict = Depends(verify_supabase_jwt),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Generate an organized shopping list for a recipe.
    Optionally scale to different serving sizes.
    """
    try:
        recipe_data = await recipe_assistant.get_recipe_context(recipe_id, user_data["user_id"], db)
        
        ingredients = recipe_data.get("ingredients", [])
        
        # Scale ingredients if different serving size requested
        if servings and servings != recipe_data.get("servings"):
            ingredients = recipe_assistant.scale_ingredients(
                ingredients,
                recipe_data.get("servings", 4),
                servings
            )
        
        shopping_list = recipe_assistant.generate_shopping_list(ingredients, servings)
        
        return {
            "recipe_id": recipe_id,
            "recipe_title": recipe_data.get("title"),
            "servings": servings or recipe_data.get("servings"),
            "shopping_list": shopping_list,
            "total_items": sum(len(items) for items in shopping_list.values()),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Shopping list error: {e}")
        raise HTTPException(status_code=500, detail="Could not generate shopping list")
