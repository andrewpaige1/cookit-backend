# assistant.py - AI Recipe Assistant Backend
import os
import json
import re
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from fastapi import HTTPException
import openai
from sqlalchemy.ext.asyncio import AsyncSession
from database import Recipe, get_recipe_hybrid
from dotenv import load_dotenv

load_dotenv()

class RecipeAssistant:
    """
    Contextual AI assistant for recipe cooking guidance.
    Provides intelligent responses based on specific recipe context.
    """
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    async def get_recipe_context(self, recipe_id: str, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Fetch complete recipe context from database"""
        recipe_data = await get_recipe_hybrid(recipe_id, user_id, db)
        if not recipe_data:
            raise HTTPException(status_code=404, detail="Recipe not found")
        return recipe_data
    
    def create_system_prompt(self, recipe_data: Dict[str, Any]) -> str:
        """Create contextual system prompt with full recipe details"""
        ingredients_text = "\n".join([f"- {ing}" for ing in recipe_data.get("ingredients", [])])
        instructions_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(recipe_data.get("instructions", []))])
        
        return f"""You are an expert culinary assistant helping someone cook this specific recipe:

**Recipe: {recipe_data.get('title', 'Unknown Recipe')}**
{recipe_data.get('description', '')}

**Ingredients:**
{ingredients_text}

**Instructions:**
{instructions_text}

**Recipe Details:**
- Servings: {recipe_data.get('servings', 'Unknown')}
- Prep Time: {recipe_data.get('prep_time', 'Unknown')}
- Cook Time: {recipe_data.get('cook_time', 'Unknown')}
- Difficulty: {recipe_data.get('difficulty', 'Unknown')}
- Cuisine: {recipe_data.get('cuisine_type', 'Unknown')}
- Meal Type: {recipe_data.get('meal_type', 'Unknown')}

**Your Capabilities:**
- Scale servings and adjust ingredient quantities proportionally
- Suggest ingredient substitutions based on dietary needs or availability
- Provide step-by-step guidance with timing and technique tips
- Help with cooking techniques and troubleshooting
- Generate organized shopping lists
- Recommend wine/beverage pairings
- Provide nutritional estimates and allergen warnings
- Suggest recipe variations and modifications

**Communication Style:**
- Be concise and actionable for voice interaction
- Use kitchen-friendly language (no complex measurements)
- Provide specific quantities when scaling or substituting
- Break complex answers into digestible steps
- Always reference the specific recipe context when relevant

**Current Recipe Context:** Always keep this exact recipe's ingredients, quantities, and instructions in mind when responding. When scaling or substituting, use the exact ingredient list provided above."""

    def extract_quick_command_intent(self, message: str) -> Dict[str, Any]:
        """Extract structured intent from common quick commands"""
        message_lower = message.lower().strip()
        
        # Scaling patterns
        scale_patterns = [
            r"scale.*?(\d+)\s*serv",
            r"make.*?(\d+)\s*serv", 
            r"(\d+)\s*people",
            r"double.*recipe",
            r"half.*recipe",
            r"triple.*recipe"
        ]
        
        for pattern in scale_patterns:
            match = re.search(pattern, message_lower)
            if match:
                if "double" in message_lower:
                    return {"intent": "scale", "servings": 2, "action": "multiply"}
                elif "triple" in message_lower:
                    return {"intent": "scale", "servings": 3, "action": "multiply"}
                elif "half" in message_lower:
                    return {"intent": "scale", "servings": 0.5, "action": "multiply"}
                else:
                    servings = int(match.group(1))
                    return {"intent": "scale", "servings": servings, "action": "set"}
        
        # Substitution patterns
        if any(word in message_lower for word in ["substitute", "replace", "swap", "instead of", "alternative"]):
            return {"intent": "substitute", "context": message}
        
        # Shopping list patterns
        if any(phrase in message_lower for phrase in ["shopping list", "grocery list", "what to buy"]):
            return {"intent": "shopping_list"}
        
        # Navigation patterns
        if any(phrase in message_lower for phrase in ["next step", "step", "continue", "what's next"]):
            return {"intent": "navigation", "direction": "next"}
        
        if any(phrase in message_lower for phrase in ["previous step", "back", "repeat", "again"]):
            return {"intent": "navigation", "direction": "previous"}
        
        # Wine/pairing patterns  
        if any(word in message_lower for word in ["wine", "drink", "beverage", "pairing", "what to drink"]):
            return {"intent": "pairing"}
        
        # Nutritional patterns
        if any(word in message_lower for word in ["calories", "nutrition", "healthy", "allergen", "gluten", "dairy"]):
            return {"intent": "nutrition"}
        
        # Technique/troubleshooting
        if any(word in message_lower for word in ["how to", "technique", "help", "wrong", "fix", "problem"]):
            return {"intent": "help", "context": message}
        
        return {"intent": "general", "context": message}

    def scale_ingredients(self, ingredients: List[str], current_servings: int, target_servings: int) -> List[str]:
        """Scale ingredient quantities proportionally"""
        if current_servings <= 0:
            current_servings = 4  # Default assumption
            
        scale_factor = target_servings / current_servings
        scaled_ingredients = []
        
        for ingredient in ingredients:
            # Pattern to match quantities at start of ingredient
            quantity_pattern = r'^(\d+(?:\.\d+)?(?:/\d+)?)\s*([a-zA-Z]*)\s*(.*)'
            match = re.match(quantity_pattern, ingredient.strip())
            
            if match:
                quantity_str, unit, rest = match.groups()
                
                # Handle fractions like "1/2"
                if '/' in quantity_str:
                    num, denom = quantity_str.split('/')
                    quantity = float(num) / float(denom)
                else:
                    quantity = float(quantity_str)
                
                scaled_quantity = quantity * scale_factor
                
                # Format scaled quantity nicely
                if scaled_quantity == int(scaled_quantity):
                    scaled_quantity_str = str(int(scaled_quantity))
                else:
                    # Convert to common fractions if close
                    if abs(scaled_quantity - 0.5) < 0.1:
                        scaled_quantity_str = "1/2"
                    elif abs(scaled_quantity - 0.25) < 0.1:
                        scaled_quantity_str = "1/4"
                    elif abs(scaled_quantity - 0.75) < 0.1:
                        scaled_quantity_str = "3/4"
                    elif abs(scaled_quantity - 1.5) < 0.1:
                        scaled_quantity_str = "1 1/2"
                    else:
                        scaled_quantity_str = f"{scaled_quantity:.2f}".rstrip('0').rstrip('.')
                
                scaled_ingredient = f"{scaled_quantity_str} {unit} {rest}".strip()
                scaled_ingredients.append(scaled_ingredient)
            else:
                # No quantity found, include as-is
                scaled_ingredients.append(ingredient)
        
        return scaled_ingredients

    def generate_shopping_list(self, ingredients: List[str], servings: Optional[int] = None) -> Dict[str, List[str]]:
        """Organize ingredients into shopping categories"""
        categories = {
            "Produce": [],
            "Meat & Seafood": [],
            "Dairy & Eggs": [],
            "Pantry & Dry Goods": [],
            "Frozen": [],
            "Other": []
        }
        
        # Categorization keywords
        produce_keywords = ["onion", "garlic", "tomato", "bell pepper", "carrot", "celery", "lettuce", "spinach", "herb", "lemon", "lime", "ginger", "mushroom", "potato", "apple", "avocado"]
        meat_keywords = ["chicken", "beef", "pork", "fish", "salmon", "shrimp", "turkey", "bacon", "sausage"]
        dairy_keywords = ["milk", "cheese", "butter", "cream", "yogurt", "egg", "sour cream"]
        pantry_keywords = ["oil", "vinegar", "salt", "pepper", "flour", "sugar", "rice", "pasta", "sauce", "broth", "stock", "spice", "vanilla"]
        frozen_keywords = ["frozen", "ice"]
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            categorized = False
            
            for keyword in produce_keywords:
                if keyword in ingredient_lower:
                    categories["Produce"].append(ingredient)
                    categorized = True
                    break
            
            if not categorized:
                for keyword in meat_keywords:
                    if keyword in ingredient_lower:
                        categories["Meat & Seafood"].append(ingredient)
                        categorized = True
                        break
            
            if not categorized:
                for keyword in dairy_keywords:
                    if keyword in ingredient_lower:
                        categories["Dairy & Eggs"].append(ingredient)
                        categorized = True
                        break
            
            if not categorized:
                for keyword in frozen_keywords:
                    if keyword in ingredient_lower:
                        categories["Frozen"].append(ingredient)
                        categorized = True
                        break
            
            if not categorized:
                for keyword in pantry_keywords:
                    if keyword in ingredient_lower:
                        categories["Pantry & Dry Goods"].append(ingredient)
                        categorized = True
                        break
            
            if not categorized:
                categories["Other"].append(ingredient)
        
        # Remove empty categories
        return {cat: items for cat, items in categories.items() if items}

    async def analyze_nutrition(self, recipe_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze nutritional content and allergens of a recipe"""
        ingredients = recipe_data.get("ingredients", [])
        servings = recipe_data.get("servings", 4)
        
        # Create a focused prompt for nutritional analysis
        nutrition_prompt = f"""Analyze the nutritional content of this recipe:

Title: {recipe_data.get('title')}
Servings: {servings}
Ingredients:
{chr(10).join([f'- {ing}' for ing in ingredients])}

Provide a JSON response with estimated nutritional information per serving:
{{
    "calories_per_serving": 350,
    "macronutrients": {{
        "protein_g": 25,
        "carbs_g": 30,
        "fat_g": 15,
        "fiber_g": 5
    }},
    "allergens": ["gluten", "dairy"],
    "dietary_tags": ["high-protein", "vegetarian"],
    "healthiness_score": 7,
    "health_notes": "Good source of protein and fiber. Moderate calories."
}}

Be realistic with estimates based on typical ingredient portions."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": nutrition_prompt}],
                response_format={"type": "json_object"},
                max_tokens=400,
                temperature=0.1
            )
            
            response_content = response.choices[0].message.content
            if response_content:
                nutrition_data = json.loads(response_content)
                return nutrition_data
            else:
                raise ValueError("Empty response from AI")
            
        except Exception as e:
            print(f"Nutrition analysis error: {e}")
            return {
                "error": "Could not analyze nutrition",
                "calories_per_serving": "Unknown",
                "allergens": [],
                "dietary_tags": []
            }

    async def suggest_wine_pairing(self, recipe_data: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest wine and beverage pairings for a recipe"""
        cuisine = recipe_data.get("cuisine_type", "")
        meal_type = recipe_data.get("meal_type", "")
        ingredients = recipe_data.get("ingredients", [])[:5]  # First 5 for context
        
        pairing_prompt = f"""Suggest wine and beverage pairings for this recipe:

Recipe: {recipe_data.get('title')}
Cuisine: {cuisine}
Meal Type: {meal_type}
Key Ingredients: {', '.join(ingredients)}

Provide a JSON response with pairing suggestions:
{{
    "wine_pairings": [
        {{"type": "Pinot Noir", "reason": "Complements the earthy flavors"}},
        {{"type": "Chardonnay", "reason": "Balances the richness"}}
    ],
    "non_alcoholic": [
        {{"type": "Sparkling water with lemon", "reason": "Cleanses palate"}},
        {{"type": "Herbal tea", "reason": "Aids digestion"}}
    ],
    "beer_options": [
        {{"type": "IPA", "reason": "Cuts through rich flavors"}}
    ]
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": pairing_prompt}],
                response_format={"type": "json_object"},
                max_tokens=300,
                temperature=0.3
            )
            
            response_content = response.choices[0].message.content
            if response_content:
                pairing_data = json.loads(response_content)
                return pairing_data
            else:
                raise ValueError("Empty response from AI")
            
        except Exception as e:
            print(f"Wine pairing error: {e}")
            return {
                "error": "Could not generate pairings",
                "wine_pairings": [],
                "non_alcoholic": []
            }

    async def process_message(
        self, 
        message: str, 
        recipe_id: str, 
        user_id: str, 
        db: AsyncSession,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Process user message with recipe context and return structured response"""
        
        # Get recipe context
        recipe_data = await self.get_recipe_context(recipe_id, user_id, db)
        
        # Extract intent from message
        intent_data = self.extract_quick_command_intent(message)
        
        # Handle structured intents directly
        if intent_data["intent"] == "scale":
            current_servings = recipe_data.get("servings", 4)
            if intent_data["action"] == "multiply":
                target_servings = int(current_servings * intent_data["servings"])
            else:
                target_servings = intent_data["servings"]
            
            scaled_ingredients = self.scale_ingredients(
                recipe_data.get("ingredients", []), 
                current_servings, 
                target_servings
            )
            
            return {
                "type": "scaling",
                "response": f"Here's the recipe scaled for {target_servings} servings:",
                "data": {
                    "original_servings": current_servings,
                    "new_servings": target_servings,
                    "scaled_ingredients": scaled_ingredients
                }
            }
        
        elif intent_data["intent"] == "shopping_list":
            shopping_list = self.generate_shopping_list(recipe_data.get("ingredients", []))
            
            return {
                "type": "shopping_list", 
                "response": f"Here's your shopping list for {recipe_data.get('title')}:",
                "data": shopping_list
            }
        
        # For complex intents, use LLM with recipe context
        system_prompt = self.create_system_prompt(recipe_data)
        
        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            for msg in conversation_history:
                if "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": message})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective for conversational responses
                messages=messages,  # type: ignore
                max_tokens=500,
                temperature=0.3,  # Balanced creativity for cooking advice
                stream=False
            )
            
            assistant_response = response.choices[0].message.content
            
            return {
                "type": "conversational",
                "response": assistant_response,
                "intent": intent_data["intent"],
                "recipe_context": {
                    "recipe_id": recipe_id,
                    "recipe_title": recipe_data.get("title"),
                    "servings": recipe_data.get("servings")
                }
            }
            
        except Exception as e:
            print(f"Error generating assistant response: {e}")
            return {
                "type": "error",
                "response": "I'm having trouble right now. Please try your question again.",
                "error": str(e)
            }

# Global assistant instance
recipe_assistant = RecipeAssistant()
