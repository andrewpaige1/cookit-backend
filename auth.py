# auth.py - Supabase JWT Authentication
import os
import jwt
from jwt.api_jwk import PyJWK
import httpx
import json
from fastapi import HTTPException, Depends, Header
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_JWKS_URL = os.environ.get("SUPABASE_JWKS_URL")

if not SUPABASE_URL or not SUPABASE_JWKS_URL:
    print("⚠️  Warning: Supabase configuration not found in .env file")
    print("   Add SUPABASE_URL and SUPABASE_JWKS_URL to enable authentication")

class AuthManager:
    def __init__(self):
        self.supabase_url = SUPABASE_URL
        self.jwks_url = SUPABASE_JWKS_URL
        self.jwks_cache = None
        self.jwks_last_fetch = 0
        self.jwks_cache_ttl = 600  # 10 minutes cache TTL
        
    async def fetch_jwks(self) -> Dict[str, Any]:
        """Fetch JSON Web Key Set from Supabase"""
        now = datetime.now().timestamp()
        
        # Return cached JWKS if not expired
        if self.jwks_cache and now - self.jwks_last_fetch < self.jwks_cache_ttl:
            return self.jwks_cache
        
        if not self.jwks_url:
            print("❌ AUTH DEBUG: JWKS URL is not configured")
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: JWKS URL is not set"
            )
            
        print(f"🔍 AUTH DEBUG: Fetching JWKS from {self.jwks_url}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(str(self.jwks_url))
                if response.status_code != 200:
                    print(f"❌ AUTH DEBUG: Failed to fetch JWKS: HTTP {response.status_code}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to fetch JWKS: HTTP {response.status_code}"
                    )
                    
                jwks = response.json()
                self.jwks_cache = jwks
                self.jwks_last_fetch = now
                print(f"✅ AUTH DEBUG: JWKS fetched successfully")
                return jwks
        except Exception as e:
            print(f"❌ AUTH DEBUG: Error fetching JWKS: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching JWKS: {str(e)}"
            )
    
    def get_public_key_for_token(self, token: str, jwks: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key ID from token header and find matching public key in JWKS"""
        try:
            # Decode header without verification
            header = jwt.get_unverified_header(token)
            if 'kid' not in header:
                print("❌ AUTH DEBUG: No 'kid' in token header")
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token: No 'kid' in header"
                )
                
            kid = header['kid']
            print(f"🔍 AUTH DEBUG: Token key ID (kid): {kid}")
            
            # Find matching key in JWKS
            for key in jwks.get('keys', []):
                if key.get('kid') == kid:
                    return key
                    
            print(f"❌ AUTH DEBUG: No matching key found for kid {kid}")
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token: No matching key found for kid {kid}"
            )
        except Exception as e:
            print(f"❌ AUTH DEBUG: Error extracting key ID: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail=f"Error processing token: {str(e)}"
            )
        
    async def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify Supabase JWT token and extract user data"""
        if not self.jwks_url:
            print("❌ AUTH DEBUG: JWKS URL not configured")
            raise HTTPException(
                status_code=500, 
                detail="Server configuration error: Supabase JWKS URL not configured. Contact administrator."
            )
        
        print(f"🔍 AUTH DEBUG: Starting JWT verification")
        print(f"🔍 AUTH DEBUG: Token length: {len(token)}")
            
        try:
            # Fetch JWKS
            jwks = await self.fetch_jwks()
            
            # Get public key for token
            public_key = self.get_public_key_for_token(token, jwks)
            
            # Prepare key for PyJWT
            jwk = PyJWK.from_dict(public_key)
            
            print("🔍 AUTH DEBUG: Attempting JWT decode...")
            payload = jwt.decode(
                token, 
                jwk.key, 
                algorithms=["ES256"],  # Supabase now uses ES256 for JWT
                audience="authenticated"  # Supabase audience
            )
            
            print("✅ AUTH DEBUG: JWT decode successful")
            print(f"🔍 AUTH DEBUG: Payload keys: {list(payload.keys())}")
            
            # Extract user information
            user_data = {
                "user_id": payload.get("sub"),  # Supabase user ID
                "email": payload.get("email"),
                "role": payload.get("role", "authenticated"),
                "exp": payload.get("exp"),  # Expiration time
                "iat": payload.get("iat"),  # Issued at time
            }
            
            print(f"✅ AUTH DEBUG: User data extracted: {user_data}")
            
            # Check if token is expired with proper timezone handling and clock skew tolerance
            if user_data["exp"]:
                current_timestamp = datetime.now(timezone.utc).timestamp()
                token_exp_timestamp = user_data["exp"]
                
                # Add 30 second buffer for clock skew between client and server
                clock_skew_buffer = 30
                effective_exp_time = token_exp_timestamp - clock_skew_buffer
                
                print(f"🔍 AUTH DEBUG: Timestamp comparison:")
                print(f"   - Current UTC timestamp: {current_timestamp}")
                print(f"   - Token exp timestamp: {token_exp_timestamp}")
                print(f"   - Effective exp (with buffer): {effective_exp_time}")
                print(f"   - Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"   - Token exp time: {datetime.fromtimestamp(token_exp_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"   - Time difference: {token_exp_timestamp - current_timestamp:.2f} seconds")
                
                if current_timestamp > effective_exp_time:
                    if current_timestamp > token_exp_timestamp:
                        # Token is actually expired
                        exp_date = datetime.fromtimestamp(token_exp_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                        print(f"❌ AUTH DEBUG: Token genuinely expired on {exp_date}")
                        raise HTTPException(
                            status_code=401, 
                            detail=f"Authentication failed: Token expired on {exp_date}. Please sign in again to get a new token."
                        )
                    else:
                        # Within buffer zone - accept but warn
                        print(f"⚠️ AUTH DEBUG: Token expires soon (within {clock_skew_buffer}s buffer) but accepting")
                else:
                    print(f"✅ AUTH DEBUG: Token is still valid for {(token_exp_timestamp - current_timestamp)/60:.1f} minutes")
            
            print("✅ AUTH DEBUG: Token verification successful")
            return user_data
            
        except jwt.ExpiredSignatureError:
            print("❌ AUTH DEBUG: JWT ExpiredSignatureError")
            raise HTTPException(
                status_code=401, 
                detail="Authentication failed: JWT token has expired. Please sign in again to get a fresh token."
            )
        except jwt.InvalidSignatureError:
            print("❌ AUTH DEBUG: JWT InvalidSignatureError")
            raise HTTPException(
                status_code=401, 
                detail="Authentication failed: Invalid token signature. This token may have been tampered with or is from a different system."
            )
        except jwt.InvalidAudienceError:
            print("❌ AUTH DEBUG: JWT InvalidAudienceError")
            raise HTTPException(
                status_code=401, 
                detail="Authentication failed: Token audience mismatch. This token is not intended for this service."
            )
        except jwt.InvalidIssuerError:
            print("❌ AUTH DEBUG: JWT InvalidIssuerError")
            raise HTTPException(
                status_code=401, 
                detail="Authentication failed: Token issuer mismatch. This token was not issued by the expected Supabase instance."
            )
        except jwt.DecodeError as e:
            print(f"❌ AUTH DEBUG: JWT DecodeError: {e}")
            raise HTTPException(
                status_code=401, 
                detail="Authentication failed: Malformed JWT token. The token structure is invalid or corrupted."
            )
        except jwt.InvalidTokenError as e:
            print(f"❌ AUTH DEBUG: JWT InvalidTokenError: {e}")
            raise HTTPException(
                status_code=401, 
                detail=f"Authentication failed: Invalid JWT token - {str(e)}. Please ensure you're using a valid Supabase access token."
            )
        except Exception as e:
            print(f"❌ AUTH DEBUG: Unexpected error: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=401, 
                detail=f"Authentication failed: Unexpected error during token verification - {str(e)}. Please try signing in again."
            )

# Global auth manager instance
auth_manager = AuthManager()

async def verify_supabase_jwt(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """FastAPI dependency to verify Supabase JWT from Authorization header"""
    
    print(f"🔍 AUTH DEBUG: Received authorization header: {authorization}")
    
    if not authorization:
        print("❌ AUTH DEBUG: No authorization header provided")
        raise HTTPException(
            status_code=401, 
            detail="Authentication required: Missing Authorization header. Please include 'Authorization: Bearer <your_jwt_token>' in your request headers."
        )
    
    if not authorization.startswith("Bearer "):
        print(f"❌ AUTH DEBUG: Invalid header format. Header: '{authorization}'")
        raise HTTPException(
            status_code=401, 
            detail="Authentication failed: Invalid Authorization header format. Expected format: 'Authorization: Bearer <your_jwt_token>'. Current format appears to be missing 'Bearer ' prefix."
        )
    
    # Extract token from "Bearer <token>"
    token_parts = authorization.split(" ", 1)
    if len(token_parts) != 2:
        print(f"❌ AUTH DEBUG: Malformed header. Parts: {token_parts}")
        raise HTTPException(
            status_code=401, 
            detail="Authentication failed: Malformed Authorization header. Expected format: 'Authorization: Bearer <your_jwt_token>'."
        )
    
    token = token_parts[1]
    
    if not token or token.strip() == "":
        print("❌ AUTH DEBUG: Empty token after Bearer")
        raise HTTPException(
            status_code=401, 
            detail="Authentication failed: Empty JWT token provided. Please ensure your Authorization header includes a valid Supabase access token after 'Bearer '."
        )
    
        print(f"✅ AUTH DEBUG: Extracted token (length: {len(token)})")
    print(f"🔍 AUTH DEBUG: Token preview: {token[:50]}...")
    print(f"🔍 AUTH DEBUG: JWKS URL configured: {bool(auth_manager.jwks_url)}")
    
    # Verify token and return user data
    return await auth_manager.verify_jwt_token(token)

async def optional_auth(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Optional authentication - returns None if no auth provided, validates if provided"""
    
    if not authorization:
        return None
    
    try:
        return await verify_supabase_jwt(authorization)
    except HTTPException as e:
        # For optional auth, we could log the error but not raise it
        # This allows endpoints to handle unauthenticated users gracefully
        print(f"Optional authentication failed: {e.detail}")
        return None

# Helper function to check if authentication is configured
def is_auth_configured() -> bool:
    """Check if Supabase authentication is properly configured"""
    return bool(SUPABASE_URL and SUPABASE_JWKS_URL)
