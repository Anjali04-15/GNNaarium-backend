from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse
from datetime import datetime
import os
import json
from app.database import get_user_by_email, create_user, create_jwt_token
from urllib.parse import urlencode
import base64

router = APIRouter()

@router.get("/auth/test")
async def test_oauth_config(request: Request):
    """Test OAuth configuration"""
    
    redirect_uri = request.url_for("google_callback")

    # Manual OAuth URL for testing
    params = {
        'client_id': os.getenv('GOOGLE_CLIENT_ID'),
        'redirect_uri': redirect_uri,
        'response_type': 'code',
        'scope': 'openid email profile',
        'access_type': 'offline'
    }
    
    manual_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    
    return {
        "client_id": os.getenv('GOOGLE_CLIENT_ID'),
        "client_secret_configured": bool(os.getenv('GOOGLE_CLIENT_SECRET')),
        "redirect_uri": redirect_uri,
        "manual_oauth_url": manual_url
    }

@router.get("/auth/google/login")
async def google_login(request: Request, redirect_to: str = None):
    """Redirect to Google OAuth login"""
    try:
        # Check if Google OAuth is configured
        client_id = os.getenv('GOOGLE_CLIENT_ID')
        if not client_id or not os.getenv('GOOGLE_CLIENT_SECRET'):
            raise HTTPException(status_code=500, detail="Google OAuth not configured")

        redirect_uri = request.url_for("google_callback")
        
        # Encode redirect_to in state parameter
        state = base64.b64encode((redirect_to or '/').encode()).decode() if redirect_to else None
        
        params = {
            'client_id': client_id,
            'redirect_uri': redirect_uri,
            'response_type': 'code',
            'scope': 'openid email profile',
            'access_type': 'offline'
        }
        
        if state:
            params['state'] = state
        
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
        return RedirectResponse(url=auth_url)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OAuth error: {str(e)}")

@router.get("/auth/google/callback", name="google_callback")
async def google_callback(request: Request):
    FRONTEND_FALLBACK = "http://localhost:3000"
    redirect_to = FRONTEND_FALLBACK
    try:
        if 'error' in request.query_params:
            raise HTTPException(
                status_code=400,
                detail=request.query_params.get('error_description', 'Google OAuth error')
            )

        code = request.query_params.get('code')
        if not code:
            raise HTTPException(status_code=400, detail="No authorization code received")

        redirect_uri = request.url_for("google_callback")

        import httpx
        token_data = {
            'code': code,
            'client_id': os.getenv('GOOGLE_CLIENT_ID'),
            'client_secret': os.getenv('GOOGLE_CLIENT_SECRET'),
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code'
        }

        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                'https://oauth2.googleapis.com/token',
                data=token_data
            )
            token_json = token_response.json()

            if 'access_token' not in token_json:
                raise HTTPException(status_code=400, detail="Failed to get access token")

            user_response = await client.get(
                'https://www.googleapis.com/oauth2/v2/userinfo',
                headers={'Authorization': f'Bearer {token_json["access_token"]}'}
            )
            user_info = user_response.json()

        if not user_info or not user_info.get('email'):
            raise HTTPException(status_code=400, detail="Failed to get user info")

        existing_user = await get_user_by_email(user_info['email'])

        if existing_user:
            jwt_token = create_jwt_token(existing_user)
            user_data = {
                "email": existing_user["email"],
                "name": existing_user["name"],
                "profile_pic": existing_user.get("profile_pic")
            }
        else:
            new_user_data = {
                "email": user_info['email'],
                "name": user_info['name'],
                "google_id": user_info['id'],
                "profile_pic": user_info.get('picture'),
                "created_at": datetime.utcnow()
            }

            await create_user(new_user_data)
            jwt_token = create_jwt_token(new_user_data)
            user_data = {
                "email": new_user_data["email"],
                "name": new_user_data["name"],
                "profile_pic": new_user_data.get("profile_pic")
            }

        import base64, json
        state = request.query_params.get("state")

        if state:
            try:
                decoded = base64.b64decode(state).decode()
                if decoded.startswith("http"):
                    redirect_to = decoded
            except Exception:
                pass

        if not redirect_to or not redirect_to.startswith("http"):
            raise HTTPException(status_code=400, detail="Invalid redirect URL")

        from urllib.parse import urlencode
        redirect_url = f"{redirect_to}?{urlencode({'token': jwt_token, 'user': json.dumps(user_data)})}"

        return RedirectResponse(url=redirect_url)

    except Exception as e:
        return RedirectResponse(
            url=f"{FRONTEND_FALLBACK}?error={str(e)}"
        )

@router.post("/auth/logout")
async def logout():
    """Logout endpoint (clears session on client side)"""
    return {"status": "success", "message": "Logged out successfully"}