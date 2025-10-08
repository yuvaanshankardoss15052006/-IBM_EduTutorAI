# IBM_EduTutorAI
```python
# edututor_ai.py
# EduTutor AI — Personalized Learning with Generative AI & LMS Integration
# Single-file FastAPI prototype. Run: uvicorn edututor_ai:app --reload

import os
import uuid
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import requests
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField, SQLModel, create_engine, Session, select

# Optional: transformers for local model usage
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

####################
# Config / Settings
####################
DB_FILE = "edututor.db"
USE_LOCAL_MODEL = os.environ.get("USE_LOCAL_MODEL", "0") == "1"
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "ibm-granite/granite-3.2-2b-instruct")
WATSONX_APIKEY = os.environ.get("WATSONX_APIKEY")
WATSONX_URL = os.environ.get("WATSONX_URL")  # e.g. https://api.us-south.watsonx.ai
TOKEN_EXPIRE_MINUTES = 8 * 60  # 8 hours

####################
# DB Models
####################
class User(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    username: str = SQLField(index=True)
    full_name: Optional[str] = None
    email: Optional[str] = None
    password: str  # NOTE: store hashed passwords in production
    role: str = "student"  # student / teacher / admin
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Profile(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(index=True)
    level: str = "beginner"  # beginner / intermediate / advanced
    learning_goals: Optional[str] = None
    preferred_style: Optional[str] = "visual"  # visual / textual / hands-on
    last_activity: Optional[datetime] = None

class Lesson(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(index=True)
    title: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Quiz(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    lesson_id: int = SQLField(index=True)
    user_id: int = SQLField(index=True)
    questions_json: str  # JSON list of questions
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Progress(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(index=True)
    lesson_id: Optional[int]
    score: Optional[float]
    completed: bool = False
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Token store (very simple)
class TokenStore(SQLModel, table=True):
    token: str = SQLField(primary_key=True)
    user_id: int = SQLField(index=True)
    expires_at: datetime

####################
# Pydantic Schemas
####################
class RegisterReq(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None
    email: Optional[str] = None

class LoginReq(BaseModel):
    username: str
    password: str

class ProfileReq(BaseModel):
    level: Optional[str]
    learning_goals: Optional[str]
    preferred_style: Optional[str]

class GenerateLessonReq(BaseModel):
    title: Optional[str]
    topic: str
    length_minutes: Optional[int] = 15

class GenerateQuizReq(BaseModel):
    lesson_id: Optional[int]
    topic: Optional[str]
    num_questions: Optional[int] = 5
    difficulty: Optional[str] = "medium"  # easy/medium/hard

class SubmitQuizReq(BaseModel):
    quiz_id: int
    answers: List[Any]  # user answers - server will grade if possible

####################
# Engine / DB Init
####################
engine = create_engine(f"sqlite:///{DB_FILE}")
SQLModel.metadata.create_all(engine)

####################
# App
####################
app = FastAPI(title="EduTutor AI — NM2025TMIDO")

####################
# Helper: Auth
####################
def create_token(user_id: int) -> str:
    token = str(uuid.uuid4())
    expires = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    with Session(engine) as s:
        ts = TokenStore(token=token, user_id=user_id, expires_at=expires)
        s.add(ts)
        s.commit()
    return token

def get_user_by_token(authorization: Optional[str] = Header(None)) -> User:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.replace("Bearer ", "")
    with Session(engine) as s:
        res = s.get(TokenStore, token)
        if not res or res.expires_at < datetime.utcnow():
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        user = s.get(User, res.user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user

####################
# Helper: Generative Client (model-agnostic)
####################
# If USE_LOCAL_MODEL==1 and transformers available -> use local HF model
_local_tokenizer = None
_local_model = None
if USE_LOCAL_MODEL and TRANSFORMERS_AVAILABLE:
    _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
    _local_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    if _local_tokenizer.pad_token is None:
        _local_tokenizer.pad_token = _local_tokenizer.eos_token

def generate_text(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    # Local model path
    if USE_LOCAL_MODEL and _local_model is not None:
        inputs = _local_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = {k: v.to(_local_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outs = _local_model.generate(**inputs, max_length=max_tokens, temperature=temperature, do_sample=True, pad_token_id=_local_tokenizer.eos_token_id)
        text = _local_tokenizer.decode(outs[0], skip_special_tokens=True)
        return text.replace(prompt, "").strip()
    # IBM watsonx API path
    if WATSONX_APIKEY and WATSONX_URL:
        headers = {"Authorization": f"Bearer {WATSONX_APIKEY}", "Content-Type": "application/json"}
        # This path may need adjustment depending on watsonx API version; using a generic /v1/generate
        payload = {"model": "ibm/granite-3.2-2b-instruct", "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        resp = requests.post(f"{WATSONX_URL.rstrip('/')}/v1/generate", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # try common response shapes
        if "generated_text" in data:
            return data["generated_text"]
        if isinstance(data.get("outputs"), list):
            return " ".join(o.get("text", "") for o in data["outputs"])
        if "content" in data:
            return data["content"]
        return json.dumps(data)
    # Fallback: simple echo / placeholder
    return f"[GENERATOR_UNAVAILABLE] Could not generate. Prompt was: {prompt[:200]}"

####################
# Endpoints: Auth & Users
####################
@app.post("/register")
def register(info: RegisterReq):
    with Session(engine) as s:
        q = s.exec(select(User).where(User.username == info.username)).first()
        if q:
            raise HTTPException(status_code=400, detail="Username already exists")
        user = User(username=info.username, password=info.password, full_name=info.full_name, email=info.email)
        s.add(user)
        s.commit()
        s.refresh(user)
        # create profile
        profile = Profile(user_id=user.id)
        s.add(profile)
        s.commit()
    token = create_token(user.id)
    return {"token": token, "user_id": user.id, "username": user.username}

@app.post("/login")
def login(creds: LoginReq):
    with Session(engine) as s:
        user = s.exec(select(User).where(User.username == creds.username, User.password == creds.password)).first()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user.id)
    return {"token": token, "user_id": user.id, "username": user.username}

@app.get("/me")
def me(user: User = Depends(get_user_by_token)):
    return {"id": user.id, "username": user.username, "full_name": user.full_name, "email": user.email, "role": user.role}

####################
# Profile endpoints
####################
@app.get("/profile")
def get_profile(user: User = Depends(get_user_by_token)):
    with Session(engine) as s:
        profile = s.exec(select(Profile).where(Profile.user_id == user.id)).first()
        return profile

@app.post("/profile")
def update_profile(cfg: ProfileReq, user: User = Depends(get_user_by_token)):
    with Session(engine) as s:
        profile = s.exec(select(Profile).where(Profile.user_id == user.id)).first()
        if not profile:
            profile = Profile(user_id=user.id)
            s.add(profile)
        if cfg.level:
            profile.level = cfg.level
        if cfg.learning_goals:
            profile.learning_goals = cfg.learning_goals
        if cfg.preferred_style:
            profile.preferred_style = cfg.preferred_style
        profile.last_activity = datetime.utcnow()
        s.add(profile)
        s.commit()
        s.refresh(profile)
        return profile

####################
# Generate Lesson
####################
@app.post("/generate_lesson")
def generate_lesson(req: GenerateLessonReq, user: User = Depends(get_user_by_token)):
    # Compose prompt using profile
    with Session(engine) as s:
        profile = s.exec(select(Profile).where(Profile.user_id == user.id)).first()
    level = profile.level if profile else "beginner"
    style = profile.preferred_style if profile else "textual"
    goals = profile.learning_goals or ""
    title = req.title or f"Lesson: {req.topic}"
    prompt = (
        f"Create a personalized lesson for a {level} learner with preferred style '{style}'.\n"
        f"Topic: {req.topic}\nLearning goals: {goals}\nLength (minutes): {req.length_minutes}\n\n"
        "Structure the lesson with: Learning objectives, short introduction, 3-6 step-by-step explanations, examples, quick practice tasks, and suggested resources.\n\nLesson:\n"
    )
    content = generate_text(prompt, max_tokens=700, temperature=0.6)
    with Session(engine) as s:
        lesson = Lesson(user_id=user.id, title=title, content=content)
        s.add(lesson)
        s.commit()
        s.refresh(lesson)
    return {"lesson_id": lesson.id, "title": lesson.title, "content": lesson.content}

####################
# Generate Quiz
####################
@app.post("/generate_quiz")
def generate_quiz(req: GenerateQuizReq, user: User = Depends(get_user_by_token)):
    # If lesson_id provided, try to use lesson content as context
    context = ""
    if req.lesson_id:
        with Session(engine) as s:
            lesson = s.get(Lesson, req.lesson_id)
            if lesson:
                context = lesson.content
    prompt = (
        f"Create {req.num_questions} quiz questions (question, 4 options, correct option index) for difficulty {req.difficulty}.\n"
        f"Context/Topic: {req.topic or 'Use lesson content'}\n\n{context}\n\nReturn JSON array of objects like {{'q':'...','opts':['a','b','c','d'],'answer':1}}"
    )
    raw = generate_text(prompt, max_tokens=600, temperature=0.3)
    # Try to extract JSON from the response
    try:
        # some models include markdown or text. Find first '[' and last ']'
        start = raw.find('[')
        end = raw.rfind(']') + 1
        if start != -1 and end != -1:
            json_text = raw[start:end]
        else:
            json_text = raw
        questions = json.loads(json_text)
    except Exception:
        # Fallback: simple heuristic: create empty questions stub
        questions = [{"q": f"Sample question about {req.topic} #{i+1}", "opts": ["A", "B", "C", "D"], "answer": 0} for i in range(req.num_questions)]
    # store quiz
    with Session(engine) as s:
        quiz = Quiz(lesson_id=req.lesson_id or 0, user_id=user.id, questions_json=json.dumps(questions))
        s.add(quiz)
        s.commit()
        s.refresh(quiz)
    return {"quiz_id": quiz.id, "questions": questions}

####################
# Submit & Grade Quiz
####################
@app.post("/submit_quiz")
def submit_quiz(payload: SubmitQuizReq, user: User = Depends(get_user_by_token)):
    with Session(engine) as s:
        quiz = s.get(Quiz, payload.quiz_id)
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")
        questions = json.loads(quiz.questions_json)
        # Simple auto-grader for multiple choice
        correct_count = 0
        for i, q in enumerate(questions):
            try:
                correct = int(q.get("answer", 0))
                user_ans = payload.answers[i]
                if int(user_ans) == correct:
                    correct_count += 1
            except Exception:
                pass
        score = (correct_count / max(1, len(questions))) * 100.0
        progress = Progress(user_id=user.id, lesson_id=quiz.lesson_id, score=score, completed=True, updated_at=datetime.utcnow())
        s.add(progress)
        s.commit()
        s.refresh(progress)
    return {"score": score, "correct": correct_count, "total": len(questions)}

####################
# Progress & History
####################
@app.get("/progress")
def get_progress(user: User = Depends(get_user_by_token)):
    with Session(engine) as s:
        items = s.exec(select(Progress).where(Progress.user_id == user.id)).all()
        return items

@app.get("/lessons")
def list_lessons(user: User = Depends(get_user_by_token)):
    with Session(engine) as s:
        lessons = s.exec(select(Lesson).where(Lesson.user_id == user.id)).all()
        return lessons

####################
# LMS Integration Endpoint (webhook / sync)
####################
@app.post("/lms_sync")
async def lms_sync(request: Request, x_lms_key: Optional[str] = Header(None)):
    """
    Simple endpoint to receive LMS updates or to push content from EduTutor to LMS.
    Expect JSON body: { "action": "sync_lesson", "lesson_id": 1, "lms_endpoint": "https://lms.example/api/..." }
    If lms_endpoint provided, EduTutor will POST lesson content to that endpoint (simulate integration).
    """
    payload = await request.json()
    action = payload.get("action")
    if action == "sync_lesson":
        lesson_id = payload.get("lesson_id")
        lms_endpoint = payload.get("lms_endpoint")
        with Session(engine) as s:
            lesson = s.get(Lesson, lesson_id)
            if not lesson:
                raise HTTPException(status_code=404, detail="Lesson not found")
            # POST to LMS endpoint if provided
            if lms_endpoint:
                try:
                    resp = requests.post(lms_endpoint, json={"title": lesson.title, "content": lesson.content, "meta": {"source": "EduTutorAI", "lesson_id": lesson.id}}, timeout=10)
                    return {"status": "synced", "lms_status": resp.status_code, "lms_response": resp.text}
                except Exception as e:
                    return {"status": "failed", "reason": str(e)}
            return {"status": "ok", "lesson": {"id": lesson.id, "title": lesson.title}}
    return {"status": "no_action", "received": payload}

####################
# Admin: List Users (simple)
####################
@app.get("/admin/users")
def admin_list_users(user: User = Depends(get_user_by_token)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="admin only")
    with Session(engine) as s:
        users = s.exec(select(User)).all()
        return users

####################
# Utilities: seed an admin user if none exists
####################
def seed_admin():
    with Session(engine) as s:
        admin = s.exec(select(User).where(User.role == "admin")).first()
        if not admin:
            a = User(username="admin", password="adminpass", full_name="Administrator", email="admin@example.com", role="admin")
            s.add(a)
            s.commit()
            print("Seeded admin user: admin / adminpass")

if __name__ == "__main__":
    seed_admin()
    import uvicorn
    uvicorn.run("edututor_ai:app", host="0.0.0.0", port=8000, reload=True)
```
