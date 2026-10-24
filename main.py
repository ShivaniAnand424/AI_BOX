from pathlib import Path
import os
import sqlite3
import hashlib
import hmac
import secrets
import json
from urllib.parse import parse_qs

import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR / "ml"
MODEL_FILE = ML_DIR / "adherence_model.joblib"
ISOLATION_MODEL_FILE = ML_DIR / "isolation_forest.joblib"
DATABASE_FILE = BASE_DIR / "smartmedbox_users.db"

SESSION_SECRET = os.environ.get("SESSION_SECRET")
FIREBASE_PROJECT_ID = os.environ.get(
    "FIREBASE_PROJECT_ID",
    "medication-schedule-manager",
)

if not SESSION_SECRET:
    # Local development fallback. Set SESSION_SECRET in production.
    SESSION_SECRET = secrets.token_urlsafe(32)

SESSION_SECRET = SESSION_SECRET.encode("utf-8")

app = FastAPI(title="Smart MedBox API")

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# --------------------------------------------------
# FIREBASE ADMIN INITIALIZATION
# --------------------------------------------------

firebase_app = None
firebase_init_error = None

try:
    try:
        firebase_app = firebase_admin.get_app()
    except ValueError:
        credential_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

        if credential_path:
            service_account_file = Path(credential_path).expanduser()

            if not service_account_file.is_file():
                raise FileNotFoundError(
                    "Firebase service-account JSON was not found at "
                    f"{service_account_file}"
                )

            firebase_credential = credentials.Certificate(
                str(service_account_file)
            )
        else:
            # Allows Google Application Default Credentials when configured.
            firebase_credential = credentials.ApplicationDefault()
            firebase_credential.get_credential()

        firebase_app = firebase_admin.initialize_app(
            firebase_credential,
            {"projectId": FIREBASE_PROJECT_ID},
        )

except Exception as exc:
    firebase_init_error = str(exc)
    print("WARNING: Firebase Admin is not configured:", firebase_init_error)
    print("Firebase initialization error:", firebase_init_error)
    print("Firebase app initialized:", firebase_app is not None)
    print("Firebase credential path:", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))


def require_firebase():
    if firebase_app is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Firebase Admin is not configured. Set "
                "GOOGLE_APPLICATION_CREDENTIALS to the Firebase service "
                "account JSON file path and restart the server."
            ),
        )


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def get_db():
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


initialize_database()


# --------------------------------------------------
# PASSWORD HELPERS FOR LEGACY SQLITE ACCOUNTS
# --------------------------------------------------

def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        600_000,
    ).hex()


def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    calculated_hash = hash_password(password, salt)
    return hmac.compare_digest(calculated_hash, stored_hash)


# --------------------------------------------------
# SIGNED SESSION COOKIE
# --------------------------------------------------

def create_session_token(email: str) -> str:
    signature = hmac.new(
        SESSION_SECRET,
        email.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return f"{email}|{signature}"


def get_session_email(request: Request):
    token = request.cookies.get("smartmedbox_session")

    if not token or "|" not in token:
        return None

    email, supplied_signature = token.rsplit("|", 1)

    expected_signature = hmac.new(
        SESSION_SECRET,
        email.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(supplied_signature, expected_signature):
        return None

    with get_db() as connection:
        user = connection.execute(
            "SELECT email FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    return user["email"] if user else None


def get_current_user(request: Request):
    email = get_session_email(request)

    if not email:
        return None

    with get_db() as connection:
        user = connection.execute(
            "SELECT id, full_name, email FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    return dict(user) if user else None


def set_session_cookie(response, email: str):
    response.set_cookie(
        key="smartmedbox_session",
        value=create_session_token(email),
        httponly=True,
        secure=os.environ.get("COOKIE_SECURE", "false").lower() == "true",
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
        path="/",
    )
    # --------------------------------------------------
# FIREBASE SESSION ENDPOINT
# --------------------------------------------------

class FirebaseSessionInput(BaseModel):
    idToken: str = Field(min_length=1)


@app.post("/auth/session")
def create_firebase_session(data: FirebaseSessionInput):
    require_firebase()

    try:
        decoded_token = auth.verify_id_token(
            data.idToken,
            app=firebase_app,
            check_revoked=True,
            clock_skew_seconds=5,
        )

        email = decoded_token.get("email", "").strip().lower()
        uid = decoded_token.get("uid", "")

        if not email or not uid:
            raise HTTPException(
                status_code=401,
                detail="The Firebase token has no verified email or user ID.",
            )

        firebase_user = auth.get_user(uid, app=firebase_app)

        full_name = (
            firebase_user.display_name or email.split("@")[0]
        ).strip()

        # Firebase authenticates the user.
        # Random placeholders satisfy the existing SQLite schema.
        random_salt = secrets.token_hex(16)
        random_hash = hash_password(
            secrets.token_urlsafe(32),
            random_salt,
        )

        with get_db() as connection:
            existing = connection.execute(
                "SELECT id FROM users WHERE email = ?",
                (email,),
            ).fetchone()

            if existing:
                connection.execute(
                    "UPDATE users SET full_name = ? WHERE email = ?",
                    (full_name, email),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO users (
                        full_name,
                        email,
                        password_hash,
                        password_salt
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (full_name, email, random_hash, random_salt),
                )

        response = JSONResponse(
            {
                "success": True,
                "message": "Session created successfully.",
                "redirect": "/",
            }
        )

        set_session_cookie(response, email)
        return response

    except HTTPException:
        raise

    except Exception as exc:
        print("Firebase session error:", repr(exc))

        raise HTTPException(
            status_code=401,
            detail=(
                "Firebase token verification failed. Check your Firebase "
                "project, service-account credentials, and ID token."
            ),
        )


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

def logout_response():
    response = RedirectResponse(
        url="/login",
        status_code=303,
    )

    response.delete_cookie(
        key="smartmedbox_session",
        path="/",
        httponly=True,
        samesite="lax",
    )

    return response


@app.get("/logout")
def logout():
    return logout_response()


@app.post("/logout")
def logout_post():
    return logout_response()


# --------------------------------------------------
# TEMPLATE HELPERS
# --------------------------------------------------

def render_template(
    request: Request,
    template_name: str,
    context: dict,
):
    template_context = dict(context)
    template_context["request"] = request

    try:
        return templates.TemplateResponse(
            template_name,
            template_context,
        )

    except TypeError:
        return templates.TemplateResponse(
            request,
            template_name,
            template_context,
        )


def redirect_to_login():
    return RedirectResponse(
        url="/login",
        status_code=303,
    )


async def read_form(request: Request):
    body = (await request.body()).decode("utf-8")
    values = parse_qs(body, keep_blank_values=True)

    return {
        key: items[-1] if items else ""
        for key, items in values.items()
    }
# --------------------------------------------------
# MEDICATION INPUT MODEL
# --------------------------------------------------

class MedicationInput(BaseModel):
    medication_name: str = Field(min_length=1, max_length=150)
    dosage: str = Field(min_length=1, max_length=100)
    scheduled_time: str = Field(min_length=1, max_length=50)
    frequency: str = Field(min_length=1, max_length=50)
    notes: str = Field(default="", max_length=1000)

    model_config = ConfigDict(str_strip_whitespace=True)


ALLOWED_FREQUENCIES = {
    "daily",
    "twice_daily",
    "three_times_daily",
    "weekly",
}


def validate_medication(data: MedicationInput):
    medication_name = data.medication_name.strip()
    dosage = data.dosage.strip()
    scheduled_time = data.scheduled_time.strip()
    frequency = data.frequency.strip().lower()
    notes = data.notes.strip()

    if not medication_name:
        raise HTTPException(
            status_code=422,
            detail="Please enter a medication name.",
        )

    if not dosage:
        raise HTTPException(
            status_code=422,
            detail="Please enter the medication dosage.",
        )

    if not scheduled_time:
        raise HTTPException(
            status_code=422,
            detail="Please select a medication time.",
        )

    if frequency not in ALLOWED_FREQUENCIES:
        raise HTTPException(
            status_code=422,
            detail="Please select a valid medication frequency.",
        )

    return {
        "medication_name": medication_name,
        "dosage": dosage,
        "scheduled_time": scheduled_time,
        "frequency": frequency,
        "notes": notes,
    }


# --------------------------------------------------
# FIRESTORE HELPERS
# --------------------------------------------------

def get_firestore_client():
    require_firebase()

    try:
        return firestore.client(app=firebase_app)

    except Exception as exc:
        print("Firestore connection error:", repr(exc))

        raise HTTPException(
            status_code=503,
            detail=(
                "Could not connect to Firestore. Check that Firestore is "
                "enabled and the Firebase service account has permission "
                "to access the database."
            ),
        )


def get_user_medications(email: str):
    db = get_firestore_client()

    try:
        documents = (
            db.collection("medications")
            .where("user_email", "==", email)
            .stream()
        )

        medications = []

        for document in documents:
            item = document.to_dict() or {}
            item["id"] = document.id
            medications.append(item)

        medications.sort(
            key=lambda item: (
                str(item.get("scheduled_time", "")),
                str(item.get("medication_name", "")).lower(),
            )
        )

        return medications

    except HTTPException:
        raise

    except Exception as exc:
        print("Loading medications error:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail="Could not load your medications from Firestore.",
        )


def serialize_medication(document):
    item = document.to_dict() or {}
    item["id"] = document.id
    return item
# --------------------------------------------------
# SAVE MEDICATION API
# --------------------------------------------------

@app.post("/api/medications")
def save_medication(
    data: MedicationInput,
    request: Request,
):
    user = get_current_user(request)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Please log in before saving a medication.",
        )

    medication = validate_medication(data)
    db = get_firestore_client()

    try:
        medication["user_email"] = user["email"]
        medication["created_at"] = (
            __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat()
        )

        document_ref = db.collection("medications").document()
        document_ref.set(medication)

        return {
            "success": True,
            "message": "Medication saved successfully!",
            "medication": {
                "id": document_ref.id,
                **medication,
            },
        }

    except Exception as exc:
        print("Saving medication error:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail="Could not save the medication. Please try again.",
        )


# --------------------------------------------------
# LOAD MEDICATIONS API
# --------------------------------------------------

@app.get("/api/medications")
def load_medications(request: Request):
    user = get_current_user(request)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Please log in to view your medications.",
        )

    medications = get_user_medications(user["email"])

    return {
        "success": True,
        "count": len(medications),
        "medications": medications,
    }
# --------------------------------------------------
# HOME / DASHBOARD
# --------------------------------------------------

@app.get("/")
def home(request: Request):
    user = get_current_user(request)

    if not user:
        return redirect_to_login()

    return render_template(request, "index.html", {
        "active_page": "dashboard",
        "user": user,
        "device_online": True,
        "last_seen": "Just now",
        "adherence_score": "95%",
        "next_dose": {
            "medicine_name": "Paracetamol 500mg",
            "scheduled_time": "08:00 PM",
            "dosage": "1 Tablet",
            "rfid_tag": "A1B2C3D4",
            "compartment_id": 1,
        },
        "alert": None,
    })


@app.get("/dashboard")
def dashboard_page(request: Request):
    return RedirectResponse(url="/", status_code=303)


# --------------------------------------------------
# MEDICATION SCHEDULE PAGE
# --------------------------------------------------

@app.get("/schedule")
def schedule_page(request: Request):
    user = get_current_user(request)

    if not user:
        return redirect_to_login()

    try:
        schedules = get_user_medications(user["email"])
        alert = None

    except HTTPException:
        schedules = []
        alert = "Unable to load medications. Please try again."

    return render_template(request, "schedule.html", {
        "active_page": "schedule",
        "user": user,
        "schedules": schedules,
        "alert": alert,
    })


# --------------------------------------------------
# HISTORY AND LOGS
# --------------------------------------------------

@app.get("/logs")
def logs_page(request: Request):
    user = get_current_user(request)

    if not user:
        return redirect_to_login()

    return render_template(request, "logs.html", {
        "active_page": "logs",
        "user": user,
        "logs": [],
        "alerts": [],
    })
# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

@app.get("/login")
def login_page(request: Request):
    if get_current_user(request):
        return RedirectResponse(url="/", status_code=303)

    return render_template(request, "login.html", {
        "active_page": "login",
        "alert": None,
    })


# --------------------------------------------------
# LOGIN ACTION
# --------------------------------------------------

@app.post("/login")
async def login_action(request: Request):
    form = await read_form(request)

    email = form.get("email", "").strip().lower()
    password = form.get("password", "")

    if not email or not password:
        return render_template(request, "login.html", {
            "active_page": "login",
            "alert": "Please enter your email address and password.",
        })

    with get_db() as connection:
        user = connection.execute(
            """
            SELECT email, password_hash, password_salt
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

    if not user or not verify_password(
        password,
        user["password_salt"],
        user["password_hash"],
    ):
        return render_template(request, "login.html", {
            "active_page": "login",
            "alert": "Invalid email or password.",
        })

    response = RedirectResponse(url="/", status_code=303)
    set_session_cookie(response, email)

    return response


# --------------------------------------------------
# REGISTRATION PAGE
# --------------------------------------------------

@app.get("/register")
def register_page(request: Request):
    if get_current_user(request):
        return RedirectResponse(url="/", status_code=303)

    return render_template(request, "register.html", {
        "active_page": "register",
        "alert": None,
    })


# --------------------------------------------------
# REGISTRATION ACTION
# --------------------------------------------------

@app.post("/register")
async def register_action(request: Request):
    form = await read_form(request)

    full_name = form.get("full_name", "").strip()
    email = form.get("email", "").strip().lower()
    password = form.get("password", "")
    confirm_password = form.get("confirm_password", "")

    if not full_name or not email or not password:
        return render_template(request, "register.html", {
            "active_page": "register",
            "alert": "Please fill in all required fields.",
        })

    if "@" not in email or "." not in email.split("@")[-1]:
        return render_template(request, "register.html", {
            "active_page": "register",
            "alert": "Please enter a valid email address.",
        })

    if len(password) < 8:
        return render_template(request, "register.html", {
            "active_page": "register",
            "alert": "Password must contain at least 8 characters.",
        })

    if confirm_password and password != confirm_password:
        return render_template(request, "register.html", {
            "active_page": "register",
            "alert": "Passwords do not match.",
        })

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:
        with get_db() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    full_name,
                    email,
                    password_hash,
                    password_salt
                )
                VALUES (?, ?, ?, ?)
                """,
                (full_name, email, password_hash, salt),
            )

    except sqlite3.IntegrityError:
        return render_template(request, "register.html", {
            "active_page": "register",
            "alert": "An account with this email already exists.",
        })

    response = RedirectResponse(url="/", status_code=303)
    set_session_cookie(response, email)

    return response
# --------------------------------------------------
# LOAD ML MODEL
# --------------------------------------------------

model_bundle = None
model_load_error = None

try:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    model_bundle = joblib.load(MODEL_FILE)

    if not isinstance(model_bundle, dict):
        raise ValueError(
            "Unexpected model file format."
        )

    if (
        "model" not in model_bundle
        or "features" not in model_bundle
    ):
        raise ValueError(
            "Model file must contain 'model' and 'features'."
        )

    print("Random Forest adherence model loaded successfully.")

except Exception as exc:
    model_bundle = None
    model_load_error = str(exc)

    print(
        "WARNING: Adherence model could not be loaded:",
        model_load_error,
    )

# --------------------------------------------------
# ISOLATION FOREST ANOMALY DETECTION MODEL
# --------------------------------------------------

isolation_model_bundle = None
isolation_model_load_error = None

try:
    if not ISOLATION_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Isolation Forest model not found: "
            f"{ISOLATION_MODEL_FILE}"
        )

    isolation_model_bundle = joblib.load(
        ISOLATION_MODEL_FILE
    )

    if (
        not isinstance(isolation_model_bundle, dict)
        or "model" not in isolation_model_bundle
        or "features" not in isolation_model_bundle
    ):
        raise ValueError(
            "Isolation Forest model file must contain "
            "'model' and 'features'."
        )

    print(
        "Isolation Forest model loaded successfully."
    )

except Exception as exc:
    isolation_model_bundle = None
    isolation_model_load_error = str(exc)

    print(
        "WARNING: Isolation Forest model could not "
        "be loaded:",
        isolation_model_load_error,
    )


# --------------------------------------------------
# ML INPUT MODEL
# --------------------------------------------------

class AdherenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    AGE: int = Field(ge=0, le=120)
    ANNUALCONTRIBUTION: float = Field(
        ge=0, allow_inf_nan=False
    )
    ANNUALCLAIMAMOUNT: float = Field(
        ge=0, allow_inf_nan=False
    )
    UNITSTOTAL: float = Field(
        ge=0, allow_inf_nan=False
    )

    GENDER_M: bool
    SCHEMETYPE_MEDIUM: bool
    SCHEMETYPE_PREMIUM: bool
    DIAGNOSIS_HYPERTENSION: bool
    COVERTYPE_STANDARD: bool
    COMORBIDITY_NO_COMORBIDITY: bool
    COMPLICATIONDEVELOPMENT_NO_COMPLICATION: bool


# --------------------------------------------------
# ML MODEL STATUS API
# --------------------------------------------------

@app.get("/api/ml/status")

@app.get("/api/ml/isolation-status")
def isolation_status(request: Request):
    if not get_current_user(request):
        raise HTTPException(
            status_code=401,
            detail="Please sign in."
        )

    return {
        "model_loaded": isolation_model_bundle is not None,
        "message": (
            "Isolation Forest ready"
            if isolation_model_bundle is not None
            else "Isolation Forest unavailable"
        ),
    }

def ml_status(request: Request):
    if not get_current_user(request):
        raise HTTPException(
            status_code=401,
            detail="Please sign in."
        )

    return {
        "model_loaded": model_bundle is not None,
        "message": (
            "Model ready"
            if model_bundle is not None
            else "Model unavailable"
        ),
    }


# --------------------------------------------------
# RANDOM FOREST ADHERENCE PREDICTION API
# --------------------------------------------------

@app.post("/api/ml/predict-adherence")
def predict_adherence(
    data: AdherenceInput,
    request: Request
):
    if not get_current_user(request):
        raise HTTPException(
            status_code=401,
            detail="Please sign in."
        )

    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Adherence model is unavailable. "
                "Check the server terminal."
            ),
        )

    try:
        model = model_bundle["model"]
        feature_names = model_bundle["features"]
        input_values = data.model_dump()

        missing_features = [
            name
            for name in feature_names
            if name not in input_values
        ]

        if missing_features:
            raise ValueError(
                f"Missing model features: {missing_features}"
            )

        input_df = pd.DataFrame(
            [[input_values[name] for name in feature_names]],
            columns=feature_names,
        )

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        probability_by_class = {
            int(label): float(probability)
            for label, probability in zip(
                classes, probabilities
            )
        }

        return {
            "prediction": (
                "ADHERENT"
                if prediction == 1
                else "NON-ADHERENT"
            ),
            "class_probabilities": {
                "ADHERENT": round(
                    probability_by_class.get(1, 0.0), 4
                ),
                "NON-ADHERENT": round(
                    probability_by_class.get(0, 0.0), 4
                ),
            },
            "model": "Random Forest",
            "note": (
                "Experimental prediction from a "
                "claims-related dataset; not a clinical "
                "assessment or dose-level prediction."
            ),
        }

    except Exception as exc:
        print(f"Prediction error: {exc}")

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. "
                "Check the server terminal."
            ),
        )
    
# --------------------------------------------------
# ISOLATION FOREST ANOMALY DETECTION API
# --------------------------------------------------

@app.post("/api/ml/detect-anomalies")
def detect_anomalies(request: Request):
    if not get_current_user(request):
        raise HTTPException(
            status_code=401,
            detail="Please sign in."
        )

    if isolation_model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Isolation Forest model is unavailable. "
                "Check the server terminal."
            ),
        )

    try:
        model = isolation_model_bundle["model"]
        feature_names = isolation_model_bundle["features"]

        # Load the same dataset used to train Isolation Forest.
        data_file = (
            ML_DIR / "data" /
            "Final Prepared Dataset - Diabetes and Hypertension Data.xlsx"
        )

        if not data_file.exists():
            raise FileNotFoundError(
                f"Dataset not found: {data_file}"
            )

        df = pd.read_excel(data_file)
        X = df.drop(
            columns=["ADHERENCE"],
            errors="ignore"
        )

        X = X.reindex(columns=feature_names)
        X = X.select_dtypes(include="number")
        X = X.replace([float("inf"), -float("inf")], float("nan"))
        X = X.fillna(X.median()).fillna(0)

        predictions = model.predict(X)

        return {
            "model": "Isolation Forest",
            "records_tested": int(len(predictions)),
            "anomalies_detected": int((predictions == -1).sum()),
            "normal_records": int((predictions == 1).sum()),
            "note": (
                "Anomalies are unusual statistical patterns, "
                "not proof of medication non-adherence or "
                "a clinical problem."
            ),
        }

    except Exception as exc:
        print(f"Isolation Forest error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Anomaly detection failed. Check the server terminal."
        )



# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "firebase_admin_configured": firebase_app is not None,
        "model_loaded": model_bundle is not None,
    }
