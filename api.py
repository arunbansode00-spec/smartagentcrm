"""SmartAgentCRM demo API (FastAPI).  Run:  uvicorn api:app --reload --port 8000
Swagger docs:  http://localhost:8000/docs      Demo page:  http://localhost:8000/"""
import os
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from src.service import ModelStore

STORE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STORE["m"] = ModelStore(os.environ.get("DATA_DIR", "data"))   # trains all models once (about 15-30 s)
    yield


app = FastAPI(title="SmartAgentCRM ML API", version="1.0",
              description="Fair-value estimate, client risk score, commission forecast and anomaly check "
                          "for independent real estate agents. Course demo (AI in Finance).",
              lifespan=lifespan)


def m() -> ModelStore:
    return STORE["m"]


class HouseIn(BaseModel):
    bedrooms: int = Field(3, ge=1, le=33)
    bathrooms: float = Field(2.0, ge=0.5, le=8)
    living_area: float = Field(1800, ge=300, le=14000, description="square feet")
    lot_area: float = Field(6000, ge=500, le=1100000)
    floors: float = Field(1.0, ge=1, le=3.5)
    waterfront: Literal[0, 1] = 0
    views: int = Field(0, ge=0, le=4)
    condition: int = Field(3, ge=1, le=5)
    grade: int = Field(7, ge=1, le=13)
    built_year: int = Field(1975, ge=1900, le=2016)
    latitude: float = Field(52.8, description="dataset range about 52.4 to 53.0")
    longitude: float = Field(-114.4, description="dataset range about -114.7 to -113.5")


class RiskIn(BaseModel):
    Income: float = Field(5000000, ge=0)
    Age: int = Field(30, ge=18, le=100)
    Experience: int = Field(5, ge=0, le=60)
    CURRENT_JOB_YRS: int = Field(3, ge=0, le=60)
    CURRENT_HOUSE_YRS: int = Field(11, ge=0, le=60)
    Marital: Literal["single", "married"] = "single"
    House_Ownership: Literal["rented", "owned", "norent_noown"] = "rented"
    Car_Ownership: Literal["no", "yes"] = "no"
    Profession: str = "Engineer"
    CITY: str = "Pune"
    STATE: str = "Maharashtra"


class SaleIn(BaseModel):
    postcode: int = Field(2600, description="one of the 27 postcodes in raw_sales.csv")
    property_type: Literal["house", "unit"] = "house"
    bedrooms: int = Field(3, ge=0, le=5)
    price: float = Field(900000, gt=0, description="AUD")


def guard(fn, *a, **k):
    try:
        return fn(*a, **k)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {"fair_value_holdout": m().fv_metrics, "australian_postcodes": m().au_postcodes,
            "risk_note": "AUC about 0.50 on deduplicated data (no better than chance)"}


@app.post("/fair-value")
def fair_value(body: HouseIn):
    return guard(m().fair_value, body.model_dump())


@app.get("/risk/options")
def risk_options():
    return m().risk_options


@app.post("/risk")
def risk(body: RiskIn):
    return guard(m().risk, body.model_dump())


@app.get("/forecast")
def forecast(months: int = Query(3, ge=1, le=12), commission_rate: float = Query(0.02, gt=0, le=0.2)):
    return m().forecast(months, commission_rate)


@app.post("/anomaly")
def anomaly(body: SaleIn):
    return guard(m().anomaly, body.postcode, body.property_type, body.bedrooms, body.price)
