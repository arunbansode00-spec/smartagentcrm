"""Quick check of the demo logic without starting the web server:  python test_service.py"""
from src.service import ModelStore
import time
t = time.time(); m = ModelStore("data"); print(f"models trained in {time.time()-t:.0f}s")
print("fair-value demo model, held-out metrics:", m.fv_metrics)
print(m.fair_value(dict(bedrooms=3, bathrooms=2.0, living_area=1800, lot_area=6000, floors=1.0, waterfront=0,
                        views=0, condition=3, grade=7, built_year=1975, latitude=52.8, longitude=-114.4)))
print(m.risk(dict(Income=5000000, Age=30, Experience=5, CURRENT_JOB_YRS=3, CURRENT_HOUSE_YRS=11, Marital="single",
                  House_Ownership="rented", Car_Ownership="no", Profession="Engineer", CITY="Pune", STATE="Maharashtra")))
print(m.forecast(3))
print(m.anomaly(2600, "house", 3, 900000)); print(m.anomaly(2600, "house", 3, 4000000))
