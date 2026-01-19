from pydantic import BaseModel

class CarInput(BaseModel):
    make: str
    model: str
    year: int
    engine_hp: float
    engine_cylinders: float
    transmission_type: str  # <--- MUST BE THIS
    vehicle_style: str
    highway_mpg: int
    city_mpg: int