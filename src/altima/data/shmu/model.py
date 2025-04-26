from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class RadarImage(Base):
    __tablename__ = 'radar_images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    fname = Column(String, nullable=False, unique=True)
    dt = Column(Integer, nullable=False)
    dt_utc = Column(Integer, nullable=False)
    dt_f_tz = Column(String, nullable=False)
    product = Column(String, nullable=False, index=True)


class TemperatureImage(Base):
    __tablename__ = 'temperature_images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    fname = Column(String, nullable=False, unique=True)
    dt_utc = Column(Integer, nullable=False, index=True)  # UTC timestamp parsed from filename


class HumidityImage(Base):
    __tablename__ = 'humidity_images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    fname = Column(String, nullable=False, unique=True)
    dt_utc = Column(Integer, nullable=False, index=True)


class Rainfall1HImage(Base):
    __tablename__ = 'rainfall_1h_images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    fname = Column(String, nullable=False, unique=True)
    dt_utc = Column(Integer, nullable=False, index=True)
