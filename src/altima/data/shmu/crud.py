from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
from altima.data.shmu.model import Base, RadarImage, TemperatureImage


def init_db(db_url: str = 'sqlite:///shmu.db') -> create_engine:
    """
    Initialize the SQLite database, create tables, and return the engine.
    """
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine) -> Session:
    """
    Create and return a new SQLAlchemy Session.
    """
    return sessionmaker(bind=engine)()


class BaseCRUD:
    """
    Generic CRUD operations for a SQLAlchemy model.
    """
    def __init__(self, session: Session, model):
        self.session = session
        self.model = model

    def create(self, data: dict):
        """
        Create a new record for the model.
        :param data: dict of attributes for the model
        :return: the created record
        """
        record = self.model(**data)
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def delete(self, *, record_id: int = None, **filters) -> int:
        """
        Delete record(s) by `id` or other filters.
        :param record_id: primary key of the record
        :param filters: other column-based filters (e.g., fname='...')
        :return: number of rows deleted
        """
        query = self.session.query(self.model)
        if record_id is not None:
            query = query.filter(self.model.id == record_id)
        elif filters:
            query = query.filter_by(**filters)
        else:
            raise ValueError("Must provide `record_id` or filters to delete.")
        count = query.delete()
        self.session.commit()
        return count

    def get(self, **filters):
        """
        Retrieve records matching given filters.
        :param filters: column-based filters
        :return: list of matching records
        """
        return self.session.query(self.model).filter_by(**filters).all()

    def get_by_id(self, record_id: int):
        """
        Retrieve a single record by primary key.
        :param record_id: primary key
        :return: the record or None
        """
        return self.session.query(self.model).get(record_id)


class RadarImageCRUD(BaseCRUD):
    """
    CRUD operations specifically for RadarImage.
    """
    def __init__(self, session: Session):
        super().__init__(session, RadarImage)


class TemperatureImageCRUD(BaseCRUD):
    """
    CRUD for TemperatureImage, plus helper to fetch the latest timestamp.
    """
    def __init__(self, session: Session):
        super().__init__(session, TemperatureImage)

    def get_last_dt(self) -> int:
        """
        Return the maximum dt_utc in the table, or 0 if empty.
        """
        last = self.session.query(func.max(self.model.dt_utc)).scalar()
        return last or 0


# if __name__ == '__main__':
#     # Example usage:
#     engine = init_db()
#     session = get_session(engine)
#     radar_repo = RadarImageCRUD(session)
#
#     # Create
#     rec = radar_repo.create({
#         'fname': 'cmax.kruh.20250425.1340.0.png',
#         'dt': 1745588400,
#         'dt_utc': 1745581200,
#         'dt_f_tz': '2025-04-25T15:40:00+02:00'
#     })
#     # Delete
#     # deleted_count = radar_repo.delete(fname='cmax.kruh.20250425.1340.0.png')
#     session.close()
