from abc import ABC, abstractmethod
from pathlib import Path
import requests
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

from altima.data.shmu.crud import init_db, get_session, TemperatureImageCRUD, HumidityImageCRUD

from altima.data.shmu.model import RadarImage

# Constants
API_URL = 'https://www.shmu.sk/api/v1/meteo/getradardata'
BASE_HOST = 'https://www.shmu.sk'
BASE_DATA_DIR = "/home/mike/Data/meteo"


class RadarDataLoader:
    """
    Loader for SHMU radar data: downloads images and syncs metadata to the database.
    """
    def __init__(
        self,
        base_dir: Path = Path(BASE_DATA_DIR),
        batch_size: int = 100,
        api_url: str = API_URL,
        base_host: str = BASE_HOST,
        db_url: str = 'sqlite:///shmu.db'
    ):
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.api_url = api_url
        self.base_host = base_host
        self.db_url = db_url

    def fetch_json(self) -> list:
        """
        Fetch radar JSON data from SHMU API.
        """
        resp = requests.get(self.api_url)
        resp.raise_for_status()
        return resp.json()

    def process_data(self, data: list, session):
        """
        Download images and batch-insert records into the database.
        """
        # Preload existing filenames to skip re-downloads and DB hits
        existing = {row[0] for row in session.query(RadarImage.fname).all()}
        pending = []
        sink_count = 0

        for product in data:
            prod_key = product['product']
            base_url = product['base_url']
            for item in tqdm(product.get('items', []), desc=prod_key, unit='file'):
                fname = item['fname']
                if fname in existing:
                    continue

                # Download image
                dt_obj = datetime.fromtimestamp(item['dt'])
                dir_path = self.base_dir / 'radar' / dt_obj.strftime('%Y-%m-%d') / prod_key
                dir_path.mkdir(parents=True, exist_ok=True)
                file_path = dir_path / fname

                image_url = f"{self.base_host}{base_url}{fname}"
                try:
                    with requests.get(image_url, stream=True) as r:
                        r.raise_for_status()
                        with open(file_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                except Exception as e:
                    print(f"Failed to download {fname}: {e}")
                    continue

                # Queue record for insertion
                pending.append({
                    'fname':    fname,
                    'dt':       item['dt'],
                    'dt_utc':   item['dt_utc'],
                    'dt_f_tz':  item['dt_f_tz'],
                    'product':  prod_key,
                })
                existing.add(fname)
                sink_count += 1

                # Batch insert
                if sink_count % self.batch_size == 0:
                    session.bulk_insert_mappings(RadarImage, pending)
                    session.commit()
                    print(f"Inserted batch of {len(pending)} records")
                    pending.clear()

        # Final insert
        if pending:
            session.bulk_insert_mappings(RadarImage, pending)
            session.commit()
            print(f"Inserted final batch of {len(pending)} records")

    def run(self):
        """
        Main entry point: init DB session, fetch data, process it, then close session.
        """
        engine = init_db(self.db_url)
        session = get_session(engine)
        try:
            data = self.fetch_json()
            self.process_data(data, session)
        finally:
            session.close()


class BaseHourlyLoader(ABC):
    """
    Abstract base for hourly image loaders.
    Subclasses must define IMAGE_PATH, FILE_TEMPLATE, SUBFOLDER, CRUD_CLASS,
    and implement fallback_start().
    """
    IMAGE_PATH: str
    FILE_TEMPLATE: str
    SUBFOLDER: str
    CRUD_CLASS: type

    def __init__(
        self,
        base_dir: Path = Path(BASE_DATA_DIR),
        db_url: str = 'sqlite:///shmu.db',
        timeout: int = 10
    ):
        self.base_dir = base_dir
        self.db_url = db_url
        self.timeout = timeout

    @abstractmethod
    def fallback_start(self) -> datetime:
        """
        Return a naive UTC datetime to start from if no records exist.
        """
        ...

    def run(self):
        # Initialize DB and repository
        engine = init_db(self.db_url)
        session = get_session(engine)
        repo = self.CRUD_CLASS(session)

        # Determine start datetime (UTC)
        last_ts = repo.get_last_dt()
        if last_ts:
            start_dt = datetime.fromtimestamp(last_ts, timezone.utc) + timedelta(hours=1)
        else:
            start_dt = self.fallback_start().replace(tzinfo=timezone.utc)
        start_dt = start_dt.replace(minute=0, second=0, microsecond=0)

        # Determine end datetime (current hour UTC)
        end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        # Iterate hourly with progress bar
        total_hours = int((end_dt - start_dt) / timedelta(hours=1))
        for i in tqdm(range(total_hours + 1), desc=self.SUBFOLDER, unit='hours'):
            dt = start_dt + timedelta(hours=i)
            dt_str = dt.strftime('%Y%m%d%H%M')
            fname = self.FILE_TEMPLATE.format(dt_str=dt_str)

            # Skip if already recorded
            if repo.get(fname=fname):
                continue

            # Build download URL and local path
            url = f"{BASE_HOST}{self.IMAGE_PATH}/{fname}"
            date_dir = self.base_dir / self.SUBFOLDER / dt.strftime('%Y-%m-%d')
            date_dir.mkdir(parents=True, exist_ok=True)
            file_path = date_dir / fname

            # Download image
            try:
                resp = requests.get(url, timeout=self.timeout)
                if resp.status_code == 404:
                    # print(f"Not found: {fname}")
                    continue
                resp.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(resp.content)

                # Record metadata
                repo.create({
                    'fname':  fname,
                    'dt_utc': int(dt.timestamp())
                })
                # print(f"Saved and recorded: {fname}")
            except requests.RequestException as e:
                print(f"Error fetching {fname}: {e}")

        session.close()


class TemperatureDataLoader(BaseHourlyLoader):
    """
    Loader for T2M temperature images.
    """
    IMAGE_PATH = '/data/datainca/T2M/R7'
    FILE_TEMPLATE = 'T2M_oper_iso_R7_{dt_str}-0000.png'
    SUBFOLDER = 'temperature'
    CRUD_CLASS = TemperatureImageCRUD

    def fallback_start(self) -> datetime:
        # Default start: 2025-04-25 12:00 UTC
        return datetime(2025, 4, 25, 12)


class HumidityDataLoader(BaseHourlyLoader):
    """
    Loader for RH humidity images.
    """
    IMAGE_PATH = '/data/datainca/RH/R7'
    FILE_TEMPLATE = 'RH_oper_iso_R7_{dt_str}-0000.png'
    SUBFOLDER = 'humidity'
    CRUD_CLASS = HumidityImageCRUD

    def fallback_start(self) -> datetime:
        # Default start: 2025-04-25 12:00 UTC
        return datetime(2025, 4, 25, 12)


if __name__ == '__main__':
    loader = HumidityDataLoader()
    loader.run()
