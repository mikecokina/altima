from datetime import timedelta, datetime
from pathlib import Path

import requests


class TemperatureDownloader:
    """
    Simple downloader for T2M temperature images, without DB.
    Iterates hourly from a given start to end datetime, ignores 404s.
    """
    def __init__(self,
                 base_dir: Path,
                 base_host: str = 'https://www.shmu.sk',
                 image_path: str = '/data/datainca/T2M/R7',
                 timeout: int = 10):
        self.base_dir = base_dir
        self.base_host = base_host.rstrip('/')
        self.image_path = image_path.strip('/')
        self.timeout = timeout

    def run(self, start_dt: datetime, end_dt: datetime):
        """
        Download images hourly from start_dt to end_dt.
        :param start_dt: datetime to start (inclusive)
        :param end_dt: datetime to end (inclusive)
        """
        dt = start_dt.replace(minute=0, second=0, microsecond=0)
        end = end_dt.replace(minute=0, second=0, microsecond=0)

        while dt <= end:
            dt_str = dt.strftime('%Y%m%d%H%M')
            fname = f"T2M_oper_iso_R7_{dt_str}-0000.png"
            url = f"{self.base_host}/{self.image_path}/{fname}"

            # Prepare directory and filepath
            date_dir = self.base_dir / 'temperature' / dt.strftime('%Y-%m-%d')
            date_dir.mkdir(parents=True, exist_ok=True)
            file_path = date_dir / fname

            # Skip if already exists
            if file_path.exists():
                print(f"Exists, skip: {file_path}")
                dt += timedelta(hours=1)
                continue

            # Attempt download
            try:
                resp = requests.get(url, timeout=self.timeout)
                if resp.status_code == 404:
                    print(f"Not found (404): {url}")
                else:
                    resp.raise_for_status()
                    with open(file_path, 'wb') as f:
                        f.write(resp.content)
                    print(f"Downloaded: {file_path}")
            except requests.RequestException as e:
                print(f"Error downloading {url}: {e}")

            dt += timedelta(hours=1)


if __name__ == '__main__':
    # Mock start date at 2025-04-25 12:00
    start = datetime(2025, 4, 25, 12)
    # End at current hour
    end = datetime.now().replace(minute=0, second=0, microsecond=0)

    downloader = TemperatureDownloader(
        base_dir=Path('/home/mike/Data/meteo'),
    )
    downloader.run(start, end)
