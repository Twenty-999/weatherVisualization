from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from pyecharts import options as opts
from pyecharts.charts import Bar, HeatMap, Line, Radar, Scatter
from pyecharts.commons.utils import JsCode
from pyecharts.globals import CurrentConfig, ThemeType


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
API_CACHE_DIR = BASE_DIR / ".api_cache"
ADMIN_DIVISION_CACHE_DIR = API_CACHE_DIR / "admin_divisions"
FAILED_CITIES_FILE = API_CACHE_DIR / "failed_cities.json"
DASHBOARD_STATUS_FILE = API_CACHE_DIR / "dashboard_status.json"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ALIYUN_BOUND_URL = "https://geo.datav.aliyun.com/areas_v3/bound/{code}_full.json"
DEFAULT_DASHBOARD_API_PORT = 8765

DAILY_FILE = PROCESSED_DIR / "china_weather_daily_5y.csv"
MONTHLY_FILE = PROCESSED_DIR / "china_weather_monthly_5y.csv"
DASHBOARD_FILE = FIGURES_DIR / "china_weather_dashboard_5y.html"
REPORT_FILE = REPORTS_DIR / "china_weather_report_5y.md"

PROVINCE_NAME_MAP = {
    "北京": "北京市",
    "天津": "天津市",
    "上海": "上海市",
    "重庆": "重庆市",
    "河北": "河北省",
    "山西": "山西省",
    "辽宁": "辽宁省",
    "吉林": "吉林省",
    "黑龙江": "黑龙江省",
    "江苏": "江苏省",
    "浙江": "浙江省",
    "安徽": "安徽省",
    "福建": "福建省",
    "江西": "江西省",
    "山东": "山东省",
    "河南": "河南省",
    "湖北": "湖北省",
    "湖南": "湖南省",
    "广东": "广东省",
    "海南": "海南省",
    "四川": "四川省",
    "贵州": "贵州省",
    "云南": "云南省",
    "陕西": "陕西省",
    "甘肃": "甘肃省",
    "青海": "青海省",
    "台湾": "台湾省",
    "内蒙古": "内蒙古自治区",
    "广西": "广西壮族自治区",
    "西藏": "西藏自治区",
    "宁夏": "宁夏回族自治区",
    "新疆": "新疆维吾尔自治区",
    "香港": "香港特别行政区",
    "澳门": "澳门特别行政区",
}


@dataclass(frozen=True)
class City:
    city: str
    province: str
    region: str
    lat: float
    lon: float
    elevation: int = 0
    city_code: str = ""


FALLBACK_CITIES: List[City] = [
    City("北京", "北京", "华北", 39.9042, 116.4074, 43),
    City("天津", "天津", "华北", 39.0842, 117.2009, 5),
    City("石家庄", "河北", "华北", 38.0428, 114.5149, 83),
    City("太原", "山西", "华北", 37.8706, 112.5489, 800),
    City("呼和浩特", "内蒙古", "华北", 40.8426, 111.7492, 1065),
    City("沈阳", "辽宁", "东北", 41.8057, 123.4315, 55),
    City("长春", "吉林", "东北", 43.8171, 125.3235, 236),
    City("哈尔滨", "黑龙江", "东北", 45.8038, 126.5350, 150),
    City("上海", "上海", "华东", 31.2304, 121.4737, 4),
    City("南京", "江苏", "华东", 32.0603, 118.7969, 15),
    City("杭州", "浙江", "华东", 30.2741, 120.1551, 41),
    City("合肥", "安徽", "华东", 31.8206, 117.2290, 37),
    City("福州", "福建", "华东", 26.0745, 119.2965, 14),
    City("南昌", "江西", "华东", 28.6829, 115.8582, 48),
    City("济南", "山东", "华东", 36.6512, 117.1201, 23),
    City("郑州", "河南", "华中", 34.7466, 113.6254, 110),
    City("武汉", "湖北", "华中", 30.5928, 114.3055, 37),
    City("长沙", "湖南", "华中", 28.2282, 112.9388, 63),
    City("广州", "广东", "华南", 23.1291, 113.2644, 21),
    City("南宁", "广西", "华南", 22.8170, 108.3669, 72),
    City("海口", "海南", "华南", 20.0442, 110.1999, 12),
    City("香港", "香港", "华南", 22.3193, 114.1694, 35),
    City("台北", "台湾", "华东", 25.0330, 121.5654, 9),
    City("台中", "台湾", "华东", 24.1477, 120.6736, 84),
    City("高雄", "台湾", "华东", 22.6273, 120.3014, 9),
    City("重庆", "重庆", "西南", 29.5630, 106.5516, 244),
    City("成都", "四川", "西南", 30.5728, 104.0668, 500),
    City("贵阳", "贵州", "西南", 26.6470, 106.6302, 1100),
    City("昆明", "云南", "西南", 25.0389, 102.7183, 1890),
    City("拉萨", "西藏", "西南", 29.6520, 91.1721, 3650),
    City("西安", "陕西", "西北", 34.3416, 108.9398, 405),
    City("兰州", "甘肃", "西北", 36.0611, 103.8343, 1520),
    City("西宁", "青海", "西北", 36.6171, 101.7782, 2275),
    City("银川", "宁夏", "西北", 38.4872, 106.2309, 1110),
    City("乌鲁木齐", "新疆", "西北", 43.8256, 87.6168, 800),
]

PROVINCE_CODE_MAP = {
    "北京市": "110000",
    "天津市": "120000",
    "河北省": "130000",
    "山西省": "140000",
    "内蒙古自治区": "150000",
    "辽宁省": "210000",
    "吉林省": "220000",
    "黑龙江省": "230000",
    "上海市": "310000",
    "江苏省": "320000",
    "浙江省": "330000",
    "安徽省": "340000",
    "福建省": "350000",
    "江西省": "360000",
    "山东省": "370000",
    "河南省": "410000",
    "湖北省": "420000",
    "湖南省": "430000",
    "广东省": "440000",
    "广西壮族自治区": "450000",
    "海南省": "460000",
    "重庆市": "500000",
    "四川省": "510000",
    "贵州省": "520000",
    "云南省": "530000",
    "西藏自治区": "540000",
    "陕西省": "610000",
    "甘肃省": "620000",
    "青海省": "630000",
    "宁夏回族自治区": "640000",
    "新疆维吾尔自治区": "650000",
}

PROVINCE_SHORT_NAME_MAP = {
    "北京": "北京市",
    "天津": "天津市",
    "河北": "河北省",
    "山西": "山西省",
    "内蒙古": "内蒙古自治区",
    "辽宁": "辽宁省",
    "吉林": "吉林省",
    "黑龙江": "黑龙江省",
    "上海": "上海市",
    "江苏": "江苏省",
    "浙江": "浙江省",
    "安徽": "安徽省",
    "福建": "福建省",
    "江西": "江西省",
    "山东": "山东省",
    "河南": "河南省",
    "湖北": "湖北省",
    "湖南": "湖南省",
    "广东": "广东省",
    "广西": "广西壮族自治区",
    "海南": "海南省",
    "重庆": "重庆市",
    "四川": "四川省",
    "贵州": "贵州省",
    "云南": "云南省",
    "西藏": "西藏自治区",
    "陕西": "陕西省",
    "甘肃": "甘肃省",
    "青海": "青海省",
    "宁夏": "宁夏回族自治区",
    "新疆": "新疆维吾尔自治区",
}

PROVINCE_REGION_MAP = {
    "北京市": "华北",
    "天津市": "华北",
    "河北省": "华北",
    "山西省": "华北",
    "内蒙古自治区": "华北",
    "辽宁省": "东北",
    "吉林省": "东北",
    "黑龙江省": "东北",
    "上海市": "华东",
    "江苏省": "华东",
    "浙江省": "华东",
    "安徽省": "华东",
    "福建省": "华东",
    "江西省": "华东",
    "山东省": "华东",
    "河南省": "华中",
    "湖北省": "华中",
    "湖南省": "华中",
    "广东省": "华南",
    "广西壮族自治区": "华南",
    "海南省": "华南",
    "重庆市": "西南",
    "四川省": "西南",
    "贵州省": "西南",
    "云南省": "西南",
    "西藏自治区": "西南",
    "陕西省": "西北",
    "甘肃省": "西北",
    "青海省": "西北",
    "宁夏回族自治区": "西北",
    "新疆维吾尔自治区": "西北",
}

MUNICIPALITY_PROVINCES = {"北京市", "天津市", "上海市", "重庆市"}


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    API_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ADMIN_DIVISION_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def save_failed_cities(cities: list[City]) -> None:
    payload = [
        {
            "city": city.city,
            "province": city.province,
            "region": city.region,
            "lat": city.lat,
            "lon": city.lon,
            "elevation": city.elevation,
            "city_code": city.city_code,
        }
        for city in cities
    ]
    FAILED_CITIES_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_failed_cities() -> list[City]:
    if not FAILED_CITIES_FILE.exists():
        return []
    payload = json.loads(FAILED_CITIES_FILE.read_text(encoding="utf-8"))
    cities: list[City] = []
    for item in payload:
        cities.append(
            City(
                city=item["city"],
                province=item["province"],
                region=item["region"],
                lat=float(item["lat"]),
                lon=float(item["lon"]),
                elevation=int(item.get("elevation", 0)),
                city_code=str(item.get("city_code", "")),
            )
        )
    return cities


def write_dashboard_status(status: dict) -> None:
    DASHBOARD_STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def read_dashboard_status() -> dict:
    if not DASHBOARD_STATUS_FILE.exists():
        return {
            "running": False,
            "mode": "",
            "message": "暂无任务",
            "updated_at": "",
            "last_success": "",
            "last_error": "",
            "failed_count": len(load_failed_cities()),
            "total": 0,
            "completed": 0,
            "success_count": 0,
            "current_city": "",
        }
    return json.loads(DASHBOARD_STATUS_FILE.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取并分析近五年中国天气数据")
    parser.add_argument("--force-refresh", action="store_true", help="忽略缓存并重新抓取")
    parser.add_argument("--workers", type=int, default=2, help="并发抓取线程数")
    parser.add_argument("--serve-dashboard", action="store_true", help="启动大屏按钮控制服务")
    parser.add_argument("--api-port", type=int, default=DEFAULT_DASHBOARD_API_PORT, help="大屏按钮控制服务端口")
    return parser.parse_args()


def calc_date_range() -> tuple[date, date]:
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 5)
    return start_date, end_date


def city_cache_file(city: City, start_date: date, end_date: date) -> Path:
    suffix = f"{start_date.isoformat()}_{end_date.isoformat()}"
    cache_key = city.city_code or f"{normalize_province_name(city.province)}_{city.city}"
    safe_key = str(cache_key).replace("/", "_").replace("\\", "_").replace(" ", "_")
    return RAW_DIR / f"{safe_key}_{suffix}.csv"


def find_latest_city_cache(city: City) -> Path | None:
    cache_key = city.city_code or f"{normalize_province_name(city.province)}_{city.city}"
    safe_key = str(cache_key).replace("/", "_").replace("\\", "_").replace(" ", "_")
    candidates = sorted(RAW_DIR.glob(f"{safe_key}_*.csv"))
    return candidates[-1] if candidates else None


def normalize_province_name(name: str) -> str:
    return PROVINCE_SHORT_NAME_MAP.get(name, name)


def admin_cache_file(code: str) -> Path:
    return ADMIN_DIVISION_CACHE_DIR / f"{code}_full.json"


def fetch_admin_geojson(code: str, force_refresh: bool = False) -> dict:
    cache_file = admin_cache_file(code)
    if cache_file.exists() and not force_refresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))

    response = requests.get(
        ALIYUN_BOUND_URL.format(code=code),
        timeout=60,
        headers={"User-Agent": "china-weather-analysis/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def extract_feature_center(feature: dict) -> tuple[float, float] | None:
    props = feature.get("properties", {})
    center = props.get("center") or props.get("centroid") or props.get("cp")
    if isinstance(center, list) and len(center) >= 2:
        return float(center[1]), float(center[0])
    return None


def build_prefecture_level_cities(force_refresh: bool = False) -> list[City]:
    cities: list[City] = []
    seen_codes: set[str] = set()

    for province_name, province_code in PROVINCE_CODE_MAP.items():
        region = PROVINCE_REGION_MAP.get(province_name, "其他")
        try:
            geojson = fetch_admin_geojson(province_code, force_refresh=force_refresh)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {province_name} 行政区划加载失败，稍后回退到内置城市样本: {exc}")
            return FALLBACK_CITIES

        features = geojson.get("features", [])
        province_cities: list[City] = []
        for feature in features:
            props = feature.get("properties", {})
            level = str(props.get("level", ""))
            if level != "city":
                continue
            center = extract_feature_center(feature)
            if center is None:
                continue
            city_name = props.get("name")
            city_code = str(props.get("adcode", ""))
            if not city_name or not city_code or city_code in seen_codes:
                continue
            lat, lon = center
            province_cities.append(
                City(
                    city=city_name,
                    province=province_name,
                    region=region,
                    lat=lat,
                    lon=lon,
                    city_code=city_code,
                )
            )
            seen_codes.add(city_code)

        if province_cities:
            cities.extend(province_cities)
            continue

        if province_name in MUNICIPALITY_PROVINCES:
            province_feature = next(iter(features), None)
            center = extract_feature_center(province_feature) if province_feature else None
            if center is not None:
                lat, lon = center
                cities.append(
                    City(
                        city=province_name,
                        province=province_name,
                        region=region,
                        lat=lat,
                        lon=lon,
                        city_code=province_code,
                    )
                )
                seen_codes.add(province_code)

    return cities or FALLBACK_CITIES


def fetch_city_weather(city: City, start_date: date, end_date: date, force_refresh: bool = False) -> pd.DataFrame:
    if dashboard_cancel_event.is_set():
        raise InterruptedError("任务已中断")
    cache_file = city_cache_file(city, start_date, end_date)
    if cache_file.exists() and not force_refresh:
        df = pd.read_csv(cache_file, parse_dates=["time"])
        return df

    if not force_refresh:
        fallback_cache = find_latest_city_cache(city)
        if fallback_cache is not None and fallback_cache != cache_file:
            df = pd.read_csv(fallback_cache, parse_dates=["time"])
            print(f"[CACHE] {city.city} 使用历史缓存 {fallback_cache.name}")
            return df

    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "Asia/Shanghai",
        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "apparent_temperature_mean",
                "precipitation_sum",
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
                "wind_direction_10m_dominant",
            ]
        ),
    }
    payload = None
    last_error = None
    for attempt in range(4):
        try:
            response = requests.get(
                OPEN_METEO_ARCHIVE_URL,
                params=params,
                timeout=60,
                headers={"User-Agent": "china-weather-analysis/1.0"},
            )
            if response.status_code == 429:
                wait_seconds = 2 * (attempt + 1)
                time.sleep(wait_seconds)
                last_error = RuntimeError(f"{city.city} 请求过快，已重试 {attempt + 1} 次")
                continue
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))

    if payload is None:
        fallback_cache = find_latest_city_cache(city)
        if fallback_cache is not None:
            df = pd.read_csv(fallback_cache, parse_dates=["time"])
            print(f"[CACHE] {city.city} 网络失败，回退到历史缓存 {fallback_cache.name}")
            return df
        raise RuntimeError(str(last_error) if last_error else f"{city.city} 抓取失败")

    daily_data = payload.get("daily", {})
    if not daily_data or not daily_data.get("time"):
        raise RuntimeError(f"{city.city} 未抓取到天气数据")

    df = pd.DataFrame(daily_data)
    df["city"] = city.city
    df["province"] = normalize_province_name(city.province)
    df["region"] = city.region
    df["lat"] = city.lat
    df["lon"] = city.lon
    df["city_code"] = city.city_code
    df.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return df


def crawl_weather_data(cities: Iterable[City], start_date: date, end_date: date, force_refresh: bool, workers: int) -> pd.DataFrame:
    city_list = list(cities)
    total = len(city_list)
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    failed_city_objs: list[City] = []
    completed = 0

    status_snapshot = read_dashboard_status()
    write_dashboard_status(
        {
            **status_snapshot,
            "running": True,
            "message": "抓取进行中",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": total,
            "completed": 0,
            "success_count": 0,
            "failed_count": 0,
            "current_city": "",
        }
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_city_weather, city, start_date, end_date, force_refresh): city
            for city in city_list
        }
        for future in as_completed(futures):
            if dashboard_cancel_event.is_set():
                remaining = [futures[item] for item in futures if not item.done()]
                failed_city_objs.extend(remaining)
                for item in futures:
                    if not item.done():
                        item.cancel()
                save_failed_cities(failed_city_objs)
                raise InterruptedError("任务已中断")
            city = futures[future]
            completed += 1
            try:
                frames.append(future.result())
                print(f"[OK] {city.city} 抓取完成")
            except InterruptedError:
                failed_city_objs.append(city)
                save_failed_cities(failed_city_objs)
                raise
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{city.city}: {exc}")
                failed_city_objs.append(city)
                print(f"[WARN] {city.city} 抓取失败: {exc}")
            write_dashboard_status(
                {
                    **read_dashboard_status(),
                    "running": True,
                    "message": "抓取进行中",
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total": total,
                    "completed": completed,
                    "success_count": len(frames),
                    "failed_count": len(failed_city_objs),
                    "current_city": city.city,
                }
            )

    if not frames:
        save_failed_cities(failed_city_objs)
        raise RuntimeError("所有城市抓取均失败，无法继续分析")

    if errors:
        print("\n部分城市抓取失败：")
        for msg in errors:
            print(f"- {msg}")

    save_failed_cities(failed_city_objs)
    df = pd.concat(frames, ignore_index=True)
    return clean_weather_data(df)


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "time": "date",
        "temperature_2m_mean": "avg_temp",
        "temperature_2m_min": "min_temp",
        "temperature_2m_max": "max_temp",
        "precipitation_sum": "precipitation",
        "wind_speed_10m_max": "wind_speed",
        "apparent_temperature_mean": "apparent_temp",
        "wind_gusts_10m_max": "wind_gusts",
        "wind_direction_10m_dominant": "wind_direction",
        "weather_code": "weather_code",
    }
    df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"])
    df["province"] = df["province"].map(lambda value: normalize_province_name(str(value)))
    if "city_code" not in df.columns:
        df["city_code"] = ""
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    df["season"] = df["month"].map({
        12: "冬季", 1: "冬季", 2: "冬季",
        3: "春季", 4: "春季", 5: "春季",
        6: "夏季", 7: "夏季", 8: "夏季",
        9: "秋季", 10: "秋季", 11: "秋季",
    })

    metric_cols = ["avg_temp", "min_temp", "max_temp", "precipitation", "wind_speed", "apparent_temp", "wind_gusts"]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df.groupby("city")[col].transform(lambda s: s.interpolate(limit_direction="both"))

    df["temp_range"] = (df["max_temp"] - df["min_temp"]).round(1)
    df["comfort_index"] = (
        df["apparent_temp"].fillna(df["avg_temp"])
        - 0.25 * df["wind_speed"].fillna(df["wind_speed"].median())
        - 0.015 * df["precipitation"].fillna(0)
    ).round(2)

    df = df.dropna(subset=["avg_temp"]).sort_values(["city", "date"]).reset_index(drop=True)
    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(["city", "city_code", "province", "region", "lat", "lon", "year", "month", "year_month"], as_index=False)
        .agg(
            avg_temp=("avg_temp", "mean"),
            min_temp=("min_temp", "mean"),
            max_temp=("max_temp", "mean"),
            precipitation=("precipitation", "sum"),
            wind_speed=("wind_speed", "mean"),
            apparent_temp=("apparent_temp", "mean"),
            wind_gusts=("wind_gusts", "mean"),
            weather_code=("weather_code", lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
            comfort_index=("comfort_index", "mean"),
        )
        .round(2)
    )
    return monthly


def compute_city_trends(monthly_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, city_df in monthly_df.groupby("city"):
        city_df = city_df.sort_values(["year", "month"]).reset_index(drop=True)
        x = np.arange(len(city_df))
        slope = float(np.polyfit(x, city_df["avg_temp"], 1)[0]) if len(city_df) >= 2 else 0.0
        rows.append(
            {
                "city": city,
                "city_code": city_df["city_code"].iloc[0] if "city_code" in city_df.columns else "",
                "province": city_df["province"].iloc[0],
                "region": city_df["region"].iloc[0],
                "lat": city_df["lat"].iloc[0],
                "lon": city_df["lon"].iloc[0],
                "temp_slope_per_month": round(slope, 4),
                "avg_temp_5y": round(city_df["avg_temp"].mean(), 2),
                "precipitation_5y": round(city_df["precipitation"].mean(), 2),
                "comfort_5y": round(city_df["comfort_index"].mean(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("temp_slope_per_month", ascending=False)


def make_geo_scatter(trend_df: pd.DataFrame) -> Scatter:
    chart = Scatter(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="100%", height="560px"))
    chart.add_xaxis([])
    chart.add_yaxis(
        "城市气候样本",
        [
            opts.ScatterItem(
                name=row["city"],
                value=[
                    round(row["lon"], 2),
                    round(row["lat"], 2),
                    round(row["avg_temp_5y"], 2),
                    round(row["temp_slope_per_month"], 4),
                ],
                symbol_size=max(10, min(28, 10 + row["avg_temp_5y"] * 0.45)),
            )
            for _, row in trend_df.iterrows()
        ],
        color="#d05f3f",
    )
    chart.set_series_opts(
        label_opts=opts.LabelOpts(
            is_show=False
        ),
        itemstyle_opts=opts.ItemStyleOpts(opacity=0.9),
    )
    chart.set_global_opts(
        title_opts=opts.TitleOpts(
            title="中国主要城市近五年气候空间分布",
            subtitle="横轴为经度，纵轴为纬度，颜色与点径共同反映近五年均温",
            pos_left="3%",
            title_textstyle_opts=opts.TextStyleOpts(font_size=22, font_weight="bold", color="#17324d"),
            subtitle_textstyle_opts=opts.TextStyleOpts(color="#5f7488"),
        ),
        xaxis_opts=opts.AxisOpts(
            name="经度",
            min_=80,
            max_=130,
            splitline_opts=opts.SplitLineOpts(is_show=True),
            axislabel_opts=opts.LabelOpts(color="#567086"),
            name_textstyle_opts=opts.TextStyleOpts(color="#3f5d73"),
        ),
        yaxis_opts=opts.AxisOpts(
            name="纬度",
            min_=18,
            max_=48,
            splitline_opts=opts.SplitLineOpts(is_show=True),
            axislabel_opts=opts.LabelOpts(color="#567086"),
            name_textstyle_opts=opts.TextStyleOpts(color="#3f5d73"),
        ),
        visualmap_opts=opts.VisualMapOpts(
            min_=-5,
            max_=28,
            dimension=2,
            pos_left="2%",
            pos_top="28%",
            is_calculable=True,
            range_text=["高温", "低温"],
        ),
        tooltip_opts=opts.TooltipOpts(
            formatter=JsCode(
                "function(params){return params.name"
                "+ '<br/>经度: ' + params.value[0] + '°'"
                "+ '<br/>纬度: ' + params.value[1] + '°'"
                "+ '<br/>近五年均温: ' + params.value[2] + ' ℃'"
                "+ '<br/>月度趋势斜率: ' + params.value[3];}"
            )
        )
    )
    return chart


def make_trend_line(monthly_df: pd.DataFrame) -> Line:
    national = (
        monthly_df.groupby("year_month", as_index=False)
        .agg(avg_temp=("avg_temp", "mean"), precipitation=("precipitation", "mean"))
        .sort_values("year_month")
    )
    temp_baseline = national["avg_temp"].mean()
    national["temp_anomaly"] = (national["avg_temp"] - temp_baseline).round(2)

    line = Line(init_opts=opts.InitOpts(theme=ThemeType.ROMA, width="100%", height="420px"))
    line.add_xaxis(national["year_month"].tolist())
    line.add_yaxis(
        "全国样本平均气温",
        national["avg_temp"].round(2).tolist(),
        is_smooth=True,
        symbol_size=5,
        label_opts=opts.LabelOpts(is_show=False),
    )
    line.add_yaxis(
        "气温距平",
        national["temp_anomaly"].round(2).tolist(),
        is_smooth=True,
        symbol_size=4,
        label_opts=opts.LabelOpts(is_show=False),
    )
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="近五年全国样本月度气温趋势与距平"),
        yaxis_opts=opts.AxisOpts(name="温度 (℃)"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
    )
    return line


def make_heatmap(monthly_df: pd.DataFrame) -> HeatMap:
    region_month = (
        monthly_df.groupby(["region", "month"], as_index=False)["avg_temp"]
        .mean()
        .round(2)
    )
    regions = ["东北", "华北", "华东", "华中", "华南", "西南", "西北"]
    months = [f"{m}月" for m in range(1, 13)]
    heat_data = []
    for region_idx, region in enumerate(regions):
        sub = region_month[region_month["region"] == region]
        for _, row in sub.iterrows():
            heat_data.append([int(row["month"]) - 1, region_idx, float(row["avg_temp"])])

    chart = HeatMap(init_opts=opts.InitOpts(theme=ThemeType.MACARONS, width="100%", height="420px"))
    chart.add_xaxis(months)
    chart.add_yaxis("月均温", regions, heat_data)
    chart.set_global_opts(
        title_opts=opts.TitleOpts(title="区域月度平均气温热力图"),
        visualmap_opts=opts.VisualMapOpts(min_=-20, max_=32, pos_right="10"),
    )
    return chart


def make_warming_bar(trend_df: pd.DataFrame) -> Bar:
    top = trend_df.head(12).sort_values("temp_slope_per_month")
    chart = Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="420px"))
    chart.add_xaxis(top["city"].tolist())
    chart.add_yaxis("温度趋势斜率", top["temp_slope_per_month"].round(4).tolist(), color="#457b9d")
    chart.reversal_axis()
    chart.set_series_opts(label_opts=opts.LabelOpts(position="right"))
    chart.set_global_opts(
        title_opts=opts.TitleOpts(title="近五年升温趋势最明显城市 TOP12"),
        xaxis_opts=opts.AxisOpts(name="每月升温斜率"),
        yaxis_opts=opts.AxisOpts(name="城市"),
        legend_opts=opts.LegendOpts(is_show=False),
    )
    return chart


def make_radar(monthly_df: pd.DataFrame) -> Radar:
    region_stats = (
        monthly_df.groupby("region", as_index=False)
        .agg(
            avg_temp=("avg_temp", "mean"),
            precipitation=("precipitation", "mean"),
            wind_speed=("wind_speed", "mean"),
            apparent_temp=("apparent_temp", "mean"),
            wind_gusts=("wind_gusts", "mean"),
            comfort_index=("comfort_index", "mean"),
        )
        .round(2)
    )
    show_regions = region_stats[region_stats["region"].isin(["华东", "华南", "西南", "西北"])]
    chart = Radar(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="100%", height="500px"))
    chart.add_schema(
        schema=[
            opts.RadarIndicatorItem(name="气温", max_=30),
            opts.RadarIndicatorItem(name="降水", max_=260),
            opts.RadarIndicatorItem(name="风速", max_=6),
            opts.RadarIndicatorItem(name="体感温度", max_=30),
            opts.RadarIndicatorItem(name="舒适度", max_=28),
        ],
        shape="circle",
        center=["50%", "56%"],
        radius="68%",
        splitarea_opt=opts.SplitAreaOpts(
            is_show=True,
            areastyle_opts=opts.AreaStyleOpts(opacity=0.08, color=["#dbeafe", "#eff6ff"]),
        ),
        textstyle_opts=opts.TextStyleOpts(color="#28465e", font_size=13),
    )
    for _, row in show_regions.iterrows():
        chart.add(
            row["region"],
            [[row["avg_temp"], row["precipitation"], row["wind_speed"], row["apparent_temp"], row["comfort_index"]]],
            linestyle_opts=opts.LineStyleOpts(width=2),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.16),
        )
    chart.set_global_opts(
        title_opts=opts.TitleOpts(
            title="典型区域气候特征雷达图",
            pos_left="3%",
            title_textstyle_opts=opts.TextStyleOpts(font_size=20, font_weight="bold", color="#17324d"),
        ),
        legend_opts=opts.LegendOpts(pos_top="4%", pos_right="4%"),
    )
    return chart


def make_scatter(monthly_df: pd.DataFrame) -> Scatter:
    latest_year = int(monthly_df["year"].max())
    sample = (
        monthly_df[monthly_df["year"] == latest_year]
        .groupby("city", as_index=False)
        .agg(
            avg_temp=("avg_temp", "mean"),
            precipitation=("precipitation", "mean"),
            wind_speed=("wind_speed", "mean"),
        )
        .round(2)
    )
    chart = Scatter(init_opts=opts.InitOpts(theme=ThemeType.VINTAGE, width="100%", height="380px"))
    chart.add_xaxis(sample["avg_temp"].round(2).tolist())
    chart.add_yaxis(
        f"{latest_year} 年城市样本",
        [
            opts.ScatterItem(
                name=row["city"],
                value=[round(row["avg_temp"], 2), round(row["precipitation"], 2), round(row["wind_speed"], 2)],
                symbol_size=max(10, min(26, 8 + row["wind_speed"] * 0.6)),
            )
            for _, row in sample.iterrows()
        ],
        symbol_size=14,
    )
    chart.set_series_opts(
        label_opts=opts.LabelOpts(
            is_show=False
        )
    )
    chart.set_global_opts(
        title_opts=opts.TitleOpts(title=f"{latest_year} 年城市温度与降水关系散点图"),
        xaxis_opts=opts.AxisOpts(name="年平均气温 (℃)"),
        yaxis_opts=opts.AxisOpts(name="月平均降水量 (mm)"),
        tooltip_opts=opts.TooltipOpts(
            formatter=JsCode(
                "function(params){return params.name + '<br/>年平均气温: ' + params.value[0] + ' ℃<br/>月平均降水量: ' + params.value[1] + ' mm<br/>平均风速: ' + params.value[2] + ' m/s';}"
            )
        ),
    )
    return chart


def build_map_timeline_config(monthly_df: pd.DataFrame) -> dict:
    metric_labels = {
        "avg_temp": "平均气温",
        "precipitation": "降水量",
        "wind_speed": "平均风速",
    }
    metric_units = {
        "avg_temp": "℃",
        "precipitation": "mm",
        "wind_speed": "m/s",
    }
    province_month = (
        monthly_df.groupby(["province", "year_month"], as_index=False)
        .agg(
            avg_temp=("avg_temp", "mean"),
            precipitation=("precipitation", "mean"),
            wind_speed=("wind_speed", "mean"),
        )
        .round(2)
    )
    provinces = sorted(province_month["province"].dropna().unique().tolist())
    periods = sorted(province_month["year_month"].dropna().unique().tolist())
    province_display_names = {province: PROVINCE_NAME_MAP.get(province, province) for province in provinces}
    reverse_display_names = {display: short for short, display in province_display_names.items()}

    metrics_payload: dict[str, dict] = {}
    for metric in metric_labels:
        metric_series = []
        metric_values: list[float] = []
        for period in periods:
            period_df = province_month[province_month["year_month"] == period]
            value_map = {row["province"]: row[metric] for _, row in period_df.iterrows()}
            period_metric_values: list[float] = []
            series_data = []
            for province in provinces:
                value = value_map.get(province)
                display_name = province_display_names[province]
                if pd.notna(value):
                    numeric_value = round(float(value), 2)
                    metric_values.append(float(value))
                    period_metric_values.append(float(value))
                    series_data.append({"name": display_name, "value": numeric_value})
                else:
                    series_data.append({"name": display_name, "value": None})
            period_min = round(min(period_metric_values), 2) if period_metric_values else 0
            period_max = round(max(period_metric_values), 2) if period_metric_values else 1
            if period_min == period_max:
                period_max = round(period_min + 1, 2)
            metric_series.append(
                {
                    "period": period,
                    "data": series_data,
                    "min": period_min,
                    "max": period_max,
                }
            )

        metrics_payload[metric] = {
            "label": metric_labels[metric],
            "unit": metric_units[metric],
            "min": round(min(metric_values), 2) if metric_values else 0,
            "max": round(max(metric_values), 2) if metric_values else 1,
            "series": metric_series,
        }

    national_series = (
        province_month.groupby("year_month", as_index=False)
        .agg(
            avg_temp=("avg_temp", "mean"),
            precipitation=("precipitation", "mean"),
            wind_speed=("wind_speed", "mean"),
        )
        .round(2)
    )

    province_trends: dict[str, dict[str, list[float | None]]] = {}
    for province in provinces:
        province_df = province_month[province_month["province"] == province].set_index("year_month")
        province_trends[province] = {
            metric: [
                round(float(province_df.loc[period, metric]), 2) if period in province_df.index and pd.notna(province_df.loc[period, metric]) else None
                for period in periods
            ]
            for metric in metric_labels
        }

    return {
        "periods": periods,
        "metrics": metrics_payload,
        "province_trends": province_trends,
        "province_display_names": province_display_names,
        "province_name_lookup": reverse_display_names,
        "national_trends": {
            metric: [
                round(float(national_series.set_index("year_month").loc[period, metric]), 2)
                if period in national_series.set_index("year_month").index
                else None
                for period in periods
            ]
            for metric in metric_labels
        },
    }


def build_map_timeline_config_v2(monthly_df: pd.DataFrame) -> dict:
    monthly_df = monthly_df.copy()
    if "city_code" not in monthly_df.columns:
        monthly_df["city_code"] = ""
    monthly_df["province"] = monthly_df["province"].map(lambda value: normalize_province_name(str(value)))
    metric_labels = {
        "avg_temp": "average temperature",
        "precipitation": "precipitation",
        "wind_speed": "wind speed",
    }
    metric_units = {
        "avg_temp": "°C",
        "precipitation": "mm",
        "wind_speed": "m/s",
    }
    province_month = (
        monthly_df.groupby(["province", "year_month"], as_index=False)
        .agg(
            avg_temp=("avg_temp", "mean"),
            precipitation=("precipitation", "mean"),
            wind_speed=("wind_speed", "mean"),
        )
        .round(2)
    )
    city_month = (
        monthly_df.groupby(["province", "city", "city_code", "lat", "lon", "year_month"], as_index=False)
        .agg(
            avg_temp=("avg_temp", "mean"),
            precipitation=("precipitation", "mean"),
            wind_speed=("wind_speed", "mean"),
        )
        .round(2)
    )
    provinces = sorted(province_month["province"].dropna().unique().tolist())
    periods = sorted(province_month["year_month"].dropna().unique().tolist())
    province_display_names = {province: province for province in provinces}
    reverse_display_names = {display: short for short, display in province_display_names.items()}

    metrics_payload: dict[str, dict] = {}
    for metric in metric_labels:
        metric_series = []
        metric_values: list[float] = []
        for period in periods:
            period_df = province_month[province_month["year_month"] == period]
            value_map = {row["province"]: row[metric] for _, row in period_df.iterrows()}
            period_metric_values: list[float] = []
            series_data = []
            for province in provinces:
                value = value_map.get(province)
                display_name = province_display_names[province]
                if pd.notna(value):
                    numeric_value = round(float(value), 2)
                    metric_values.append(float(value))
                    period_metric_values.append(float(value))
                    series_data.append({"name": display_name, "value": numeric_value})
                else:
                    series_data.append({"name": display_name, "value": None})
            period_min = round(min(period_metric_values), 2) if period_metric_values else 0
            period_max = round(max(period_metric_values), 2) if period_metric_values else 1
            if period_min == period_max:
                period_max = round(period_min + 1, 2)
            metric_series.append({"period": period, "data": series_data, "min": period_min, "max": period_max})

        metrics_payload[metric] = {
            "label": metric_labels[metric],
            "unit": metric_units[metric],
            "min": round(min(metric_values), 2) if metric_values else 0,
            "max": round(max(metric_values), 2) if metric_values else 1,
            "series": metric_series,
        }

    national_series = (
        province_month.groupby("year_month", as_index=False)
        .agg(
            avg_temp=("avg_temp", "mean"),
            precipitation=("precipitation", "mean"),
            wind_speed=("wind_speed", "mean"),
        )
        .round(2)
        .set_index("year_month")
    )

    province_trends: dict[str, dict[str, list[float | None]]] = {}
    province_city_metrics: dict[str, dict[str, dict]] = {}
    for province in provinces:
        province_df = province_month[province_month["province"] == province].set_index("year_month")
        province_trends[province] = {
            metric: [
                round(float(province_df.loc[period, metric]), 2) if period in province_df.index and pd.notna(province_df.loc[period, metric]) else None
                for period in periods
            ]
            for metric in metric_labels
        }

        province_city_metrics[province] = {}
        province_city_df = city_month[city_month["province"] == province]
        for metric in metric_labels:
            metric_series = []
            metric_values: list[float] = []
            for period in periods:
                period_df = province_city_df[province_city_df["year_month"] == period]
                period_metric_values: list[float] = []
                series_data = []
                for _, row in period_df.iterrows():
                    value = row[metric]
                    if pd.isna(value):
                        continue
                    series_data.append(
                        {
                            "name": row["city"],
                            "value": [round(float(row["lon"]), 4), round(float(row["lat"]), 4), round(float(value), 2)],
                            "city_code": row["city_code"] if pd.notna(row["city_code"]) else "",
                        }
                    )
                    metric_values.append(float(value))
                    period_metric_values.append(float(value))
                period_min = round(min(period_metric_values), 2) if period_metric_values else 0
                period_max = round(max(period_metric_values), 2) if period_metric_values else 1
                if period_min == period_max:
                    period_max = round(period_min + 1, 2)
                metric_series.append({"period": period, "data": series_data, "min": period_min, "max": period_max})

            province_city_metrics[province][metric] = {
                "label": metric_labels[metric],
                "unit": metric_units[metric],
                "min": round(min(metric_values), 2) if metric_values else 0,
                "max": round(max(metric_values), 2) if metric_values else 1,
                "series": metric_series,
            }

    return {
        "periods": periods,
        "metrics": metrics_payload,
        "province_city_metrics": province_city_metrics,
        "province_codes": {province: PROVINCE_CODE_MAP.get(province, "") for province in provinces},
        "province_trends": province_trends,
        "province_display_names": province_display_names,
        "province_name_lookup": reverse_display_names,
        "national_trends": {
            metric: [round(float(national_series.loc[period, metric]), 2) if period in national_series.index else None for period in periods]
            for metric in metric_labels
        },
    }


def build_dashboard(monthly_df: pd.DataFrame, trend_df: pd.DataFrame) -> None:
    geo_chart = make_geo_scatter(trend_df)
    line_chart = make_trend_line(monthly_df)
    heatmap_chart = make_heatmap(monthly_df)
    warming_chart = make_warming_bar(trend_df)
    radar_chart = make_radar(monthly_df)
    scatter_chart = make_scatter(monthly_df)
    map_timeline_config = build_map_timeline_config_v2(monthly_df)

    hottest_city = trend_df.sort_values("avg_temp_5y", ascending=False).iloc[0]
    coolest_city = trend_df.sort_values("avg_temp_5y", ascending=True).iloc[0]
    wettest_city = trend_df.sort_values("precipitation_5y", ascending=False).iloc[0]
    fastest_warming = trend_df.sort_values("temp_slope_per_month", ascending=False).iloc[0]
    city_count = int(monthly_df["city"].nunique())
    month_count = int(monthly_df["year_month"].nunique())

    dependencies = sorted(
        set(
            list(geo_chart.js_dependencies.items)
            + list(line_chart.js_dependencies.items)
            + list(heatmap_chart.js_dependencies.items)
            + list(warming_chart.js_dependencies.items)
            + list(radar_chart.js_dependencies.items)
            + list(scatter_chart.js_dependencies.items)
        )
    )
    scripts = "\n".join(
        f'<script src="{CurrentConfig.ONLINE_HOST}{dep}.js"></script>' for dep in dependencies
    )
    map_config_json = json.dumps(map_timeline_config, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>中国近五年天气可视化分析</title>
    {scripts}
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
            color: #13283b;
            background:
                radial-gradient(circle at top left, rgba(103, 163, 255, 0.16), transparent 24%),
                radial-gradient(circle at right 20%, rgba(55, 198, 171, 0.14), transparent 20%),
                linear-gradient(180deg, #eef5fb 0%, #e4eef8 100%);
        }}
        .dashboard {{
            max-width: 1660px;
            margin: 0 auto;
            padding: 28px 24px 36px;
        }}
        .hero {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 24px;
            margin-bottom: 22px;
        }}
        .hero h1 {{
            margin: 0 0 8px;
            font-size: 34px;
            letter-spacing: 0.5px;
        }}
        .hero p {{
            margin: 0;
            color: #5c7388;
            font-size: 15px;
        }}
        .hero-badge {{
            padding: 10px 16px;
            border-radius: 999px;
            background: rgba(19, 40, 59, 0.08);
            color: #26455c;
            font-size: 13px;
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
            margin-bottom: 18px;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(255, 255, 255, 0.65);
            border-radius: 20px;
            padding: 18px 20px;
            box-shadow: 0 18px 40px rgba(36, 67, 98, 0.08);
            backdrop-filter: blur(10px);
        }}
        .card-label {{
            color: #6d8498;
            font-size: 13px;
            margin-bottom: 10px;
        }}
        .card-value {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 6px;
        }}
        .card-sub {{
            color: #527089;
            font-size: 13px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(12, minmax(0, 1fr));
            gap: 18px;
        }}
        .panel {{
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.68);
            border-radius: 24px;
            box-shadow: 0 22px 52px rgba(27, 57, 86, 0.08);
            padding: 10px 10px 4px;
            overflow: hidden;
        }}
        .panel-head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 14px;
            padding: 12px 16px 0;
            flex-wrap: wrap;
        }}
        .panel-title {{
            font-size: 22px;
            font-weight: 700;
            color: #17324d;
        }}
        .panel-sub {{
            color: #61788d;
            font-size: 13px;
            margin-top: 4px;
        }}
        .metric-switch {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .map-actions {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .job-actions {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .metric-btn {{
            border: 0;
            border-radius: 999px;
            padding: 10px 16px;
            background: #e9f1f9;
            color: #2a4b63;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.18s ease;
        }}
        .metric-btn.active {{
            background: linear-gradient(135deg, #1f6aa5 0%, #37a2b4 100%);
            color: #fff;
            box-shadow: 0 12px 24px rgba(41, 115, 168, 0.24);
        }}
        .ghost-btn {{
            border: 0;
            border-radius: 999px;
            padding: 10px 14px;
            background: rgba(31, 106, 165, 0.1);
            color: #25506f;
            font-size: 13px;
            cursor: pointer;
        }}
        .job-btn {{
            border: 0;
            border-radius: 999px;
            padding: 10px 14px;
            background: linear-gradient(135deg, #17486a 0%, #26758f 100%);
            color: #fff;
            font-size: 13px;
            cursor: pointer;
            box-shadow: 0 10px 22px rgba(34, 100, 129, 0.22);
        }}
        .status-pill {{
            padding: 9px 14px;
            border-radius: 999px;
            background: rgba(19, 40, 59, 0.08);
            color: #35546d;
            font-size: 13px;
        }}
        .job-status {{
            padding: 9px 14px;
            border-radius: 999px;
            background: rgba(20, 67, 108, 0.08);
            color: #2f5877;
            font-size: 13px;
        }}
        .job-progress-wrap {{
            min-width: 320px;
            flex: 1;
            max-width: 520px;
        }}
        .job-progress-head {{
            display: flex;
            justify-content: space-between;
            color: #60788d;
            font-size: 12px;
            margin-bottom: 8px;
        }}
        .job-progress-track {{
            position: relative;
            height: 10px;
            border-radius: 999px;
            background: rgba(81, 118, 145, 0.18);
            overflow: hidden;
        }}
        .job-progress-bar {{
            position: absolute;
            inset: 0 auto 0 0;
            width: 0%;
            border-radius: 999px;
            background: linear-gradient(90deg, #1f6aa5 0%, #2fbec0 100%);
            box-shadow: 0 0 18px rgba(47, 190, 192, 0.3);
            transition: width 0.35s ease;
        }}
        .map-meta {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 0 16px 8px;
            flex-wrap: wrap;
        }}
        .month-chip {{
            padding: 10px 16px;
            border-radius: 16px;
            background: linear-gradient(135deg, #16324b 0%, #246b9a 100%);
            color: #fff;
            box-shadow: 0 14px 28px rgba(28, 86, 128, 0.26);
        }}
        .month-chip-label {{
            font-size: 12px;
            opacity: 0.8;
            margin-bottom: 4px;
        }}
        .month-chip-value {{
            font-size: 24px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        .progress-wrap {{
            min-width: 280px;
            flex: 1;
        }}
        .progress-head {{
            display: flex;
            justify-content: space-between;
            color: #60788d;
            font-size: 12px;
            margin-bottom: 8px;
        }}
        .progress-track {{
            position: relative;
            height: 10px;
            border-radius: 999px;
            background: rgba(81, 118, 145, 0.18);
            overflow: hidden;
        }}
        .progress-bar {{
            position: absolute;
            inset: 0 auto 0 0;
            width: 0%;
            border-radius: 999px;
            background: linear-gradient(90deg, #2f7fbe 0%, #41c0c3 100%);
            box-shadow: 0 0 18px rgba(65, 192, 195, 0.35);
            transition: width 0.9s ease;
        }}
        #timeline-map {{
            width: 100%;
            height: 640px;
        }}
        #province-trend {{
            width: 100%;
            height: 420px;
        }}
        .span-12 {{ grid-column: span 12; }}
        .span-7 {{ grid-column: span 7; }}
        .span-5 {{ grid-column: span 5; }}
        .span-6 {{ grid-column: span 6; }}
        @media (max-width: 1200px) {{
            .cards {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
            .span-7, .span-5, .span-6 {{
                grid-column: span 12;
            }}
        }}
        @media (max-width: 720px) {{
            .dashboard {{
                padding: 18px 12px 28px;
            }}
            .hero {{
                flex-direction: column;
                align-items: flex-start;
            }}
            .cards {{
                grid-template-columns: 1fr;
            }}
            .hero h1 {{
                font-size: 28px;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="hero">
            <div>
                <h1>中国近五年天气可视化分析</h1>
                <p>基于中国主要城市近五年历史天气数据生成，覆盖均温、降水、升温趋势、区域差异与体感特征。</p>
            </div>
            <div class="hero-badge">样本城市 {city_count} 个 · 月度样本 {month_count} 个月</div>
        </div>

        <div class="cards">
            <div class="card">
                <div class="card-label">近五年均温最高城市</div>
                <div class="card-value">{hottest_city['city']}</div>
                <div class="card-sub">{hottest_city['avg_temp_5y']:.2f} ℃</div>
            </div>
            <div class="card">
                <div class="card-label">近五年均温最低城市</div>
                <div class="card-value">{coolest_city['city']}</div>
                <div class="card-sub">{coolest_city['avg_temp_5y']:.2f} ℃</div>
            </div>
            <div class="card">
                <div class="card-label">月均降水最强城市</div>
                <div class="card-value">{wettest_city['city']}</div>
                <div class="card-sub">{wettest_city['precipitation_5y']:.2f} mm</div>
            </div>
            <div class="card">
                <div class="card-label">升温趋势最明显城市</div>
                <div class="card-value">{fastest_warming['city']}</div>
                <div class="card-sub">月度斜率 {fastest_warming['temp_slope_per_month']:.4f}</div>
            </div>
        </div>

        <div class="grid">
            <section class="panel span-12">
                <div class="panel-head">
                    <div>
                        <div class="panel-title">中国分省气候时序地图</div>
                        <div class="panel-sub">拖动时间轴查看不同月份各省气温、降水和风速变化，点击右侧按钮切换指标。</div>
                    </div>
                    <div class="map-actions">
                        <div class="metric-switch">
                            <button class="metric-btn active" data-metric="avg_temp">平均气温</button>
                            <button class="metric-btn" data-metric="precipitation">降水量</button>
                            <button class="metric-btn" data-metric="wind_speed">平均风速</button>
                        </div>
                        <div class="job-actions">
                            <button class="job-btn" id="refresh-all-btn">更新全部数据</button>
                            <button class="job-btn" id="refresh-failed-btn">仅重抓失败数据</button>
                            <button class="ghost-btn" id="cancel-job-btn" disabled>中断任务</button>
                        </div>
                        <button class="ghost-btn" id="timeline-play">自动播放</button>
                        <button class="ghost-btn" id="timeline-reset">查看全国</button>
                        <div class="status-pill" id="selected-province-label">当前联动：全国平均</div>
                        <div class="job-status" id="job-status-label">数据任务：未连接按钮服务</div>
                    </div>
                </div>
                <div class="map-meta">
                    <div class="month-chip">
                        <div class="month-chip-label">当前月份</div>
                        <div class="month-chip-value" id="current-period-chip">2021-04</div>
                    </div>
                    <div class="job-progress-wrap">
                        <div class="job-progress-head">
                            <span>数据更新进度</span>
                            <span id="job-progress-text">0 / 0</span>
                        </div>
                        <div class="job-progress-track">
                            <div class="job-progress-bar" id="job-progress-bar"></div>
                        </div>
                    </div>
                    <div class="progress-wrap">
                        <div class="progress-head">
                            <span>播放进度</span>
                            <span id="progress-text">1 / 60</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-bar" id="timeline-progress-bar"></div>
                        </div>
                    </div>
                </div>
                <div id="timeline-map"></div>
            </section>
            <section class="panel span-12">
                <div class="panel-head">
                    <div>
                        <div class="panel-title">省份联动月度趋势</div>
                        <div class="panel-sub">点击上方地图任一省份后，下方趋势自动切换为该省；默认展示全国平均趋势。</div>
                    </div>
                </div>
                <div id="province-trend"></div>
            </section>
            <section class="panel span-12">{geo_chart.render_embed()}</section>
            <section class="panel span-12">{line_chart.render_embed()}</section>
            <section class="panel span-7">{heatmap_chart.render_embed()}</section>
            <section class="panel span-5">{warming_chart.render_embed()}</section>
            <section class="panel span-6">{radar_chart.render_embed()}</section>
            <section class="panel span-6">{scatter_chart.render_embed()}</section>
        </div>
    </div>
    <script>
        const timelineMapConfig = {map_config_json};
        const dashboardApiBase = 'http://127.0.0.1:{DEFAULT_DASHBOARD_API_PORT}';
        const timelineMapEl = document.getElementById('timeline-map');
        const provinceTrendEl = document.getElementById('province-trend');
        const metricButtons = Array.from(document.querySelectorAll('.metric-btn'));
        const playBtn = document.getElementById('timeline-play');
        const resetBtn = document.getElementById('timeline-reset');
        const refreshAllBtn = document.getElementById('refresh-all-btn');
        const refreshFailedBtn = document.getElementById('refresh-failed-btn');
        const cancelJobBtn = document.getElementById('cancel-job-btn');
        const jobStatusLabel = document.getElementById('job-status-label');
        const jobProgressText = document.getElementById('job-progress-text');
        const jobProgressBar = document.getElementById('job-progress-bar');
        const provinceLabel = document.getElementById('selected-province-label');
        const currentPeriodChip = document.getElementById('current-period-chip');
        const progressText = document.getElementById('progress-text');
        const progressBar = document.getElementById('timeline-progress-bar');
        const timelineMapChart = echarts.init(timelineMapEl, null, {{ renderer: 'canvas' }});
        const provinceTrendChart = echarts.init(provinceTrendEl, null, {{ renderer: 'canvas' }});
        let lastJobRunning = false;
        let lastJobUpdatedAt = '';
        let currentTimelineIndex = 0;
        let autoPlayTimer = null;
        let autoPlayEnabled = false;
        let isInterpolating = false;
        let suppressTimelineEvent = false;
        let mapMode = 'national';
        let drilldownProvince = null;
        const provinceMapCache = new Map();

        async function loadChinaGeoJson() {{
            const urls = [
                'https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json',
                'https://geo.datav.aliyun.com/areas_v3/bound/geojson?code=100000_full'
            ];
            for (const url of urls) {{
                try {{
                    const res = await fetch(url);
                    if (res.ok) {{
                        return await res.json();
                    }}
                }} catch (err) {{}}
            }}
            throw new Error('中国地图 GeoJSON 加载失败');
        }}

        function makeTimelineOption(metric) {{
            const meta = timelineMapConfig.metrics[metric];
            return {{
                baseOption: {{
                    backgroundColor: 'transparent',
                    animation: true,
                    animationDuration: 1100,
                    animationDurationUpdate: 1400,
                    animationEasing: 'quarticOut',
                    animationEasingUpdate: 'cubicInOut',
                    timeline: {{
                        axisType: 'category',
                        autoPlay: false,
                        playInterval: 2300,
                        data: timelineMapConfig.periods,
                        bottom: 18,
                        left: 60,
                        right: 60,
                        label: {{
                            color: '#597287'
                        }},
                        lineStyle: {{
                            color: '#8db2cf'
                        }},
                        checkpointStyle: {{
                            color: '#1f6aa5',
                            borderColor: '#ffffff',
                            borderWidth: 2
                        }},
                        tooltip: {{
                            show: true,
                            formatter: function(params) {{
                                return '时间：' + (params.name || params.value || '');
                            }}
                        }},
                        controlStyle: {{
                            color: '#1f6aa5',
                            borderColor: '#1f6aa5'
                        }}
                    }},
                    title: {{
                        text: '指标：' + meta.label,
                        left: 20,
                        top: 10,
                        textStyle: {{
                            color: '#17324d',
                            fontSize: 18,
                            fontWeight: 'bold'
                        }}
                    }},
                    tooltip: {{
                        trigger: 'item',
                        formatter: function(params) {{
                            const period = timelineMapConfig.periods[currentTimelineIndex] || '';
                            const value = params.value == null ? '暂无数据' : params.value + ' ' + meta.unit;
                            return params.name + '<br/>' + meta.label + '：' + value + '<br/>时间：' + period;
                        }}
                    }},
                    visualMap: {{
                        min: meta.min,
                        max: meta.max,
                        left: 26,
                        bottom: 84,
                        calculable: true,
                        text: ['高', '低'],
                        textStyle: {{
                            color: '#4b657c'
                        }},
                        inRange: {{
                            color: ['#08306b', '#2171b5', '#6baed6', '#deebf7', '#fee0d2', '#fc9272', '#de2d26', '#a50f15']
                        }}
                    }},
                    series: [
                        {{
                            name: meta.label,
                            type: 'map',
                            map: 'china-custom',
                            roam: true,
                            zoom: 1.12,
                            animationDuration: 1100,
                            animationDurationUpdate: 1400,
                            animationEasing: 'quarticOut',
                            animationEasingUpdate: 'cubicInOut',
                            universalTransition: true,
                            emphasis: {{
                                label: {{
                                    color: '#0e2233'
                                }},
                                itemStyle: {{
                                    areaColor: '#ffd166'
                                }}
                            }},
                            itemStyle: {{
                                borderColor: 'rgba(255,255,255,0.72)',
                                borderWidth: 0.8,
                                areaColor: '#edf4fb'
                            }},
                            label: {{
                                show: true,
                                fontSize: 10,
                                color: '#3e6078'
                            }}
                        }}
                    ]
                }},
                options: meta.series.map(item => {{
                    return {{
                        title: {{
                            subtext: '时间：' + item.period,
                            subtextStyle: {{
                                color: '#627b91'
                            }}
                        }},
                        visualMap: {{
                            min: item.min,
                            max: item.max
                        }},
                        series: [{{ data: item.data }}]
                    }};
                }})
            }};
        }}

        async function loadProvinceGeoJson(province) {{
            const code = timelineMapConfig.province_codes[province];
            if (!code) {{
                throw new Error('missing province code');
            }}
            const cacheKey = 'province-' + code;
            if (provinceMapCache.has(cacheKey)) {{
                return provinceMapCache.get(cacheKey);
            }}
            const urls = [
                `https://geo.datav.aliyun.com/areas_v3/bound/${{code}}_full.json`,
                `https://geo.datav.aliyun.com/areas_v3/bound/geojson?code=${{code}}_full`
            ];
            for (const url of urls) {{
                try {{
                    const res = await fetch(url);
                    if (res.ok) {{
                        const geoJson = await res.json();
                        echarts.registerMap(cacheKey, geoJson);
                        provinceMapCache.set(cacheKey, cacheKey);
                        return cacheKey;
                    }}
                }} catch (err) {{}}
            }}
            throw new Error('province geojson load failed');
        }}

        function getProvinceMetricSeriesData(province, metric, index) {{
            return timelineMapConfig.province_city_metrics?.[province]?.[metric]?.series?.[index]?.data || [];
        }}

        function getProvinceMetricRange(province, metric, index) {{
            const meta = timelineMapConfig.province_city_metrics?.[province]?.[metric] || {{}};
            const item = meta.series?.[index] || {{}};
            const min = typeof item.min === 'number' ? item.min : (typeof meta.min === 'number' ? meta.min : 0);
            const max = typeof item.max === 'number' ? item.max : (typeof meta.max === 'number' ? meta.max : 1);
            return {{
                min: min,
                max: min === max ? max + 1 : max
            }};
        }}

        function makeProvinceTimelineOption(metric, province, mapKey) {{
            const meta = timelineMapConfig.province_city_metrics?.[province]?.[metric];
            if (!meta) {{
                return makeTimelineOption(metric);
            }}
            return {{
                baseOption: {{
                    backgroundColor: 'transparent',
                    animation: true,
                    animationDuration: 700,
                    animationDurationUpdate: 900,
                    timeline: {{
                        axisType: 'category',
                        autoPlay: false,
                        playInterval: 2300,
                        data: timelineMapConfig.periods,
                        bottom: 18,
                        left: 60,
                        right: 60
                    }},
                    title: {{
                        text: province + ' · ' + meta.label,
                        left: 20,
                        top: 10,
                        textStyle: {{
                            color: '#17324d',
                            fontSize: 18,
                            fontWeight: 'bold'
                        }}
                    }},
                    tooltip: {{
                        trigger: 'item',
                        formatter: function(params) {{
                            const value = Array.isArray(params.value) ? params.value[2] : params.value;
                            const period = timelineMapConfig.periods[currentTimelineIndex] || '';
                            return params.name + '<br/>' + meta.label + ': ' + value + ' ' + meta.unit + '<br/>time: ' + period;
                        }}
                    }},
                    visualMap: {{
                        min: meta.min,
                        max: meta.max,
                        left: 26,
                        bottom: 84,
                        calculable: true,
                        textStyle: {{
                            color: '#4b657c'
                        }},
                        inRange: {{
                            color: ['#08306b', '#2171b5', '#6baed6', '#deebf7', '#fee0d2', '#fc9272', '#de2d26', '#a50f15']
                        }}
                    }},
                    geo: {{
                        map: mapKey,
                        roam: true,
                        zoom: 1.05,
                        label: {{
                            show: true,
                            color: '#406175',
                            fontSize: 10
                        }},
                        itemStyle: {{
                            borderColor: 'rgba(255,255,255,0.72)',
                            borderWidth: 0.8,
                            areaColor: '#edf4fb'
                        }},
                        emphasis: {{
                            label: {{
                                color: '#0e2233'
                            }},
                            itemStyle: {{
                                areaColor: '#dceef9'
                            }}
                        }}
                    }},
                    series: [
                        {{
                            name: meta.label,
                            type: 'scatter',
                            coordinateSystem: 'geo',
                            symbolSize: function(value) {{
                                return Math.max(10, Math.min(24, 10 + Number(value[2] || 0) * 0.35));
                            }},
                            encode: {{ value: 2 }},
                            label: {{
                                show: true,
                                formatter: '{{b}}',
                                position: 'right',
                                color: '#17324d',
                                fontSize: 11
                            }},
                            itemStyle: {{
                                opacity: 0.9
                            }}
                        }}
                    ]
                }},
                options: meta.series.map(item => {{
                    return {{
                        title: {{
                            subtext: 'time: ' + item.period,
                            subtextStyle: {{
                                color: '#627b91'
                            }}
                        }},
                        visualMap: {{
                            min: item.min,
                            max: item.max
                        }},
                        series: [{{ data: item.data }}]
                    }};
                }})
            }};
        }}

        async function renderCurrentMap() {{
            if (mapMode === 'province' && drilldownProvince) {{
                const mapKey = await loadProvinceGeoJson(drilldownProvince);
                timelineMapChart.setOption(makeProvinceTimelineOption(activeMetric, drilldownProvince, mapKey), true);
                return;
            }}
            timelineMapChart.setOption(makeTimelineOption(activeMetric), true);
        }}

        let activeMetric = 'avg_temp';
        let selectedProvince = null;

        async function fetchJobStatus() {{
            try {{
                const res = await fetch(dashboardApiBase + '/api/status');
                if (!res.ok) throw new Error('status request failed');
                const data = await res.json();
                const failedCount = typeof data.failed_count === 'number' ? data.failed_count : 0;
                const successCount = typeof data.success_count === 'number' ? data.success_count : 0;
                const total = typeof data.total === 'number' ? data.total : 0;
                const completed = typeof data.completed === 'number' ? data.completed : 0;
                const currentCity = data.current_city || '';
                const progressPercent = total > 0 ? Math.min(100, (completed / total) * 100) : 0;
                jobProgressText.textContent = completed + ' / ' + total + ' · 成功 ' + successCount + ' · 失败 ' + failedCount;
                jobProgressBar.style.width = progressPercent.toFixed(2) + '%';
                refreshAllBtn.disabled = !!data.running;
                refreshFailedBtn.disabled = !!data.running;
                cancelJobBtn.disabled = !data.running;
                if (data.running) {{
                    jobStatusLabel.textContent = '数据任务：运行中，已完成 ' + completed + '/' + total + '，成功 ' + successCount + '，失败 ' + failedCount + (currentCity ? '，当前 ' + currentCity : '');
                }} else if (data.last_error) {{
                    jobStatusLabel.textContent = '数据任务：失败，成功 ' + successCount + '，失败 ' + failedCount + ' - ' + data.last_error;
                }} else if (data.message === '任务已中断') {{
                    jobStatusLabel.textContent = '数据任务：已中断，成功 ' + successCount + '，失败待重抓 ' + failedCount;
                }} else if (data.last_success) {{
                    jobStatusLabel.textContent = '数据任务：空闲，最近成功 ' + data.last_success + '，成功 ' + successCount + '，失败待重抓 ' + failedCount;
                }} else {{
                    jobStatusLabel.textContent = '数据任务：空闲，成功 ' + successCount + '，失败待重抓 ' + failedCount;
                }}

                if (lastJobRunning && !data.running && data.updated_at && data.updated_at !== lastJobUpdatedAt) {{
                    if (data.last_error) {{
                        window.alert('数据任务失败\\n成功：' + successCount + '\\n失败：' + failedCount + '\\n错误：' + data.last_error);
                    }} else if (data.message === '任务已中断') {{
                        window.alert('数据任务已中断\\n成功：' + successCount + '\\n失败待重抓：' + failedCount);
                    }} else {{
                        window.alert('数据任务完成\\n成功：' + successCount + '\\n失败：' + failedCount);
                        window.setTimeout(() => window.location.reload(), 800);
                    }}
                }}
                lastJobRunning = !!data.running;
                lastJobUpdatedAt = data.updated_at || '';
            }} catch (err) {{
                jobStatusLabel.textContent = '数据任务：未连接按钮服务';
                jobProgressText.textContent = '0 / 0 · 成功 0 · 失败 0';
                jobProgressBar.style.width = '0%';
                refreshAllBtn.disabled = false;
                refreshFailedBtn.disabled = false;
                cancelJobBtn.disabled = true;
            }}
        }}

        async function triggerDataJob(path) {{
            try {{
                if (path === '/api/update/cancel') {{
                    cancelJobBtn.disabled = true;
                }} else {{
                    refreshAllBtn.disabled = true;
                    refreshFailedBtn.disabled = true;
                    cancelJobBtn.disabled = false;
                }}
                const res = await fetch(dashboardApiBase + path, {{
                    method: 'POST'
                }});
                const data = await res.json();
                jobStatusLabel.textContent = '数据任务：' + data.message;
                lastJobRunning = false;
                lastJobUpdatedAt = '';
                fetchJobStatus();
            }} catch (err) {{
                jobStatusLabel.textContent = '数据任务：按钮服务不可用';
                refreshAllBtn.disabled = false;
                refreshFailedBtn.disabled = false;
                cancelJobBtn.disabled = true;
            }}
        }}

        function updateTimelineMeta() {{
            const total = timelineMapConfig.periods.length || 1;
            const current = currentTimelineIndex + 1;
            const period = timelineMapConfig.periods[currentTimelineIndex] || '-';
            currentPeriodChip.textContent = period;
            progressText.textContent = current + ' / ' + total;
            progressBar.style.width = ((current / total) * 100).toFixed(2) + '%';
        }}

        function getMetricSeriesData(metric, index) {{
            if (mapMode === 'province' && drilldownProvince) {{
                return getProvinceMetricSeriesData(drilldownProvince, metric, index);
            }}
            return timelineMapConfig.metrics[metric].series[index]?.data || [];
        }}

        function getMetricRange(metric, index) {{
            if (mapMode === 'province' && drilldownProvince) {{
                return getProvinceMetricRange(drilldownProvince, metric, index);
            }}
            const item = timelineMapConfig.metrics[metric].series[index] || {{}};
            const min = typeof item.min === 'number' ? item.min : timelineMapConfig.metrics[metric].min;
            const max = typeof item.max === 'number' ? item.max : timelineMapConfig.metrics[metric].max;
            return {{
                min: min,
                max: min === max ? max + 1 : max
            }};
        }}

        function interpolateSeriesData(metric, fromIndex, toIndex, progress) {{
            const eased = 0.5 - 0.5 * Math.cos(Math.PI * progress);
            const fromSeries = getMetricSeriesData(metric, fromIndex);
            const toSeries = getMetricSeriesData(metric, toIndex);
            return fromSeries.map((fromItem, idx) => {{
                const toItem = toSeries[idx] || fromItem;
                const fromVal = Array.isArray(fromItem.value)
                    ? Number(fromItem.value[2])
                    : (typeof fromItem.value === 'number' ? fromItem.value : null);
                const toVal = Array.isArray(toItem.value)
                    ? Number(toItem.value[2])
                    : (typeof toItem.value === 'number' ? toItem.value : null);
                let value = null;
                if (fromVal != null && toVal != null) {{
                    value = Number((fromVal + (toVal - fromVal) * eased).toFixed(2));
                }} else if (eased < 0.5) {{
                    value = fromVal;
                }} else {{
                    value = toVal;
                }}
                const fromCoords = Array.isArray(fromItem.value) ? fromItem.value.slice(0, 2) : null;
                const toCoords = Array.isArray(toItem.value) ? toItem.value.slice(0, 2) : fromCoords;
                return {{
                    name: toItem.name || fromItem.name,
                    value: fromCoords || toCoords ? [...(toCoords || fromCoords || []), value] : value
                }};
            }});
        }}

        function applyInterpolatedFrame(metric, fromIndex, toIndex, progress) {{
            const meta = mapMode === 'province' && drilldownProvince
                ? timelineMapConfig.province_city_metrics?.[drilldownProvince]?.[metric]
                : timelineMapConfig.metrics[metric];
            const fromPeriod = timelineMapConfig.periods[fromIndex] || '-';
            const toPeriod = timelineMapConfig.periods[toIndex] || '-';
            const blended = interpolateSeriesData(metric, fromIndex, toIndex, progress);
            const eased = 0.5 - 0.5 * Math.cos(Math.PI * progress);
            const fromRange = getMetricRange(metric, fromIndex);
            const toRange = getMetricRange(metric, toIndex);
            const blendedMin = Number((fromRange.min + (toRange.min - fromRange.min) * eased).toFixed(2));
            const blendedMax = Number((fromRange.max + (toRange.max - fromRange.max) * eased).toFixed(2));
            timelineMapChart.setOption({{
                title: {{
                    subtext: '过渡：' + fromPeriod + ' → ' + toPeriod
                }},
                visualMap: {{
                    min: blendedMin,
                    max: blendedMax
                }},
                series: [{{ data: blended }}]
            }}, false, true);
            currentPeriodChip.textContent = fromPeriod + ' → ' + toPeriod;
            const total = timelineMapConfig.periods.length || 1;
            const progressValue = ((fromIndex + eased + 1) / total) * 100;
            progressBar.style.width = progressValue.toFixed(2) + '%';
            progressText.textContent = (fromIndex + 1) + ' → ' + (toIndex + 1) + ' / ' + total;
        }}

        function stopAutoPlay() {{
            if (autoPlayTimer) {{
                clearTimeout(autoPlayTimer);
                autoPlayTimer = null;
            }}
            autoPlayEnabled = false;
            playBtn.textContent = '自动播放';
        }}

        function animateToPeriod(targetIndex, withInterpolation = false, interactionMode = 'auto') {{
            if (isInterpolating) return;
            if (targetIndex === currentTimelineIndex) {{
                updateTimelineMeta();
                return;
            }}
            isInterpolating = true;
            suppressTimelineEvent = true;
            currentTimelineIndex = targetIndex;
            renderCurrentMap().then(() => {{
                timelineMapChart.dispatchAction({{
                    type: 'timelineChange',
                    currentIndex: targetIndex
                }});
                provinceTrendChart.setOption(makeProvinceTrendOption(activeMetric, selectedProvince), true);
                isInterpolating = false;
                updateTimelineMeta();
                if (autoPlayEnabled) {{
                    queueNextAutoPlay();
                }}
            }});
        }}

        function queueNextAutoPlay() {{
            if (!autoPlayEnabled) return;
            autoPlayTimer = window.setTimeout(() => {{
                const nextIndex = (currentTimelineIndex + 1) % timelineMapConfig.periods.length;
                animateToPeriod(nextIndex);
            }}, 820);
        }}

        function jumpToPeriod(targetIndex) {{
            if (targetIndex == null || targetIndex < 0 || targetIndex >= timelineMapConfig.periods.length) return;
            if (autoPlayTimer) {{
                clearTimeout(autoPlayTimer);
                autoPlayTimer = null;
            }}
            autoPlayEnabled = false;
            playBtn.textContent = '自动播放';
            isInterpolating = false;
            currentTimelineIndex = targetIndex;
            renderCurrentMap().then(() => {{
                timelineMapChart.dispatchAction({{
                    type: 'timelineChange',
                    currentIndex: targetIndex
                }});
                provinceTrendChart.setOption(makeProvinceTrendOption(activeMetric, selectedProvince), true);
                updateTimelineMeta();
            }});
        }}

        function startAutoPlay() {{
            stopAutoPlay();
            autoPlayEnabled = true;
            queueNextAutoPlay();
            playBtn.textContent = '暂停播放';
        }}

        function makeProvinceTrendOption(metric, province) {{
            const meta = timelineMapConfig.metrics[metric];
            const values = province
                ? timelineMapConfig.province_trends[province][metric]
                : timelineMapConfig.national_trends[metric];
            const title = province ? province + '月度趋势' : '全国平均月度趋势';
            return {{
                animationDuration: 500,
                animationDurationUpdate: 500,
                title: {{
                    text: title + ' · ' + meta.label,
                    left: 20,
                    top: 10,
                    textStyle: {{
                        color: '#17324d',
                        fontSize: 18,
                        fontWeight: 'bold'
                    }},
                    subtext: province ? '已与地图选中省份联动' : '当前未选中省份'
                }},
                grid: {{
                    left: 58,
                    right: 24,
                    top: 70,
                    bottom: 48
                }},
                tooltip: {{
                    trigger: 'axis'
                }},
                xAxis: {{
                    type: 'category',
                    data: timelineMapConfig.periods,
                    axisLabel: {{
                        color: '#587186',
                        rotate: 45
                    }}
                }},
                yAxis: {{
                    type: 'value',
                    name: meta.unit,
                    axisLabel: {{
                        color: '#587186'
                    }},
                    splitLine: {{
                        lineStyle: {{
                            color: 'rgba(91, 122, 146, 0.18)'
                        }}
                    }}
                }},
                dataZoom: [
                    {{
                        type: 'inside',
                        start: 0,
                        end: 100
                    }},
                    {{
                        type: 'slider',
                        bottom: 8,
                        height: 18
                    }}
                ],
                series: [
                    {{
                        type: 'line',
                        smooth: true,
                        data: values,
                        symbolSize: 7,
                        lineStyle: {{
                            width: 3,
                            color: '#1f6aa5'
                        }},
                        itemStyle: {{
                            color: '#1f6aa5'
                        }},
                        areaStyle: {{
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                {{ offset: 0, color: 'rgba(31,106,165,0.30)' }},
                                {{ offset: 1, color: 'rgba(31,106,165,0.03)' }}
                            ])
                        }},
                        z: 3
                    }}
                ]
            }};
        }}

        function getTrendValues(metric, province) {{
            return province
                ? timelineMapConfig.province_trends[province][metric]
                : timelineMapConfig.national_trends[metric];
        }}

        function applyInterpolatedTrendFrame(metric, province, fromIndex, toIndex, progress) {{
            provinceTrendChart.setOption({{
                title: {{
                    subtext: (province ? '已与地图选中省份联动' : '当前未选中省份') + ' · 动态插值过渡中'
                }}
            }}, false, true);
        }}

        function activateMetric(metric) {{
            activeMetric = metric;
            metricButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.metric === metric));
            renderCurrentMap();
            provinceTrendChart.setOption(makeProvinceTrendOption(metric, selectedProvince), true);
            updateTimelineMeta();
        }}

        loadChinaGeoJson().then(geoJson => {{
            echarts.registerMap('china-custom', geoJson);
            activateMetric(activeMetric);
        }}).catch(err => {{
            timelineMapEl.innerHTML = '<div style="padding:48px;color:#6b7f92;font-size:14px;">地图底图加载失败，请检查网络后刷新页面。</div>';
        }});

        metricButtons.forEach(btn => {{
            btn.addEventListener('click', () => activateMetric(btn.dataset.metric));
        }});

        refreshAllBtn.addEventListener('click', () => {{
            triggerDataJob('/api/update/full');
        }});

        refreshFailedBtn.addEventListener('click', () => {{
            triggerDataJob('/api/update/failed');
        }});

        cancelJobBtn.addEventListener('click', () => {{
            triggerDataJob('/api/update/cancel');
        }});

        playBtn.addEventListener('click', () => {{
            if (autoPlayEnabled) {{
                stopAutoPlay();
            }} else {{
                startAutoPlay();
            }}
        }});

        resetBtn.addEventListener('click', () => {{
            mapMode = 'national';
            drilldownProvince = null;
            selectedProvince = null;
            provinceLabel.textContent = '当前联动：全国平均';
            renderCurrentMap();
            provinceTrendChart.setOption(makeProvinceTrendOption(activeMetric, selectedProvince), true);
        }});

        timelineMapChart.on('timelinechanged', params => {{
            if (suppressTimelineEvent) {{
                suppressTimelineEvent = false;
                return;
            }}
            currentTimelineIndex = params.currentIndex || 0;
            provinceTrendChart.setOption(makeProvinceTrendOption(activeMetric, selectedProvince), true);
            updateTimelineMeta();
        }});

        timelineMapChart.on('click', params => {{
            if (mapMode === 'province') return;
            if (!params.name) return;
            selectedProvince = timelineMapConfig.province_name_lookup[params.name] || params.name;
            const selectedDisplay = timelineMapConfig.province_display_names[selectedProvince] || params.name;
            provinceLabel.textContent = '当前联动：' + selectedDisplay;
            provinceTrendChart.setOption(makeProvinceTrendOption(activeMetric, selectedProvince), true);
        }});

        timelineMapChart.on('dblclick', params => {{
            if (mapMode !== 'national' || !params.name) return;
            const province = timelineMapConfig.province_name_lookup[params.name] || params.name;
            if (!timelineMapConfig.province_city_metrics?.[province]) return;
            mapMode = 'province';
            drilldownProvince = province;
            selectedProvince = province;
            provinceLabel.textContent = '当前联动：' + province + ' / city view';
            renderCurrentMap().then(() => {{
                timelineMapChart.dispatchAction({{
                    type: 'timelineChange',
                    currentIndex: currentTimelineIndex
                }});
                provinceTrendChart.setOption(makeProvinceTrendOption(activeMetric, selectedProvince), true);
            }});
        }});

        provinceTrendChart.on('click', params => {{
            let targetIndex = null;
            if (typeof params.dataIndex === 'number') {{
                targetIndex = params.dataIndex;
            }} else if (params.name) {{
                const idx = timelineMapConfig.periods.indexOf(params.name);
                if (idx >= 0) targetIndex = idx;
            }}
            if (targetIndex == null || targetIndex < 0) return;
            jumpToPeriod(targetIndex);
        }});

        window.addEventListener('resize', () => {{
            timelineMapChart.resize();
            provinceTrendChart.resize();
        }});

        updateTimelineMeta();
        fetchJobStatus();
        window.setInterval(fetchJobStatus, 4000);
    </script>
</body>
</html>"""

    DASHBOARD_FILE.write_text(html, encoding="utf-8")


def build_report(daily_df: pd.DataFrame, monthly_df: pd.DataFrame, trend_df: pd.DataFrame, start_date: date, end_date: date) -> None:
    hottest = trend_df.sort_values("avg_temp_5y", ascending=False).head(5)
    wettest = trend_df.sort_values("precipitation_5y", ascending=False).head(5)
    warming = trend_df.sort_values("temp_slope_per_month", ascending=False).head(5)
    comfort = trend_df.sort_values("comfort_5y", ascending=False).head(5)
    national = monthly_df.groupby("year", as_index=False)["avg_temp"].mean().round(2)

    lines = [
        "# 中国近五年天气分析摘要",
        "",
        f"- 数据区间：`{start_date.isoformat()}` 至 `{end_date.isoformat()}`",
        f"- 城市数量：`{daily_df['city'].nunique()}`",
        f"- 日度记录数：`{len(daily_df):,}`",
        f"- 月度记录数：`{len(monthly_df):,}`",
        "",
        "## 全国样本年度平均气温",
        "",
    ]
    lines.extend([f"- {row.year} 年：`{row.avg_temp:.2f} ℃`" for row in national.itertuples(index=False)])
    lines.extend([
        "",
        "## 近五年平均气温最高城市 TOP5",
        "",
    ])
    lines.extend([f"- {row.city}：`{row.avg_temp_5y:.2f} ℃`" for row in hottest.itertuples(index=False)])
    lines.extend([
        "",
        "## 近五年月均降水最高城市 TOP5",
        "",
    ])
    lines.extend([f"- {row.city}：`{row.precipitation_5y:.2f} mm`" for row in wettest.itertuples(index=False)])
    lines.extend([
        "",
        "## 近五年升温趋势最明显城市 TOP5",
        "",
    ])
    lines.extend([f"- {row.city}：`{row.temp_slope_per_month:.4f}` / 月" for row in warming.itertuples(index=False)])
    lines.extend([
        "",
        "## 综合舒适度最高城市 TOP5",
        "",
    ])
    lines.extend([f"- {row.city}：`{row.comfort_5y:.2f}`" for row in comfort.itertuples(index=False)])

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def save_outputs(daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> None:
    daily_df.to_csv(DAILY_FILE, index=False, encoding="utf-8-sig")
    monthly_df.to_csv(MONTHLY_FILE, index=False, encoding="utf-8-sig")


def run_pipeline(force_refresh: bool = False, workers: int = 2, failed_only: bool = False) -> dict:
    ensure_dirs()
    start_date, end_date = calc_date_range()
    all_cities = build_prefecture_level_cities(force_refresh=force_refresh)
    if failed_only:
        failed_codes = {city.city_code for city in load_failed_cities() if city.city_code}
        failed_names = {(city.province, city.city) for city in load_failed_cities() if not city.city_code}
        cities = [
            city
            for city in all_cities
            if (city.city_code and city.city_code in failed_codes)
            or ((not city.city_code) and (city.province, city.city) in failed_names)
        ]
        if not cities:
            write_dashboard_status(
                {
                    "running": False,
                    "mode": "failed",
                    "message": "没有可重试的失败城市",
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_success": read_dashboard_status().get("last_success", ""),
                    "last_error": "",
                    "failed_count": len(load_failed_cities()),
                    "total": 0,
                    "completed": 0,
                    "success_count": 0,
                    "current_city": "",
                }
            )
            return {"ok": True, "message": "没有可重试的失败城市", "city_count": 0}
    else:
        cities = all_cities

    print(f"抓取区间: {start_date.isoformat()} -> {end_date.isoformat()}")
    print(f"城市样本: {len(cities)}")
    daily_df = crawl_weather_data(cities, start_date, end_date, force_refresh, workers)
    if failed_only and DAILY_FILE.exists():
        existing_daily = pd.read_csv(DAILY_FILE, parse_dates=["date"])
        target_keys = {(city.city_code or "", city.city) for city in cities}
        existing_daily["city_code"] = existing_daily["city_code"].fillna("") if "city_code" in existing_daily.columns else ""
        remaining_daily = existing_daily[
            ~existing_daily.apply(lambda row: (str(row.get("city_code", "")), row["city"]) in target_keys, axis=1)
        ]
        daily_df = pd.concat([remaining_daily, daily_df], ignore_index=True)
        daily_df = clean_weather_data(daily_df)

    monthly_df = aggregate_monthly(daily_df)
    trend_df = compute_city_trends(monthly_df)

    save_outputs(daily_df, monthly_df)
    build_dashboard(monthly_df, trend_df)
    build_report(daily_df, monthly_df, trend_df, start_date, end_date)

    status = read_dashboard_status()
    write_dashboard_status(
        {
            "running": False,
            "mode": "failed" if failed_only else "full",
            "message": "更新完成",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_success": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_error": "",
            "failed_count": len(load_failed_cities()),
            "total": len(cities),
            "completed": len(cities),
            "success_count": len(cities) - len(load_failed_cities()),
            "current_city": "",
        }
    )
    return {
        "ok": True,
        "message": "更新完成",
        "city_count": len(cities),
        "failed_count": len(load_failed_cities()),
        "last_success": status.get("last_success", ""),
    }


dashboard_job_lock = threading.Lock()
dashboard_cancel_event = threading.Event()


def start_dashboard_job(force_refresh: bool, workers: int, failed_only: bool) -> tuple[bool, str]:
    if dashboard_job_lock.locked():
        return False, "已有更新任务正在运行"

    def runner() -> None:
        dashboard_job_lock.acquire()
        dashboard_cancel_event.clear()
        try:
            write_dashboard_status(
                {
                    "running": True,
                    "mode": "failed" if failed_only else "full",
                    "message": "任务运行中",
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_success": read_dashboard_status().get("last_success", ""),
                    "last_error": "",
                    "failed_count": len(load_failed_cities()),
                    "total": 0,
                    "completed": 0,
                    "success_count": 0,
                    "current_city": "",
                }
            )
            run_pipeline(force_refresh=force_refresh, workers=workers, failed_only=failed_only)
        except InterruptedError:
            current_status = read_dashboard_status()
            write_dashboard_status(
                {
                    **current_status,
                    "running": False,
                    "message": "任务已中断",
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_error": "",
                    "current_city": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            write_dashboard_status(
                {
                    "running": False,
                    "mode": "failed" if failed_only else "full",
                    "message": "任务失败",
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_success": read_dashboard_status().get("last_success", ""),
                    "last_error": str(exc),
                    "failed_count": len(load_failed_cities()),
                    "total": read_dashboard_status().get("total", 0),
                    "completed": read_dashboard_status().get("completed", 0),
                    "success_count": read_dashboard_status().get("success_count", 0),
                    "current_city": "",
                }
            )
        finally:
            dashboard_job_lock.release()

    threading.Thread(target=runner, daemon=True).start()
    return True, "任务已启动"


def cancel_dashboard_job() -> tuple[bool, str]:
    if not dashboard_job_lock.locked():
        return False, "当前没有运行中的任务"
    dashboard_cancel_event.set()
    current_status = read_dashboard_status()
    write_dashboard_status(
        {
            **current_status,
            "running": True,
            "message": "已发送中断请求，等待当前城市结束",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return True, "已发送中断请求"


def make_dashboard_handler(workers: int):  # type: ignore[no-untyped-def]
    class DashboardHandler(BaseHTTPRequestHandler):
        def _write_json(self, payload: dict, status_code: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def _write_file(self, file_path: Path) -> None:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", f"{mime_type or 'text/html'}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._write_json({}, 200)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/dashboard"}:
                self._write_file(DASHBOARD_FILE)
                return
            if path == "/api/status":
                self._write_json(read_dashboard_status())
                return
            self._write_json({"ok": False, "message": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/update/full":
                ok, message = start_dashboard_job(force_refresh=True, workers=workers, failed_only=False)
                self._write_json({"ok": ok, "message": message}, 200 if ok else 409)
                return
            if path == "/api/update/failed":
                ok, message = start_dashboard_job(force_refresh=True, workers=workers, failed_only=True)
                self._write_json({"ok": ok, "message": message}, 200 if ok else 409)
                return
            if path == "/api/update/cancel":
                ok, message = cancel_dashboard_job()
                self._write_json({"ok": ok, "message": message}, 200 if ok else 409)
                return
            self._write_json({"ok": False, "message": "not found"}, 404)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return DashboardHandler


def serve_dashboard(api_port: int, workers: int) -> None:
    write_dashboard_status(read_dashboard_status())
    server = ThreadingHTTPServer(("127.0.0.1", api_port), make_dashboard_handler(workers))
    print(f"Dashboard control server: http://127.0.0.1:{api_port}/dashboard")
    server.serve_forever()


def main() -> None:
    args = parse_args()
    result: dict | None = None
    try:
        result = run_pipeline(force_refresh=args.force_refresh, workers=args.workers, failed_only=False)

        print("\n分析完成")
        print(f"- 日度数据: {DAILY_FILE}")
        print(f"- 月度数据: {MONTHLY_FILE}")
        print(f"- 可视化大屏: {DASHBOARD_FILE}")
        print(f"- 分析摘要: {REPORT_FILE}")
        if result.get("ok"):
            print(f"- 失败待重抓城市: {result.get('failed_count', 0)}")
    except Exception:
        if not args.serve_dashboard:
            raise
        print("初始更新失败，但将继续启动大屏按钮服务。")

    if args.serve_dashboard:
        serve_dashboard(api_port=args.api_port, workers=args.workers)


if __name__ == "__main__":
    main()
