from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List

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
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

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


CITIES: List[City] = [
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


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    API_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取并分析近五年中国天气数据")
    parser.add_argument("--force-refresh", action="store_true", help="忽略缓存并重新抓取")
    parser.add_argument("--workers", type=int, default=2, help="并发抓取线程数")
    return parser.parse_args()


def calc_date_range() -> tuple[date, date]:
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 5)
    return start_date, end_date


def city_cache_file(city: City, start_date: date, end_date: date) -> Path:
    suffix = f"{start_date.isoformat()}_{end_date.isoformat()}"
    return RAW_DIR / f"{city.city}_{suffix}.csv"


def find_latest_city_cache(city: City) -> Path | None:
    candidates = sorted(RAW_DIR.glob(f"{city.city}_*.csv"))
    return candidates[-1] if candidates else None


def fetch_city_weather(city: City, start_date: date, end_date: date, force_refresh: bool = False) -> pd.DataFrame:
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
    df["province"] = city.province
    df["region"] = city.region
    df["lat"] = city.lat
    df["lon"] = city.lon
    df.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return df


def crawl_weather_data(cities: Iterable[City], start_date: date, end_date: date, force_refresh: bool, workers: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_city_weather, city, start_date, end_date, force_refresh): city
            for city in cities
        }
        for future in as_completed(futures):
            city = futures[future]
            try:
                frames.append(future.result())
                print(f"[OK] {city.city} 抓取完成")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{city.city}: {exc}")
                print(f"[WARN] {city.city} 抓取失败: {exc}")

    if not frames:
        raise RuntimeError("所有城市抓取均失败，无法继续分析")

    if errors:
        print("\n部分城市抓取失败：")
        for msg in errors:
            print(f"- {msg}")

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
        df.groupby(["city", "province", "region", "lat", "lon", "year", "month", "year_month"], as_index=False)
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


def build_dashboard(monthly_df: pd.DataFrame, trend_df: pd.DataFrame) -> None:
    geo_chart = make_geo_scatter(trend_df)
    line_chart = make_trend_line(monthly_df)
    heatmap_chart = make_heatmap(monthly_df)
    warming_chart = make_warming_bar(trend_df)
    radar_chart = make_radar(monthly_df)
    scatter_chart = make_scatter(monthly_df)
    map_timeline_config = build_map_timeline_config(monthly_df)

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
        .status-pill {{
            padding: 9px 14px;
            border-radius: 999px;
            background: rgba(19, 40, 59, 0.08);
            color: #35546d;
            font-size: 13px;
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
                        <button class="ghost-btn" id="timeline-play">自动播放</button>
                        <button class="ghost-btn" id="timeline-reset">查看全国</button>
                        <div class="status-pill" id="selected-province-label">当前联动：全国平均</div>
                    </div>
                </div>
                <div class="map-meta">
                    <div class="month-chip">
                        <div class="month-chip-label">当前月份</div>
                        <div class="month-chip-value" id="current-period-chip">2021-04</div>
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
        const timelineMapEl = document.getElementById('timeline-map');
        const provinceTrendEl = document.getElementById('province-trend');
        const metricButtons = Array.from(document.querySelectorAll('.metric-btn'));
        const playBtn = document.getElementById('timeline-play');
        const resetBtn = document.getElementById('timeline-reset');
        const provinceLabel = document.getElementById('selected-province-label');
        const currentPeriodChip = document.getElementById('current-period-chip');
        const progressText = document.getElementById('progress-text');
        const progressBar = document.getElementById('timeline-progress-bar');
        const timelineMapChart = echarts.init(timelineMapEl, null, {{ renderer: 'canvas' }});
        const provinceTrendChart = echarts.init(provinceTrendEl, null, {{ renderer: 'canvas' }});
        let currentTimelineIndex = 0;
        let autoPlayTimer = null;
        let autoPlayEnabled = false;
        let isInterpolating = false;
        let suppressTimelineEvent = false;

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

        let activeMetric = 'avg_temp';
        let selectedProvince = null;

        function updateTimelineMeta() {{
            const total = timelineMapConfig.periods.length || 1;
            const current = currentTimelineIndex + 1;
            const period = timelineMapConfig.periods[currentTimelineIndex] || '-';
            currentPeriodChip.textContent = period;
            progressText.textContent = current + ' / ' + total;
            progressBar.style.width = ((current / total) * 100).toFixed(2) + '%';
        }}

        function getMetricSeriesData(metric, index) {{
            return timelineMapConfig.metrics[metric].series[index]?.data || [];
        }}

        function getMetricRange(metric, index) {{
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
                const fromVal = typeof fromItem.value === 'number' ? fromItem.value : null;
                const toVal = typeof toItem.value === 'number' ? toItem.value : null;
                let value = null;
                if (fromVal != null && toVal != null) {{
                    value = Number((fromVal + (toVal - fromVal) * eased).toFixed(2));
                }} else if (eased < 0.5) {{
                    value = fromVal;
                }} else {{
                    value = toVal;
                }}
                return {{
                    name: toItem.name || fromItem.name,
                    value: value
                }};
            }});
        }}

        function applyInterpolatedFrame(metric, fromIndex, toIndex, progress) {{
            const meta = timelineMapConfig.metrics[metric];
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
            const fromIndex = currentTimelineIndex;
            if (targetIndex === fromIndex) {{
                updateTimelineMeta();
                return;
            }}
            isInterpolating = true;

            const startInterpolation = () => {{
                if (!withInterpolation) {{
                    suppressTimelineEvent = true;
                    currentTimelineIndex = targetIndex;
                    timelineMapChart.dispatchAction({{
                        type: 'timelineChange',
                        currentIndex: targetIndex
                    }});
                    window.setTimeout(() => {{
                        provinceTrendChart.setOption(makeProvinceTrendOption(activeMetric, selectedProvince), true);
                        isInterpolating = false;
                        updateTimelineMeta();
                    }}, 450);
                    return;
                }}

                const isManual = interactionMode === 'manual';
                const duration = isManual ? 780 : 1700;
                const steps = isManual ? 6 : 16;
                let step = 0;
                const tick = () => {{
                    step += 1;
                    const progress = Math.min(1, step / steps);
                    applyInterpolatedFrame(activeMetric, fromIndex, targetIndex, progress);
                    applyInterpolatedTrendFrame(activeMetric, selectedProvince, fromIndex, targetIndex, progress);
                    if (progress < 1) {{
                        autoPlayTimer = window.setTimeout(tick, duration / steps);
                    }} else {{
                        suppressTimelineEvent = true;
                        currentTimelineIndex = targetIndex;
                        timelineMapChart.dispatchAction({{
                            type: 'timelineChange',
                            currentIndex: targetIndex
                        }});
                        window.setTimeout(() => {{
                            provinceTrendChart.setOption(makeProvinceTrendOption(activeMetric, selectedProvince), true);
                            isInterpolating = false;
                            updateTimelineMeta();
                            if (autoPlayEnabled) {{
                                queueNextAutoPlay();
                            }}
                        }}, 260);
                    }}
                }};
                tick();
            }};

            autoPlayTimer = window.setTimeout(startInterpolation, 420);
        }}

        function queueNextAutoPlay() {{
            if (!autoPlayEnabled) return;
            autoPlayTimer = window.setTimeout(() => {{
                const nextIndex = (currentTimelineIndex + 1) % timelineMapConfig.periods.length;
                animateToPeriod(nextIndex, true);
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
            timelineMapChart.setOption(makeTimelineOption(activeMetric), true);
            timelineMapChart.dispatchAction({{
                type: 'timelineChange',
                currentIndex: targetIndex
            }});
            provinceTrendChart.setOption(makeProvinceTrendOption(activeMetric, selectedProvince), true);
            updateTimelineMeta();
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
            timelineMapChart.setOption(makeTimelineOption(metric), true);
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

        playBtn.addEventListener('click', () => {{
            if (autoPlayEnabled) {{
                stopAutoPlay();
            }} else {{
                startAutoPlay();
            }}
        }});

        resetBtn.addEventListener('click', () => {{
            selectedProvince = null;
            provinceLabel.textContent = '当前联动：全国平均';
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
            if (!params.name) return;
            selectedProvince = timelineMapConfig.province_name_lookup[params.name] || params.name;
            const selectedDisplay = timelineMapConfig.province_display_names[selectedProvince] || params.name;
            provinceLabel.textContent = '当前联动：' + selectedDisplay;
            provinceTrendChart.setOption(makeProvinceTrendOption(activeMetric, selectedProvince), true);
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


def main() -> None:
    args = parse_args()
    ensure_dirs()
    start_date, end_date = calc_date_range()

    print(f"抓取区间: {start_date.isoformat()} -> {end_date.isoformat()}")
    daily_df = crawl_weather_data(CITIES, start_date, end_date, args.force_refresh, args.workers)
    monthly_df = aggregate_monthly(daily_df)
    trend_df = compute_city_trends(monthly_df)

    save_outputs(daily_df, monthly_df)
    build_dashboard(monthly_df, trend_df)
    build_report(daily_df, monthly_df, trend_df, start_date, end_date)

    print("\n分析完成")
    print(f"- 日度数据: {DAILY_FILE}")
    print(f"- 月度数据: {MONTHLY_FILE}")
    print(f"- 可视化大屏: {DASHBOARD_FILE}")
    print(f"- 分析摘要: {REPORT_FILE}")


if __name__ == "__main__":
    main()
