# 中国近五年天气抓取与可视化分析

这是一个基于 Python 的中国天气数据抓取与分析项目。项目会抓取中国主要城市近五年的历史天气数据，进行清洗、聚合和分析，并输出交互式可视化大屏与文本摘要报告。

详细说明请查看：

- [PROJECT_DOC.md](D:/Users/Twenty/Desktop/天气可视化/PROJECT_DOC.md)

## 快速开始

安装依赖：

```powershell
pip install -r requirements.txt
```

运行项目：

```powershell
python china_weather_spider_analysis.py
```

忽略缓存并重新抓取：

```powershell
python china_weather_spider_analysis.py --force-refresh
```

建议在当前网络环境下优先使用单线程：

```powershell
python china_weather_spider_analysis.py --workers 1
```

## 输出文件

- `data/raw/`：城市级原始抓取缓存
- `data/processed/china_weather_daily_5y.csv`：日度汇总数据
- `data/processed/china_weather_monthly_5y.csv`：月度聚合数据
- `outputs/figures/china_weather_dashboard_5y.html`：交互式可视化大屏
- `outputs/reports/china_weather_report_5y.md`：文本摘要报告

## 当前状态

- 中国主要城市近五年天气抓取与分析已可运行
- 可视化大屏已支持地图、时间轴、指标切换、趋势联动
- 台湾城市已加入采样，但当前代理环境下尚未成功抓取到台湾数据
- 趋势图高亮功能已按当前版本要求移除，以保证页面稳定性
