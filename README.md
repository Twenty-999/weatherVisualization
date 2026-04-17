# 中国近五年天气抓取与可视化分析

这是一个基于 Python 的中国天气数据抓取与分析项目。项目会抓取中国主要城市近五年的历史天气数据，进行清洗、聚合和分析，并输出交互式可视化大屏与文本摘要报告。

详细说明请查看：

- [PROJECT_DOC.md](weatherVisualization/PROJECT_DOC.md)

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

如需启用大屏中的“更新全部数据”“仅重抓失败数据”“中断任务”按钮，请启动本地控制服务：

```powershell
python china_weather_spider_analysis.py --serve-dashboard --workers 1
```

如需“先启动大屏服务，再按页面按钮按需更新”，上面这条命令已经会直接进入本地服务，不再先执行整套全量抓取。
如需在启动服务前先跑一次全量更新，可使用：

```powershell
python china_weather_spider_analysis.py --serve-dashboard --workers 1 --init-update
```

也可以直接双击：

- [start_dashboard.bat](weatherVisualization/start_dashboard.bat)

## 输出文件

- `data/raw/`：城市级原始抓取缓存
- `data/processed/china_weather_daily_5y.csv`：日度汇总数据
- `data/processed/china_weather_monthly_5y.csv`：月度聚合数据
- `outputs/figures/china_weather_dashboard_5y.html`：交互式可视化大屏
- `outputs/reports/china_weather_report_5y.md`：文本摘要报告

## 当前状态

- 中国主要城市近五年天气抓取与分析已可运行
- 可视化大屏已支持地图、时间轴、指标切换、趋势联动
- 代码层面已支持按行政区划动态生成地级市抓取列表
- 地图已支持双击省份进入省内城市视图
- 地图自动播放已改为直接切换月份，不再做过渡插值
- 城市缓存命名已优先使用行政区代码，避免重名城市互相覆盖
- 可视化大屏已支持本地按钮触发“更新全部数据”和“仅重抓失败数据”
- 大屏更新任务已支持显示总进度、成功数、失败数和当前处理城市
- 任务运行中会临时禁用两个更新按钮，并启用“中断任务”按钮，避免重复启动
- 抓取失败城市会写入 `.api_cache/failed_cities.json`，便于后续只补抓失败部分
- 项目已提供 [start_dashboard.bat](weatherVisualization/start_dashboard.bat) 用于一键启动本地服务并打开大屏
- 天气接口与行政区划接口已补充统一重试、退避和可中断等待，降低瞬时失败率
- 行政区划接口失败时已改为“只回退当前省”，不再因为单省失败整批退回旧样本
- “仅重抓失败数据”已改为直接使用失败清单，不再依赖重新匹配当前城市列表
- 中断任务按钮已补充即时状态反馈，并优化了启动与中断之间的竞态问题
- 大屏除地图外的摘要卡片和图表说明已调整为“当前样本”口径，更符合实际落地数据状态
- 核心抓取与控制逻辑已补充中文注释，便于后续维护
- 台湾城市已加入采样，但当前代理环境下尚未成功抓取到台湾数据
- 趋势图高亮功能已按当前版本要求移除，以保证页面稳定性
- 但现有 `data/processed` 是否已经更新为完整地级市数据，取决于最近一次全量抓取是否成功
- 当前环境若网络或代理失败，行政区划接口与天气接口都可能无法访问
- 在这种情况下，页面里看到的仍可能是旧的样本城市数据
