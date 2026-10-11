from pathlib import Path
import subprocess,re,html,json
R=Path(__file__).resolve().parents[1]
css='''
:root{--ink:#162f46;--accent:#126776;--muted:#5b6873;--line:#d3dee5;--paper:#ffffff;--bg:#f3f6f8;}
html{scroll-behavior:smooth;}body{font-family:system-ui,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;margin:0;background:var(--bg);color:#172938;line-height:1.85;}
body>header,body>nav,body>h1,body>h2,body>h3,body>h4,body>p,body>ul,body>ol,body>blockquote,body>pre,body>table,body>div,body>hr{max-width:1150px;margin-left:auto;margin-right:auto;}
body{padding:30px 34px 60px;}#title-block-header{padding:28px 34px;background:var(--ink);color:white;border-radius:9px;margin-bottom:26px;}h1,h2,h3{line-height:1.45;color:var(--ink);}#title-block-header h1{color:white;}h1{border-top:3px solid var(--accent);padding-top:30px;margin-top:52px;}h2{margin-top:30px;}a{color:var(--accent);text-decoration:none;}a:hover{text-decoration:underline;}#TOC{background:white;border:1px solid var(--line);padding:18px 26px;border-radius:6px;}table{display:block;overflow-x:auto;border-collapse:collapse;background:white;margin-top:20px;margin-bottom:25px;font-size:.94rem;}thead{background:#eaf0f4;}th,td{border:1px solid var(--line);padding:9px 12px;vertical-align:top;}th{text-align:left;}code{background:#e8edf2;padding:.12em .32em;border-radius:3px;font-size:.9em;}pre{overflow-x:auto;background:#e8edf2;padding:20px;}pre code{padding:0;}blockquote{border-left:4px solid var(--accent);padding-left:22px;color:var(--muted);}math{font-size:1.03em;}hr{border:0;border-top:1px solid var(--line);margin-top:30px;margin-bottom:30px;}.subtitle{font-size:1.1rem;}@media print{body{background:white;font-size:10pt;padding:0;}a{color:inherit;}h1{page-break-before:always;}table{display:table;width:100%;font-size:8.5pt;}pre{white-space:pre-wrap;}}
'''
(R/'reference/reader.css').write_text(css)
files=[R/'docs/02_EXECUTIVE_PLAN_CN.md',R/'docs/03_THEORY_AND_CLAIM_MAP_CN.md',R/'docs/04_SHARED_EVALUATION_CN.md',R/'docs/05_STAGE_PLAN_AND_DECISIONS_CN.md']
for folder in sorted((R/'directions').iterdir()):
 files.extend([folder/'LITERATURE_REVIEW_CN.md',folder/'SERVER_EXPERIMENTS_CN.md'])
files.extend([R/'benchmarks/LITERATURE_PROTOCOL_CARDS_CN.md',R/'benchmarks/METRIC_SPECIFICATIONS_CN.md',R/'benchmarks/BASELINE_MATRIX_CN.md',R/'datasets/DATASET_ATLAS_CN.md',R/'literature/VERIFICATION_AND_SEARCH_CN.md',R/'literature/SOURCE_CATALOG_CN.md',R/'sources/USER_SOURCES_CN.md'])
parts=['# VCS · VL 六方向研究与实施全稿\n\n2026-10-11 · CVPR V6 / r7。各章节区分文献事实、项目来源与本轮研究建议。完整执行先读`01_SERVER_AGENT_PROMPT_CN.md`。\n']
for p in files:
 s=p.read_text()
 if s.startswith('---\n'):
  s=s.split('---\n',2)[2]
 s=s.replace('\\newpage','').replace('../../datasets/','datasets/').replace('../../literature/','literature/')
 # Paragraph equations in \[ \] work with pandoc's tex_math_single_backslash.
 parts.append(f'\n\n---\n\n<!-- source: {p.relative_to(R)} -->\n\n'+s)
full='\n'.join(parts)
(R/'FULL_RESEARCH_AND_SERVER_CN.md').write_text(full)
subprocess.run(['pandoc',str(R/'FULL_RESEARCH_AND_SERVER_CN.md'),'--from','markdown+tex_math_single_backslash','--standalone','--mathml','--toc','--toc-depth=2','--metadata','title=VCS · VL 六方向研究与服务器全稿','--include-in-header',str(R/'reference/html_head.html'),'-o',str(R/'FULL_RESEARCH_AND_SERVER_CN.html')],check=True) if (R/'reference/html_head.html').exists() else None
(R/'reference/html_head.html').write_text('<style>'+css+'</style>')
subprocess.run(['pandoc',str(R/'FULL_RESEARCH_AND_SERVER_CN.md'),'--from','markdown+tex_math_single_backslash','--standalone','--mathml','--toc','--toc-depth=2','--metadata','title=VCS · VL 六方向研究与服务器全稿','--include-in-header',str(R/'reference/html_head.html'),'-o',str(R/'FULL_RESEARCH_AND_SERVER_CN.html')],check=True)
# Per-direction offline HTML makes review links usable outside Markdown tools.
for folder in sorted((R/'directions').iterdir()):
 for fn in ['LITERATURE_REVIEW_CN','SERVER_EXPERIMENTS_CN']:
  subprocess.run(['pandoc',str(folder/(fn+'.md')),'--from','markdown+tex_math_single_backslash','--standalone','--mathml','--toc','--toc-depth=2','--metadata','title='+folder.name+' · '+('文献综述' if fn.startswith('LITERATURE') else '服务器方案'),'--include-in-header',str(R/'reference/html_head.html'),'-o',str(folder/(fn+'.html'))],check=True)
rows=[]
labels=['D1 视觉证据压缩','D2 事实支持与忠实性','D3 条件生成分布','D4 多对多关系','D5 数据选择与再描述','D6 视频时间支持']
for lab,folder in zip(labels,sorted((R/'directions').iterdir())):
 rel=str(folder.relative_to(R))
 rows.append(f'<tr><td>{lab}</td><td><a href="{rel}/LITERATURE_REVIEW_CN.html">文献综述</a> / <a href="{rel}/LITERATURE_REVIEW_CN.md">Markdown</a></td><td><a href="{rel}/SERVER_EXPERIMENTS_CN.html">实验实施</a> / <a href="{rel}/SERVER_EXPERIMENTS_CN.md">Markdown</a></td></tr>')
idx=f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>VCS r7 资料包入口</title><style>{css}</style></head><body><header id="title-block-header"><h1>VCS · Vision–Language研究图谱</h1><p>CVPR V6 · r7 · 2026-10-11</p></header><p>六份文献综述、六份实施方案；67项来源、35个数据/协议条目、57条按路线组织的对照记录。没有新的自然数据结果或已运行的六方向训练。</p><p><a href="VCS_VL_Research_Atlas_r7_Overview_CN.pdf">6页执行总览 PDF</a>　|　<a href="FULL_RESEARCH_AND_SERVER_CN.html">全部内容（离线HTML）</a>　|　<a href="01_SERVER_AGENT_PROMPT_CN.md">Agent接入消息</a></p><table><thead><tr><th>方向</th><th>综述</th><th>执行</th></tr></thead><tbody>{''.join(rows)}</tbody></table><p>详细数据集：<a href="datasets/DATASET_ATLAS_CN.md">数据图谱</a>；<a href="datasets/datasets.csv">CSV</a>。相关工作：<a href="literature/SOURCE_CATALOG_CN.md">来源目录</a>；<a href="literature/papers.csv">文献表</a>。<a href="benchmarks/LITERATURE_PROTOCOL_CARDS_CN.md">原文实验协议卡</a>与<a href="benchmarks/METRIC_SPECIFICATIONS_CN.md">指标规范</a>。</p><p>服务器先审数据、许可和最近基线，选择一条主线，不自动同时启动六套网格。参考测试只验证定义/指标fixture，不替代官方评价器。</p></body></html>'''
(R/'INDEX.html').write_text(idx)
print('full chars',len(full),'section files',len(files))
