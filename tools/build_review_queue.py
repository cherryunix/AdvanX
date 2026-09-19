#!/usr/bin/env python3
"""Build a focused offline review page for a segment JSON queue."""

from __future__ import annotations

import argparse
import html
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from build_segment_review import card, extract_preview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segments", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="AdvanX 边界片段复核")
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 2))
    args = parser.parse_args()
    rows = json.loads(args.segments.read_text())
    rows.sort(key=lambda row: abs(row.get("distance_from_threshold", 0)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    images = {}
    for row in rows:
        relative = Path("thumbnails") / f"{row['segment_id']}.jpg"
        images[row["segment_id"]] = relative.as_posix()
        jobs.append((row, args.output_dir / relative))
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(extract_preview, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            target, error = future.result()
            if error:
                failures.append({"target": str(target), "error": error})
            if index % 20 == 0 or index == len(futures):
                print(f"previews {index}/{len(futures)}", flush=True)
    (args.output_dir / "preview_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n"
    )
    cards = "".join(
        card(row, images[row["segment_id"]], args.output_dir, "review")
        for row in rows
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(args.title)}</title>
<style>
:root{{--bg:#080a08;--card:#111510;--line:#283026;--text:#f2f4ef;--muted:#aeb7aa;--green:#76b900;--red:#ff675c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif}}header,main{{padding:28px max(22px,5vw)}}h1{{font-size:clamp(30px,5vw,58px);margin:0}}header p,.candidate p{{color:var(--muted)}}
.bar{{position:sticky;top:0;z-index:3;background:#080a08ed;padding:12px 0;display:flex;gap:8px;flex-wrap:wrap}}button,input{{font:inherit;border:1px solid #3c4738;background:#171d15;color:var(--text);padding:8px 12px;border-radius:8px}}button{{cursor:pointer}}button:hover{{border-color:var(--green)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}}.candidate{{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}}.candidate[data-decision=keep]{{border-color:var(--green)}}.candidate[data-decision=drop]{{opacity:.45;border-color:var(--red)}}.candidate img{{width:100%;display:block;aspect-ratio:720/320;object-fit:cover;background:#000}}.candidate-body{{padding:14px}}.candidate-top{{display:flex;justify-content:space-between}}.candidate h3{{font-size:15px;margin:8px 0;overflow-wrap:anywhere}}.candidate p{{font-size:13px;margin:4px 0}}.badge{{font-size:12px;padding:2px 8px;border-radius:20px;background:#3b3514;color:#ffe48b}}.actions{{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}}.keep{{color:#bde979}}.drop{{color:#ffaaa3}}
dialog{{width:min(920px,94vw);background:#0b0d0a;color:var(--text);border:1px solid var(--line);border-radius:14px;padding:16px}}dialog video{{width:100%;max-height:76vh;background:#000}}dialog::backdrop{{background:#000c}}
</style></head><body><header><h1>{html.escape(args.title)}</h1><p>{len(rows)} 段位于人工学得的 yaw 边界附近，按距离阈值从近到远排列。原有 92 条人工标签已经锁定，不在此页重复出现。</p></header>
<main><div class="bar"><input id="search" placeholder="筛选路径"><button onclick="exportDecisions()">导出这批决定</button><span id="progress"></span></div><div class="grid">{cards}</div></main>
<dialog id="player"><h3 id="playerTitle"></h3><video id="video" controls></video><p><button onclick="closePlayer()">关闭</button></p></dialog>
<script>
const key='advanx-boundary-review-decisions',decisions=JSON.parse(localStorage.getItem(key)||'{{}}');
function paint(){{document.querySelectorAll('.candidate').forEach(c=>c.dataset.decision=decisions[c.dataset.id]||'');const n=Object.keys(decisions).length;document.getElementById('progress').textContent=`已决定 ${{n}} / {len(rows)}`;filterCards()}}
function decide(id,value){{if(value)decisions[id]=value;else delete decisions[id];localStorage.setItem(key,JSON.stringify(decisions));paint()}}
function filterCards(){{const q=document.getElementById('search').value.toLowerCase();document.querySelectorAll('.candidate').forEach(c=>c.style.display=!q||c.dataset.path.toLowerCase().includes(q)?'':'none')}}document.getElementById('search').oninput=filterCards;
let stopAt=0;const video=document.getElementById('video');video.addEventListener('timeupdate',()=>{{if(stopAt&&video.currentTime>=stopAt)video.pause()}});function playSegment(src,start,end,title){{stopAt=end;document.getElementById('playerTitle').textContent=title+' · '+start.toFixed(2)+'s–'+end.toFixed(2)+'s';video.src=src;video.onloadedmetadata=()=>{{video.currentTime=start;video.play()}};document.getElementById('player').showModal()}}function closePlayer(){{video.pause();video.removeAttribute('src');video.load();document.getElementById('player').close()}}
function exportDecisions(){{const blob=new Blob([JSON.stringify({{generated_at:new Date().toISOString(),decisions}},null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='advanx-boundary-review-decisions.json';a.click();URL.revokeObjectURL(a.href)}}paint();
</script></body></html>"""
    (args.output_dir / "index.html").write_text(page)
    print(json.dumps({"segments": len(rows), "failures": len(failures)}))


if __name__ == "__main__":
    main()
