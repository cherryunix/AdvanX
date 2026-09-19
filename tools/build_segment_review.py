#!/usr/bin/env python3
"""Build a local visual review report for pose-driven segment candidates."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote


def read_json(path: Path):
    return json.loads(path.read_text())


def best_per_source(rows: list[dict]) -> list[dict]:
    selected: dict[str, dict] = {}
    for row in rows:
        previous = selected.get(row["source_id"])
        if previous is None or row["score"] > previous["score"]:
            selected[row["source_id"]] = row
    return sorted(selected.values(), key=lambda row: -row["score"])


def clock(seconds: float) -> str:
    whole = round(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def video_url(report_dir: Path, source_path: str) -> str:
    relative = os.path.relpath(source_path, report_dir).replace(os.sep, "/")
    return quote(relative, safe="/.-_()")


def preview_command(row: dict, target: Path) -> list[str]:
    start = float(row["source_start_s"])
    duration = float(row["source_end_s"]) - start
    times = [start + duration * fraction for fraction in (0.15, 0.50, 0.85)]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    for moment in times:
        command.extend(
            ["-ss", f"{moment:.6f}", "-noautorotate", "-i", row["source_path"]]
        )
    rotation = int(row.get("rotation_clockwise", 0))
    transforms = {
        0: "",
        90: "transpose=1,",
        180: "hflip,vflip,",
        270: "transpose=2,",
    }[rotation]
    chains = []
    for index in range(3):
        chains.append(
            f"[{index}:v]{transforms}"
            "scale=240:320:force_original_aspect_ratio=decrease,"
            "pad=240:320:(ow-iw)/2:(oh-ih)/2:color=0x111111,"
            f"setsar=1[v{index}]"
        )
    chains.append("[v0][v1][v2]hstack=inputs=3[out]")
    command.extend(
        [
            "-filter_complex",
            ";".join(chains),
            "-map",
            "[out]",
            "-frames:v",
            "1",
            "-q:v",
            "3",
            "-y",
            str(target),
        ]
    )
    return command


def extract_preview(job: tuple[dict, Path]) -> tuple[Path, str | None]:
    row, target = job
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target, None
    result = subprocess.run(
        preview_command(row, target), capture_output=True, text=True, check=False
    )
    if result.returncode:
        return target, result.stderr.strip()
    return target, None


def card(row: dict, image_path: str, report_dir: Path, label: str) -> str:
    segment_id = html.escape(row["segment_id"])
    path = html.escape(row["relative_path"])
    source = html.escape(video_url(report_dir, row["source_path"]))
    start = float(row["source_start_s"])
    end = float(row["source_end_s"])
    return f"""
    <article class="candidate" data-id="{segment_id}" data-shot="{row['shot']}"
      data-score="{row['score']:.3f}" data-path="{path}">
      <img loading="lazy" src="{html.escape(image_path)}" alt="{path}">
      <div class="candidate-body">
        <div class="candidate-top"><span class="badge {label}">{label}</span>
          <strong>{row['score']:.1f}</strong></div>
        <h3>{path}</h3>
        <p>{clock(start)} – {clock(end)} · {row['duration_s']:.1f}s · {row['shot']}</p>
        <p>头部 P95：pitch {row['p95_abs_pitch_deg']:.1f}° · yaw {row['p95_abs_yaw_deg']:.1f}° · roll {row['p95_abs_roll_deg']:.1f}°</p>
        <div class="actions">
          <button onclick="playSegment('{source}',{start:.6f},{end:.6f},'{path}')">播放区间</button>
          <button class="keep" onclick="decide('{segment_id}','keep')">保留</button>
          <button class="drop" onclick="decide('{segment_id}','drop')">淘汰</button>
          <button onclick="decide('{segment_id}','')">清除</button>
        </div>
      </div>
    </article>"""


def zero_card(row: dict, image_path: str, report_dir: Path) -> str:
    ident = html.escape("source:" + row["source_id"])
    path = html.escape(row["relative_path"])
    source = html.escape(video_url(report_dir, row["source_path"]))
    end = float(row["seconds"])
    top = sorted(
        row["rejected_frame_counts"].items(), key=lambda pair: -pair[1]
    )[:3]
    reasons = " · ".join(f"{name} {count / row['frames']:.0%}" for name, count in top)
    return f"""
    <article class="candidate rejected" data-id="{ident}" data-shot="excluded"
      data-score="0" data-path="{path}">
      <img loading="lazy" src="{html.escape(image_path)}" alt="{path}">
      <div class="candidate-body">
        <div class="candidate-top"><span class="badge excluded">标准档无连续片段</span></div>
        <h3>{path}</h3>
        <p>原片 {clock(end)} · 原始合格帧 {row['raw_pass_frames']/row['frames']:.1%}</p>
        <p>主要触发：{html.escape(reasons)}</p>
        <div class="actions">
          <button onclick="playSegment('{source}',0,{end:.6f},'{path}')">播放原片</button>
          <button class="keep" onclick="decide('{ident}','keep')">人工加入</button>
          <button class="drop" onclick="decide('{ident}','drop')">确认淘汰</button>
        </div>
      </div>
    </article>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 2))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = read_json(args.candidates / "summary.json")
    standard_all = read_json(args.candidates / "standard" / "segments.json")
    strict_all = read_json(args.candidates / "strict" / "segments.json")
    standard_sources = read_json(
        args.candidates / "standard" / "source_summary.json"
    )
    duplicates = read_json(args.candidates / "duplicates.json")
    catalog = {row["id"]: row for row in read_json(args.catalog)}

    standard = best_per_source(standard_all)
    strict = best_per_source(strict_all)
    zeros = []
    for row in standard_sources:
        if row["segments"]:
            continue
        source = catalog[row["source_id"]]
        zeros.append(
            {
                **row,
                "source_path": source["path"],
                "rotation_clockwise": source.get(
                    "rotation_correction_clockwise", 0
                ),
            }
        )

    jobs: list[tuple[dict, Path]] = []
    images: dict[tuple[str, str], str] = {}
    for group, rows in (("standard", standard), ("strict", strict)):
        for row in rows:
            relative = Path("thumbnails") / group / f"{row['segment_id']}.jpg"
            images[(group, row["segment_id"])] = relative.as_posix()
            jobs.append((row, args.output_dir / relative))
    for row in zeros:
        preview_row = {
            **row,
            "source_start_s": row["seconds"] * 0.40,
            "source_end_s": row["seconds"] * 0.60,
        }
        relative = Path("thumbnails") / "excluded" / f"{row['source_id']}.jpg"
        images[("excluded", row["source_id"])] = relative.as_posix()
        jobs.append((preview_row, args.output_dir / relative))

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

    standard_cards = "".join(
        card(
            row,
            images[("standard", row["segment_id"])],
            args.output_dir,
            "standard",
        )
        for row in standard
    )
    strict_cards = "".join(
        card(
            row,
            images[("strict", row["segment_id"])],
            args.output_dir,
            "strict",
        )
        for row in strict
    )
    zero_cards = "".join(
        zero_card(
            row,
            images[("excluded", row["source_id"])],
            args.output_dir,
        )
        for row in zeros
    )
    duplicate_rows = "".join(
        "<tr><td>" + html.escape(row["relative_path"]) + "</td><td>" +
        html.escape(row["canonical_relative_path"]) + "</td></tr>"
        for row in duplicates
    )
    total_seconds = summaries["standard"]["input_frames"] / 24
    standard_seconds = summaries["standard"]["selected_seconds"]
    strict_seconds = summaries["strict"]["selected_seconds"]
    generated = {
        "independent_sources": summaries["standard"]["sources"],
        "duplicate_sources": len(duplicates),
        "standard_representatives": len(standard),
        "strict_representatives": len(strict),
        "standard_zero_sources": len(zeros),
        "preview_failures": len(failures),
    }
    (args.output_dir / "report_summary.json").write_text(
        json.dumps(generated, ensure_ascii=False, indent=2) + "\n"
    )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AdvanX 姿态分段筛选报告</title>
<style>
:root{{--bg:#080a08;--card:#111510;--line:#283026;--text:#f2f4ef;--muted:#aeb7aa;--green:#76b900;--red:#ff675c}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
header{{padding:48px max(24px,5vw) 28px;border-bottom:1px solid var(--line)}} h1{{font-size:clamp(30px,5vw,64px);margin:0 0 8px;letter-spacing:-.04em}}
.lead{{color:var(--muted);max-width:920px}} .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:28px}}
.stat{{background:var(--card);border:1px solid var(--line);padding:18px;border-radius:14px}} .stat b{{display:block;font-size:28px;color:var(--green)}}
main{{padding:24px max(24px,5vw) 80px}} nav{{position:sticky;top:0;z-index:4;background:#080a08ee;padding:12px 0;display:flex;gap:8px;flex-wrap:wrap}}
button,input,select{{font:inherit}} button{{border:1px solid #3c4738;background:#171d15;color:var(--text);padding:8px 12px;border-radius:8px;cursor:pointer}}
button:hover,.active{{border-color:var(--green)}} .toolbar{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}} input,select{{background:#111510;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:10px}}
.panel{{display:none}} .panel.active{{display:block}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}}
.candidate{{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}} .candidate[data-decision=keep]{{border-color:var(--green)}} .candidate[data-decision=drop]{{opacity:.45;border-color:var(--red)}}
.candidate img{{display:block;width:100%;aspect-ratio:720/320;object-fit:cover;background:#000}} .candidate-body{{padding:14px}} .candidate-top{{display:flex;justify-content:space-between}}
.candidate h3{{font-size:15px;margin:8px 0;overflow-wrap:anywhere}} .candidate p{{margin:4px 0;color:var(--muted);font-size:13px}} .badge{{font-size:12px;padding:2px 8px;border-radius:20px;background:#233118;color:#bde979}}
.badge.strict{{background:#304c0d;color:#dcff9a}} .badge.excluded{{background:#44201e;color:#ffaaa3}} .actions{{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}} .keep{{color:#bde979}} .drop{{color:#ffaaa3}}
table{{width:100%;border-collapse:collapse;background:var(--card)}} td,th{{text-align:left;padding:10px;border-bottom:1px solid var(--line)}} .notice{{border-left:4px solid var(--green);padding:12px 16px;background:#111510;margin:16px 0;color:var(--muted)}}
dialog{{width:min(920px,94vw);background:#0b0d0a;color:var(--text);border:1px solid var(--line);border-radius:14px;padding:16px}} dialog video{{width:100%;max-height:76vh;background:#000}} dialog::backdrop{{background:#000c}}
</style></head><body>
<header><h1>姿态驱动的分段与筛选</h1><p class="lead">按 24 fps 的逐帧姿态、头部位姿、视线、眼睛开合、人物尺度与画面位置寻找连续可用区间。预览图依次取片段前 15%、中点、后 15%。</p>
<div class="stats"><div class="stat"><b>{summaries['standard']['sources']}</b>独立原片</div><div class="stat"><b>{clock(total_seconds)}</b>去重后时长</div><div class="stat"><b>{summaries['standard']['segments']}</b>标准候选</div><div class="stat"><b>{clock(standard_seconds)}</b>标准保留</div><div class="stat"><b>{summaries['strict']['segments']}</b>严格候选</div><div class="stat"><b>{clock(strict_seconds)}</b>严格保留</div></div></header>
<main><div class="notice">当前缓存只可靠描述主角。face_candidates 在全量结果中始终为 1，电视画面或远处背景人物仍需人工看预览，不能据此自动排除。</div>
<nav><button class="tab active" data-panel="standard">标准档 · 每原片最佳</button><button class="tab" data-panel="strict">严格档 · 每原片最佳</button><button class="tab" data-panel="excluded">标准档零产出</button><button class="tab" data-panel="duplicates">重复素材</button><button onclick="exportDecisions()">导出人工决定</button></nav>
<div class="toolbar"><input id="search" placeholder="筛选路径"><select id="shot"><option value="">全部景别</option><option value="close">近景</option><option value="medium">中景</option><option value="wide">远景</option></select><select id="decision"><option value="">全部状态</option><option value="keep">已保留</option><option value="drop">已淘汰</option><option value="pending">未决定</option></select></div>
<section id="standard" class="panel active"><h2>标准档：{len(standard)} 个原片有候选</h2><div class="grid">{standard_cards}</div></section>
<section id="strict" class="panel"><h2>严格档：{len(strict)} 个原片有候选</h2><div class="grid">{strict_cards}</div></section>
<section id="excluded" class="panel"><h2>标准档零产出：{len(zeros)} 个原片</h2><div class="grid">{zero_cards}</div></section>
<section id="duplicates" class="panel"><h2>姿态轨迹完全一致的重复素材：{len(duplicates)} 条</h2><table><thead><tr><th>排除路径</th><th>保留路径</th></tr></thead><tbody>{duplicate_rows}</tbody></table></section>
</main>
<dialog id="player"><h3 id="playerTitle"></h3><video id="video" controls></video><p><button onclick="closePlayer()">关闭</button></p></dialog>
<script>
const decisions=JSON.parse(localStorage.getItem('advanx-segment-decisions')||'{{}}');
function paint(){{document.querySelectorAll('.candidate').forEach(c=>{{c.dataset.decision=decisions[c.dataset.id]||''}});filterCards()}}
function decide(id,value){{if(value)decisions[id]=value;else delete decisions[id];localStorage.setItem('advanx-segment-decisions',JSON.stringify(decisions));paint()}}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{{document.querySelectorAll('.tab,.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.panel).classList.add('active');filterCards()}});
function filterCards(){{const q=document.getElementById('search').value.toLowerCase(),shot=document.getElementById('shot').value,d=document.getElementById('decision').value;document.querySelectorAll('.panel.active .candidate').forEach(c=>{{const state=c.dataset.decision||'pending';c.style.display=(!q||c.dataset.path.toLowerCase().includes(q))&&(!shot||c.dataset.shot===shot)&&(!d||state===d)?'':'none'}})}}
document.querySelectorAll('#search,#shot,#decision').forEach(x=>x.oninput=filterCards);
let stopAt=0;const video=document.getElementById('video');video.addEventListener('timeupdate',()=>{{if(stopAt&&video.currentTime>=stopAt)video.pause()}});
function playSegment(src,start,end,title){{stopAt=end;document.getElementById('playerTitle').textContent=title+' · '+start.toFixed(2)+'s–'+end.toFixed(2)+'s';video.src=src;video.onloadedmetadata=()=>{{video.currentTime=start;video.play()}};document.getElementById('player').showModal()}}
function closePlayer(){{video.pause();video.removeAttribute('src');video.load();document.getElementById('player').close()}}
function exportDecisions(){{const blob=new Blob([JSON.stringify({{generated_at:new Date().toISOString(),decisions}},null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='advanx-review-decisions.json';a.click();URL.revokeObjectURL(a.href)}}
paint();
</script></body></html>"""
    (args.output_dir / "index.html").write_text(page)
    print(json.dumps(generated, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
