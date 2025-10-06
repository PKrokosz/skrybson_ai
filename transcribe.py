import os, json, sys, glob, pathlib, re
from datetime import datetime
from faster_whisper import WhisperModel

# --- konfiguracja ---
SESSION_DIR = os.environ.get("SESSION_DIR")
RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "/app/recordings")
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")

def pick_latest_session(recordings_dir: str) -> str:
    sessions = sorted([p for p in pathlib.Path(recordings_dir).glob("*") if p.is_dir()],
                      key=lambda p: p.stat().st_mtime, reverse=True)
    return str(sessions[0]) if sessions else ""

def norm_text(t: str) -> str:
    t = t.strip().lower()
    t = re.sub(r"[^\wąćęłńóśżź ]+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def main():
    session_dir = SESSION_DIR or pick_latest_session(RECORDINGS_DIR)
    if not session_dir:
        print("Brak sesji do transkrypcji.")
        sys.exit(1)

    raw_dir = os.path.join(session_dir, "raw")
    out_dir = os.path.join(session_dir, "transcripts")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, "*.wav")))
    files = [f for f in files if os.path.getsize(f) >= 1024]
    if not files:
        print(f"[!] Brak sensownych plików WAV w {raw_dir} (>=1KB).")
        sys.exit(0)

    print(f"[i] Sesja: {session_dir}")
    print(f"[i] Model: {MODEL_SIZE}")

    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    buckets = {}
    for f in files:
        name = os.path.basename(f)
        user_prefix = name.rsplit("_seg", 1)[0]
        buckets.setdefault(user_prefix, []).append(f)

    summary_index = []

    for user_prefix, wavs in buckets.items():
        wavs.sort(key=lambda x: os.path.getmtime(x))
        timeline = []
        print(f"\n=== START USER {user_prefix} ===")

        for wav in wavs:
            try:
                file_t0 = os.path.getmtime(wav)
                print(f"[file] {os.path.basename(wav)} mtime={file_t0}")
                segments, _info = model.transcribe(
                    wav,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                for seg in segments:
                    pseudo_t = file_t0 + float(seg.start)
                    print(f"  [seg] {wav} {seg.start:.2f}-{seg.end:.2f}s :: {seg.text.strip()}")
                    timeline.append({
                        "pseudo_t": pseudo_t,
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": seg.text.strip(),
                        "file": os.path.basename(wav),
                    })
            except Exception as e:
                print(f"[!] Pomijam uszkodzony plik: {wav} ({e})")
                continue

        timeline.sort(key=lambda x: x["pseudo_t"])

        deduped = []
        last_norm = ""
        last_t = -1e9
        for item in timeline:
            nt = norm_text(item["text"])
            if deduped and nt == last_norm and (item["pseudo_t"] - last_t) < 1.5:
                print(f"  [dup] Pomijam duplikat: {item['text']}")
                continue
            deduped.append(item)
            last_norm = nt
            last_t = item["pseudo_t"]

        t0 = deduped[0]["pseudo_t"]
        segments_all = [{
            "start": round(it["pseudo_t"] - t0, 2),
            "end": round(it["pseudo_t"] - t0 + (it["end"] - it["start"]), 2),
            "text": it["text"]
        } for it in deduped]

        user_json_path = os.path.join(out_dir, f"{user_prefix}.json")
        with open(user_json_path, "w", encoding="utf-8") as f:
            json.dump({"user": user_prefix, "segments": segments_all}, f, ensure_ascii=False, indent=2)
        print(f"[✓] Zapisano: {user_json_path} ({len(segments_all)} segmentów)")
        summary_index.append({"user": user_prefix, "segments": len(segments_all)})

        print(f"=== END USER {user_prefix} ===\n")

    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "session_dir": session_dir,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "items": summary_index
        }, f, ensure_ascii=False, indent=2)
    print(f"[✓] Gotowe. Index: {index_path}")

if __name__ == "__main__":
    main()
