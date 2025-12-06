import io
import json
import multiprocessing
import os
import tarfile
import time
from pathlib import Path

import fsspec
from img2dataset.resizer import Resizer


def _run_single_shard_resize(resizer, row):
    shard_id, shard_file = row
    proc = multiprocessing.current_process()
    print(f"[PROC {proc.name} pid={os.getpid()}] START RESIZE shard {shard_id}", flush=True)

    resize_start = time.time()
    failed_to_resize = 0
    error_counts = {}

    fs, shard_path = fsspec.core.url_to_fs(shard_file)
    with fs.open(shard_path, "rb") as f:
        tar_bytes = f.read()

    tar = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:")

    new_buf = io.BytesIO()
    new_tar = tarfile.open(fileobj=new_buf, mode="w:")

    members = tar.getmembers()
    failed_basenames = set()
    image_results = {}

    for m in members:
        if not m.isfile():
            continue

        if m.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            raw = tar.extractfile(m).read()
            img_stream = io.BytesIO(raw)
            img_stream.seek(0)

            t0 = time.time()
            resized_bytes, w, h, ow, oh, err = resizer(img_stream)
            dt = time.time() - t0

            if dt > 5:
                print(
                    f"[SLOW-RESIZE] {dt:.1f}s shard={shard_id} key={m.name} "
                    f"pid={os.getpid()}",
                    flush=True,
                )

            base, _ = os.path.splitext(m.name)

            if err:
                print(f"[RESIZE FAIL] {m.name}: {err}", flush=True)
                failed_to_resize += 1
                msg = str(err)
                error_counts[msg] = error_counts.get(msg, 0) + 1
                failed_basenames.add(base)
                image_results[m.name] = None
            else:
                image_results[m.name] = resized_bytes

    for m in members:
        if not m.isfile():
            new_tar.addfile(m)
            continue

        base, ext = os.path.splitext(m.name)
        ext_lower = ext.lower()

        if base in failed_basenames:
            continue

        if m.name in image_results:
            final = image_results[m.name]
            if final is None:
                continue

            info = tarfile.TarInfo(name=m.name)
            info.size = len(final)
            new_tar.addfile(info, io.BytesIO(final))
            continue

        if ext_lower == ".json":
            raw = tar.extractfile(m).read()
            try:
                meta = json.loads(raw.decode("utf-8"))
            except Exception:
                info = tarfile.TarInfo(name=m.name)
                info.size = len(raw)
                new_tar.addfile(info, io.BytesIO(raw))
                continue

            meta["width"] = resizer.image_size
            meta["height"] = resizer.image_size

            updated = json.dumps(meta, ensure_ascii=False).encode("utf-8")
            info = tarfile.TarInfo(name=m.name)
            info.size = len(updated)
            new_tar.addfile(info, io.BytesIO(updated))
            continue

        raw = tar.extractfile(m).read()
        info = tarfile.TarInfo(name=m.name)
        info.size = len(raw)
        new_tar.addfile(info, io.BytesIO(raw))

    tar.close()
    new_tar.close()

    with fs.open(shard_path, "wb") as f_out:
        f_out.write(new_buf.getvalue())

    resize_end = time.time()
    resize_duration = resize_end - resize_start

    shard_dir = os.path.dirname(shard_path)
    stats_path = os.path.join(shard_dir, f"{shard_id}_stats.json")

    try:
        with fs.open(stats_path, "r") as sf:
            stats = json.load(sf)

        stats["failed_to_resize"] = stats.get("failed_to_resize", 0) + failed_to_resize
        stats["resize_duration"] = resize_duration

        status_dict = stats.get("status_dict", {})
        for msg, count in error_counts.items():
            status_dict[msg] = status_dict.get(msg, 0) + count
        stats["status_dict"] = status_dict

        with fs.open(stats_path, "w") as sf:
            json.dump(stats, sf, indent=4)

    except FileNotFoundError:
        print(f"Stats file not found for tar: {shard_path}", flush=True)

    print(
        f"[PROC {proc.name}] FINISH RESIZE shard {shard_id} | "
        f"failed_to_resize={failed_to_resize} | resize_duration={resize_duration:.2f}s",
        flush=True,
    )


def iter_shards(root_dir, start_at=0):
    """Yield one shard row at a time, skipping those below start_at."""
    for tar_path in Path(root_dir).glob("*.tar"):
        try:
            shard_num = int(tar_path.stem)  # convert "123" → 123
        except ValueError:
            continue

        if shard_num < start_at:
            continue
        yield tar_path.stem, str(tar_path)


def multiprocessing_resizer(processes_count, resizer, gen, shard_timeout_seconds=600):
    """Run _run_single_shard_resize on all rows in gen with up to processes_count workers."""
    ctx = multiprocessing.get_context("spawn")
    failed_shards = []
    active = []
    gen_iter = iter(gen)

    def start_new_if_possible():
        while len(active) < processes_count:
            try:
                row = next(gen_iter)
            except StopIteration:
                break
            p = ctx.Process(target=_run_single_shard_resize, args=(resizer, row))
            p.start()
            active.append((p, row, time.time()))

    start_new_if_possible()

    while active:
        # now = time.time()
        for p, row, t0 in list(active):
            shard_id, shard_file = row

            if not p.is_alive():
                p.join()
                active.remove((p, row, t0))
                if p.exitcode != 0:
                    print(
                        f"[RESIZER] shard {shard_id} finished with exitcode {p.exitcode}, "
                        f"marking as failed",
                        flush=True,
                    )
                    failed_shards.append(row)
                else:
                    print(
                        f"[RESIZER] shard {shard_id} finished successfully",
                        flush=True,
                    )
                continue

            # if now - t0 > shard_timeout_seconds:
            #     print(
            #         f"[RESIZER TIMEOUT] shard {shard_id} from {shard_file} "
            #         f"exceeded {shard_timeout_seconds} seconds, terminating process",
            #         flush=True,
            #     )
            #     p.terminate()
            #     p.join()
            #     active.remove((p, row, t0))
            #     failed_shards.append(row)

        start_new_if_possible()
        time.sleep(0.1)

    return failed_shards


if __name__ == "__main__":
    resizer = Resizer(
        image_size=256,
        resize_mode="border",
        resize_only_if_bigger=False,
        encode_quality=95,
    )

    gen = iter_shards(".", start_at=1)

    multiprocessing_resizer(processes_count=8, resizer=resizer, gen=gen)
