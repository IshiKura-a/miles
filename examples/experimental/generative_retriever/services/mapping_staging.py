"""Local /tmp staging and static snapshot helpers for columnar DocID-to-keyword mappings."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import threading

from filelock import FileLock
import pyarrow as pa
import pyarrow.parquet as pq

from examples.experimental.generative_retriever.index.id_text_mapping import COLUMNAR_INDEX_FILES
from examples.experimental.generative_retriever.reward.relevance_reward import is_doc2k_disabled
from examples.experimental.generative_retriever.utils.paths import absolute_path

RESUME_OVERLAY_FILENAME = "gr_doc2k_resume_overlay.parquet"


def stage_local_mapping_dir(mapping_path: str | Path) -> Path:
    resolved = Path(mapping_path).resolve()
    local_root = Path(tempfile.gettempdir())
    target = resolved if resolved.is_relative_to(local_root) else local_root / f"gr_mapping_{resolved.name}"
    sentinel = target / ".staged_arrow_ok"
    if not sentinel.is_file():
        with FileLock(f"{target}.lock"):
            if not sentinel.is_file():
                target.mkdir(parents=True, exist_ok=True)
                for fname in COLUMNAR_INDEX_FILES:
                    src = resolved / fname
                    if not src.is_file():
                        raise FileNotFoundError(f"Columnar mapping file missing: {src}")
                    dst = target / fname
                    if dst != src and (not dst.is_file() or dst.stat().st_size != src.stat().st_size):
                        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{fname}.", dir=target)
                        os.close(tmp_fd)
                        tmp_path = Path(tmp_name)
                        shutil.copyfile(src, tmp_path)
                        tmp_path.replace(dst)
                ipc_dst = target / "keywords_only.arrow"
                if not ipc_dst.is_file():
                    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".keywords_only.arrow.", dir=target)
                    os.close(tmp_fd)
                    tmp_ipc = Path(tmp_name)
                    tbl = pq.read_table(target / "keywords.parquet", columns=["keyword"])
                    with pa.OSFile(str(tmp_ipc), "wb") as sink:
                        with pa.ipc.new_file(sink, tbl.schema) as writer:
                            writer.write_table(tbl)
                    tmp_ipc.replace(ipc_dst)
                sentinel.write_text(str(resolved), encoding="utf-8")
    return target


def stage_local_doc2k_state(
    checkpoint_dir: str | Path | None = None,
    *,
    resume: bool = False,
) -> None:
    local_root = Path(tempfile.gettempdir())
    for gpu_idx in range(16):
        (local_root / f"gr_doc2k_local_gpu{gpu_idx}.pkl").unlink(missing_ok=True)
        (local_root / f"gr_doc2k_local_gpu{gpu_idx}.delta.bin").unlink(missing_ok=True)
        (local_root / f"gr_doc2k_global_gpu{gpu_idx}.pkl").unlink(missing_ok=True)
    resume_dst = local_root / RESUME_OVERLAY_FILENAME
    if is_doc2k_disabled() or not resume or checkpoint_dir is None:
        resume_dst.unlink(missing_ok=True)
        return
    ckpt_root = Path(checkpoint_dir)
    tracker = ckpt_root / "latest_checkpointed_iteration.txt"
    raw_tracker = tracker.read_text(encoding="utf-8").strip() if tracker.is_file() else ""
    iteration = int(raw_tracker) if raw_tracker.isdigit() else -1
    if iteration <= 0:
        resume_dst.unlink(missing_ok=True)
        return
    overlay_src = ckpt_root / f"iter_{iteration:07d}" / "mapping" / "overlay.parquet"
    if not overlay_src.is_file():
        resume_dst.unlink(missing_ok=True)
        return
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".gr_resume_overlay.", dir=local_root)
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    shutil.copyfile(overlay_src, tmp_path)
    tmp_path.replace(resume_dst)


def save_static_mapping_snapshot(
    output_dir: str | Path,
    *,
    base_mapping: str | Path,
    stage_dir: str | Path | None = None,
    async_transfer: bool = True,
) -> dict[str, str]:
    target_dir = absolute_path(output_dir)
    resolved_base = Path(base_mapping).resolve()
    staged_marker = resolved_base / ".staged_arrow_ok"
    if staged_marker.is_file():
        origin_text = staged_marker.read_text(encoding="utf-8").strip()
        if origin_text and Path(origin_text).is_dir():
            resolved_base = Path(origin_text).resolve()

    overlay_table = pa.table({
        "docid_u32": pa.array([], type=pa.uint32()),
        "keyword_ids": pa.array([], type=pa.list_(pa.uint32())),
    }).replace_schema_metadata({b"base_mapping": str(resolved_base).encode()})

    def _link_target_for(fname: str) -> str:
        src_f = resolved_base / fname
        try:
            common = os.path.commonpath([str(target_dir), str(src_f)])
        except ValueError:
            return str(src_f)
        if common not in {"", os.sep}:
            return os.path.relpath(src_f, start=target_dir)
        return str(src_f)

    def _populate_dir(local_d: Path) -> None:
        local_d.mkdir(parents=True, exist_ok=True)
        pq.write_table(overlay_table, local_d / "overlay.parquet", compression="zstd")
        for link_name in COLUMNAR_INDEX_FILES:
            src_f = resolved_base / link_name
            dst_f = local_d / link_name
            if not dst_f.exists() and not dst_f.is_symlink() and src_f.is_file():
                dst_f.symlink_to(_link_target_for(link_name))

    if stage_dir is not None:
        _populate_dir(Path(stage_dir))
        return {"output_dir": str(target_dir), "staged_dir": str(stage_dir)}

    tmp_dir = Path(tempfile.mkdtemp(prefix="gr_save_static_map_"))
    _populate_dir(tmp_dir)

    def _sync_worker() -> None:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            dst_tmp = target_dir / "overlay.parquet.tmp"
            shutil.copyfile(tmp_dir / "overlay.parquet", dst_tmp)
            dst_tmp.replace(target_dir / "overlay.parquet")
            for link_name in COLUMNAR_INDEX_FILES:
                src_link = tmp_dir / link_name
                dst_link = target_dir / link_name
                if src_link.is_symlink() and not dst_link.exists() and not dst_link.is_symlink():
                    dst_link.symlink_to(os.readlink(src_link))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if async_transfer:
        threading.Thread(
            target=_sync_worker,
            name=f"gr-static-map-sync-{target_dir.name}",
            daemon=False,
        ).start()
    else:
        _sync_worker()
    return {"output_dir": str(target_dir)}
