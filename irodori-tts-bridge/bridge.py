"""
Irodori-TTS HTTP bridge.

Endpoints expected by FlowMinds://AI's IrodoriTtsService:

    POST /api/irodori/v1/tts
        { text, num_steps?, seed?, cfg_scale_speaker?, truncation_factor?, ref_wav? }
        -> audio/wav  (voice clone with reference audio, or no-ref)

    POST /api/voicedesign/v1/tts
        { text, caption?, num_steps?, seed?, cfg_scale_caption?, truncation_factor?, ref_wav? }
        -> audio/wav

v4 対応 (2026-08-05):
  - 既定チェックポイントは統合モデル Aratako/Irodori-TTS-v4-Small。
    v4 は text + ref + caption の 3-branch を1チェックポイントで扱うため、
    両エンドポイントとも同一ランタイムを共有する（ロード1回・VRAM半減）。
  - ランタイムのキャッシュキーはチェックポイントソース。環境変数で v3 の
    2モデル構成にも戻せる:
        IRODORI_CHECKPOINT              … 両エンドポイント共通の上書き
        IRODORI_CHECKPOINT_IRODORI      … /api/irodori 側のみ上書き
        IRODORI_CHECKPOINT_VOICEDESIGN  … /api/voicedesign 側のみ上書き
    (v3 に戻す例: IRODORI_CHECKPOINT_IRODORI=Aratako/Irodori-TTS-500M-v3
                 IRODORI_CHECKPOINT_VOICEDESIGN=Aratako/Irodori-TTS-600M-v3-VoiceDesign)
  - チェックポイント取得は download_hf_checkpoint() を使用
    （v4 はリポジトリ同梱の tokenizer/ アセットも必要なため）。
  - max_ref_seconds=None でチェックポイント推奨値に委譲
    （v4-Small=120秒、メタデータの無い旧チェックポイントは30秒）。

`ref_wav` in the irodori endpoint is a path on the *bridge host* filesystem
(i.e. this PC) — since FlowMinds and the bridge run on the same machine this
avoids multipart uploads and the per-request DACVAE latent encode cost stays
amortised across calls with the same reference.
"""

from __future__ import annotations

import io
import os
import traceback
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    download_hf_checkpoint,
    resolve_cfg_scales,
)

# duration predictor により出力長は自動推定される（seconds=None）。
# 念のため上限を MAX_SECONDS でキャップする。
MAX_SECONDS = 30.0

_DEFAULT_CHECKPOINT = "Aratako/Irodori-TTS-v4-Small"
_shared_override = os.environ.get("IRODORI_CHECKPOINT", _DEFAULT_CHECKPOINT)
_CHECKPOINT_SOURCES = {
    "irodori": os.environ.get("IRODORI_CHECKPOINT_IRODORI", _shared_override),
    "voicedesign": os.environ.get("IRODORI_CHECKPOINT_VOICEDESIGN", _shared_override),
}

_device = os.environ.get("IRODORI_DEVICE", default_runtime_device())
# bf16 は CUDA / XPU でのみ許可。CPU / MPS では fp32 を既定にする。
_default_precision = "bf16" if _device in ("cuda", "xpu") else "fp32"
_precision = os.environ.get("IRODORI_PRECISION", _default_precision)
_codec_repo = os.environ.get("IRODORI_CODEC_REPO", "Aratako/Semantic-DACVAE-Japanese-32dim")

# チェックポイントソース単位でキャッシュ（v4 では両モードが同一ランタイムを共有する）
_runtimes: dict[str, InferenceRuntime] = {}

app = FastAPI(title="Irodori-TTS Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_checkpoint_path(source: str) -> str:
    """ローカルの .pt/.safetensors パスはそのまま、それ以外は HF repo id として取得する。"""
    if source.lower().endswith((".pt", ".safetensors")) and os.path.isfile(source):
        return source
    return download_hf_checkpoint(source)


def _get_runtime(mode: str) -> InferenceRuntime:
    if mode not in _CHECKPOINT_SOURCES:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")
    source = _CHECKPOINT_SOURCES[mode]
    rt = _runtimes.get(source)
    if rt is not None:
        return rt

    print(f"[bridge] Loading runtime for mode={mode} from {source} ...", flush=True)
    checkpoint_path = _resolve_checkpoint_path(source)
    rt = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=_device,
            codec_repo=_codec_repo,
            model_precision=_precision,
            codec_device=_device,
            codec_precision=_precision,
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        )
    )
    _runtimes[source] = rt
    print(f"[bridge] Runtime ready ({source})", flush=True)
    return rt


def _use_speaker_condition(runtime: InferenceRuntime) -> bool:
    # v4 の ModelConfig は use_speaker_condition が None (=自動解決) の場合がある。
    # 解決済みプロパティを優先し、旧チェックポイントの生フラグへフォールバックする。
    resolved = getattr(runtime.model_cfg, "use_speaker_condition_resolved", None)
    if resolved is not None:
        return bool(resolved)
    return bool(runtime.model_cfg.use_speaker_condition)


def _build_request(
    runtime: InferenceRuntime,
    text: str,
    num_steps: int,
    caption: Optional[str],
    ref_wav: Optional[str],
    seed: Optional[int],
    cfg_scale_text: float,
    cfg_scale_caption: float,
    cfg_scale_speaker: float,
    truncation_factor: Optional[float],
) -> SamplingRequest:
    use_caption = bool(
        runtime.model_cfg.use_caption_condition
        and caption is not None
        and caption.strip() != ""
    )
    use_speaker = _use_speaker_condition(runtime)
    no_ref = not (use_speaker and ref_wav is not None and ref_wav.strip() != "")

    if ref_wav is not None and ref_wav.strip() != "":
        if not os.path.isfile(ref_wav):
            raise HTTPException(
                status_code=400,
                detail=f"ref_wav file not found: {ref_wav}",
            )

    cfg_text, cfg_caption, cfg_speaker, _messages = resolve_cfg_scales(
        cfg_guidance_mode="independent",
        cfg_scale_text=float(cfg_scale_text),
        cfg_scale_caption=float(cfg_scale_caption),
        cfg_scale_speaker=float(cfg_scale_speaker),
        cfg_scale=None,
        use_caption_condition=use_caption,
        use_speaker_condition=use_speaker,
    )

    return SamplingRequest(
        text=text,
        caption=caption if use_caption else None,
        ref_wav=ref_wav if (use_speaker and not no_ref) else None,
        ref_latent=None,
        no_ref=no_ref,
        ref_normalize_db=-16.0,
        ref_ensure_max=True,
        num_candidates=1,
        decode_mode="sequential",
        seconds=None,  # duration predictor による自動長予測
        duration_scale=1.0,
        max_seconds=MAX_SECONDS,
        max_ref_seconds=None,  # チェックポイント推奨値（v4-Small=120s、旧=30s）
        max_text_len=None,
        max_caption_len=None,
        num_steps=int(num_steps),
        cfg_scale_text=cfg_text,
        cfg_scale_caption=cfg_caption,
        cfg_scale_speaker=cfg_speaker,
        cfg_guidance_mode="independent",
        cfg_scale=None,
        cfg_min_t=0.5,
        cfg_max_t=1.0,
        truncation_factor=truncation_factor,
        rescale_k=None,
        rescale_sigma=None,
        context_kv_cache=True,
        speaker_kv_scale=None,
        speaker_kv_min_t=None,
        speaker_kv_max_layers=None,
        seed=seed,
        trim_tail=True,
        tail_window_size=20,
        tail_std_threshold=0.05,
        tail_mean_threshold=0.1,
    )


def _synthesize_wav(
    mode: str,
    text: str,
    num_steps: int,
    caption: Optional[str],
    ref_wav: Optional[str],
    seed: Optional[int],
    cfg_scale_text: float,
    cfg_scale_caption: float,
    cfg_scale_speaker: float,
    truncation_factor: Optional[float],
) -> bytes:
    # リクエスト内容を可視化（参照音声が使われているか、seedが固定されているか、等の確認用）
    ref_info = f"ref_wav={ref_wav!r}" if ref_wav else "no_ref"
    print(
        f"[bridge] mode={mode} steps={num_steps} seed={seed} "
        f"cfg(t/c/s)={cfg_scale_text}/{cfg_scale_caption}/{cfg_scale_speaker} "
        f"trunc={truncation_factor} {ref_info} text={text[:40]!r}",
        flush=True,
    )
    runtime = _get_runtime(mode)
    request = _build_request(
        runtime=runtime,
        text=text,
        num_steps=num_steps,
        caption=caption,
        ref_wav=ref_wav,
        seed=seed,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_caption=cfg_scale_caption,
        cfg_scale_speaker=cfg_scale_speaker,
        truncation_factor=truncation_factor,
    )
    result = runtime.synthesize(request, log_fn=None)

    audio = result.audio
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio[0]

    buf = io.BytesIO()
    sf.write(buf, audio, int(result.sample_rate), format="WAV", subtype="PCM_16")
    return buf.getvalue()


class IrodoriRequest(BaseModel):
    text: str
    num_steps: int = 40
    ref_wav: Optional[str] = None
    seed: Optional[int] = None
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 5.0
    truncation_factor: Optional[float] = None


class VoiceDesignRequest(BaseModel):
    text: str
    caption: str = ""
    num_steps: int = 40
    # 3-branch (text + ref_wav + caption)。
    # ref_wav を指定すると参照音声で声質を固定しつつ caption でスタイル制御できる。
    ref_wav: Optional[str] = None
    seed: Optional[int] = None
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_speaker: float = 5.0
    truncation_factor: Optional[float] = None


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "irodori-tts-bridge",
        "device": _device,
        "precision": _precision,
        "checkpoints": _CHECKPOINT_SOURCES,
        "loaded_checkpoints": list(_runtimes.keys()),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/irodori/v1/tts")
def tts_irodori(req: IrodoriRequest):
    try:
        wav = _synthesize_wav(
            mode="irodori",
            text=req.text,
            num_steps=req.num_steps,
            caption=None,
            ref_wav=req.ref_wav,
            seed=req.seed,
            cfg_scale_text=req.cfg_scale_text,
            cfg_scale_caption=3.0,
            cfg_scale_speaker=req.cfg_scale_speaker,
            truncation_factor=req.truncation_factor,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print("[bridge] synthesis failed:\n" + tb, flush=True)
        raise HTTPException(status_code=500, detail=f"synthesis failed: {exc}") from exc
    return Response(content=wav, media_type="audio/wav")


@app.post("/api/voicedesign/v1/tts")
def tts_voicedesign(req: VoiceDesignRequest):
    try:
        wav = _synthesize_wav(
            mode="voicedesign",
            text=req.text,
            num_steps=req.num_steps,
            caption=req.caption,
            ref_wav=req.ref_wav,
            seed=req.seed,
            cfg_scale_text=req.cfg_scale_text,
            cfg_scale_caption=req.cfg_scale_caption,
            cfg_scale_speaker=req.cfg_scale_speaker,
            truncation_factor=req.truncation_factor,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print("[bridge] synthesis failed:\n" + tb, flush=True)
        raise HTTPException(status_code=500, detail=f"synthesis failed: {exc}") from exc
    return Response(content=wav, media_type="audio/wav")
