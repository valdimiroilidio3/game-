#!/usr/bin/env python3
"""jarvis-show — acorda o OpenJarvis com duas palmas.

Fica ouvindo o microfone; quando ouve duas palmas em sequência, toca o jingle
(se existir), garante que o servidor web está no ar, abre a interface no
navegador e inicia `uv run jarvis chat --voice`.

Coloque este arquivo na raiz do checkout do OpenJarvis (a pasta que tem
pyproject.toml) e rode com:

    uv run python jarvis-show.py

Variáveis de ambiente
---------------------
JARVIS_CLAP_THRESH  Pico mínimo (0.0-1.0) para um transiente contar como palma.
                    Padrão: 0.20
JARVIS_MIC          Índice (ex. "2") ou parte do nome (ex. "Microfone") do
                    dispositivo de entrada. Padrão: dispositivo do sistema.

Modos de uso
------------
    --selftest        Testa o detector com palmas sintéticas (não usa microfone)
    --calibrate       Mede o ruído e as palmas do seu microfone e sugere JARVIS_CLAP_THRESH
    --list-devices    Lista as entradas de áudio e os índices para JARVIS_MIC
    --wav ARQUIVO     Analisa um .wav gravado, em vez do microfone ao vivo
    --no-launch       Só detecta e mostra o que faria (não abre chat/navegador)

Como o detector evita falso positivo
------------------------------------
Uma palma tem três assinaturas que a fala não tem juntas:
  1. ataque instantâneo (sobe ao pico em < 16 ms);
  2. timbre de ruído largo (muita energia em alta frequência -> alta taxa de
     cruzamento por zero);
  3. curta: decai abaixo da metade do pico em menos de ~65 ms.
O detector exige as três, em duas ocorrências separadas por 0,12-0,70 s, com
silêncio entre elas.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Parâmetros de áudio / detecção
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
FRAME = 256  # 16 ms por quadro

DEFAULT_THRESH = 0.20
ENV_THRESH = "JARVIS_CLAP_THRESH"
ENV_MIC = "JARVIS_MIC"
ENV_MAXMS = "JARVIS_CLAP_MAXMS"
ENV_MIN_ZCR = "JARVIS_CLAP_MIN_ZCR"

MIN_GAP = 0.12      # duas palmas mais próximas que isto = re-trigger da mesma
MAX_GAP = 0.70      # duas palmas mais distantes que isto = não é "duas palmas"
QUIET_RATIO = 0.50  # entre as palmas o nível tem de cair abaixo de thresh*isto
MAX_BURST_FRAMES = 4   # ~64 ms; acima disso é fala/ruído, não palma
MIN_ZCR = 0.20      # taxa de cruzamento por zero mínima (palma é ruído largo)

WEB_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Detector (núcleo puro — testável sem microfone)
# ---------------------------------------------------------------------------


def zero_crossing_rate(frame: np.ndarray) -> float:
    """Fração de amostras que trocam de sinal. Palma ≈ 0,5; vogal ≈ 0,05-0,15."""
    if frame.size < 2:
        return 0.0
    signs = np.signbit(frame)
    return float(np.mean(signs[1:] != signs[:-1]))


class DoubleClapDetector:
    """Duas palmas = dois transientes curtos e agudos, com silêncio entre eles."""

    def __init__(
        self,
        thresh: float = DEFAULT_THRESH,
        *,
        min_gap: float = MIN_GAP,
        max_gap: float = MAX_GAP,
        quiet_ratio: float = QUIET_RATIO,
        max_burst_frames: int = MAX_BURST_FRAMES,
        min_zcr: float = MIN_ZCR,
        sample_rate: int = SAMPLE_RATE,
        frame: int = FRAME,
    ) -> None:
        self.thresh = float(thresh)
        self.min_gap = float(min_gap)
        self.max_gap = float(max_gap)
        self.quiet_ratio = float(quiet_ratio)
        self.max_burst_frames = int(max_burst_frames)
        self.min_zcr = float(min_zcr)
        self.sample_rate = int(sample_rate)
        self.frame = int(frame)

        self.time = 0.0
        self.onsets: list[float] = []      # palmas confirmadas
        self.rejected: list[str] = []      # diagnóstico do que foi descartado
        self._prev_peak = 0.0
        self._pending: dict | None = None  # transiente em avaliação
        self._min_since_onset = float("inf")
        self._last_onset: float | None = None

    # -- utilidades ---------------------------------------------------------
    def reset(self) -> None:
        self.time = 0.0
        self.onsets.clear()
        self.rejected.clear()
        self._prev_peak = 0.0
        self._pending = None
        self._min_since_onset = float("inf")
        self._last_onset = None

    # -- lógica do par ------------------------------------------------------
    def _accept_clap(self, t: float) -> bool:
        """Registra uma palma confirmada; True se ela fecha um par."""
        if self._last_onset is None:
            self._last_onset = t
            self._min_since_onset = float("inf")
            self.onsets.append(t)
            return False

        gap = t - self._last_onset
        quiet = self._min_since_onset < self.thresh * self.quiet_ratio

        if gap < self.min_gap:
            return False  # mesma palma, eco ou re-trigger
        if gap > self.max_gap:
            self._last_onset = t  # tarde demais: começa um par novo
            self._min_since_onset = float("inf")
            self.onsets.append(t)
            return False
        if not quiet:
            self._last_onset = t  # sem silêncio entre eles: som contínuo
            self._min_since_onset = float("inf")
            return False

        self.onsets.append(t)
        return True

    def _settle_pending(self) -> bool:
        """Fecha a avaliação do transiente pendente. True se ele era palma."""
        pending, self._pending = self._pending, None
        if pending is None:
            return False
        if pending["too_long"]:
            self.rejected.append("longo")
            return False
        if pending["zcr"] < self.min_zcr:
            self.rejected.append("voz")
            return False
        return self._accept_clap(pending["onset"])

    # -- entrada ------------------------------------------------------------
    def feed(self, block: np.ndarray) -> bool:
        """Consome um bloco mono float32 [-1, 1]. True quando ouve duas palmas."""
        samples = np.asarray(block, dtype=np.float32).reshape(-1)
        n_frames = len(samples) // self.frame
        fired = False

        for i in range(n_frames):
            chunk = samples[i * self.frame : (i + 1) * self.frame]
            peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
            self.time += self.frame / self.sample_rate

            rising = peak >= self.thresh and self._prev_peak < min(
                self.thresh * 0.5, peak * 0.5
            )
            if rising:
                # começa a avaliar; se já havia um pendente, resolve antes
                if self._pending is not None and self._settle_pending():
                    fired = True
                self._pending = {
                    "onset": self.time,
                    "run": 1,
                    "too_long": False,
                    "zcr": zero_crossing_rate(chunk),
                }
            elif self._pending is not None:
                if peak >= self.thresh * 0.5:
                    self._pending["run"] += 1
                    if self._pending["run"] > max(self.max_burst_frames, 1):
                        self._pending["too_long"] = True
                else:
                    if self._settle_pending():
                        fired = True

            self._min_since_onset = min(self._min_since_onset, peak)
            self._prev_peak = peak

            if fired:
                return True
        return False

    def finish(self) -> bool:
        """Encerra a análise (fim do arquivo): resolve um pendente em aberto."""
        return self._settle_pending()


def detect_in_array(
    samples: np.ndarray, detector: DoubleClapDetector | None = None, **kwargs
) -> tuple[bool, DoubleClapDetector]:
    """Roda o detector sobre um sinal inteiro (usado no --wav e no --selftest)."""
    det = detector or DoubleClapDetector(**kwargs)
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    fired = False
    step = det.frame * 16
    for start in range(0, len(samples), step):
        if det.feed(samples[start : start + step]):
            fired = True
    if det.finish():
        fired = True
    return fired, det


# ---------------------------------------------------------------------------
# Áudio: microfone, calibragem, dispositivos
# ---------------------------------------------------------------------------


def _sd():
    try:
        import sounddevice as sd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depende do ambiente
        sys.exit(
            "sounddevice não está instalado. Rode dentro da pasta do OpenJarvis:\n"
            "  uv sync --extra desktop --extra voice\n"
            f"(erro original: {exc})"
        )
    return sd


def resolve_device(spec: str | None) -> int | None:
    """Traduz JARVIS_MIC (índice ou parte do nome) em um índice do sounddevice."""
    if not spec:
        return None
    sd = _sd()
    devices = sd.query_devices()
    if str(spec).isdigit():
        idx = int(spec)
        if not 0 <= idx < len(devices):
            sys.exit(f"JARVIS_MIC={spec}: só existem {len(devices)} dispositivos.")
        return idx
    needle = str(spec).lower()
    for idx, dev in enumerate(devices):
        if needle in str(dev["name"]).lower() and dev["max_input_channels"] > 0:
            return idx
    sys.exit(
        f"JARVIS_MIC={spec!r}: nenhum dispositivo de entrada com esse nome.\n"
        "Rode com --list-devices para ver as opções."
    )
    return None


def list_devices() -> None:
    sd = _sd()
    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
    print("Dispositivos de áudio (use o número em JARVIS_MIC):")
    for idx, dev in enumerate(sd.query_devices()):
        kind = "entrada" if dev["max_input_channels"] > 0 else "saída  "
        mark = "  <== padrão" if idx == default_in else ""
        print(f"  {idx:>2}  {kind}  {dev['name']}{mark}")


def live_peaks(seconds: float, device: int | None) -> np.ndarray:
    """Grava *seconds* e devolve os picos por quadro."""
    sd = _sd()
    frames = int(seconds * SAMPLE_RATE / FRAME)
    peaks: list[float] = []
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device, blocksize=FRAME
    ) as stream:
        for _ in range(frames):
            block, _ = stream.read(FRAME)
            peaks.append(float(np.max(np.abs(block[:, 0]))))
    return np.array(peaks, dtype=np.float32)


def calibrate(device: int | None) -> None:
    print("Calibragem — passo 1 de 2: fique em SILÊNCIO por 3 segundos…")
    time.sleep(1.0)
    noise = live_peaks(3.0, device)
    noise_peak = float(np.max(noise)) if noise.size else 0.0
    noise_med = float(np.median(noise)) if noise.size else 0.0
    print(f"  ruído de fundo: pico {noise_peak:.3f} | mediana {noise_med:.3f}")

    print("Calibragem — passo 2 de 2: bata DUAS PALMAS agora (3 s)…")
    time.sleep(0.5)
    claps = live_peaks(3.0, device)
    clap_peak = float(np.max(claps)) if claps.size else 0.0
    print(f"  palmas: pico {clap_peak:.3f}")

    if clap_peak <= noise_peak * 1.2:
        print(
            "\n  As palmas não se destacaram do ruído. Chegue mais perto do microfone,\n"
            "  bata mais forte e repita. Valor atual mantido."
        )
        return

    suggested = max(0.05, min(0.90, (noise_peak * 3 + clap_peak) / 4))
    print(
        f"\n  Sugestão: JARVIS_CLAP_THRESH={suggested:.2f}"
        f"  (entre ruído {noise_peak:.2f} e palma {clap_peak:.2f})"
    )
    print("  PowerShell (vale em terminal novo):")
    print(f'    setx JARVIS_CLAP_THRESH "{suggested:.2f}"')


# ---------------------------------------------------------------------------
# Acordar: jingle, servidor web, chat por voz
# ---------------------------------------------------------------------------


def play_jingle(path: Path) -> None:
    if not path.is_file():
        print(f"  (jingle não encontrado em {path} — seguindo sem som)")
        return
    print(f"  tocando {path.name}…")
    try:
        import sounddevice as sd  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415

        data, sr = sf.read(str(path), dtype="float32")
        sd.play(data, sr)
        sd.wait()
    except Exception as exc:
        print(f"  (não consegui tocar por dentro: {exc})")
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc2:
            print(f"  (jingle ignorado: {exc2})")


def port_open(url: str, timeout: float = 0.4) -> bool:
    host, _, port = url.split("://", 1)[1].partition(":")
    try:
        with socket.create_connection((host, int(port or 80)), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_server(repo: Path) -> bool:
    """Sobe o daemon se a porta 8000 estiver fechada. True se ficou no ar."""
    if port_open(WEB_URL):
        print("  servidor web já está no ar")
        return True
    print("  subindo o servidor web…")
    try:
        subprocess.run(
            ["uv", "run", "jarvis", "start"],
            cwd=str(repo),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=90,
        )
    except Exception as exc:
        print(f"  (falha ao iniciar: {exc})")
        return False
    for _ in range(40):  # ~20 s
        if port_open(WEB_URL):
            print("  servidor pronto")
            return True
        time.sleep(0.5)
    print("  servidor não respondeu em 20 s — veja ~/.openjarvis/server.log")
    return False


def wake_up(repo: Path, args: argparse.Namespace) -> int:
    play_jingle(repo / "voice" / "assets" / "jingle.mp3")
    if ensure_server(repo):
        try:
            webbrowser.open(WEB_URL)
        except Exception:
            pass
    if args.no_launch:
        print("  --no-launch: não vou abrir o chat.")
        return 0

    cmd = ["uv", "run", "jarvis", "chat", "--voice"]
    if args.num_ctx:
        cmd += ["--num-ctx", str(args.num_ctx)]
    if args.num_gpu is not None:
        cmd += ["--num-gpu", str(args.num_gpu)]
    print(f"  iniciando: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd, cwd=str(repo))
    except KeyboardInterrupt:
        return 0


# ---------------------------------------------------------------------------
# Selftest: palmas e fala sintéticas, sem microfone
# ---------------------------------------------------------------------------


def _clap(sr: int, amp: float, rng: np.random.Generator, decay: float = 0.008) -> np.ndarray:
    n = int(0.05 * sr)
    env = np.exp(-np.arange(n) / (sr * decay))
    noise = rng.standard_normal(n)
    wave = noise * env
    return (amp * wave / np.max(np.abs(wave))).astype(np.float32)


def _noise(sr: int, seconds: float, amp: float, rng: np.random.Generator) -> np.ndarray:
    return (amp * rng.standard_normal(int(seconds * sr))).astype(np.float32)


def _speech(sr: int, seconds: float, amp: float, rng: np.random.Generator,
            burst: float, gap: float) -> np.ndarray:
    """Fala sintética: rajadas harmônicas (180 Hz + ruído) separadas por pausas."""
    n = int(seconds * sr)
    out = np.zeros(n, dtype=np.float32)
    t = 0.0
    while t < seconds:
        s = int(t * sr)
        e = min(n, s + int(burst * sr))
        if e > s:
            tt = np.arange(e - s) / sr
            out[s:e] = amp * (
                0.6 * np.sin(2 * np.pi * 180 * tt) + 0.4 * rng.standard_normal(e - s)
            )
        t += burst + gap
    return out


def _place(buf: np.ndarray, clap: np.ndarray, at_seconds: float, sr: int) -> None:
    start = int(at_seconds * sr)
    end = min(len(buf), start + len(clap))
    buf[start:end] += clap[: end - start]


def selftest() -> int:
    sr = SAMPLE_RATE
    rng = np.random.default_rng(7)
    failures: list[str] = []

    def run(name: str, signal: np.ndarray, expected: bool, thresh: float = DEFAULT_THRESH) -> None:
        fired, det = detect_in_array(signal, thresh=thresh)
        ok = fired == expected
        status = "ok  " if ok else "FALHA"
        print(
            f"  [{status}] {name}: esperado={expected} obtido={fired} "
            f"(palmas={len(det.onsets)}, rejeitados={det.rejected})"
        )
        if not ok:
            failures.append(name)

    # --- casos que DEVEM disparar -----------------------------------------
    s1 = _noise(sr, 2.0, 0.01, rng)
    _place(s1, _clap(sr, 0.8, rng), 0.60, sr)
    _place(s1, _clap(sr, 0.8, rng), 0.90, sr)
    run("duas palmas (0.8) a 300 ms, ruído 0.01", s1, True)

    s4 = _noise(sr, 2.0, 0.02, rng)
    _place(s4, _clap(sr, 0.35, rng), 0.50, sr)
    _place(s4, _clap(sr, 0.35, rng), 0.75, sr)
    run("duas palmas fracas (0.35), ruído 0.02", s4, True)

    s8 = _noise(sr, 2.5, 0.01, rng)
    for at in (0.50, 0.80, 1.10):
        _place(s8, _clap(sr, 0.8, rng), at, sr)
    run("três palmas (dispara no par)", s8, True)

    s9 = _noise(sr, 2.0, 0.02, rng)
    _place(s9, _clap(sr, 0.25, rng), 0.50, sr)
    _place(s9, _clap(sr, 0.25, rng), 0.80, sr)
    run("duas palmas no limiar (0.25), ruído 0.02", s9, True)

    s10 = _noise(sr, 2.0, 0.01, rng)
    _place(s10, _clap(sr, 0.8, rng), 0.50, sr)
    _place(s10, _clap(sr, 0.8, rng), 0.63, sr)
    run("duas palmas a 130 ms (borda inferior)", s10, True)

    s11 = _noise(sr, 2.5, 0.01, rng)
    _place(s11, _clap(sr, 0.8, rng), 0.50, sr)
    _place(s11, _clap(sr, 0.8, rng), 1.10, sr)
    run("duas palmas a 600 ms (borda superior)", s11, True)

    # --- casos que NÃO devem disparar -------------------------------------
    s2 = _noise(sr, 2.0, 0.01, rng)
    _place(s2, _clap(sr, 0.8, rng), 0.60, sr)
    run("uma palma só", s2, False)

    run("ruído contínuo em 0.30", _noise(sr, 2.0, 0.30, rng), False)

    s5 = _noise(sr, 2.5, 0.01, rng)
    _place(s5, _clap(sr, 0.8, rng), 0.50, sr)
    _place(s5, _clap(sr, 0.8, rng), 1.70, sr)
    run("duas palmas a 1.2 s (longe demais)", s5, False)

    s6 = _noise(sr, 2.0, 0.01, rng)
    _place(s6, _clap(sr, 0.8, rng), 0.60, sr)
    _place(s6, _clap(sr, 0.8, rng), 0.65, sr)
    run("duas palmas a 50 ms (coladas)", s6, False)

    run("silêncio absoluto", np.zeros(sr * 2, dtype=np.float32), False)

    # fala: rajadas longas (250 ms) — antes disparava, agora deve ser rejeitada
    run("fala sintética 250 ms/150 ms, amp 0.45",
        _speech(sr, 3.0, 0.45, rng, burst=0.25, gap=0.15), False)

    # vogais curtas: curtíssimas, mas graves (ZCR baixo) -> rejeitadas
    run("vogais curtas 80 ms, amp 0.50",
        _speech(sr, 3.0, 0.50, rng, burst=0.08, gap=0.20), False)

    if failures:
        print(f"\n{len(failures)} caso(s) falharam: {', '.join(failures)}")
        return 1
    print("\nTodos os casos passaram. O detector está são.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    """Lê um float de variável de ambiente, com aviso se estiver inválido."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"{name}={raw!r} não é número — usando {default}.")
        return default


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jarvis-show",
        description="Acorda o OpenJarvis com duas palmas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exemplos:\n"
            "  uv run python jarvis-show.py --selftest\n"
            "  uv run python jarvis-show.py --calibrate\n"
            "  uv run python jarvis-show.py --list-devices\n"
            "  uv run python jarvis-show.py            # ficar ouvindo\n"
        ),
    )
    p.add_argument("--thresh", type=float, default=None, help="pico mínimo de palma")
    p.add_argument("--max-ms", type=float, default=None,
                   help="duração máxima do transiente em ms (padrão 64)")
    p.add_argument("--min-zcr", type=float, default=None,
                   help="cruzamento por zero mínimo, 0-1 (padrão 0.20)")
    p.add_argument("--mic", default=None, help="índice/nome do microfone")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="para depois de N segundos ouvindo (0 = para sempre)")
    p.add_argument("--num-ctx", type=int, default=8192,
                   help="contexto do chat (padrão 8192; 0 = padrão do motor)")
    p.add_argument("--num-gpu", type=int, default=0,
                   help="camadas na GPU (-1=todas, 0=CPU; padrão 0)")
    p.add_argument("--no-launch", action="store_true",
                   help="detecta e não abre chat/navegador")
    p.add_argument("--selftest", action="store_true", help="testa com palmas sintéticas")
    p.add_argument("--calibrate", action="store_true", help="mede ruído e palmas reais")
    p.add_argument("--list-devices", action="store_true", help="lista dispositivos de áudio")
    p.add_argument("--wav", default=None, help="analisa um .wav em vez do microfone")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.selftest:
        return selftest()

    repo = Path(__file__).resolve().parent
    if not (repo / "pyproject.toml").is_file():
        print(
            "Aviso: não achei pyproject.toml nesta pasta.\n"
            f"  pasta atual: {repo}\n"
            "  copie jarvis-show.py para a raiz do checkout do OpenJarvis."
        )

    if args.list_devices:
        list_devices()
        return 0

    raw_thresh = args.thresh
    if raw_thresh is None:
        env = os.environ.get(ENV_THRESH, "").strip()
        try:
            raw_thresh = float(env) if env else DEFAULT_THRESH
        except ValueError:
            print(f"{ENV_THRESH}={env!r} não é número — usando {DEFAULT_THRESH}.")
            raw_thresh = DEFAULT_THRESH

    mic_spec = args.mic or os.environ.get(ENV_MIC, "").strip() or None
    device = resolve_device(mic_spec) if mic_spec else None

    if args.calibrate:
        calibrate(device)
        return 0

    max_ms = args.max_ms if args.max_ms is not None else _env_float(
        ENV_MAXMS, MAX_BURST_FRAMES * 1000.0 * FRAME / SAMPLE_RATE
    )
    min_zcr = args.min_zcr if args.min_zcr is not None else _env_float(
        ENV_MIN_ZCR, MIN_ZCR
    )
    max_frames = max(1, int(round(max_ms * SAMPLE_RATE / (1000.0 * FRAME))))

    det = DoubleClapDetector(
        thresh=raw_thresh, max_burst_frames=max_frames, min_zcr=min_zcr
    )
    origin = "padrão" if args.thresh is None else "--thresh"
    print(
        f"jarvis-show | limiar {det.thresh:.2f} ({origin}) | "
        f"mic {mic_spec or 'padrão'} | janela {det.min_gap:.2f}-{det.max_gap:.2f}s | "
        f"transiente <= {max_frames * FRAME * 1000 // SAMPLE_RATE} ms | zcr >= {min_zcr:.2f}"
    )

    if args.wav:
        try:
            import soundfile as sf  # noqa: PLC0415
        except Exception as exc:
            sys.exit(f"para --wav preciso do soundfile ({exc})")
        data, sr = sf.read(args.wav, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            idx = np.linspace(0, len(mono) - 1, int(len(mono) * SAMPLE_RATE / sr))
            mono = np.interp(idx, np.arange(len(mono)), mono).astype(np.float32)
        fired, d = detect_in_array(mono, detector=det)
        print(f"  palmas confirmadas em {[round(t, 2) for t in d.onsets]} s")
        print(f"  descartes: {d.rejected}")
        print("  resultado:", "DUAS PALMAS" if fired else "nada")
        if fired and not args.no_launch:
            return wake_up(repo, args)
        return 0 if fired else 2

    sd = _sd()
    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    print("Ouvindo… bata duas palmas. (Ctrl+C para sair)")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            device=device, blocksize=FRAME,
        ) as stream:
            while True:
                block, _ = stream.read(FRAME)
                if det.feed(block[:, 0]):
                    print("\n  ▸ duas palmas detectadas")
                    break
                if deadline is not None and time.monotonic() > deadline:
                    print("\n  tempo esgotado sem palmas")
                    return 2
    except KeyboardInterrupt:
        print("\n  cancelado")
        return 130

    return wake_up(repo, args)


if __name__ == "__main__":
    sys.exit(main())
