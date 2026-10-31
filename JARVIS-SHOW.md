# jarvis-show — acordar o Jarvis com duas palmas

Dois arquivos, ambos copiados para a **raiz do checkout do OpenJarvis**
(`C:\Users\<você>\jarvis`):

| Arquivo | Para que serve |
|---|---|
| `jarvis-show.py` | detector de palmas + sequência de acordar |
| `jarvis-show.cmd` | atalho para rodar no Windows sem digitar `uv run python` |

## Uso

```powershell
cd C:\Users\<você>\jarvis
.\jarvis-show.cmd
```

Fica ouvindo. Ao ouvir **duas palmas** em 0,12–0,70 s uma da outra:

1. toca `voice/assets/jingle.mp3`, se existir;
2. sobe o servidor web (`uv run jarvis start`) se a porta 8000 estiver fechada;
3. abre `http://127.0.0.1:8000` no navegador;
4. inicia `uv run jarvis chat --voice --num-ctx 8192 --num-gpu 0`.

Ctrl+C encerra.

## Verificações rápidas

```powershell
.\jarvis-show.cmd --selftest      # 13 casos sintéticos, sem microfone
.\jarvis-show.cmd --list-devices  # nomes e índices de entrada de áudio
.\jarvis-show.cmd --calibrate     # mede ruído x palma e sugere o limiar
.\jarvis-show.cmd --wav payload.wav --no-launch   # analisa um arquivo gravado
```

## Ajustes (variáveis de ambiente)

| Variável | Padrão | Quando mexer |
|---|---|---|
| `JARVIS_CLAP_THRESH` | `0.20` | palmas não disparam → desça para 0.12–0.15; dispara com qualquer barulho → suba para 0.30–0.40 |
| `JARVIS_MIC` | padrão do sistema | microfone errado; use `--list-devices` para achar índice ou nome |
| `JARVIS_CLAP_MAXMS` | `64` | palmas com eco/sala grande rejeitadas como "longo" → suba para 90–120 |
| `JARVIS_CLAP_MIN_ZCR` | `0.20` | palmas abafadas rejeitadas como "voz" → desça para 0.12–0.15 |

```powershell
setx JARVIS_CLAP_THRESH "0.15"       # vale em terminal novo
setx JARVIS_MIC "Microfone"
```

## Como o detector decide

Um transiente só conta como palma se tiver, ao mesmo tempo:

1. **ataque instantâneo** — pico ≥ limiar com o quadro anterior abaixo de metade dele (16 ms);
2. **timbre largo** — taxa de cruzamento por zero ≥ 0.20 (palma ≈ 0.47; vogal ≈ 0.01–0.10);
3. **curto** — cai abaixo da metade do pico em ≤ 64 ms (fala dura 100–300 ms).

Duas palmas válidas, separadas por 0,12–0,70 s, com silêncio entre elas, acordam o Jarvis.
