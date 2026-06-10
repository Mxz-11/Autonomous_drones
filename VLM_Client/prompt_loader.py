import os
import sys

from mission_config import DEFAULT_DIRECT_PROMPT


def load_prompt(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Fichero de prompt no encontrado: {path}\n"
            f"Créalo o pasa otra ruta con --prompt."
        )
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:].lstrip("\n")

    return text.strip()


def parse_prompt_arg(argv: list[str] | None = None) -> str:
    args = argv if argv is not None else sys.argv[1:]
    path = DEFAULT_DIRECT_PROMPT
    for i, arg in enumerate(args):
        if arg == "--prompt" and i + 1 < len(args):
            path = args[i + 1]
            break

        if arg.startswith("--prompt="):
            path = arg.split("=", 1)[1]
            break
        
    return path
