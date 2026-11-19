#!/usr/bin/env python3
"""Monta o PDF de entrega da Fase 3 (capa + documento de entrega + relatório).

    pip install markdown weasyprint
    python scripts/gerar_entrega_pdf.py

Saída: ENTREGA_TECH_CHALLENGE_FASE3.pdf na raiz do projeto.

O PDF é gerado a partir dos mesmos arquivos markdown versionados no repositório —
não há conteúdo digitado direto aqui. Assim o documento entregue e o repositório
não divergem: corrigir o relatório e rodar de novo basta.

O diagrama do fluxo (bloco mermaid) é renderizado como imagem pelo Chrome em modo
headless, já que o PDF não roda JavaScript. Sem Chrome, ele cai para bloco de
código — o PDF continua sendo gerado.
"""

from __future__ import annotations

import base64
import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown
from weasyprint import HTML

RAIZ = Path(__file__).resolve().parent.parent
ENTREGA = RAIZ / "docs" / "entrega-fase3.md"
RELATORIO = RAIZ / "RELATORIO_TECH_CHALLENGE_FASE3.md"
SAIDA = RAIZ / "ENTREGA_TECH_CHALLENGE_FASE3.pdf"
REPO = "https://github.com/otaviano/fiap-tech-challenge-fase3/blob/main/"

TITULO = "Assistente Médico Virtual"
SUBTITULO = "LLM fine-tunada com dados internos + LangChain + LangGraph"
CURSO = "FIAP Postech — IA para Devs"
FASE = "3 — Assistente médico com LLM customizada"
DATA = "Novembro de 2026"
INTEGRANTES = (
    ("Otaviano Montes Zibetti", "RM374369", "otavianomontes@gmail.com"),
    ("Felipe Squarizi", "RM374950", "fesqua@yahoo.com.br"),
)

VINHO = "#8f1a3f"
VINHO_ESCURO = "#5c0f28"
MERMAID_JS = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"

CSS = f"""
@page {{
    size: A4;
    margin: 2.2cm 2cm 2cm 2cm;
    @bottom-center {{
        content: "Tech Challenge — Fase 3 | {CURSO}";
        font-family: "DejaVu Serif", Georgia, serif;
        font-size: 8.5pt; color: #8a8a8a;
    }}
    @bottom-right {{
        content: "Pág. " counter(page) " / " counter(pages);
        font-family: "DejaVu Serif", Georgia, serif;
        font-size: 8.5pt; color: #8a8a8a;
    }}
}}
/* A capa não leva rodapé. */
@page capa {{ margin: 0; @bottom-center {{ content: none; }} @bottom-right {{ content: none; }} }}

body {{
    font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif;
    font-size: 10pt; line-height: 1.5; color: #24242a;
}}

.capa {{
    page: capa;
    background: linear-gradient(160deg, {VINHO_ESCURO} 0%, #2a0713 55%, #14040a 100%);
    color: #fff; height: 297mm; padding: 45mm 22mm 20mm 22mm;
    box-sizing: border-box; page-break-after: always;
}}
.capa h1 {{ font-size: 30pt; margin: 0 0 6mm 0; color: #fff; border: none; padding: 0; }}
.capa .regua {{ height: 2px; background: #e0577f; width: 78%; margin-bottom: 9mm; }}
.capa h2 {{ font-size: 15pt; color: #f4c3d3; margin: 0 0 2mm 0; border: none; padding: 0; }}
.capa .sub {{ font-size: 11pt; color: #d8a4b8; margin-bottom: 9mm; }}
.capa .curso {{ font-size: 11pt; color: #efe2e8; margin-bottom: 12mm; }}
.capa .rotulo {{ font-size: 9.5pt; color: #e0577f; font-weight: bold;
                 letter-spacing: .4px; margin-bottom: 2mm; }}
.capa .pessoa {{ font-size: 10pt; color: #f2e9ee; margin-bottom: 1.5mm; }}
.capa .meta {{ font-size: 10pt; color: #f2e9ee; margin-top: 9mm; }}
.capa .meta b {{ color: #e0577f; }}
.capa .meta a {{ color: #f4c3d3; }}

h1 {{ font-size: 17pt; color: {VINHO}; border-bottom: 2px solid #d9d0d4;
     padding-bottom: 2mm; margin-top: 10mm; page-break-after: avoid; }}
h2 {{ font-size: 13pt; color: {VINHO}; margin-top: 7mm; page-break-after: avoid; }}
h3 {{ font-size: 11pt; color: #3c3c44; margin-top: 5mm; page-break-after: avoid; }}
h1 + h2, h2 + h3 {{ margin-top: 3mm; }}

p {{ text-align: justify; margin: 0 0 2.6mm 0; }}
ul, ol {{ margin: 0 0 3mm 0; padding-left: 5mm; }}
li {{ margin-bottom: 1.2mm; }}

table {{ border-collapse: collapse; width: 100%; margin: 3mm 0 5mm 0;
         font-size: 8.6pt; page-break-inside: avoid; }}
th {{ background: {VINHO}; color: #fff; text-align: left;
      padding: 2mm 2.4mm; font-weight: bold; }}
td {{ padding: 1.8mm 2.4mm; border-bottom: 1px solid #e4dee1; vertical-align: top; }}
tr:nth-child(even) td {{ background: #faf6f7; }}

code {{ font-family: "DejaVu Sans Mono", monospace; font-size: 8.4pt;
        background: #f3eef0; padding: 0.4mm 1mm; border-radius: 2px; }}
pre {{ background: #f7f4f5; border-left: 3px solid {VINHO}; padding: 2.5mm 3mm;
       font-size: 8.2pt; line-height: 1.35; overflow-wrap: break-word;
       white-space: pre-wrap; page-break-inside: avoid; margin: 2mm 0 4mm 0; }}
pre code {{ background: none; padding: 0; }}
/* Dentro do cabeçalho vinho, código e negrito precisam continuar legíveis. */
th code {{ background: rgba(255, 255, 255, .18); color: #fff; }}
th strong {{ color: #fff; }}

.diagrama {{ text-align: center; margin: 3mm 0 5mm 0; page-break-inside: avoid; }}
.diagrama img {{ max-width: 100%; max-height: 175mm; }}

blockquote {{ border-left: 3px solid #d9a8bb; margin: 2mm 0 4mm 0;
              padding: 1mm 0 1mm 3mm; color: #55505a; font-style: italic; }}
hr {{ border: none; border-top: 1px solid #e0dade; margin: 6mm 0; }}
a {{ color: {VINHO}; text-decoration: none; word-break: break-all; }}
strong {{ color: #1d1d22; }}
"""


def capa_html() -> str:
    pessoas = "\n".join(
        f'<div class="pessoa">{nome} — {rm} — {email}</div>' for nome, rm, email in INTEGRANTES)
    return f"""
<div class="capa">
  <h1>TECH CHALLENGE — FASE 3</h1>
  <div class="regua"></div>
  <h2>{TITULO}</h2>
  <div class="sub">{SUBTITULO}</div>
  <div class="curso">{CURSO}</div>
  <div class="rotulo">INTEGRANTES DO GRUPO</div>
  {pessoas}
  <div class="meta"><b>Fase:</b> {FASE}</div>
  <div class="meta" style="margin-top:2mm"><b>Data de entrega:</b> {DATA}</div>
</div>
"""


def _chrome() -> str | None:
    for nome in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if caminho := shutil.which(nome):
            return caminho
    return None


def renderizar_mermaid(codigo: str, pasta: Path) -> str | None:
    """Renderiza o diagrama num PNG e devolve como data URI; None se não der."""
    chrome = _chrome()
    if chrome is None:
        return None
    pagina = pasta / "diagrama.html"
    png = pasta / "diagrama.png"
    pagina.write_text(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="{MERMAID_JS}"></script>
<style>body {{ margin: 0; background: #fff; }} #d {{ display: inline-block; padding: 12px; }}</style>
</head><body><div id="d" class="mermaid">{html.escape(codigo)}</div>
<script>
mermaid.initialize({{ startOnLoad: true, theme: "base", flowchart: {{ curve: "basis", rankSpacing: 34, nodeSpacing: 28, padding: 10 }},
  themeVariables: {{ fontFamily: "DejaVu Sans, Arial, sans-serif", fontSize: "17px",
    primaryColor: "#faf6f7", primaryBorderColor: "{VINHO}", primaryTextColor: "#24242a",
    lineColor: "#6b5a61", edgeLabelBackground: "#ffffff" }} }});
</script></body></html>""", encoding="utf-8")
    try:
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             "--force-device-scale-factor=2", "--window-size=1100,1300",
             "--virtual-time-budget=15000", f"--screenshot={png}", pagina.as_uri()],
            check=True, capture_output=True, timeout=90)
    except (subprocess.SubprocessError, OSError):
        return None
    if not png.exists():
        return None

    from PIL import Image, ImageChops  # dependência do weasyprint

    with Image.open(png) as imagem:
        imagem = imagem.convert("RGB")
        caixa = ImageChops.difference(imagem, Image.new("RGB", imagem.size, "white")).getbbox()
        if caixa is None:  # o mermaid não carregou (sem rede): só página em branco
            return None
        margem = 16
        caixa = (max(caixa[0] - margem, 0), max(caixa[1] - margem, 0),
                 min(caixa[2] + margem, imagem.width), min(caixa[3] + margem, imagem.height))
        imagem.crop(caixa).save(png)
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()


def preparar(texto: str, pasta: Path) -> str:
    """Ajustes de markdown que só fazem sentido no PDF."""
    def mermaid(m: re.Match) -> str:
        imagem = renderizar_mermaid(m.group(1), pasta)
        if imagem is None:
            print("aviso: diagrama mermaid mantido como código (sem Chrome ou sem rede)",
                  file=sys.stderr)
            return f"```\n{m.group(1)}```"
        return f'<div class="diagrama"><img src="{imagem}" alt="Fluxo LangGraph"></div>'

    texto = re.sub(r"```mermaid\n(.*?)```", mermaid, texto, flags=re.S)
    # Links relativos do repositório apontam para o GitHub no documento impresso.
    texto = re.sub(r"\]\((?!https?://|#)([^)]+)\)", rf"]({REPO}\1)", texto)
    return texto


def main() -> int:
    for caminho in (ENTREGA, RELATORIO):
        if not caminho.exists():
            print(f"arquivo não encontrado: {caminho}", file=sys.stderr)
            return 1

    conversor = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        corpo = conversor.convert(preparar(ENTREGA.read_text(encoding="utf-8"), pasta))
        conversor.reset()
        corpo += '\n<div style="page-break-before: always"></div>\n'
        corpo += conversor.convert(preparar(RELATORIO.read_text(encoding="utf-8"), pasta))

    documento = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Tech Challenge Fase 3 — {TITULO}</title>
<style>{CSS}</style></head>
<body>{capa_html()}{corpo}</body></html>"""

    HTML(string=documento, base_url=str(RAIZ)).write_pdf(SAIDA)
    print(f"PDF gerado: {SAIDA} ({SAIDA.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
