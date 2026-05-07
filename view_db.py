"""
view_db.py — Inspect what's stored in SQLite and ChromaDB.

Usage:
    python view_db.py                      # summary of everything
    python view_db.py --papers             # all extracted papers
    python view_db.py --sections           # section counts per paper
    python view_db.py --sections --paper Besen_2016  # sections for one paper
    python view_db.py --chunks             # chunk count in ChromaDB
    python view_db.py --search "lactate"   # semantic search
"""

import argparse
import config  # noqa: F401 — creates dirs, loads env
from rich.console import Console
from rich.table import Table
import pandas as pd
from sqlalchemy import create_engine, text

console = Console()


def _engine():
    return create_engine(f"sqlite:///{config.DB_PATH}", echo=False)


def _chroma_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection("chunks")


def show_summary():
    with _engine().connect() as conn:
        n_papers   = conn.execute(text("SELECT COUNT(*) FROM papers")).scalar()
        n_sections = conn.execute(text("SELECT COUNT(*) FROM sections")).scalar()
    col = _chroma_collection()
    n_chunks = col.count()

    console.print(f"\n[bold]Sepsis Atlas — DB Summary[/bold]")
    console.print(f"  Papers   (SQLite):  [green]{n_papers}[/green]")
    console.print(f"  Sections (SQLite):  [green]{n_sections}[/green]")
    console.print(f"  Chunks   (ChromaDB):[green]{n_chunks}[/green]\n")


def show_papers():
    df = pd.read_sql("SELECT paper_id, meta_title, meta_year, meta_journal, pop_sample_size, out_mortality, overall_confidence FROM papers", _engine())
    if df.empty:
        console.print("[yellow]No papers yet.[/yellow]")
        return
    t = Table(title=f"{len(df)} papers", show_lines=True)
    for col in ["paper_id", "meta_title", "meta_year", "pop_sample_size", "out_mortality", "overall_confidence"]:
        t.add_column(col, max_width=35, overflow="fold")
    for _, row in df.iterrows():
        t.add_row(*[str(row[c])[:60] if pd.notna(row[c]) else "[dim]-[/dim]" for c in ["paper_id", "meta_title", "meta_year", "pop_sample_size", "out_mortality", "overall_confidence"]])
    console.print(t)


def show_sections(paper_id=None):
    query = "SELECT paper_id, section_name, LENGTH(section_text) as chars FROM sections"
    params = {}
    if paper_id:
        query += " WHERE paper_id = :pid"
        params = {"pid": paper_id}
    df = pd.read_sql(query, _engine(), params=params)
    if df.empty:
        console.print("[yellow]No sections yet.[/yellow]")
        return
    t = Table(title=f"{len(df)} sections", show_lines=True)
    for col in ["paper_id", "section_name", "chars"]:
        t.add_column(col)
    for _, row in df.iterrows():
        t.add_row(row["paper_id"], row["section_name"], str(row["chars"]))
    console.print(t)


def show_chunks():
    col = _chroma_collection()
    count = col.count()
    console.print(f"\n[bold]ChromaDB chunks:[/bold] [green]{count}[/green]")
    if count > 0:
        sample = col.peek(5)
        t = Table(title="Sample chunks (first 5)", show_lines=True)
        t.add_column("chunk_id", width=25)
        t.add_column("paper_id", width=18)
        t.add_column("section", width=12)
        t.add_column("text", max_width=70, overflow="fold")
        for i, cid in enumerate(sample["ids"]):
            meta = sample["metadatas"][i]
            text = sample["documents"][i][:200]
            t.add_row(cid, meta.get("paper_id", ""), meta.get("section_name", ""), text)
        console.print(t)


def do_search(query, section=None):
    from utils.db import search_chunks
    results = search_chunks(query, n_results=10, section_name=section)
    t = Table(title=f'Search: "{query}"', show_lines=True)
    t.add_column("Score", width=6)
    t.add_column("Paper", width=18)
    t.add_column("Section", width=12)
    t.add_column("Text", max_width=80, overflow="fold")
    for r in results:
        t.add_row(str(r["score"]), r["paper_id"], r["section_name"], r["chunk_text"][:300])
    console.print(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers",   action="store_true")
    parser.add_argument("--sections", action="store_true")
    parser.add_argument("--paper",    type=str, help="Filter --sections to a paper_id")
    parser.add_argument("--chunks",   action="store_true")
    parser.add_argument("--search",   type=str)
    parser.add_argument("--section",  type=str, help="Filter --search by section name")
    args = parser.parse_args()

    if not any([args.papers, args.sections, args.chunks, args.search]):
        show_summary()
        return

    if args.papers:   show_papers()
    if args.sections: show_sections(args.paper)
    if args.chunks:   show_chunks()
    if args.search:   do_search(args.search, args.section)


if __name__ == "__main__":
    main()
