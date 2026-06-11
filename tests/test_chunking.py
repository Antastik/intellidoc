from app.rag.ingest import chunk_pages, split_text


def test_split_short_text_single_chunk():
    assert split_text("hello world", 900, 150) == ["hello world"]


def test_split_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # size + a little slack for boundaries


def test_chunk_pages_tracks_page_numbers():
    pages = ["First page text about cats.", "Second page text about dogs."]
    chunks = chunk_pages("animals.pdf", pages, 900, 150)
    assert [c.page for c in chunks] == [1, 2]
    assert chunks[0].source == "animals.pdf"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    # all chunks share the document id
    assert len({c.doc_id for c in chunks}) == 1
