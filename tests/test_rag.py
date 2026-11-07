from pathlib import Path
import pytest
from rag.answer import answer, validate_citations
from rag.index import Index, chunk_markdown

DOCS = Path(__file__).resolve().parent.parent / "docs"


@pytest.fixture(scope="module")
def idx():
    i = Index(); i.build(DOCS); return i


def test_chunks_carry_section_sources():
    srcs = [s for s, _ in chunk_markdown(DOCS / "errors.md")]
    assert "errors.md#E12 – High pressure" in srcs and len(srcs) == 3


def test_answer_cites_correct_source(idx):
    a = answer(idx, "What does error E31 mean?")
    assert a.grounded and a.sources == ["errors.md#E31 – Outdoor sensor"] and "[1]" in a.text


def test_unanswerable_question_is_refused(idx):
    assert not answer(idx, "Who won the football world cup in 2014?").grounded


def test_llm_with_invented_citation_is_rejected(idx):
    a = answer(idx, "Which circuit breaker is required?", llm=lambda p: "Use 32 A [7].")
    assert not a.grounded


def test_llm_refusal_is_respected(idx):
    assert answer(idx, "What is the warranty?", llm=lambda p: "I don't know.").text == "I don't know."


def test_citation_validator():
    assert validate_citations("A [1] and B [2]", 3)
    assert not validate_citations("No citation", 3)
