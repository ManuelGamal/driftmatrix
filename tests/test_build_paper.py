"""Page-limit and undefined-reference checks of the paper build script."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_paper.py"
spec = importlib.util.spec_from_file_location("build_paper", SCRIPT)
build_paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_paper)


def test_body_pages_reads_the_page_of_the_endofbody_label():
    aux = "\\newlabel{sec:intro}{{1}{1}{Introduction}{section.1}{}}\n" \
          "\\newlabel{endofbody}{{8}{6}{Conclusion}{section.8}{}}\n"
    assert build_paper.body_pages(aux) == 6


def test_body_pages_fails_when_the_label_is_missing():
    with pytest.raises(ValueError, match="endofbody"):
        build_paper.body_pages("\\newlabel{sec:intro}{{1}{1}{Introduction}{section.1}{}}\n")


def test_log_problems_finds_undefined_and_duplicate_references():
    log = ("LaTeX Warning: Citation `tracelab' on page 1 undefined on input line 12.\n"
           "LaTeX Warning: Reference `fig:grid' on page 4 undefined on input line 40.\n"
           "LaTeX Warning: Label `tab:h' multiply defined.\n"
           "Overfull \\hbox (2.0pt too wide) in paragraph at lines 3--4\n")
    assert len(build_paper.log_problems(log)) == 3


def test_log_problems_finds_the_summary_line_when_a_warning_wraps():
    log = ("LaTeX Warning: Citation `a-very-long-citation-key-that-makes-the-log-line-wrap' on page 1 undefi\n"
           "ned on input line 12.\n"
           "LaTeX Warning: There were undefined references.\n")
    assert build_paper.log_problems(log) == ["LaTeX Warning: There were undefined references."]


def test_log_problems_finds_the_multiply_defined_summary():
    assert len(build_paper.log_problems("LaTeX Warning: There were multiply-defined labels.\n")) == 1


AUX_FLOATS = (
    "\\@writefile{lot}{\\contentsline {table}{\\numberline {1}{\\ignorespaces Models.\\relax }}{2}{table.caption.9}\\protected@file@percent }\n"
    "\\@writefile{lof}{\\contentsline {figure}{\\numberline {1}{\\ignorespaces Turns {nested}.\\relax }}{3}{figure.caption.18}\\protected@file@percent }\n"
    "\\newlabel{endofbody}{{8}{6}{Conclusion}{section*.42}{}}\n"
    "\\bodyfloats{2}{1}\n"
    "\\@writefile{lof}{\\contentsline {figure}{\\numberline {2}{\\ignorespaces Deferred.\\relax }}{8}{figure.caption.25}\\protected@file@percent }\n"
    "\\@writefile{lot}{\\contentsline {table}{\\numberline {2}{\\ignorespaces Appendix table.\\relax }}{9}{table.caption.45}\\protected@file@percent }\n")


def test_body_float_pages_include_a_body_float_deferred_past_the_label():
    assert build_paper.body_float_pages(AUX_FLOATS) == [2, 3, 8]


def test_body_float_pages_fail_when_the_float_count_is_missing():
    with pytest.raises(ValueError, match="bodyfloats"):
        build_paper.body_float_pages("\\newlabel{endofbody}{{8}{6}{Conclusion}{section*.42}{}}\n")


def test_log_problems_is_empty_for_a_clean_log():
    assert build_paper.log_problems("Output written on main.pdf (6 pages).\n") == []
