# Primary-source research record

Date: 2026-10-28. Research state: OCP format contract verified from primary specification; BADA bibliographic identity verified, exact RS definition unavailable. No paper encoding inferred from its abstract.

| Source | Verified scope | Precise location | Access / limitation |
|---|---|---|---|
| [BADA author publication page](https://insuofficial.github.io/publication/bada/) | Title, authors, HPCA2025 identity, novel8-bit slice representation in abstract | Publication abstract and PDF link | Readable, but no encoding details |
| [BADA DOI](https://doi.org/10.1109/HPCA61900.2025.00067) / [publisher PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10946711&tp=) | Primary paper target | Exact RS section/page pending | IEEE stamp HTTP418; document page JavaScript challenge. No fulltext read |
| [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) | Block32; E8M0; MXINT8TC implicit2^-6; native FP4E2M1; shared-scale dot and conversion | §§5.1–5.4, Tables1/5/6/7, §§6.1–6.3; printed pp9–15 | Primary PDF read; conversion and arithmetic defaults frozen in SPEC |
| [Microsoft MX reference emulator](https://github.com/microsoft/microxcaling) | Candidate independently maintained format emulator | README, full commit not yet pinned | No library installed/executed; not an oracle result |

The author's PDF link resolves to IEEE article10946711. Searches included exact RS-INT8, paper title, DOI, author preprint/repository and local project filenames. No local manuscript was found. Author raw GitHub content guesses returned404/SSL failures; API was rate-limited. No contact message was sent to any author.

Before RS activation obtain a user-provided or publicly accessible primary manuscript. Record SHA256, exact edition, printed page, section/figure/equation, encoding bit/slice widths, sign and carry metadata, mapping for all source integers and four-bit reconstruction. Verify255 source values exhaustively against a separately implemented decoder. If the paper does not define contiguous high4 or attention truncation, explicitly separate paper facts from the proposed attention adapter and resolve its budget before coding. Never label an invented adapter as the paper's RS rule.

This is a research-access blocker, not evidence that BADA lacks an RS representation. The exact expansion of “RS” remains unasserted until the primary text is available.
