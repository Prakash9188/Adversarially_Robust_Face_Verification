# CVPR-format report (Overleaf)

Built on the CVPR 2027 author kit (`cvpr.sty`, `ieeenat_fullname.bst`, `preamble.tex` come from
https://github.com/cvpr-org/author-kit). Compiler: pdfLaTeX; bibliography: BibTeX.

## Compile on Overleaf
1. Overleaf -> New Project -> Upload Project -> upload `report_overleaf.zip` (or this `report/` folder zipped).
2. Menu -> Main document = `main.tex`, Compiler = pdfLaTeX. Recompile (run twice for references).
3. **Fill in two macros** near the top of `main.tex` after the final commit:
   `\hashthree` and `\timethree` (hash and timestamp of Commit 3; a commit cannot contain its own hash).
   Get them with `git log -1 --format="%h" --date=iso-strict --pretty=format:"%h %ad"` on `main`.
4. **Check the page count.** CVPR allows 8 pages of content plus references. We could not compile locally,
   so the length is unverified. If it is over, first uncomment `\usepackage{microtype}` in `preamble.tex`,
   then shorten Related Work / the ablation table.

## Regenerate numbers, tables and figures
Tables (`tables/*.tex`) and figures (`figures/*.pdf`) are generated from `experiments/results/*.csv`:

    python scripts/make_report_assets.py

Do not edit the generated files by hand.
