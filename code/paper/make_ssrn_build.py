# make_ssrn_build.py -- derive the SSRN reading copy from the journal-format source.
# Same text; single spacing, footnotes and floats inline, no endnote block. Writes
# preprint/paper_A_ssrn.tex and preprint/paper_A_ssrn_online_appendix.tex.
# Usage: python code/paper/make_ssrn_build.py   (run from the repository root; then build both PDFs in preprint/)
import re
src="submission/paper_A_jfec.tex"; s=open(src,encoding="utf-8").read()
reps=[("\\usepackage{endnotes}\n",""),("\\usepackage[nolists]{endfloat}\n",""),("\\setstretch{2}","\\setstretch{1.15}"),
      ("\\let\\footnote\\endnote\n",""),
      ("\\begingroup\n\\parindent 0pt\\parskip 1ex\n\\renewcommand{\\enotesize}{\\normalsize}\n\\theendnotes\n\\endgroup\n","")]
for a,b in reps:
    assert s.count(a)==1,a; s=s.replace(a,b)
s=s.replace("\\bibliography{refs_v3}","\\bibliography{refs_v3}",1)
open("preprint/paper_A_ssrn.tex","w",encoding="utf-8").write(s)
o=open("submission/paper_A_jfec_online_appendix.tex",encoding="utf-8").read()
open("preprint/paper_A_ssrn_online_appendix.tex","w",encoding="utf-8").write(o)
print("wrote preprint/paper_A_ssrn.tex and preprint/paper_A_ssrn_online_appendix.tex")
