"""Compose native-resolution beam atlas and vector geographic reference.
Use base conda Python. Input figures retain their original raster sampling;
PDF placement never resamples the embedded data images.
"""
import argparse
from pathlib import Path
import pymupdf as fitz
p=argparse.ArgumentParser();p.add_argument('figures',type=Path);a=p.parse_args()
atlas=fitz.open(a.figures/'all_beams.pdf');geom=fitz.open(a.figures/'geometry_top.pdf')
r=atlas[0].rect;g=geom[0].rect
# Keep the original atlas size and place a sizeable overhead reference beside it.
gap=16.;gw=r.width*.62;gh=gw*g.height/g.width
out=fitz.open();page=out.new_page(width=r.width+gap+gw,height=r.height)
page.show_pdf_page(r,atlas,0)
page.show_pdf_page(fitz.Rect(r.width+gap,(r.height-gh)/2,r.width+gap+gw,(r.height+gh)/2),geom,0)
path=a.figures/'all_beams_with_reference.pdf';out.save(path,deflate=True)
# Preview is for screen review; the PDF contains the full native pixels.
page.get_pixmap(matrix=fitz.Matrix(1.1,1.1),alpha=False).save(a.figures/'all_beams_with_reference_preview.png')
print(path)
