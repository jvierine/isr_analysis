"""Package the native-raster forward comparisons, with explicit interpretation."""
from pathlib import Path
import sys
import fitz,h5py,numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
root=Path(sys.argv[1]);out=root/'figures';cover=out/'interpretation.pdf'
style=ParagraphStyle('body',fontName='Times-Roman',fontSize=13,leading=17)
c=canvas.Canvas(str(cover),pagesize=(330*mm,210*mm));c.setTitle('EPB depletion: observed and predicted range-time measurements')
c.setFont('Times-Bold',20);c.drawString(20*mm,190*mm,'EPB depletion: observed and predicted measurements')
c.setFont('Times-Roman',13);c.drawString(20*mm,180*mm,'SYISR, 1 March 2026, 11:45-13:45 UTC (19:45-21:45 Beijing)')
paragraphs=[
'Each beam has three panels: measured signed echo power divided by its pre-event quiet reference; a depletion-focused fit; and a prediction with that entire beam withheld. The common color scale emphasizes relative depletion. Values below zero and above 1.2 saturate at the scale endpoints; the HDF5 observations retain their original values. White denotes unavailable reference or insufficient reconstruction support, not low power.',
'The forward model advects one field-aligned deficit map eastward at 120 m/s and averages it over nine angular beam quadrature points. It uses a 10 km cross-field grid. This is a relative echo model, not an absolute electron-density estimate. It assumes frozen morphology and field-line symmetry; neither assumption is established by fitting alone.',
'The model ratio is bounded between zero and one. A one-sided Huber loss limits the influence of positive excess echo beyond a residual of 0.2 in ratio units, while retaining the usual quadratic penalty when the model predicts too much echo. Thus enhanced echoes cannot dominate as they would under squared error. Contamination can still hide a depletion or affect the quiet reference. The asymmetric loss itself can bias depletion depth.',
'Beam 9 is excluded from every fit and is predicted using the other ten beams; its middle and right panels therefore coincide. For each other beam, the right panel is a separate fit excluding both that beam and beam 9. These are conditional holdout checks: drift and the spatial penalty were selected earlier from this event. They are not independent evidence of a unique three-dimensional density cavity.',
'The comparisons use the actual inversion samples: 30 s in time and approximately 6.0 km between retained range gates. No displayed cell is discarded by raster downsampling. The quiet-reference screen restricts the supported range interval to roughly 240-680 km, depending on beam; the full high-resolution RTIs cover a larger domain. Slant range is shown here and is not interpreted as altitude.',
'In the zenith beam, the withheld prediction retains much of the observed branching depletion. Across all beams, the conditional holdout RMS discrepancy within observed ratios below 0.7 is 0.341. These low-echo measurements are not a truth mask. Changing the Huber threshold from 0.1 to 0.4 changes supported fitted ratios by RMS 0.080, demonstrating material uncertainty in depletion depth. Extending optimization changed the main predictions by at most 0.00024.',
]
y=170*mm
for txt in paragraphs:
 p=Paragraph(txt,style);w,h=p.wrap(290*mm,160*mm);p.drawOn(c,20*mm,y-h);y-=h+5*mm
assert y>12*mm,y
c.save()
doc=fitz.open();doc.insert_pdf(fitz.open(cover))
# Geographic neighborhood order, with meridional beams grouped centrally.
for b in [8,9,1,10,7,11,2,5,6,4,3]:doc.insert_pdf(fitz.open(out/f'beam_{b:02}.pdf'))
doc.save(root/'depletion-predicted-rtis.pdf',deflate=True)
# Contact sheet is for QA only; delivered figures remain at their original raster resolution.
final=fitz.open(root/'depletion-predicted-rtis.pdf')
for i,p in enumerate(final):p.get_pixmap(matrix=fitz.Matrix(1,1)).save(out/f'qa_page_{i+1:02}.png')
with h5py.File(root/'rti_predictions.h5') as h,h5py.File(root/'rti_predictions_2000.h5') as old:
 d=np.nanmax(abs(h['depletion_ratio'][()]-old['depletion_ratio'][()]));assert d<.001
 for k in ['depletion_ratio','heldout_ratio']:
  x=h[k][()];assert np.nanmin(x)>=-1e-10 and np.nanmax(x)<=1+1e-10
print('PACKAGED',len(final),'pages; stability verified')
