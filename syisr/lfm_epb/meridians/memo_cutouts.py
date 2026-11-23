"""UiT/HPLA memo source and scientific tables for the three cutouts."""
from pathlib import Path
import sys,shutil
import h5py,numpy as np
from datetime import datetime,timezone
root=Path(sys.argv[1]);repo=Path(sys.argv[2]);dest=repo/'documents/figures/memo083';dest.mkdir(parents=True,exist_ok=True)
for p in (root/'figures').glob('*.pdf'):shutil.copy2(p,dest/p.name)
h=h5py.File(root/'cutouts_metadata.h5')
s=r'''\documentclass[11pt]{article}
\makeatletter\def\input@path{{./}{../}{documents/}}\makeatother
\usepackage{uitmemo}\usepackage{float}\usepackage{pdflscape}\usepackage{booktabs}
\hidescriptprovenance
\graphicspath{{./}{../}{documents/}}
\renewcommand{\uitseries}{HIGH-POWER LARGE-APERTURE RADAR MEMO SERIES}
\renewcommand{\uitprefix}{HPLA}
\setlength{\emergencystretch}{2em}
% Provenance: isr_analysis/syisr/lfm_epb/meridians/cutouts.py and memo_cutouts.py;
% lfm_cutouts Rustplotlib renderer. Slurm6218/6219, 3 MPI ranks.
% HDF5: /gpfs/pub/data_other/juha/lfm_epb/20260301/multibeam_expanded/meridian_cutouts/.
% Native signed RTIs in cutouts.h5; current fit/marker metadata in cutouts_metadata.h5.
\begin{document}
\uitmemo{83}{22 November 2026}{SYISR and HPLA radar teams}{Juha-Pekka Vierinen}
{Three meridian cutouts of an EPB depletion and their field-line crossings in SYISR RTIs}
\makeuitmemo
\textbf{Summary.} We infer three perpendicular-to-$\mathbf B$ depletion maps
independently from western, central and eastern SYISR beam groups during
1 March 2026. The maps are viewed along $+\mathbf B$, into the field-aligned
structure. Five locations in each inferred depletion are mapped onto every
RTI in that meridian. The marker positions follow the dated magnetic geometry
and a common 120 m\,s$^{-1}$ eastward pattern drift; they are not shifted to match
individual echo features. This separates a geometric consistency check from
a fit that can adjust each beam independently.

\section*{1. What the three maps represent}
The western group comprises beams 8, 7 and 6; the central group comprises
9, 10, 11, 5 and 4; the eastern group comprises 1, 2 and 3. Beam 9 is shown
and receives geometric markers but is excluded from fitting because of
suspected coherent contamination. Each map uses only its group's measurements.
The groups sample the structure at different times, so their differences can
reflect evolution, imperfect drift registration, noise or contamination.
They are not simultaneous, independently resolved three-dimensional volumes.

The fits use the finite nine-point beam response and IGRF mapping from HPLA
Memo 80, a 10 km perpendicular grid, and the same drift and spatial penalty
(100) for all three groups. A one-sided Huber penalty with threshold 0.2 in
relative-echo units limits the influence of enhanced echoes. The modeled
ratio lies between zero and one. A fitted zero is the model's lower bound,
not a measurement of zero electron density. HPLA Memo 81 treats apparent
density as a separate contamination diagnostic.

The reference plane is normal to the IGRF field at 400 km above the radar.
Its coordinates $u$ and $v$ point approximately eastward and upward within
that plane; $v$ is not geographic altitude. The three plots share equal
horizontal and vertical physical scales. White regions lack sufficient
measurement support. The cross sections assume field alignment within each
group and infer neither a parallel length nor a unique absolute density.

\section*{2. How a marker reaches an RTI}
A marker $(u_0,v_0)$ labels a co-moving field-aligned plasma element at
$t_0=12{:}45$ UTC. Trace every point on beam $b$ along IGRF to the reference
plane, giving $[u_b(r),v_b(r)]$. Its predicted crossing solves
\begin{equation}
v_b(r_b)=v_0,\qquad
 t_b=t_0+\frac{u_b(r_b)-u_0}{0.120\ {\rm km\,s}^{-1}}.
\end{equation}
The coordinate $u$ therefore changes in the geographic frame as the plasma
drifts. The markers follow the same co-moving label across beams, not one
fixed geographic field line at all times. We retain the corresponding slant
range and geodetic altitude separately. Numerical forward/inverse closure is
better than $2\times10^{-8}$ km; this is an interpolation check, not physical
position accuracy.

'''
# separate independent cutout, native highresolution RTI panels for each meridian
for name,prefix,desc in [('west','W','8, 7 and 6'),('central','C','9, 10, 11, 5 and 4'),('east','E','1, 2 and 3')]:
 g=h[name];uv=g['markers_uv_km'][()];q=g['markers_fitted_ratio'][()]
 s+='\\clearpage\n\\section*{'+name.capitalize()+r'ern meridian: independent cutout}'+'\n' if name!='central' else '\\clearpage\n\\section*{Central meridian: independent cutout}\n'
 s+=r'\begin{figure}[H]\centering'+'\n'+r'\includegraphics[width=0.88\textwidth]{figures/memo083/'+name+r'_cutout.pdf}'+'\n'+r'\caption{'+name.capitalize()+r' group, beams '+desc+r'. View along $+\mathbf B$ into the inferred depletion. Dark colors indicate lower relative echo. The five labeled positions are projected onto the RTIs on the following page'+('s' if name=='central' else '')+r'. These are independent marker sets: W, C and E do not assert identification of the same plasma element between meridians.}'+'\n'+r'\end{figure}'+'\n'
 s+=r'\begin{center}\begin{tabular}{lrrr}\toprule Marker & $u_0$ (km) & $v_0$ (km) & Fitted relative echo\\\midrule'+'\n'
 for j,(xy,ratio) in enumerate(zip(uv,q),1):s+=f'{prefix}{j} & {xy[0]:.0f} & {xy[1]:.0f} & {ratio:.3f}'+r'\\'+'\n'
 s+=r'\bottomrule\end{tabular}\end{center}'+'\n'
 s+=r'Markers are spatially separated points in the largest connected supported region with fitted ratio below 0.6, restricted to the raised depletion ($-100\leq u\leq130$ km, $-70\leq v\leq180$ km). Selection uses the inferred map and geometric coverage; no marker is moved to improve its RTI alignment.'+'\n'
 for part in range(1,3 if name=='central' else 2):
  s+=r'\clearpage\begin{landscape}\begin{figure}[p]\centering'+'\n'+r'\includegraphics[width=\linewidth]{figures/memo083/'+name+f'_rtis_{part}_range.pdf'+'}\n'+r'\caption{'+name.capitalize()+r' meridian: circles mark the predicted time and slant range of the five co-moving field-aligned elements. Marker identifiers are retained in the cutout and crossing tables; RTI text labels are omitted to expose the echo structure. Small white bars show weighted RMS spreads from the nine finite-beam angular rays, not confidence intervals. Native 5 s and approximately 0.375 km range sampling are retained without raster undersampling. Signed nonpositive power maps to the lowest color, not white; stored signals remain signed. The shared storm scale shows range-squared signal, not electron density.}'+'\n'+r'\end{figure}\end{landscape}'+'\n'
s+=r'''\clearpage
\section*{3. What the overlays can test}
A field-aligned structure should place corresponding depletion features near
the corresponding marker circles across its meridian after the geometric and drift
mapping. Systematic displacement, missing branches or contradictory depths
can expose limitations of frozen drift, field alignment or the scattering
observable. A bright echo at a predicted depletion location can also be
contamination hiding the cavity; it does not by itself establish a density
increase or along-field structure.

The overlays are diagnostics of a fitted hypothesis. The points were selected
from the fitted maps, and the same group supplied those maps. Their alignment
is therefore not an independent validation. Whole-beam withholding, drift
sensitivity, support masks and comparison with an evolving or along-field
model remain necessary. This memo provides three independent cuts and explicit
mappings; it does not establish that the three structures morph uniquely into
one another. The first linear morph tested in Memo 80 did not improve the
withheld central-beam prediction.

The central marker C3 illustrates why matching enhanced power is not the
morphology target. Its predicted crossings occur near 12:32:26--12:32:40 UTC,
at slant ranges from 372 to 633 km across the five central beams. The local
normalized echo ratios are approximately 429 in beam 9, but 0.10, 0.16,
0.16 and 0.33 in beams 10, 11, 5 and 4. This is a strong inconsistency in
echo amplitude along the mapped structure. It is compatible with a depletion
being obscured in beam 9, but is not by itself a spectral identification of
coherent scatter. The comparison remains dependent on the quiet references.

\section*{4. Crossing coordinates}
The following tables give UTC, slant range $r$, geodetic altitude $z$, and
finite-beam RMS spreads $\sigma_t$ and $\sigma_r$. These spreads describe the
assumed Gaussian main beam only; they do not include drift, magnetic-model,
range-calibration or morphology uncertainty. The final column is the median
observed echo/quiet-reference ratio within 60 s and 10 km of the crossing,
when that normalized observable is available. It is an observational check,
not an absolute density. A dash denotes unavailable normalization.
'''
for name,prefix in [('west','W'),('central','C'),('east','E')]:
 s+='\\clearpage\n\\section*{'+name.capitalize()+' meridian: marker crossings}\n'
 s+=r'\begin{center}\small\begin{tabular}{llcrrrrr}\toprule Marker & Beam & UTC & $r$ (km) & $z$ (km) & $\sigma_t$ (s) & $\sigma_r$ (km) & Echo ratio\\\midrule'+'\n'
 for row in h[name]['marker_crossings'][()]:
  b,j,t,r,z,err,st,sr,sz,q=row;time=datetime.fromtimestamp(t,tz=timezone.utc).strftime('%H:%M:%S');qt=f'{q:.2f}' if np.isfinite(q) else '--'
  s+=f'{prefix}{int(j)} & {int(b)} & {time} & {r:.1f} & {z:.1f} & {st:.1f} & {sr:.1f} & {qt}'+r'\\'+'\n'
 s+=r'\bottomrule\end{tabular}\end{center}'+'\n'
s+=r'\end{document}'+'\n'
(repo/'documents/memos/memo-083.tex').write_text(s)
print('MEMO83_SOURCE_READY')
