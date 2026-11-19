"""Apply a narrow native-interface hook without vendoring external SAMI2 sources."""
from pathlib import Path
import sys
source, output = map(Path, sys.argv[1:3])
text = source.read_text()
needle = '      call vdrift_model(hrl,glon_in,param,vd)'
assert text.count(needle) == 1, 'Unsupported SAMI2 driver call layout'
assert 'call measured_drift' not in text, 'Already patched'
text = text.replace(needle, needle + '\n      call measured_drift(hrut,vd)')
text += '\n' + Path(__file__).with_name('sami2_measured_driver.f').read_text()
output.write_text(text)
