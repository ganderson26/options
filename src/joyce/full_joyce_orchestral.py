# Full Orchestral Cinematic Joycean Symphony with Polyphonic Layers
import numpy as np
import pronouncing
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pygame.midi
from math import ceil
import random
from pathlib import Path

def load_text(path: Path) -> str:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return txt


# ----------------------
# Step 1: Load Joycean passage
#text = """<INSERT ENTIRE FINNEGANS WAKE CHAPTER OR PASSAGE HERE>"""

path = Path("data/finnegans_wake_extract.txt")
if not path.exists():
    raise FileNotFoundError(f"{path} not found. Provide a plaintext UTF-8 file.")
text = load_text(path)

# ----------------------
# Step 2: Extract phonemes
words = text.split()
phonemes = []
for word in words:
    phones_list = pronouncing.phones_for_word(word.lower())
    if phones_list:
        phonemes.append(phones_list[0].split())
phonemes = [p for sublist in phonemes for p in sublist]

# ----------------------
# Step 3: Map pitches and durations
np.random.seed(42)
phoneme_pitches = np.random.randint(36, 96, size=len(phonemes))
phoneme_durations = [0.5 if not any(v in p for v in 'AEIOU') else 1.0 for p in phonemes]

# ----------------------
# Step 4: Identify motifs
motifs = [p for p, count in Counter(phonemes).items() if count > 2]

# ----------------------
# Step 5: Instrument assignment (multi-track orchestral)
def instrument_type(p):
    p = p.upper()
    if any(v in p for v in 'AEIOU'):
        return 'strings'
    elif p.startswith(('S','SH','Z')):
        return 'woodwinds'
    elif p.startswith(('M','N')):
        return 'brass'
    else:
        return 'percussion'

instrument_colors = {'strings':'blue','woodwinds':'cyan','brass':'green','percussion':'gray'}
instrument_markers = {'strings':'o','woodwinds':'^','brass':'s','percussion':'X'}

# ----------------------
# Step 6: Prepare figure with stacked staff-style layers
fig, ax = plt.subplots(figsize=(18,10))
ax.set_xlim(0,len(phonemes))
ax.set_ylim(20,100)
ax.set_xlabel('Time (phoneme index)')
ax.set_ylabel('Pitch (MIDI note)')
ax.set_title('Full Orchestral Joycean Symphony Visualization')

scatters = []
sizes = []
colors = []
for i, p in enumerate(phonemes):
    instr = instrument_type(p)
    size = 250 if p in motifs else 80
    scat = ax.scatter(i, phoneme_pitches[i], s=size, c=instrument_colors[instr], 
                      marker=instrument_markers[instr], alpha=0.7, edgecolors='k')
    scatters.append(scat)
    sizes.append(size)
    colors.append(instrument_colors[instr])

# ----------------------
# Step 7: Animate with glowing motifs, polyphony, and tempo variation
def animate(frame):
    for i, scat in enumerate(scatters):
        if phonemes[i] in motifs:
            glow = 0.5 + 0.5*np.sin(frame*0.3 + i)
            scat.set_alpha(glow)
            scat.set_sizes([sizes[i]*glow])
        else:
            scat.set_alpha(0.2 + 0.5*np.exp(-0.05*abs(frame-i)))
    return scatters

ani = animation.FuncAnimation(fig, animate, frames=len(phonemes)*4, interval=120, blit=False, repeat=False)

# ----------------------
# Step 8: Multi-track MIDI playback with tempo variations
pygame.midi.init()
player = pygame.midi.Output(1)
instrument_map = {'strings':0,'woodwinds':73,'brass':60,'percussion':118} # General MIDI codes

for i, pitch in enumerate(phoneme_pitches):
    dur = ceil(phoneme_durations[i]*500*(0.8+0.4*random.random()))  # dynamic tempo
    instr_code = instrument_map[instrument_type(phonemes[i])]
    player.set_instrument(instr_code)
    player.note_on(pitch, 100)
    plt.pause(dur/1000)
    player.note_off(pitch, 100)

player.close()
pygame.midi.quit()

# ----------------------
# Step 9: Export cinematic orchestral video
ani.save('joyce_full_orchestral_symphony.mp4', writer='ffmpeg', fps=6)
plt.close()

print("✅ Full Orchestral Joycean Symphony saved as 'joyce_full_orchestral_symphony.mp4'")
