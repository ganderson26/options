# Ultimate Joycean Orchestral Composition Tool
import numpy as np
import pronouncing
from collections import Counter
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pygame.midi
import threading
import time
from pathlib import Path

def load_text(path: Path) -> str:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return txt

# ----------------------
# Step 1: Load text
#text = """<INSERT FINNEGANS WAKE PASSAGE HERE>"""

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
# Step 3: Map pitches and instruments
np.random.seed(42)
phoneme_pitches = np.random.randint(36, 96, len(phonemes))
phoneme_durations = [0.5 if not any(v in p for v in 'AEIOU') else 1.0 for p in phonemes]

def instrument_type(p):
    p = p.upper()
    if any(v in p for v in 'AEIOU'): return 'strings'
    elif p.startswith(('S','SH','Z')): return 'woodwinds'
    elif p.startswith(('M','N')): return 'brass'
    else: return 'percussion'

instr_colors = {'strings':'blue','woodwinds':'cyan','brass':'green','percussion':'gray'}
instr_markers = {'strings':'circle','woodwinds':'triangle-up','brass':'square','percussion':'x'}
motifs = [p for p,count in Counter(phonemes).items() if count>2]
instrument_map = {'strings':0,'woodwinds':73,'brass':60,'percussion':118}

# ----------------------
# Step 4: Initialize MIDI
pygame.midi.init()
player = pygame.midi.Output(1)

# ----------------------
# Step 5: Build traces for animation
def create_traces(frame, chord_size=3, pulse_intensity=0.3):
    data = []
    trail_length = 5
    for i in range(len(phonemes)):
        cluster_indices = range(i, min(i+chord_size, len(phonemes)))
        # Compute cluster marker sizes
        cluster_sizes = []
        for idx in cluster_indices:
            size = 14 if phonemes[idx] in motifs else 8
            if phonemes[idx] in motifs:
                size *= 0.7 + pulse_intensity*np.sin(frame*0.2 + idx)
            cluster_sizes.append(size)
        # Motion trail
        trail_x = list(range(max(0,i-trail_length), i+1))
        trail_y = phoneme_pitches[max(0,i-trail_length):i+1]
        instr = instrument_type(phonemes[i])
        data.append(go.Scatter(
            x=trail_x,
            y=trail_y,
            mode='lines+markers',
            line=dict(color=instr_colors[instr], width=2, shape='hv'),
            marker=dict(size=cluster_sizes[0], color=instr_colors[instr], symbol=instr_markers[instr], line=dict(width=1)),
            hovertext=[f"Phoneme: {phonemes[j]}<br>Instrument: {instr}<br>Pitch: {phoneme_pitches[j]}<br>Motif: {'Yes' if phonemes[j] in motifs else 'No'}"
                       for j in range(max(0,i-trail_length), i+1)],
            hoverinfo='text',
            customdata=list(range(max(0,i-trail_length), i+1))
        ))
    return data

# Initial figure
initial_fig = go.Figure(data=create_traces(0))
initial_fig.update_layout(title='Ultimate Joycean Orchestral Composition',
                          xaxis_title='Time (phoneme index)',
                          yaxis_title='Pitch (MIDI)',
                          clickmode='event+select')

# ----------------------
# Step 6: Dash app layout
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("Ultimate Joycean Orchestral Composition Tool"),
    dcc.Graph(id='phoneme-graph', figure=initial_fig),
    html.Div([
        html.Label("Chord Size:"),
        dcc.Slider(id='chord-size', min=1, max=8, step=1, value=3, marks={i:str(i) for i in range(1,9)}),
        html.Label("Tempo Multiplier:"),
        dcc.Slider(id='tempo-mult', min=0.5, max=2.0, step=0.1, value=1.0),
        html.Label("Motif Pulse Intensity:"),
        dcc.Slider(id='pulse-intensity', min=0.1, max=1.0, step=0.05, value=0.3)
    ], style={'width':'60%'}),
    dcc.Interval(id='interval', interval=200, n_intervals=0),
    html.Div(id='clicked-phoneme')
])

# ----------------------
# Step 7: Animation callback
@app.callback(
    Output('phoneme-graph', 'figure'),
    Input('interval', 'n_intervals'),
    State('chord-size','value'),
    State('pulse-intensity','value')
)
def update_graph(frame, chord_size, pulse_intensity):
    fig = go.Figure(data=create_traces(frame, chord_size, pulse_intensity))
    fig.update_layout(title='Ultimate Joycean Orchestral Composition',
                      xaxis_title='Time (phoneme index)',
                      yaxis_title='Pitch (MIDI)',
                      clickmode='event+select')
    return fig

# ----------------------
# Step 8: Click callback to play phoneme/chords
@app.callback(
    Output('clicked-phoneme','children'),
    Input('phoneme-graph','clickData'),
    State('chord-size','value'),
    State('tempo-mult','value')
)
def play_phoneme(clickData, chord_size, tempo_mult):
    if clickData is None: return "Click a phoneme to play it."
    ind = clickData['points'][0]['customdata']
    cluster_indices = range(ind, min(ind+chord_size, len(phonemes)))
    for idx in cluster_indices:
        phon = phonemes[idx]
        pitch = phoneme_pitches[idx]
        instr_code = instrument_map[instrument_type(phon)]
        player.set_instrument(instr_code)
        threading.Thread(target=play_note, args=(pitch, tempo_mult)).start()
    return f"Played phoneme cluster starting at index {ind}"

def play_note(pitch, tempo_mult=1.0):
    player.note_on(pitch,100)
    time.sleep(0.3/tempo_mult)
    player.note_off(pitch,100)

# ----------------------
# Step 9: Run app
if __name__ == '__main__':
    app.run_server(debug=True)
