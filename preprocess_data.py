import sys

sys.path.insert(0, "./musicautobot")

from musicautobot.numpy_encode import *
from musicautobot.utils.file_processing import process_all, process_file
from musicautobot.utils.midifile import *
from musicautobot.config import *
from musicautobot.music_transformer import *

# Location of midi files
midi_path = Path("./data/midi/americana")

# Location of preprocesssed numpy files
data_path = Path("./data/numpy")
data_path.mkdir(parents=True, exist_ok=True)
data_save_name = "americana_data_save.pkl"

midi_files = get_files(midi_path, ".mid", recurse=True)

data = MusicDataBunch.from_files(
    midi_files,
    data_path,
    processors=[Midi2ItemProcessorVocab()],
    bs=16,
    bptt=512,
    encode_position=True,
)

data.save(data_save_name)
