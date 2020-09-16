import os
import time
import numpy as np

import magenta.music as mm
from magenta.models.score2perf import score2perf
from magenta.music.protobuf import music_pb2
import tensorflow.compat.v1 as tf 
from tensor2tensor.utils import decoding, trainer_lib
from tensor2tensor.data_generators import text_encoder

# Global variables. Add flag instead?
NUM_HIDDEN_LAYERS = 16
SAMPLING_METHOD = "random"
ALPHA = 0.0 # Alpha for decoder
BEAM_SIZE = 1 # Beam size for inference
MODEL_NAME = "transformer"

LOGGER = tf.logging

flags = tf.flags
FLAGS = flags.FLAGS

# Specify output directory
flags.DEFINE_string(
    "output_dir", None,
    "Midi output directory."
)

# Specify path to model
flags.DEFINE_string(
    "model_path", None,
    "Pre-trained model path."
)

# Specify path to primer
flags.DEFINE_string(
    "primer_path", None,
    "MIDI file path for priming."
    "Model can generate sample w/o priming."
)

# Specify decoder length
flags.DEFINE_integer(
    'decode_length', 1024,
    'Length of decode result.'
)

class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
    @property
    def add_eos_symbol(self):
        return True

def unconditional_input_generator(targets, decode_length):
    """Estimator input function for unconditional Transformer."""
    while True:
        yield {
            'targets': np.array([targets], dtype=np.int32),
            'decode_length': np.array(decode_length, dtype=np.int32)
        }

def decode(ids, encoder):
    """Decode a list of IDs."""
    ids = list(ids)
    if text_encoder.EOS_ID in ids:
        ids = ids[:ids.index(text_encoder.EOS_ID)]
    return encoder.decode(ids)

def get_primer_ns(filename):
    """
    Convert MIDI file to note sequences for priming.
    :param filename: MIDI file name.
    :return:
        Note sequences for priming.
    """
    primer_note_sequence = mm.midi_file_to_note_sequence(filename)

    # Handle sustain pedal in primer.
    primer_note_sequence = mm.apply_sustain_control_changes(
        primer_note_sequence)

    # Set primer instrument and program.
    for note in primer_note_sequence.notes:
        note.instrument = 1
        note.program = 0

    return primer_note_sequence

def generate(estimator, unconditional_encoders, decode_length, targets, primer_note_sequence):
    """
    Generate unconditioned music samples from estimator
    :param estimator: Transformer estimator
    :param unconditional_encoders: A dictionary contains key and its encoder.
    :param decode_length: A number represents the duration of music snippet.
    :param targets: Target input for Transformer.
    :param primer_note_sequence: Notesequence represents the primer.
    :return:
    """

    # Output filename
    tf.gfile.MakeDirs(FLAGS.output_dir)
    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    base_name = os.path.join(
        FLAGS.output_dir,
        f"unconditioned_{date_and_time:s}.mid"
    )

    # Generating sample 
    LOGGER.info("Generating sample.")
    input_function = decoding.make_input_fn_from_generator(
        unconditional_input_generator(targets, decode_length)
    )
    unconditional_samples = estimator.predict(
        input_function, checkpoint_path=FLAGS.model_path
    )

    # Sample events
    LOGGER.info("Generating sample events.")
    sample_ids = next(unconditional_samples)["outputs"]

    # Decode to note sequence
    LOGGER.info("Decoding sample ID")
    midi_filename = decode(
        sample_ids,
        encoder=unconditional_encoders["targets"]
    )
    unconditional_note_seqs = mm.midi_file_to_note_sequence(midi_filename)

    # Append continuation to primer
    coninuation_note_sequence = mm.concatenate_sequences(
        [primer_note_sequence, unconditional_note_seqs]
    )

    # Saving MIDI file
    mm.sequence_proto_to_midi_file(coninuation_note_sequence, base_name)

def run():
    """
    Load Transformer model according to flags and start sampling.
    :raises:
        ValueError: if required flags are missing or invalid
    """

    if FLAGS.model_path is None:
        raise ValueError(
        "Required Transformer pre-trained model path."
    )

    if FLAGS.output_dir is None:
        raise ValueError(
        "Required MIDI output directory."
    )

    if FLAGS.decode_length <= 0:
        raise ValueError(
        "Decode length must be > 0."
    )

    problem = PianoPerformanceLanguageModelProblem()
    unconditional_encoders = problem.get_feature_encoders()

    primer_note_sequence = music_pb2.NoteSequence()
    # It should be possible to supply absolutely no primer.
    if FLAGS.primer_path is None:
        targets = []
    else:
        primer_note_sequence = get_primer_ns(FLAGS.primer_path)
        targets = unconditional_encoders["targets"].encode_note_sequence(
            primer_note_sequence
        )

        # Remove end token from encoded primer
        targets = targets[:-1]

        if len(targets) >= FLAGS.decode_length:
            raise ValueError(
                "Primer has more or equal events than max sequence length." 
            )

    decode_length = FLAGS.decode_length - len(targets)

    # Set up hyperparameters
    hparams = trainer_lib.create_hparams(hparams_set="transformer_tpu") # Add flag
    trainer_lib.add_problem_hparams(hparams, problem)
    hparams.num_hidden_layers = NUM_HIDDEN_LAYERS
    hparams.sampling_method = SAMPLING_METHOD

    # Set up decoding HParams
    decode_hparams = decoding.decode_hparams()
    decode_hparams.alpha = ALPHA
    decode_hparams.beam_size = BEAM_SIZE

    # Create estimator
    LOGGER.info("Loading model")
    run_config = trainer_lib.create_run_config(hparams)
    estimator = trainer_lib.create_estimator(
        MODEL_NAME,
        hparams,
        run_config,
        decode_hparams=decode_hparams,
    )

    generate(estimator, unconditional_encoders, decode_length, targets, primer_note_sequence)


def main(_):
    """Invoke run, set log level"""
    
    # This should probably be set with a flag. 
    LOGGER.set_verbosity("INFO")
    run()

def console_entry_point():
    """Call main function"""
    tf.app.run(main)

if __name__ == "__main__":
    console_entry_point()