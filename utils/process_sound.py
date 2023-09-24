import os
import subprocess

def process_files(input_dir, output_dir):
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Go through each file in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.aif'):
                input_file_path = os.path.join(root, file)

                # Extract note from filename (assuming filename format like Piano.ff.Gb7.aiff)
                note = file.split('.')[-3]

                # Create the output file path
                output_file_path = os.path.join(output_dir, f"{note}.wav")

                # Execute ffmpeg command
                command = [
                    "ffmpeg", "-i", input_file_path,
                    "-af", "loudnorm,silenceremove=start_periods=1:start_silence=0.05:start_threshold=-40dB,afade=out:st=3:d=1.5,afade=in:st=0:d=0.05",
                    "-ac", "1",  # Convert to mono
                    "-to", "4.5",
                    output_file_path
                ]

                subprocess.run(command, check=True)  # Running the command and checking for errors

if __name__ == "__main__":
    input_dir = "../notes/bells.brass.ff.stereo"
    output_dir = "../notes/Bells"
    process_files(input_dir, output_dir)
