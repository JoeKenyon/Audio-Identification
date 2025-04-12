"""
File: audio_identification.py
Author: Joseph Kenyon

Description: Audio Identification System, based on implemetation found in 
    Khatri, V., Dillingham, L. & Chen, Z., Year. "Song recognition using audio fingerprinting"
    Changed hop size, samplerate and spacing type for the band splitting.
"""
import numpy as np
from scipy.ndimage import maximum_filter
import librosa
import os
import pickle
from collections import defaultdict

class AudioIdentifier:
    def __init__(self, 
                target_samplerate=4500, 
                window_size=512, 
                hop_size=128, 
                target_zone_size=60, 
                num_bands=6, 
                max_filter_size=(3,3),
                verbose = False
                ):
        '''
        Initialize the AudioIdentifier with the specified parameters.
        
        Args:
            target_samplerate (int): Sample rate to which all audio will be resampled. Defaults to 4500.
            window_size (int): Size of the FFT window for STFT computation. Defaults to 512.
            hop_size (int): Hop length for STFT computation. Defaults to 128.
            target_zone_size (int): Number of frames to consider for pairing peaks. Defaults to 60.
            num_bands (int): Number of frequency bands to divide the spectrum into. Defaults to 6.
            max_filter_size (tuple): Size of the maximum filter for peak detection. Defaults to (3,3).
            verbose (bool): Enable printing of more verbose progress information . Defaults to False.
        '''
        self.target_samplerate = target_samplerate
        self.window_size = window_size
        self.hop_size = hop_size
        self.target_zone_size = target_zone_size
        self.num_bands = num_bands
        self.max_filter_size = max_filter_size
        self.verbose = verbose
        self.database = {}
        self.song_paths = {}



    def load_audio(self, file_path):
        '''
        Load an audio file, convert to mono, and resample to target sample rate.

        Args:
            file_path (str): Path to the audio file to load

        Returns:
            numpy.ndarray: Resampled audio data as a mono signal
        '''
        # load audio in mono
        # then downsample like the paper
        audio, sr = librosa.load(file_path, mono=True)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_samplerate)
        return audio


    def get_STFT(self,audio):
        '''
        Compute the Short-Time Fourier Transform (STFT) of the audio signal.
            
        Args:
            audio (numpy.ndarray): Audio signal data

        Returns:
            numpy.ndarray: Magnitude of STFT (spectrogram)
        '''
        # SHORT TIME FFFT
        # we want the mags
        STFT = librosa.stft(audio, 
                            n_fft=self.window_size, 
                            hop_length=self.hop_size,
                            window='hamming')
        return np.abs(STFT)


    def pick_peaks(self, STFT):
        '''
        Detect and pick the peaks in the STFT magnitude spectrogram.
        
        This function divides the spectrogram into equally spaced 
        frequency bands and picks the maximum peak in each band 
        for each time frame. It then applies a maximum filter to 
        find local maxima. A peak is then defined as both the local 
        maximma and a maximum band peak.
        
        Args:
            STFT (numpy.ndarray): Magnitude spectrogram from STFT
            
        Returns:
            list: List of tuples (frequency_bin, time_frame) representing peak locations
        '''
        # rows are freqs i think yeah
        rows = STFT.shape[0]
        band_size = rows // self.num_bands

        # matrix of peaks array
        peaks = np.zeros_like(STFT, dtype=bool)

        for band in range(self.num_bands):

            # compute band start and end index
            band_start = band * band_size
            band_end = min((band + 1) * band_size, rows)

            # max value in each band for each time frame
            band_max = np.max(STFT[band_start:band_end, :], axis=0)

            # find indices where max occurs
            for t in range(STFT.shape[1]):
                max_val = band_max[t]

                # gotta be larger than 0 RIGHT!
                if max_val > 0:
                    for f in range(band_start, band_end):
                        if STFT[f, t] == max_val:
                            peaks[f, t] = True
                            # only one peak per band per time frame
                            break

        # do some maximimum filters on the spectrogram
        # find where the original spectrogram equals the max filtered spectrogram
        # these are the local maximums
        max_filtered = maximum_filter(STFT, size=self.max_filter_size)
        local_max = (STFT == max_filtered)
        final_peaks = peaks & local_max

        # get peak coordinates
        peak_coords = np.argwhere(final_peaks)

        return [(freq, time) for freq, time in peak_coords]
        

    def hash_the_peaks(self, peaks, song_id=None):
        '''
        Generate hash pairs from peaks.

        Each peak is used as an anchor point and paired with subsequent peaks
        within a target zone to create hash keys.

        Args:
            peaks (list): List of (frequency, time) tuples representing peaks
            song_id (str, optional): ID of the song being fingerprinted. Defaults to None.

        Returns:
            list: List of hash information tuples:
                If song_id is provided: (hash_key, song_id, anchor_time)
                If song_id is None: (hash_key, anchor_time)
        '''
        hashes = []

        # sort peaks by time
        peaks.sort(key=lambda x: x[1])

        # use each peak as an anchor point
        for i, (anchor_freq, anchor_time) in enumerate(peaks):

            # define target zone based on current anchor point
            # tagetzone is number of frames to the right to consider
            target_zone_end = min(i + self.target_zone_size, len(peaks))

            # pair the anchor with points in the target zone
            for j in range(i + 1, target_zone_end):

                # calculate time delta
                target_freq, target_time = peaks[j]
                time_delta = target_time - anchor_time

                # skip if time delta is too large
                if time_delta > self.target_zone_size:
                    break

                # create hash (anchor_freq, target_freq, time_delta)
                hash_key = (anchor_freq, target_freq, time_delta)

                if song_id is not None:
                    hashes.append((hash_key, song_id, anchor_time))
                else:
                    hashes.append((hash_key, anchor_time))

        return hashes


    def generate_fingerprint(self, audio_file, song_id):
        '''
        Generate a fingerprint from an audio file and store it in the database.
        
        This function processes an audio file, gets its STFT, picks 
        the peaks, generates the hashes, and stores them in the database.
        
        Args:
            audio_file (str): Path to the audio file
            song_id (str): ID for the song
            
        Returns:
            int: Number of hashes generated for the song
        '''
        # load audio
        # get the STFT
        # pick the peaks
        # hash the peaks
        audio  = self.load_audio(audio_file)
        STFT   = self.get_STFT(audio)
        peaks  = self.pick_peaks(STFT)
        hashes = self.hash_the_peaks(peaks, song_id)

        # store hashes in database
        for hash_key, song_id, offset in hashes:

            # if not already in database, add it
            if hash_key not in self.database:
                self.database[hash_key] = []

            self.database[hash_key].append((song_id, offset))

        # store the original file path
        self.song_paths[song_id] = audio_file

        return len(hashes)


    def identify_song(self, audio_file):
        '''
        Identify a song snippet by comparing it to the fingerprint database.
            
        Args:
            audio_file (str): Path to the audio snippet to identify
            
        Returns:
            list: List of dictionaries containing match information
            {
                'song_id': str, 
                'match_count': int, 
                'histogram_max': int,
                'original_path': str 
            }
        '''
        # create fingerprint for the clip
        audio       = self.load_audio(audio_file)
        STFT        = self.get_STFT(audio)
        peaks       = self.pick_peaks(STFT)
        snip_hashes = self.hash_the_peaks(peaks)

        # count matches for each song
        matches = defaultdict(list)

        # compare clip hashes with database
        for (hash_key, clip_offset) in snip_hashes:

            # if hash is in data base
            if hash_key in self.database:

                # get all matching songs for this hash
                # calculate time difference between clip and song
                for (song_id, song_offset) in self.database[hash_key]:
                    time_diff = song_offset - clip_offset
                    matches[song_id].append(time_diff)

        results = []
        for song_id, time_diffs in matches.items():
            if time_diffs: # shift all values to be non-negative
                min_diff = min(time_diffs)
                shifted_diffs = [diff - min_diff for diff in time_diffs]
                hist = np.bincount(shifted_diffs)
                hist_max = np.max(hist)
            else:
                hist_max = 0

            results.append({
                'song_id': song_id,
                'match_count': len(time_diffs),
                'histogram_max': hist_max,
                'original_path': self.song_paths.get(song_id, "")
            })

        # sort results by histogram max
        results.sort(key=lambda x: x['histogram_max'], reverse=True)
        return results


    def fingerprintBuilder(self, database_path, fingerprints_path):
        '''       
        This function processes all audio files in the specified directory,
        generates fingerprints, and saves the database to a pickle file.
        
        Args:
            database_path (str): Path to directory containing audio files
            fingerprints_path (str): Path to save the fingerprint database
            
        Returns:
            None
        '''
        # check if the database path exists
        if not os.path.exists(database_path):
            print(f"Error: Database path '{database_path}' does not exist")
            return
        
        print(f"[!] Fingerprinting files in {database_path}")

        song_count = 0
        for filename in os.listdir(database_path):
            if filename.endswith(('.wav', '.mp3')):
                file_path = os.path.join(database_path, filename)
                song_id = filename # use filename as song_id
                num_hashes = self.generate_fingerprint(file_path, song_id)
                song_count += 1
                if self.verbose:
                    print(f"[*] Generated {num_hashes} hashes for {song_id}")

        print(f"[!] Fingerprinted {song_count} songs with {len(self.database)} unique hashes")
        print(f"[!] Saving to {fingerprints_path}")

        # save fingerprints to specified path
        database_data = {
            'database'    : self.database,
            'song_paths'  : self.song_paths
        }
        with open(fingerprints_path, 'wb') as f:
            pickle.dump(database_data, f)

        print(f"[!] Database saved to {fingerprints_path}")


    def audioIdentification(self, queryset_path, fingerprints_path, output_path):
        '''
        Load a fingerprint database from disk, process all audio files 
        in a query directory, and write identification results to an output file.
        
        Args:
            queryset_path (str): Path to directory containing query audio files
            fingerprints_path (str): Path to the fingerprint database file
            output_path (str): Path to write identification results
        '''
        with open(fingerprints_path, 'rb') as f:
            database_data = pickle.load(f)
        self.database   = database_data['database']
        self.song_paths = database_data['song_paths']

        print(f"[!] Identifying songs in {queryset_path}")

        with open(output_path, 'w') as out_file:

            # process each query file
            for filename in os.listdir(queryset_path):

                if filename.endswith(('.wav', '.mp3')):

                    # this is the file naem
                    file_path = os.path.join(queryset_path, filename)

                    # get results
                    results = self.identify_song(file_path)

                    # get top 3 matches
                    top_matches = results[:3]

                    # pad if not enough THEY SHOULD HAVE 3 IDK
                    while len(top_matches) < 3:
                        top_matches.append({'song_id': '', 'original_path': ''})

                    results_line = f"{filename} {top_matches[0]['song_id']} {top_matches[1]['song_id']} {top_matches[2]['song_id']}\n"

                    out_file.write(results_line)

                    if self.verbose:
                        print(f"[*] Top 3 matches for {filename}")
                        print(f" 1. {top_matches[0]['song_id']}")
                        print(f" 2. {top_matches[1]['song_id']}")
                        print(f" 3. {top_matches[2]['song_id']}")

        print(f"[!] Results saved to {output_path}")


def evaluate(file_path):
    '''
    Evaluate the accuracy of audio identification results.
    
    This function reads the identification results from a file and calculates
    accuracy metrics by comparing the identified songs with ground truth.
    
    Args:
        file_path (str): Path to the results file
        
    Returns:
        None: Prints evaluation results
    '''
    import re
    counts = [0, 0, 0, 0]
    entries = 0
    with open(file_path, 'r') as file:
        for line in file:
            files = line.strip().split()
            if len(files) < 2:
                continue
            correct_name = re.sub(r'-snippet-\d+-\d+', '', files[0])
            match_idx = {files[i]: i - 1 for i in range(1, 4)}.get(correct_name, 3)
            counts[match_idx] += 1
            entries += 1

    print(f"=== Evaluation ===")
    print(f"-> Times 1st result was correct: {counts[0]}")
    print(f"-> Times 2nd result was correct: {counts[1]}")
    print(f"-> Times 3rd result was correct: {counts[2]}")
    print(f"-> Times No  result was correct: {counts[3]}")
    print(f"TOP 3 ACCURACY: {((counts[0]+counts[1]+counts[2])/entries)*100:.2f}%")


# if you want these to be faster you could use the database
# already stored in memory which saves you having to
# load it up again but i dont think we are meant to do that
def fingerprintBuilder(database_path, fingerprints_path):
    ai = AudioIdentifier()
    ai.fingerprintBuilder(database_path, fingerprints_path)

def audioIdentification(queryset_path, fingerprints_path, output_path):
    ai = AudioIdentifier()
    ai.audioIdentification(queryset_path, fingerprints_path, output_path)

if __name__ == "__main__":
    print_results     = True
    database_path     = "./database"
    fingerprints_path = "./fingerprints/fingerprints.pkl"
    queryset_path     = "./queryset"
    output_path       = "./queries.txt"

    fingerprintBuilder(database_path, fingerprints_path)
    audioIdentification(queryset_path, fingerprints_path, output_path)

    if print_results:
        evaluate(output_path)