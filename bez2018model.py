import cython_bez2018 # Package must be installed in-place: `python setup.py build_ext --inplace`
import numpy as np
import scipy.signal


def get_ERB_cf_list(num_cf, min_cf=125.0, max_cf=8e3):
    '''
    Helper function to get array of num_cfs ERB-scaled CFs between min_cf and max_cf.
    
    Args
    ----
    num_cf (int): number of CFs sets length of output list
    min_cf (float): lowest CF in Hz
    max_cf (float): highest CF in Hz
    
    Returns
    -------
    cf_list (list): list of CFs in Hz (lowest to highest)
    '''
    E_start = 21.4 * np.log10(0.00437 * min_cf + 1.0)
    E_end = 21.4 * np.log10(0.00437 * max_cf + 1.0)
    cf_list = np.linspace(E_start, E_end, num=num_cf)
    cf_list = (1.0/0.00437) * (10.0 ** (cf_list / 21.4) - 1.0)
    return list(cf_list)


def nervegram(signal,
              signal_fs,
              nervegram_dur=None,
              nervegram_fs=10e3,
              buffer_start_dur=0.0,
              buffer_end_dur=0.0,
              pin_fs=100e3,
              pin_dBSPL_flag=0,
              pin_dBSPL=None,
              species=2,
              bandwidth_scale_factor=1.0,
              cf_list=None,
              num_cf=50,
              min_cf=125.0,
              max_cf=8e3,
              synapseMode=0,
              max_spikes_per_train=-1,
              num_spike_trains=40,
              cohc=1.0,
              cihc=1.0,
              IhcLowPass_cutoff=3e3,
              IhcLowPass_order=7,
              spont=70.0,
              noiseType=1,
              implnt=0,
              tabs=6e-4,
              trel=6e-4,
              random_seed=None,
              return_vihcs=True,
              return_meanrates=True,
              return_spike_times=True,
              return_spike_tensor=True):
    '''
    Main function for generating an auditory nervegram.

    Args
    ----
    signal (np.ndarray): input pressure waveform must be 1-dimensional array (units Pa)
    signal_fs (int): sampling rate of input signal (Hz)
    nervegram_dur (float or None): if not None, specifies duration of clipped nervegram
    nervegram_fs (int): sampling rate of nervegram (Hz)
    buffer_start_dur (float): period to ignore at start of nervegram (s)
    buffer_end_dur (float): period to ignore at end of nervegram (s)
    pin_fs (int): sampling rate of input signal passed to ANmodel (100000 Hz)
    pin_dBSPL_flag (int): if 1, pin will be re-scaled to specified sound pressure level
    pin_dBSPL (float): sound pressure level of input signal passed to ANmodel (dB SPL)
    species (int): 1=cat, 2=human (Shera et al. 2002), 3=human (G&M 1990), 4=custom
    bandwidth_scale_factor (float or list): cochlear filter bandwidth scaling factor
    cf_list (None or list): if not None, specifies list of characteristic frequencies
    num_cf (int): if cf_list is None, specifies number of ERB-spaced CFs
    min_cf (float): if cf_list is None, specifies minimum CF (Hz)
    max_cf (float): if cf_list is None, specifies maximum CF (Hz)
    synapseMode (float): set to 1 to re-run synapse model for each spike train (0 to re-use synout)
    max_spikes_per_train (int): max array size for spike times output (<0 to auto-select)
    num_spike_trains (int): number of spike trains to sample from spike generator
    cohc (float): OHC scaling factor: 1=normal OHC function, 0=complete OHC dysfunction
    cihc (float): IHC scaling factor: 1=normal IHC function, 0=complete IHC dysfunction
    IhcLowPass_cutoff (float): cutoff frequency for IHC lowpass filter (Hz)
    IhcLowPass_order (int): order for IHC lowpass filter
    spont (float): spontaneous firing rate in spikes per second
    noiseType (int): set to 0 for noiseless and 1 for variable fGn
    implnt (int): set to 0 for "approx" and 1 for "actual" power-law function implementation
    tabs (float): absolute refractory period in seconds
    trel (float): baseline mean relative refractory period in seconds
    random_seed (int or None): if not None, used to specify np.random.seed
    return_vihcs (bool): if True, output_dict will contain inner hair cell potentials
    return_meanrates (bool): if True, output_dict will contain instantaneous firing rates
    return_spike_times (bool): if True, output_dict will contain spike times
    return_spike_tensor (bool): if True, output_dict will contain binary spike tensor

    Returns
    -------
    output_dict (dict): contains nervegram(s), stimulus, and all parameters
    '''
    # ============ PARSE ARGUMENTS ============ #
    # If specified, set random seed (eliminates stochasticity in ANmodel noise)
    if not (random_seed == None):
        np.random.seed(random_seed)
    # BEZ2018 ANmodel requires 1 dimensional arrays with dtype np.float64
    signal = np.squeeze(signal).astype(np.float64)
    assert len(signal.shape) == 1, "signal must be a 1-dimensional array"
    signal_dur = signal.shape[0] / signal_fs
    # If `cf_list` is not provided, build list from `num_cf`, `min_cf`, and `max_cf`
    if cf_list is None:
        cf_list = get_ERB_cf_list(num_cf, min_cf=min_cf, max_cf=max_cf)
    # Convert `bandwidth_scale_factor` to list of same length as `cf_list` if needed
    bandwidth_scale_factor = np.array(bandwidth_scale_factor).reshape([-1]).tolist()
    if len(bandwidth_scale_factor) == 1:
        bandwidth_scale_factor = len(cf_list) * bandwidth_scale_factor
    msg = "cf_list and bandwidth_scale_factor must have the same length"
    assert len(bandwidth_scale_factor) == len(cf_list), msg

    # ============ RESAMPLE AND RESCALE INPUT SIGNAL ============ #
    # Resample the input signal to pin_fs (at least 100kHz) for ANmodel
    pin = scipy.signal.resample_poly(signal, int(pin_fs), int(signal_fs))
    # If pin_dBSPL_flag, scale pin to desired dB SPL (otherwise compute dB SPL)
    if pin_dBSPL_flag:
        pin = pin - np.mean(pin)
        pin_rms = np.sqrt(np.mean(np.square(pin)))
        desired_rms = 2e-5 * np.power(10, pin_dBSPL / 20)
        if pin_rms > 0:
            pin = desired_rms * (pin / pin_rms)
        else:
            pin_dBSPL = -np.inf
            print('>>> [WARNING] rms(pin) = 0 (silent input signal)')
    else:
        pin_dBSPL = 20 * np.log10(np.sqrt(np.mean(np.square(pin))) / 2e-5)

    # ============ RUN AUDITORY NERVE MODEL ============ #
    # Initialize output array lists
    if return_vihcs:
        nervegram_vihcs = []
    if return_meanrates:
        nervegram_meanrates = []
    if return_spike_times:
        nervegram_spike_times = []

    # Iterate over all CFs and run the auditory nerve model components
    for cf_idx, cf in enumerate(cf_list):
        ###### Run IHC model ######
        vihc = cython_bez2018.run_ihc(pin,
                                      pin_fs,
                                      cf,
                                      species=species,
                                      bandwidth_scale_factor=bandwidth_scale_factor[cf_idx],
                                      cohc=cohc,
                                      cihc=cihc,
                                      IhcLowPass_cutoff=IhcLowPass_cutoff,
                                      IhcLowPass_order=IhcLowPass_order)
        ###### Run IHC-ANF synapse model ######
        synapse_out = cython_bez2018.run_anf(vihc,
                                             pin_fs,
                                             cf,
                                             noiseType=noiseType,
                                             implnt=implnt,
                                             spont=spont,
                                             tabs=tabs,
                                             trel=trel,
                                             synapseMode=synapseMode,
                                             max_spikes_per_train=max_spikes_per_train,
                                             num_spike_trains=num_spike_trains)
        if return_vihcs:
            tmp_vihc = scipy.signal.resample_poly(vihc, int(nervegram_fs), int(pin_fs))
            nervegram_vihcs.append(tmp_vihc)
        if return_meanrates:
            tmp_meanrate = scipy.signal.resample_poly(synapse_out['meanrate'], int(nervegram_fs), int(pin_fs))
            tmp_meanrate[tmp_meanrate < 0] = 0
            nervegram_meanrates.append(tmp_meanrate)
        if return_spike_times:
            tmp_spike_times = synapse_out['spike_times']
            nervegram_spike_times.append(tmp_spike_times)

    # Combine output arrays across CFs
    if return_vihcs:
        nervegram_vihcs = np.stack(nervegram_vihcs, axis=0).astype(np.float32)
    if return_meanrates:
        nervegram_meanrates = np.stack(nervegram_meanrates, axis=0).astype(np.float32)
    if return_spike_times:
        nervegram_spike_times = np.stack(nervegram_spike_times, axis=1).astype(np.float32)

    # ============ APPLY TRANSFORMATIONS ============ #
    if (nervegram_dur is None) or (nervegram_dur == signal_dur):
        nervegram_dur = signal_dur
    else:
        # Compute clip segment start and end indices
        buffer_start_idx = int(buffer_start_dur*nervegram_fs)
        buffer_end_idx = int(signal_dur*nervegram_fs) - int(buffer_end_dur*nervegram_fs)
        if buffer_start_idx == buffer_end_idx - nervegram_dur*nervegram_fs:
            clip_start_nervegram = buffer_start_idx
        else:
            clip_start_nervegram = np.random.randint(buffer_start_idx,
                                                     high=buffer_end_idx-nervegram_dur*nervegram_fs)
        clip_end_nervegram = clip_start_nervegram + int(nervegram_dur*nervegram_fs)
        assert clip_end_nervegram <= buffer_end_idx, "clip_end_nervegram out of buffered range"
        # Clip segment of signal (input stimulus)
        clip_start_signal = int(clip_start_nervegram * signal_fs / nervegram_fs)
        clip_end_signal = int(clip_end_nervegram * signal_fs / nervegram_fs)
        signal = signal[clip_start_signal:clip_end_signal]
        # Clip segment of pin (stimulus provided to ANmodel)
        clip_start_pin = int(clip_start_nervegram * pin_fs / nervegram_fs)
        clip_end_pin = int(clip_end_nervegram * pin_fs / nervegram_fs)
        pin = pin[clip_start_pin:clip_end_pin]
        # Clip segment of vihcs (inner hair cell potential)
        if return_vihcs:
            nervegram_vihcs = nervegram_vihcs[:, clip_start_nervegram:clip_end_nervegram]
        # Clip segment of meanrates (instantaneous firing rate)
        if return_meanrates:
            nervegram_meanrates = nervegram_meanrates[:, clip_start_nervegram:clip_end_nervegram]
        # Adjust spike times (set t=0 to `clip_start_nervegram` and eliminate negative times)
        if return_spike_times:
            clip_start_nervegram_time = clip_start_nervegram / nervegram_fs
            clip_end_nervegram_time = clip_end_nervegram / nervegram_fs
            nervegram_spike_times[nervegram_spike_times >= clip_end_nervegram_time] = 0
            nervegram_spike_times = nervegram_spike_times - clip_start_nervegram_time
            nervegram_spike_times[nervegram_spike_times < 0] = 0
            # Re-order spike times to eliminate leading zeros (spike cannot occur at t=0)
            for itr0 in range(nervegram_spike_times.shape[0]):
                for itr1 in range(nervegram_spike_times.shape[1]):
                    spike_times = nervegram_spike_times[itr0, itr1, :]
                    spike_times = spike_times[spike_times > 0]
                    nervegram_spike_times[itr0, itr1, :] = 0
                    nervegram_spike_times[itr0, itr1, 0:spike_times.shape[0]] = spike_times
    # Generate binary tensor of spikes (with shape [spike_train, CF, time]) from spike times
    if return_spike_tensor:
        assert return_spike_times, "`return_spike_times` must be true to generate spike tensor"
        nervegram_spike_indices = (nervegram_spike_times * nervegram_fs).astype(int)
        out_shape = list(nervegram_spike_indices.shape)[:-1] + [int(nervegram_dur*nervegram_fs)]
        nervegram_spike_tensor = np.zeros(out_shape, dtype=int)
        for itr0 in range(nervegram_spike_indices.shape[0]):
            for itr1 in range(nervegram_spike_indices.shape[1]):
                indices = np.trim_zeros(nervegram_spike_indices[itr0, itr1], trim='b')
                nervegram_spike_tensor[itr0, itr1, indices] = 1

    # ============ RETURN OUTPUT AS DICTIONARY ============ #
    output_dict = {
        'signal': signal.astype(np.float32),
        'signal_fs': signal_fs,
        'pin': pin.astype(np.float32),
        'pin_fs': pin_fs,
        'nervegram_fs': nervegram_fs,
        'nervegram_dur': nervegram_dur,
        'cf_list': np.array(cf_list).astype(np.float32),
        'bandwidth_scale_factor': np.array(bandwidth_scale_factor).astype(np.float32),
        'species': species,
        'spont': spont,
        'buffer_start_dur': buffer_start_dur,
        'buffer_end_dur': buffer_end_dur,
        'pin_dBSPL_flag': pin_dBSPL_flag,
        'pin_dBSPL': pin_dBSPL,
        'synapseMode': synapseMode,
        'max_spikes_per_train': max_spikes_per_train,
        'num_spike_trains': num_spike_trains,
        'cohc': cohc,
        'cihc': cihc,
        'IhcLowPass_cutoff': IhcLowPass_cutoff,
        'IhcLowPass_order': IhcLowPass_order,
        'noiseType': noiseType,
        'implnt': implnt,
        'tabs': tabs,
        'trel': trel,
    }
    if return_vihcs:
        output_dict['nervegram_vihcs'] = nervegram_vihcs
    if return_meanrates:
        output_dict['nervegram_meanrates'] = nervegram_meanrates
    if return_spike_times:
        output_dict['nervegram_spike_times'] = nervegram_spike_times
    if return_spike_tensor:
        output_dict['nervegram_spike_tensor'] = nervegram_spike_tensor
    return output_dict
