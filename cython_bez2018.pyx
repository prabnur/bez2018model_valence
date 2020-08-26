import numpy as np
import util_bez2018
import scipy.signal

from libc.stdlib cimport malloc
cimport numpy as np
np.import_array()


cdef extern from "stdlib.h":
    void *memcpy(void *str1, void *str2, size_t n)

cdef extern from "Python.h":
    ctypedef int Py_intptr_t

cdef extern from "numpy/arrayobject.h":
    ctypedef Py_intptr_t npy_intp
    object PyArray_SimpleNewFromData(
        int nd,
        npy_intp* dims,
        int typenum,
        void* data
    )

cdef extern from "model_IHC_BEZ2018.h":
    void IHCAN(
        double *px,
        double cf,
        int nrep,
        double tdres,
        int totalstim,
        double cohc,
        double cihc,
        int species,
        double bandwidth_scale_factor,
        double IhcLowPass_cutoff,
        double IhcLowPass_order,
        double *ihcout
    )

cdef extern from "model_Synapse_BEZ2018.h":
    void SingleAN(
        double *px,
        double cf,
        int nrep,
        double tdres,
        int totalstim,
        double noiseType,
        double implnt,
        double spont,
        double tabs,
        double trel,
        double *meanrate,
        double *varrate,
        double *psth,
        double *synout,
        double *trd_vector,
        double *trel_vector
    )

cdef extern from "model_Synapse_BEZ2018.h":
    double Synapse(
        double *ihcout,
        double tdres,
        double cf,
        int totalstim,
        int nrep,
        double spont,
        double noiseType,
        double implnt,
        double sampFreq,
        double *synout
    )

cdef extern from "model_Synapse_BEZ2018.h":
    int SpikeGenerator(
        double *synout,
        double tdres,
        double t_rd_rest,
        double t_rd_init,
        double tau,
        double t_rd_jump,
        int nSites,
        double tabs,
        double trel,
        double spont,
        int totalstim,
        int nrep,
        double total_mean_rate,
        long MaxArraySizeSpikes,
        double *sptime,
        double *trd_vector
    )


def run_ihc(np.ndarray[np.float64_t, ndim=1] signal,
            double fs,
            double cf,
            int species=1,
            double bandwidth_scale_factor=1.,
            double cohc=1.,
            double cihc=1.,
            IhcLowPass_cutoff=3000.,
            IhcLowPass_order=7):
    """
    Run middle ear filter, BM filters, and IHC model.
    (based on https://github.com/mrkrd/cochlea/blob/master/cochlea/zilany2014)
    
    Args
    ----
    signal (np.float64 array): input acoustic waveform in units of Pa
    fs (float): sampling rate in Hz
    cf (float): characteristic frequency in Hz
    species (int): sets filter parameters: 1=cat, 2=human, 3=G&M1990
    bandwidth_scale_factor (float): scales cochlear filter bandwidth
    cohc (float): OHC scaling factor: 1=normal OHC function, 0=complete OHC dysfunction
    cihc (float): IHC scaling factor: 1=normal IHC function, 0=complete IHC dysfunction
    IhcLowPass_cutoff (float): cutoff frequency for IHC lowpass filter (Hz)
    IhcLowPass_order (int): order for IHC lowpass filter
    
    Returns
    -------
    ihcout (np.float64 array): IHC membrane potential (in volts)
    """
    # Check arguments
    assert species in [1, 2, 3], ("species must be in [1, 2, 3]:\n"
                                  "\t1 = cat,\n"
                                  "\t2 = human: Shera et al. (PNAS 2002)\n"
                                  "\t3 = human: Glasberg & Moore (Hear. Res. 1990)\n")
    if species == 1:
        assert (cf > 124.9) and (cf < 40e3), "CF out of range for cat (125Hz to 40kHz)"
    else:
        assert (cf > 124.9) and (cf < 20001.), "CF out of range for human (125Hz to 20kHz)"
    assert (bandwidth_scale_factor > 0), "bandwidth_scale_factor must be positive"
    assert (fs >= 100e3) and (fs <= 500e3), "Sampling rate out of range (100kHz to 500kHz)"
    assert (cohc >= 0) and (cohc <= 1), "cohc out of range ([0, 1])"
    assert (cihc >= 0) and (cihc <= 1), "cihc out of range ([0, 1])"
    
    # Ensure input array (input sound) is C contiguous and initialize pointer
    if not signal.flags['C_CONTIGUOUS']:
        signal = signal.copy(order='C')
    cdef double *signal_data = <double *>np.PyArray_DATA(signal)
    
    # Initialize output array and data pointer for IHC voltage
    ihcout = np.zeros( len(signal) )
    cdef double *ihcout_data = <double *>np.PyArray_DATA(ihcout)
    
    # Run model_IHC_BEZ2018.IHCAN (modifies ihcout_data in place)
    IHCAN(
        signal_data,            #double *px,
        cf,                     #double cf,
        1,                      #int nrep,
        1.0/fs,                 #double tdres,
        len(signal),            #int totalstim,
        cohc,                   #double cohc,
        cihc,                   #double cihc,
        species,                #int species,
        bandwidth_scale_factor, #double bandwidth_scale_factor
        IhcLowPass_cutoff,      #double IhcLowPass_cutoff
        IhcLowPass_order,       #int IhcLowPass_order
        ihcout_data             #double *ihcout
    )
    return ihcout


def run_synapse(np.ndarray[np.float64_t, ndim=1] vihc,
                double fs,
                double cf,
                double noiseType=1.,
                double implnt=0.,
                double spont=70.,
                double tabs=0.6e-3,
                double trel=0.6e-3):
    """
    Run IHC-ANF synapse model.
    (based on https://github.com/mrkrd/cochlea/blob/master/cochlea/zilany2014)

    Args
    ----
    vihc (np.float64 array): IHC membrane potential (in volts)
    fs (float): sampling rate in Hz
    cf (float): characteristic frequency in Hz
    noiseType (float): set to 0 for noiseless and 1 for variable fGn
    implnt (float): set to 0 for "approx" and 1 for "actual" power-law function implementation
    spont (float): spontaneous firing rate in spikes per second
    tabs (float): absolute refractory period in seconds
    trel (float): baseline mean relative refractory period in seconds

    Returns
    -------
    output_dict (dict): dictionary of all output variables (np.float64 arrays)
        'synout': synapse output rate in /s (before redocking effects are considered)
        'meanrate': analytical estimate of the instantaneous mean firing rate in /s
        'varrate': analytical estimate of the instantaneous variance in firing rate in /s
    'psth': peristimulus time histogram of spikes
        'trd_vector': vector of the mean redocking time in seconds
        'trel_vector': vector of the mean relative refractory period in seconds
    """
    # Ensure input array (IHC voltage) is C contiguous and initialize pointer
    if not vihc.flags['C_CONTIGUOUS']:
        vihc = vihc.copy(order='C')
    cdef double *vihc_data = <double *>np.PyArray_DATA(vihc)
    
    # Initialize output arrays and data pointers
    synout = np.zeros_like(vihc) # (spiking probabilities)
    cdef double *synout_data = <double *>np.PyArray_DATA(synout)
    meanrate = np.zeros_like(vihc) # (instantaneous mean firing rate)
    cdef double *meanrate_data = <double *>np.PyArray_DATA(meanrate)
    varrate = np.zeros_like(vihc) # (instantaneous variance of firing rate)
    cdef double *varrate_data = <double *>np.PyArray_DATA(varrate)
    psth = np.zeros_like(vihc) # (PSTH of spikes)
    cdef double *psth_data = <double *>np.PyArray_DATA(psth)
    trd_vector = np.zeros_like(vihc) # (mean synaptic redocking times)
    cdef double *trd_vector_data = <double *>np.PyArray_DATA(trd_vector)
    trel_vector = np.zeros_like(vihc) # (mean relative refractory periods)
    cdef double *trel_vector_data = <double *>np.PyArray_DATA(trel_vector)
    
    # Run model_Synapse_BEZ2018.SingleAN (modifies output arrays in place)
    SingleAN(
        vihc_data,          #double *px,
        cf,                 #double cf,
        1,                  #int nrep,
        1.0/fs,             #double tdres,
        len(vihc),          #int totalstim,
        noiseType,          #double noiseType,
        implnt,             #double implnt,
        spont,              #double spont,
        tabs,               #double tabs,
        trel,               #double trel,
        meanrate_data,      #double *meanrate,
        varrate_data,       #double *varrate,
        psth_data,          #double *psth,
        synout_data,        #double *synout,
        trd_vector_data,    #double *trd_vector,
        trel_vector_data    #double *trel_vector
    )
    output_dict = {
        'synout': synout,
        'meanrate': meanrate,
        'varrate': varrate,
        'psth': psth,
        'trd_vector': trd_vector,
        'trel_vector': trel_vector
    }
    return output_dict


def tmp_test(
        np.ndarray[np.float64_t, ndim=1] vihc,
        double fs,
        double cf,
        double noiseType=1.,
        double implnt=0.,
        double spont=70.,
        double tabs=0.6e-3,
        double trel=0.6e-3):
    """
    """
    # Ensure input array (IHC voltage) is C contiguous and initialize pointer
    if not vihc.flags['C_CONTIGUOUS']:
        vihc = vihc.copy(order='C')
    cdef double *vihc_data = <double *>np.PyArray_DATA(vihc)
    
    # Fixed parameters for Synapse and SpikeGenerator functions
    tdres = 1/fs
    totalstim = len(vihc)
    nrep = 1
    sampFreq = 10e3
    nSites = 4 # number of synpatic release sites
    t_rd_rest = 14.0e-3 # resting value of the mean redocking time
    t_rd_jump = 0.4e-3 # size of jump in mean redocking time when a redocking event occurs
    t_rd_init = t_rd_rest + 0.02e-3 * spont - t_rd_jump # initial value of the mean redocking time
    tau = 60.0e-3 # time constant for short-term adaptation (in mean redocking time)
    
    # Initialize `synout` array and data pointer
    synout = np.zeros_like(vihc) # (spiking probabilities)
    cdef double *synout_data = <double *>np.PyArray_DATA(synout)
    
    # Call the Synapse function
    I = Synapse(
        vihc_data,
        tdres,
        cf,
        totalstim,
        nrep,
        spont,
        noiseType,
        implnt,
        sampFreq,
        synout_data)
    
    # Initialize `sptime` output array with length equal to `MaxArraySizeSpikes`
    total_mean_rate = np.sum(synout) / I # calculate the overall mean synaptic rate
    MeanISI = (1 / total_mean_rate) + (t_rd_init) / nSites + tabs + trel
    SignalLength = totalstim * nrep * tdres
    MaxArraySizeSpikes = np.long(np.ceil(SignalLength / MeanISI + 3 * np.sqrt(SignalLength/MeanISI)))
    sptime = np.zeros([int(MaxArraySizeSpikes)], dtype=vihc.dtype)
    cdef double *sptime_data = <double *>np.PyArray_DATA(sptime)
    
    # Initialize `trd_vector` output array and data pointer
    trd_vector = np.zeros_like(vihc) # (mean synaptic redocking times)
    cdef double *trd_vector_data = <double *>np.PyArray_DATA(trd_vector)
    
    # Call the SpikeGenerator function
    nspikes = SpikeGenerator(
        synout_data,
        tdres,
        t_rd_rest,
        t_rd_init,
        tau,
        t_rd_jump,
        nSites,
        tabs,
        trel,
        spont,
        totalstim,
        nrep,
        total_mean_rate,
        MaxArraySizeSpikes,
        sptime_data,
        trd_vector_data)
    
    # Analytic estimate of instantaneous firing rate
    IDX = synout > 0
    meanrate = np.zeros_like(vihc)
    trel_vector = np.ones_like(vihc) * trel
    trel_vector[IDX] = trel * 100 / synout[IDX]
    trel_vector[trel_vector > trel] = trel
    meanrate[IDX] = synout[IDX] / (synout[IDX] * (tabs + trd_vector[IDX] / nSites + trel_vector[IDX]) + 1)
    
    return {'sptime':sptime, 'synout':synout, 'trd_vector':trd_vector, 'meanrate': meanrate}


cdef public double* generate_random_numbers(long length):
    """
    Wrapper for np.random.rand
    """
    arr = np.random.rand(length)
    # Ensure array is C contiguous
    if not arr.flags['C_CONTIGUOUS']:
        arr = arr.copy(order='C')
    # Copy data to output array
    cdef double *data_ptr = <double *>np.PyArray_DATA(arr)
    cdef double *out_ptr = <double *>malloc(length * sizeof(double))
    memcpy(out_ptr, data_ptr, length*sizeof(double))
    return out_ptr


cdef public double* decimate(int k, double *signal, int q):
    """
    Decimate a signal
    k: number of samples in signal
    signal: pointer to the signal
    q: decimation factor
    This implementation was inspired by scipy.signal.decimate.
    """
    # signal_arr will not own the data, signal's array has to be freed
    # after return from this function
    signal_arr = PyArray_SimpleNewFromData(
        1,                      # nd
        [k],                    # dims
        np.NPY_DOUBLE,          # typenum
        <void *>signal          # data
    )
    b = scipy.signal.firwin(q+1, 1./q, window='hamming')
    a = [1.]
    filtered = scipy.signal.filtfilt(b=b, a=a, x=signal_arr)
    resampled = filtered[::q]
    # Ensure array is C contiguous
    if not resampled.flags['C_CONTIGUOUS']:
        resampled = resampled.copy(order='C')
    # Copy data to output array
    cdef double *resampled_ptr = <double *>np.PyArray_DATA(resampled)
    cdef double *out_ptr = <double *>malloc(len(resampled)*sizeof(double))
    memcpy(out_ptr, resampled_ptr, len(resampled)*sizeof(double))
    return out_ptr


cdef public double* ffGn(int N, double tdres, double Hinput, double noiseType, double mu):
    """
    Wrapper for util_bez2018.ffGn
    """
    a = util_bez2018.ffGn(N, tdres, Hinput, noiseType, mu)
    # Ensure array is C contiguous
    if not a.flags['C_CONTIGUOUS']:
        a = a.copy(order='C')
    # Copy data to output array
    cdef double *ptr = <double *>np.PyArray_DATA(a)
    cdef double *out_ptr = <double *>malloc(len(a)*sizeof(double))
    memcpy(out_ptr, ptr, len(a)*sizeof(double))
    return out_ptr
