# Axis Labels
BITSTREAM_X = "Bit Index"
BITSTREAM_Y = "Bit Value"
SYMBOLS_X = "Symbol Index"
SYMBOLS_Y = "Amplitude"
TIMEDOMAIN_X = "Time"
TIMEDOMAIN_Y = "Amplitude"
FREQUENCY_X = "Frequency"
FREQUENCY_Y = "Magnitude ($dB$)"
IMPULSE_X = "Time ($ms$)"
IMPULSE_Y = "Amplitude"

# Axis
CONSTELLATION_XLIM = (-1.5, 1.5)
CONSTELLATION_YLIM = (-1.5, 1.5)
IMPULSE_XLIM = (-5, 5)
IMPULSE_XLIM_400 = (-10, 10)
PHASE_XLIM = (40, 300)
TIME_XLIM = (40, 140)
TIME_XLIM_RECEIVER = (100, 240)
FREQ_COMBINED_XLIM = (0, 8)
FREQ_COMPONENTS_XLIM = (-1.5, 1.5)
FREQ_MODULATED_XLIM = (-10, 10)
SYNC_XLIM = (40, 140)
CORR_XLIM = (40, 140)
SYNC_XLIM_RECEIVER = (100, 280)
CORR_XLIM_RECEIVER = (100, 280)
SYMBOLS_XLIM = (0, 60)
BITSTREAM_XLIM = (0, 60)
NOISE_DENSITY_YLIM = (-1, 1)

# Colors
COLOR_I = "darkgreen"
COLOR_Q = "firebrick"
COLOR_COMBINED = "purple"
COLOR_IMPULSE = "darkorange"
COLOR_AUX1 = "darkgoldenrod"
COLOR_AUX2 = "darkviolet"
COLOR_AUX3 = "forestgreen"
COLOR_AUX4 = "dodgerblue"
CORR_PLOT_COLOR = "dodgerblue"
SYNC_PLOT_V_LIMIT_COLOR = "darkorange"
SYNC_PLOT_V_CENTRAL_COLOR = "dodgerblue"
SYNC_PLOT_BACKGROUND_COLOR = "dimgray"
CORR_PLOT_V_LIMIT_COLOR = "darkorange"
QPSK_IDEAL_COLOR = "darkgreen"
DETECTION_THRESHOLD_COLOR = "darkgoldenrod"
LPF_CUT_OFF_COLOR = "darkgoldenrod"
LPF_PHASE_COLOR = "darkviolet"
DETECTOR_COLOR1="forestgreen"
DETECTOR_COLOR2="red"
DETECTOR_COLOR3="darkgoldenrod"
DETECTOR_COLOR4="dodgerblue"
NOISE_DENSITY_COLOR = "darkorange"


# Titles
DATAGRAM_STREAM_TITLE = "Datagram Stream"
I_CHANNEL_TITLE = "$I$ Channel Stream"
Q_CHANNEL_TITLE = "$Q$ Channel Stream"
INPUT_STREAM_TITLE = "Input Stream"
OUTPUT_STREAM_TITLE = "Output Stream"
IMPULSE_TITLE = "$g(t)$ Impulse Response"
MF_IMPULSE_TITLE = "$g(-t)$ Impulse Response"
LPF_IMPULSE_TITLE = "$h(t)$ Impulse Response"
LPF_FREQ_TITLE = "$H(f)$ Frequency Response"
LPF_PZ_TITLE = "$h(t)$ Poles and Zeros"
MODULATED_STREAM_TITLE = "Modulated Signal $s(t)$"
IQ_COMPONENTS_TITLE = "$IQ$ Components Signals"
PHASE_TITLE = "Modulated Signal $s(t)$ Phase"
IQ_CONSTELLATION_TITLE = "$IQ$ Constellation"
WATERFALL_TITLE = "Received Signal $s(t) + r(t)$ Waterfall"
WATERFALL_DETECTION_TITLE = "Detection of $s(t)$ Waterfall"
WATERFALL_DECISION_TITLE = "Decision of $s(t)$ Waterfall"
SYNC_CORR_TITLE = "Correlation of $s(t)$ with Sync Signal"
NOISE_DENSITY_TITLE = "Noise Density"
NOISE_TITLE = "Noise Signal $r(t)$"