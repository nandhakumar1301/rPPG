#!/usr/bin/env python
# coding: utf-8

# In[26]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
fs=360 # sample freq

heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
detectors = Detectors(fs)
print(len(heartbeat))
print(heartbeat.shape)

r_peaks_pan = detectors.pan_tompkins_detector(heartbeat.iloc[:,1][0:80])
print(r_peaks_pan)

plt.plot(heartbeat.iloc[:,1][0:199])
plt.plot(r_peaks_pan,heartbeat.iloc[:,1][0:199][r_peaks_pan], 'ro')


# In[27]:


def R_correction(signal, peaks):
    if isinstance(peaks, list):
        peaks = np.array(peaks)
    num_peak=peaks.shape[0]
    peaks_corrected_list=list()
    for index in range(num_peak):
        peak=peaks[index]
        if index==0:
            peak_diff=peak
        else:
            peak_diff=peak-peaks[index-1]
        if peak_diff>160:
            peak_corrected=signal[peak-140:peak+141].argmax()+peak-140
            peaks_corrected_list.append(peak_corrected)
    peaks_corrected=np.asarray(peaks_corrected_list)            
    return peaks_corrected

corrected_R_peak=R_correction(heartbeat.iloc[:,1][0:199],r_peaks_pan)
plt.plot(heartbeat.iloc[:,1][0:199])
plt.plot(corrected_R_peak,heartbeat.iloc[:,1][0:199][corrected_R_peak], 'ro')


# In[28]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
fs=360 # sample freq

heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
detectors = Detectors(fs)
print(len(heartbeat))
print(heartbeat.shape)

peaks_pan = detectors.hamilton_detector(heartbeat.iloc[:,1][0:199])
print(peaks_pan)

plt.plot(heartbeat.iloc[:,1][0:199])
plt.plot(peaks_pan,heartbeat.iloc[:,1][0:199][peaks_pan], 'ro')


# In[29]:


def R_correction(signal, peaks):
    if isinstance(peaks, list):
        peaks = np.array(peaks)
    num_peak=peaks.shape[0]
    peaks_corrected_list=list()
    for index in range(num_peak):
        peak=peaks[index]
        if index==0:
            peak_diff=peak
        else:
            peak_diff=peak-peaks[index-1]
        if peak_diff>160:
            peak_corrected=signal[peak-140:peak+141].argmax()+peak-140
            peaks_corrected_list.append(peak_corrected)
    peaks_corrected=np.asarray(peaks_corrected_list)            
    return peaks_corrected

corrected_R_peak=R_correction(heartbeat.iloc[:,1][0:199],peaks_pan)
plt.plot(heartbeat.iloc[:,1][0:199])
plt.plot(corrected_R_peak,heartbeat.iloc[:,1][0:199][corrected_R_peak], 'ro')


# In[36]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fs = 360 # sample freq

heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
detectors = Detectors(fs)
print(len(heartbeat))
print(heartbeat.shape)

# Calculate ECG gain
max_voltage = max(heartbeat.iloc[:,1]) - min(heartbeat.iloc[:,1])
input_voltage_range = 20 # Assuming the input signal has a voltage range of +/- 10 mV
ecg_gain = max_voltage / input_voltage_range
print("ECG gain:", ecg_gain)

r_peaks_pan = detectors.pan_tompkins_detector(heartbeat.iloc[:,1][0:199])
print(r_peaks_pan)

plt.plot(heartbeat.iloc[:,1][0:199])
plt.plot(r_peaks_pan,heartbeat.iloc[:,1][0:199][r_peaks_pan], 'ro')


# In[37]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fs = 360 # sample freq

heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
detectors = Detectors(fs)
print(len(heartbeat))
print(heartbeat.shape)

# Calculate ECG gain
max_voltage = max(heartbeat.iloc[:,1]) - min(heartbeat.iloc[:,1])
input_voltage_range = 20 # Assuming the input signal has a voltage range of +/- 10 mV
ecg_gain = max_voltage / input_voltage_range
print("ECG gain:", ecg_gain)

r_peaks_pan = detectors.christov_detector(heartbeat.iloc[:,1][0:199])
print(r_peaks_pan)

plt.plot(heartbeat.iloc[:,1][0:199])
plt.plot(r_peaks_pan,heartbeat.iloc[:,1][0:199][r_peaks_pan], 'ro')


# In[38]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fs = 360 # sample freq

heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
detectors = Detectors(fs)
print(len(heartbeat))
print(heartbeat.shape)

# Calculate ECG gain
max_voltage = max(heartbeat.iloc[:,1]) - min(heartbeat.iloc[:,1])
input_voltage_range = 20 # Assuming the input signal has a voltage range of +/- 10 mV
ecg_gain = max_voltage / input_voltage_range
print("ECG gain:", ecg_gain)

r_peaks_pan = detectors.hamilton_detector(heartbeat.iloc[:,1][0:199])
print(r_peaks_pan)

plt.plot(heartbeat.iloc[:,1][0:199])
plt.plot(r_peaks_pan,heartbeat.iloc[:,1][0:199][r_peaks_pan], 'ro')


# In[39]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fs = 360 # sample freq

heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
detectors = Detectors(fs)
print(len(heartbeat))
print(heartbeat.shape)

# Calculate ECG gain
max_voltage = max(heartbeat.iloc[:,1]) - min(heartbeat.iloc[:,1])
input_voltage_range = 20 # Assuming the input signal has a voltage range of +/- 10 mV
ecg_gain = max_voltage / input_voltage_range
print("ECG gain:", ecg_gain)

r_peaks_pan = detectors.swt_detector(heartbeat.iloc[:,1][0:199])
print(r_peaks_pan)

plt.plot(heartbeat.iloc[:,1][0:199])
plt.plot(r_peaks_pan,heartbeat.iloc[:,1][0:199][r_peaks_pan], 'ro')


# In[47]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fs = 360 # sample freq

heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
detectors = Detectors(fs)
print(len(heartbeat))
print(heartbeat.shape)

# Calculate ECG gain
max_voltage = max(heartbeat.iloc[:,1]) - min(heartbeat.iloc[:,1])
input_voltage_range = 20 # Assuming the input signal has a voltage range of +/- 10 mV
ecg_gain = max_voltage / input_voltage_range
print("ECG gain:", ecg_gain)

r_peaks_pan = detectors.two_average_detector(heartbeat.iloc[:,1][0:900])
print(r_peaks_pan)

plt.plot(heartbeat.iloc[:,1][0:900])
plt.plot(r_peaks_pan,heartbeat.iloc[:,1][0:900][r_peaks_pan], 'ro')


# In[40]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fs = 360 # sample freq

heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
detectors = Detectors(fs)
print(len(heartbeat))
print(heartbeat.shape)

# Calculate ECG gain
max_voltage = max(heartbeat.iloc[:,1]) - min(heartbeat.iloc[:,1])
input_voltage_range = 20 # Assuming the input signal has a voltage range of +/- 10 mV
ecg_gain = max_voltage / input_voltage_range
print("ECG gain:", ecg_gain)

r_peaks_pan = detectors.matched_filter_detector(heartbeat.iloc[:,1][0:199])
print(r_peaks_pan)

plt.plot(heartbeat.iloc[:,1][0:199])
plt.plot(r_peaks_pan,heartbeat.iloc[:,1][0:199][r_peaks_pan], 'ro')


# In[10]:


import pandas as pd

# Load ECG signal data
heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")

# Calculate signal amplitude
signal_amplitude = max(heartbeat.iloc[:,1]) - min(heartbeat.iloc[:,1])
print("Signal amplitude:", signal_amplitude)


# In[12]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Load ECG signal data and ground truth data
heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
ground_truth = pd.read_csv("C:/videopp/dew ecg.csv")

fs = 100 # sample freq

# Initialize detector object
detectors = Detectors(fs)

# Detect R-peaks using Pan-Tompkins algorithm
r_peaks_pan = detectors.pan_tompkins_detector(heartbeat.iloc[:,1])

# Convert ground truth data to list of R-peak locations
r_peaks_gt = detectors.pan_tompkins_detector(ground_truth.iloc[:,1])
print(len(r_peaks_pan))
print(r_peaks_pan)
print(len(r_peaks_gt))
print(r_peaks_gt)



# In[13]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Load ECG signal data and ground truth data
heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
ground_truth = pd.read_csv("C:/videopp/dew ecg.csv")

fs = 100 # sample freq

# Initialize detector object
detectors = Detectors(fs)

# Detect R-peaks using Pan-Tompkins algorithm
r_peaks_pan = detectors.christov_detector(heartbeat.iloc[:,1])

# Convert ground truth data to list of R-peak locations
r_peaks_gt = detectors.christov_detector(ground_truth.iloc[:,1])
print(len(r_peaks_pan))
print(r_peaks_pan)
print(len(r_peaks_gt))
print(r_peaks_gt)


# In[14]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Load ECG signal data and ground truth data
heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
ground_truth = pd.read_csv("C:/videopp/dew ecg.csv")

fs = 100 # sample freq

# Initialize detector object
detectors = Detectors(fs)

# Detect R-peaks using Pan-Tompkins algorithm
r_peaks_pan = detectors.hamilton_detector(heartbeat.iloc[:,1])

# Convert ground truth data to list of R-peak locations
r_peaks_gt = detectors.hamilton_detector(ground_truth.iloc[:,1])
print(len(r_peaks_pan))
print(r_peaks_pan)
print(len(r_peaks_gt))
print(r_peaks_gt)


# In[15]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Load ECG signal data and ground truth data
heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
ground_truth = pd.read_csv("C:/videopp/dew ecg.csv")

fs = 100 # sample freq

# Initialize detector object
detectors = Detectors(fs)

# Detect R-peaks using Pan-Tompkins algorithm
r_peaks_pan = detectors.swt_detector(heartbeat.iloc[:,1])

# Convert ground truth data to list of R-peak locations
r_peaks_gt = detectors.swt_detector(ground_truth.iloc[:,1])
print(len(r_peaks_pan))
print(r_peaks_pan)
print(len(r_peaks_gt))
print(r_peaks_gt)


# In[16]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Load ECG signal data and ground truth data
heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
ground_truth = pd.read_csv("C:/videopp/dew ecg.csv")

fs = 100 # sample freq

# Initialize detector object
detectors = Detectors(fs)

# Detect R-peaks using Pan-Tompkins algorithm
r_peaks_pan = detectors.two_average_detector(heartbeat.iloc[:,1])

# Convert ground truth data to list of R-peak locations
r_peaks_gt = detectors.two_average_detector(ground_truth.iloc[:,1])
print(len(r_peaks_pan))
print(r_peaks_pan)
print(len(r_peaks_gt))
print(r_peaks_gt)


# In[17]:


from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Load ECG signal data and ground truth data
heartbeat = pd.read_csv("C:/videopp/dew ecg.csv")
ground_truth = pd.read_csv("C:/videopp/dew ecg.csv")

fs = 360 # sample freq

# Initialize detector object
detectors = Detectors(fs)

# Detect R-peaks using Pan-Tompkins algorithm
r_peaks_pan = detectors.matched_filter_detector(heartbeat.iloc[:,1])

# Convert ground truth data to list of R-peak locations
r_peaks_gt = detectors.matched_filter_detector(ground_truth.iloc[:,1])
print(len(r_peaks_pan))
print(r_peaks_pan)
print(len(r_peaks_gt))
print(r_peaks_gt)


# In[ ]:




