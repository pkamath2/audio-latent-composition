import numpy as np
from scipy.signal import butter, lfilter

####################################################### Impacts and Scraping Model######################################################################

# Gaver Method 2 (See Paper)
def get_gaver_sounds(initial_amplitude, impulse_time, filters, total_time=2, locs=None, \
                             sample_rate=16000, hittype='hit', 
                             damping_mult=None, damping_fade_expo=None, 
                             filter_order=None):
    
    
    y_scratch = np.random.rand(int(impulse_time*sample_rate))
    
    #20%
    start_t = 0.0
    end_t = 0.05*impulse_time
    y1 = initial_amplitude*y_scratch[int(start_t*sample_rate):int(end_t*sample_rate)]
    y1 = 20*butter_bandpass_filter(y1, lowcut=filters[0][0], highcut=filters[0][1], fs=sample_rate, order=2, btype='bandpass')
    y1 = applyFBFadeFilter(forward_fadetime=0,backward_fadetime=0.1*(end_t-start_t),signal=y1,fs=sample_rate, expo=1)
    y1 = np.pad(y1, (int(start_t*sample_rate),len(y_scratch)-int(end_t*sample_rate)), mode='constant')
    
    #Remaining 80%
    start_t = 0.05*impulse_time
    end_t = 1.0*impulse_time
    y2 = initial_amplitude*y_scratch[int(start_t*sample_rate):int(end_t*sample_rate)]
    if not filter_order:
        filter_order = 1
    y2 = 10*butter_bandpass_filter(y2, lowcut=filters[1][0], highcut=filters[1][1], fs=sample_rate, order=filter_order, btype='bandpass')
    if not damping_mult:
        damping_mult = 0.1
        damping_fade_expo = 1
    y2 = applyFBFadeFilter(forward_fadetime=0,backward_fadetime=damping_mult*(end_t-start_t),signal=y2,fs=sample_rate, expo=damping_fade_expo)
    y2 = np.pad(y2, (int(start_t*sample_rate),len(y_scratch)-int(end_t*sample_rate)), mode='constant')
    
    
    y_scratch = y1+y2
    
    signal_mult = 0.00005
    if hittype == 'scratch':
        signal_mult = 0.0005
    signal = signal_mult*np.random.randn(int(total_time*sample_rate))
    
    for loc in locs:
        start_loc = int(loc*sample_rate)
        end_loc = start_loc+len(y_scratch)
        y_scratch_ = y_scratch

        if end_loc > len(signal):
            end_loc = len(signal)
            y_scratch_ = y_scratch_[0:end_loc-start_loc]

        signal[start_loc:end_loc] = y_scratch_
    return signal/np.max(signal)


# Gaver Method 1 (See Paper)
#phi: Initial amplitude - relates to mallet hardness; force or proximity
#delta: Damping - relates to material (0=Full cosine no damping; higher values more damping;)
#omega: partial frequencies - relates to size, configuration (high pos/neg values are tinnier; values around zero are woody;)

def gaver_impact_plate(initial_amplitude, impulse_time, num_partials, size='small', metal_or_plastic='metal', samplerate=16000, total_time=0, locs=None):
    single_hit = gaver_impact_plate_single(initial_amplitude, impulse_time, num_partials, size, metal_or_plastic, samplerate)
    gn=np.zeros(int(total_time*samplerate))
    for loc in locs:
        start_loc = int(loc*samplerate)
        end_loc = start_loc+len(single_hit)
        single_hit_ = single_hit

        if end_loc > len(gn):
            end_loc = len(gn)
            single_hit_ = single_hit_[0:end_loc-start_loc]

        gn[start_loc:end_loc] = applyFBFadeFilter(forward_fadetime=0.2,backward_fadetime=0,signal=single_hit_,fs=samplerate)

    return gn/np.max(gn)



def gaver_impact_plate_single(initial_amplitude, impulse_time, num_partials, size='small', metal_or_plastic='metal', samplerate=16000):
    t_ = np.linspace(0,impulse_time,int(impulse_time*samplerate))
    
    omega_1 = 60 #In Hz. Frequency is inversely proportional to size
    if size == 'small':
        omega_1 = 250
        
    D = 0.001
    if metal_or_plastic == 'plastic':
        D = 0.3
    
    h=[h_ for h_ in np.arange(-0.005, 0.005, (0.005+0.005)/num_partials)][::-1]
    
    gn=np.zeros(int(impulse_time*samplerate))
    for i in range(num_partials):
        omega_n = omega_1 + i*omega_1 + np.random.randint(50)
        omega_n_radians = 2 * np.pi * omega_n
        
        phi = initial_amplitude + h[i]*(omega_n_radians - (2*np.pi*omega_1))
        
        delta_n = omega_n_radians * D
        g = phi * np.exp(-1*delta_n*t_) * np.cos(omega_n_radians * t_)
        gn = gn + g
    return gn



# Gaver Method 1 (See Paper)
#phi: Initial amplitude - relates to mallet hardness; force or proximity
#delta: Damping - relates to material (0=Full cosine no damping; higher values more damping;)
#omega: partial frequencies - relates to size, configuration (high pos/neg values are tinnier; values around zero are woody;)

def gaver_impact_solidbar(initial_amplitude, impulse_time, num_partials, size='small', metal_or_plastic='metal', samplerate=16000, total_time=0, locs=None):
    single_hit = gaver_impact_solidbar_single(initial_amplitude, impulse_time, num_partials, size, metal_or_plastic, samplerate)
    gn=np.zeros(int(total_time*samplerate))
    for loc in locs:
        start_loc = int(loc*samplerate)
        end_loc = start_loc+len(single_hit)
        single_hit_ = single_hit

        if end_loc > len(gn):
            end_loc = len(gn)
            single_hit_ = single_hit_[0:end_loc-start_loc]

        gn[start_loc:end_loc] = applyFBFadeFilter(forward_fadetime=0.05,backward_fadetime=0,signal=single_hit_,fs=samplerate)

    return gn/np.max(gn)




def gaver_impact_solidbar_single(initial_amplitude, impulse_time, num_partials, size='small', metal_or_plastic='metal', samplerate=16000):
    t_ = np.linspace(0,impulse_time,int(impulse_time*samplerate))
    
    omega_1 = 240 #In Hz. Frequency is inversely proportional to size
    if size == 'small':
        omega_1 = 660
        
    D = 0.001
    if metal_or_plastic == 'plastic':
        D = 0.5
    
    h=[h_ for h_ in np.arange(-0.005, 0.005, (0.005+0.005)/num_partials)][::-1]
    
    gn=np.zeros(int(impulse_time*samplerate))
    for i in range(num_partials):
        omega_n_radians = omega_1*((2*i + 1)**2)/9
        
        phi = initial_amplitude + h[i]*(omega_n_radians - (2*np.pi*omega_1))
        
        delta_n = omega_n_radians * D
        g = phi * np.exp(-1*delta_n*t_) * np.cos(omega_n_radians * t_)
        gn = gn + g
    return gn


#################################################################################################################################################

####################################################### Water Filling Model######################################################################

#phi: Initial amplitude - relates to mallet hardness; force or proximity
#delta: Damping - relates to material (0=Full cosine no damping; higher values more damping;)
#omega: partial frequencies - relates to size, configuration (high pos/neg values are tinnier; values around zero are woody;)

def gaver_impact(phi, delta, omega, t, samplerate, add_variance=False):
    t_ = np.arange(t*samplerate)
    
    omega_ = omega
    if add_variance == True:
        omega_ = omega + np.random.uniform(-0.01, 0.01)
    g = phi * np.exp(-1*delta*t_) * np.cos(omega_ * t_)
    
    return g


#each level is modelled as combination of 3 omegas of drips
def model_water_filling(num_secs, level=0, drip_secs=0.1, sample_rate=16000):
    y = np.zeros(num_secs*sample_rate)
    
    if level==0:
        start_drip = 0.55
        for i in range(int(num_secs/drip_secs)):
            omega_ = 0.45 + i*0.002
            drip_NN1 = gaver_impact(phi=(2-i*0.01), delta=0.003, omega=omega_, t=drip_secs, samplerate=sample_rate, add_variance=True)
            omega__ = 0.45 - i*0.005
            drip_NN2 = gaver_impact(phi=(2-i*0.01), delta=0.003, omega=omega__, t=drip_secs, samplerate=sample_rate, add_variance=True)
            
            drip = gaver_impact(phi=1, delta=0.003, omega=0.85, t=drip_secs, samplerate=sample_rate, add_variance=True)
            drip_2 = gaver_impact(phi=0.5, delta=0.003, omega=0.1, t=drip_secs, samplerate=sample_rate, add_variance=True)
            drip_3 = gaver_impact(phi=0.2, delta=0.003, omega=0.85, t=drip_secs, samplerate=sample_rate, add_variance=True)
            
            y[int(i*drip_secs*sample_rate):int(i*drip_secs*sample_rate+(drip_secs*sample_rate))] = drip_NN1 +drip_NN2 +drip +drip_2
    
    if level==1:
        start_drip = 0.55
        for i in range(int(num_secs/drip_secs)):
            omega_ = 0.6 - i*0.005
            drip_NN1 = gaver_impact(phi=(2-i*0.01), delta=0.003, omega=omega_, t=drip_secs, samplerate=sample_rate, add_variance=True)
            omega__ = 0.85 - i*0.005
            drip_NN2 = gaver_impact(phi=(2-i*0.01), delta=0.003, omega=omega__, t=drip_secs, samplerate=sample_rate, add_variance=True)
            
            drip = gaver_impact(phi=0.2, delta=0.003, omega=0.1, t=drip_secs, samplerate=sample_rate, add_variance=True)
            drip_2 = gaver_impact(phi=0.5, delta=0.003, omega=0.45, t=drip_secs, samplerate=sample_rate, add_variance=True)
            drip_3 = gaver_impact(phi=0.2, delta=0.003, omega=0.85, t=drip_secs, samplerate=sample_rate, add_variance=True)
            
            y[int(i*drip_secs*sample_rate):int(i*drip_secs*sample_rate+(drip_secs*sample_rate))] = drip + drip_2 + drip_3  + drip_NN1 + drip_NN2 #+g
            
    if level==3:
        for i in range(int(num_secs/drip_secs)):
            drip = gaver_impact(phi=1, delta=0.003, omega=0.1, t=drip_secs, samplerate=sample_rate, add_variance=True)
            drip_2 = gaver_impact(phi=0.5, delta=0.003, omega=0.55, t=drip_secs, samplerate=sample_rate, add_variance=True)
            drip_3 = gaver_impact(phi=0.2, delta=0.003, omega=0.85, t=drip_secs, samplerate=sample_rate, add_variance=True)
            drip_4 = gaver_impact(phi=0.1, delta=0.003, omega=1.2, t=drip_secs, samplerate=sample_rate, add_variance=True)
            y[int(i*drip_secs*sample_rate):int(i*drip_secs*sample_rate+(drip_secs*sample_rate))] = drip + drip_2 + drip_3 + drip_4
            
    
    if level==5:
        for i in range(int(num_secs/drip_secs)):
            drip = gaver_impact(phi=1, delta=0.003, omega=0.1, t=drip_secs, samplerate=sample_rate, add_variance=True)
            drip_2 = gaver_impact(phi=1, delta=0.003, omega=0.15, t=drip_secs, samplerate=sample_rate, add_variance=True)
            drip_3 = gaver_impact(phi=0.7, delta=0.003, omega=0.85, t=drip_secs, samplerate=sample_rate, add_variance=True)
            y[int(i*drip_secs*sample_rate):int(i*drip_secs*sample_rate+(drip_secs*sample_rate))] = drip + drip_2 + drip_3 
            
    
    if level==10:
        for i in range(int(num_secs/(drip_secs))):
            
            omega_ = 0.55 + i*0.01
            drip_NN1 = gaver_impact(phi=1, delta=0.003, omega=omega_, t=drip_secs, samplerate=sample_rate, add_variance=True)
            
            y[int(i*drip_secs*sample_rate):int(i*drip_secs*sample_rate+(drip_secs*sample_rate))] = drip_NN1

            
    return y

#################################################################################################################################################


# Modified - Original From Chitra
def applyFBFadeFilter(forward_fadetime,backward_fadetime,signal,fs,expo=1):
    forward_num_fad_samp = int(forward_fadetime*fs) 
    backward_num_fad_samp = int(backward_fadetime*fs) 
    signal_length = len(signal) 
    fadefilter = np.ones(signal_length)
    if forward_num_fad_samp>0:
        fadefilter[0:forward_num_fad_samp]=np.linspace(0,1,forward_num_fad_samp)**expo
    if backward_num_fad_samp>0:
        fadefilter[signal_length-backward_num_fad_samp:signal_length]=np.linspace(1,0,backward_num_fad_samp)**expo
    return fadefilter*signal


def butter_bandpass(lowcut, highcut, fs, order=5,btype='bandpass'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    return b, a

def butter_lowhighpass(cut, fs, order=5, btype='lowpass'):
    nyq = 0.5 * fs
    cut = cut / nyq
    b, a = butter(order, cut, btype=btype)
    return b, a

def butter_bandpass_filter(data, highcut, fs,lowcut=None,  order=5, btype='bandpass'):
    if btype=='bandpass':
        b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype=btype)
    else:
        b, a = butter_lowhighpass(highcut, fs, order=order, btype=btype)
    y = lfilter(b, a, data)
    return y
