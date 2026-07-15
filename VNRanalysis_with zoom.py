# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 12:41:58 2026

@author: Ajay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import curve_fit
from tkinter import Tk, filedialog
import matplotlib
import json
import os

# Parameters
SR = 10000.0         # Sample Rate (Hz)
WINDOW_MS = 10       # 10ms window for STD
TAU_MS = 100.0       # 100ms Time Constant for Swim Strength
BANDPASS_LOW = 300   # Hz  Lower cutoff frequency for bandpass filter
BANDPASS_HIGH = 1000 # Hz  Upper cutoff frequency for bandpass filter

MIN_PEAK_DISTANCE_MS = 10   # Minimum time between detected peaks (in ms)
BOUT_GAP_S = 0.2            # Maximum inter-burst interval to be considered part of the same bout (in sec)
MIN_BOUT_BURSTS = 2         # Minimum number of bursts required to define a swimming bout

    
# TASK PHASE DEFINITIONS 
TASK_PHASES = {
    "forward_to_right": {
        "cycle_duration": 60,  # seconds
        "phases": [
            {"name": "Rest", "start": 0, "end": 10, "angle": 0, "velocity": 0.0},
            {"name": "Forward", "start": 10, "end": 20, "angle": 0, "velocity": 2.0},
            {"name": "Right", "start": 20, "end": 30, "angle": 90, "velocity": 2.0},
            {"name": "Forward", "start": 30, "end": 40, "angle": 0, "velocity": 2.0},
            {"name": "Rest", "start": 40, "end": 60, "angle": 0, "velocity": 0.0}]},
    
    "forward_to_left": {
        "cycle_duration": 60,
        "phases": [
            {"name": "Rest", "start": 0, "end": 10, "angle": 0, "velocity": 0.0},
            {"name": "Forward", "start": 10, "end": 20, "angle": 0, "velocity": 2.0},
            {"name": "Left", "start": 20, "end": 30, "angle": -90, "velocity": 2.0},
            {"name": "Forward", "start": 30, "end": 40, "angle": 0, "velocity": 2.0},
            {"name": "Rest", "start": 40, "end": 60, "angle": 0, "velocity": 0.0}]},
    
    "alternating_velocity": {
        "cycle_duration": 60,
        "phases": [
            {"name": "Rest", "start": 0, "end": 10, "velocity": 0.0},
            {"name": "Fast (v=3)", "start": 10, "end": 20, "velocity": 3.0},
            {"name": "Slow (v=1)", "start": 20, "end": 30, "velocity": 1.0},
            {"name": "Rest", "start": 30, "end": 40, "velocity": 0.0},
            {"name": "Slow (v=1)", "start": 40, "end": 50, "velocity": 1.0},
            {"name": "Fast (v=3)", "start": 50, "end": 60, "velocity": 3.0}]},
    
    "rest_swim_cycles": {
        "cycle_duration": 30,
        "phases": [
            {"name": "Rest", "duration": 10, "velocity": 0.0},
            {"name": "Swim", "duration": 20, "velocity": 2.0}]},
    
    "directional_bias": {
        "cycle_duration": 90,
        "phases": [
            {"name": "Forward", "start": 0, "end": 10, "angle": 0},
            {"name": "Diagonal Right (45)", "start": 10, "end": 20, "angle": 45},
            {"name": "Right (90)", "start": 20, "end": 30, "angle": 90},
            {"name": "Diagonal Right (45)", "start": 30, "end": 40, "angle": 45},
            {"name": "Forward", "start": 40, "end": 50, "angle": 0},
            {"name": "Diagonal Left (-45)", "start": 50, "end": 60, "angle": -45},
            {"name": "Left (-90)", "start": 60, "end": 70, "angle": -90},
            {"name": "Diagonal Left (-45)", "start": 70, "end": 80, "angle": -45},
            {"name": "Forward", "start": 80, "end": 90, "angle": 0}]},
    
    "alternating_right_left": {
        "cycle_duration": 90,
        "phases": [
            {"name": "Rest", "start": 0, "end": 10, "angle": 0, "velocity": 0.0},
            {"name": "Forward", "start": 10, "end": 20, "angle": 0, "velocity": 2.0},
            {"name": "Right", "start": 20, "end": 30, "angle": 90, "velocity": 2.0},
            {"name": "Forward", "start": 30, "end": 40, "angle": 0, "velocity": 2.0},
            {"name": "Rest", "start": 40, "end": 60, "angle": 0, "velocity": 0.0},
            {"name": "Forward", "start": 60, "end": 70, "angle": 0, "velocity": 2.0},
            {"name": "Left", "start": 70, "end": 80, "angle": -90, "velocity": 2.0},
            {"name": "Forward", "start": 80, "end": 90, "angle": 0, "velocity": 2.0}]},

    "dark_flash": {
        "cycle_duration": 20,  # 19s white + 1s dark
        "phases": [
            {"name": "White", "duration": 19, "phase_type": "white"},
            {"name": "Dark Flash", "duration": 1, "phase_type": "dark"}]}}

def gaussian(x, a, x0, sigma):

    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) #Gaussian function for curve fitting


def load_data():

    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    filepath = filedialog.askopenfilename(title="Select Data File", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    
    root.destroy()
    
    if not filepath:
        return None, None
    
    # Load CSV
    data = pd.read_csv(filepath)
    
    # Load metadata (same name but .json extension)
    metadata_path = filepath.replace('.csv', '.json')
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return data, filepath, metadata


def bandpass_filter(signal, sample_rate, low_freq, high_freq, order=2):

    nyq = 0.5 * sample_rate
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    
    # filter
    filtered = filtfilt(b, a, signal) # filtfilt applies the filter twice: forward then backward.
    
    return filtered


def calculate_std_dev(signal, sample_rate, window_ms):

    window_samples = int(sample_rate * window_ms / 1000) # Number of samples in 10ms window = 10000*10/1000 = 100
    
    # Standard deviation over a moving 10ms window
    std_signal = pd.Series(signal).rolling(window=window_samples, center=True, min_periods=1).std().values
    
    # Smooth with moving mean
    std_signal = uniform_filter1d(std_signal, size=window_samples, mode='nearest') #mode='nearest' - Edge handling (repeat edge values)
    
    return std_signal


def calculate_threshold(std_signal, n_bins=200):

    # Create histogram
    counts, bin_edges = np.histogram(std_signal, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    try:
        
        # Fit Gaussian to histogram
        fitted_params, _ = curve_fit(gaussian, bin_centers, counts, maxfev=5000)
        
        # Extract fitted values  
        mean = fitted_params[1] # amplitude = fitted_params[0]
        sigma = fitted_params[2]
        
        # Calculate threshold (mean + 3*sigma)
        threshold = mean + 3 * abs(sigma)
        # VALIDATION: Threshold must be within signal range
        signal_max = np.max(std_signal)
        signal_min = np.min(std_signal)
        
        if threshold > signal_max or threshold < signal_min:
            print(f"Gaussian fit gave invalid threshold ({threshold:.2f}), using fallback")
            threshold = np.median(std_signal) + 3 * np.std(std_signal)
        
    except Exception:
        print(f"Gaussian fit gave invalid threshold ({threshold:.2f}), using fallback")
        threshold = np.median(std_signal) + 3 * np.std(std_signal)
    return threshold



def smoothing_swim_strength(input_signal, decay_factor):

    output = np.zeros_like(input_signal)
    current_val = 0.0
    for i in range(len(input_signal)):
        current_val = current_val * decay_factor + input_signal[i]
        output[i] = current_val
    return output


def calculate_swim_strength(std_signal, threshold, sample_rate, tau_ms):

    gated_signal = np.where(std_signal > threshold, std_signal - threshold, 0)
    
    dt = 1.0 / sample_rate     # Time per sample = 1/10000 sec
    tau_sec = tau_ms / 1000.0  # Time constant (100ms) 
    decay = np.exp(-dt / tau_sec) # Decay factor = 0.999
    
    swim_strength = smoothing_swim_strength(gated_signal, decay)
    
    return swim_strength


def detect_bursts(std_signal, threshold, sample_rate):
    min_distance = int(sample_rate * MIN_PEAK_DISTANCE_MS / 1000)
    
    peaks, _ = find_peaks(std_signal, height=threshold, distance=min_distance)
    
    return peaks


def segment_bouts(peak_indices, sample_rate):
    if len(peak_indices) < MIN_BOUT_BURSTS:
        return []
    
    peak_times = peak_indices / sample_rate
    ibi = np.diff(peak_times)
    
    bout_breaks = np.where(ibi > BOUT_GAP_S)[0]
    
    bout_starts = np.concatenate([[0], bout_breaks + 1])
    bout_ends = np.concatenate([bout_breaks, [len(peak_indices) - 1]])
    
    bouts = []
    for start, end in zip(bout_starts, bout_ends):
        burst_count = end - start + 1
        
        if burst_count >= MIN_BOUT_BURSTS:
            bout = {
                'start_idx': peak_indices[start],
                'end_idx': peak_indices[end],
                'burst_count': burst_count}
            bouts.append(bout)
    
    return bouts


def calculate_bout_metrics(bouts, sample_rate):
    metrics = []
    
    for bout in bouts:
        start_time = bout['start_idx'] / sample_rate
        end_time = bout['end_idx'] / sample_rate
        duration = end_time - start_time
        burst_count = bout['burst_count']
        
        if duration > 0:
            frequency = burst_count / duration
        else:
            frequency = 0
        
        metrics.append({'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'burst_count': burst_count,
            'frequency': frequency})
    
    return metrics


def get_task_name_from_metadata(metadata):

    if metadata is None:
        return None
    
    if 'task_name' in metadata:
        return metadata['task_name']

    
    return None


def generate_phase_regions(task_name, total_duration):
    
    task_info = TASK_PHASES[task_name]
    regions = []
    cycle_duration = task_info["cycle_duration"]
    phases = task_info["phases"]
    
    current_cycle_start = 0
    
    while current_cycle_start < total_duration:
        current_time = current_cycle_start
        
        for phase in phases:
            # Determine start and end based on phase format
            if "start" in phase and "end" in phase:
                # Format: {"start": 0, "end": 10, ...}
                start = current_cycle_start + phase["start"]
                end = current_cycle_start + phase["end"]
            elif "duration" in phase:
                # Format: {"duration": 10, ...}
                start = current_time
                end = current_time + phase["duration"]
                current_time = end
            else:
                continue
            
            # Skip if past total duration
            if start >= total_duration:
                break
            
            # Clamp end to total duration
            end = min(end, total_duration)
            
            regions.append({
                "start": start,
                "end": end,
                "name": phase["name"],
                "info": phase})
        
        current_cycle_start += cycle_duration
    
    return regions


def add_task_overlay(ax, task_name, total_duration, y_min, y_max):

    
    regions = generate_phase_regions(task_name, total_duration)
    
    if not regions:
        return
    
    # Special handling for dark_flash task
    if task_name == "dark_flash":
        for region in regions:
            phase_type = region["info"].get("phase_type", "white")
            
            if phase_type == "white":
                color = "#ffffff"
                alpha = 0.1
            else:
                # Dark phase: light grey color
                color = "#d3d3d3"  # Light grey
                alpha = 0.7
            
            rect = Rectangle(
                (region["start"], y_min),
                region["end"] - region["start"],
                y_max - y_min,
                facecolor=color,
                alpha=alpha,
                edgecolor='none', zorder=0)
            
            ax.add_patch(rect)
        
        # No labels for dark_flash task
        return

    # Alternating_right_left task
    if task_name == "alternating_right_left":
        # 4 distinct colors: Rest, Forward, Right, Left
        for region in regions:
            phase_name = region["name"]
            angle = region["info"].get("angle", 0)
            velocity = region["info"].get("velocity", 0)
            
            # Determine color based on phase
            if velocity == 0.0 or "Rest" in phase_name:
                # Rest - light grey
                color = "#d3d3d3"
                alpha = 0.4
                label_color = "#34495e"  # Grey for rest label
            elif angle == 0:
                # Forward - off white
                color = "#FEFFED"
                alpha = 0.8
                label_color = "#34495e"  # Dark grey
            elif angle == 90:
                # Right - green
                color = "#1abc9c"
                alpha = 0.15
                label_color = color
            else:  # angle == -90
                # Left - blue
                color = "#3498db"
                alpha = 0.15
                label_color = color
            
            rect = Rectangle(
                (region["start"], y_min),
                region["end"] - region["start"],
                y_max - y_min,
                facecolor=color,
                alpha=alpha,
                edgecolor='none', zorder=0)
            
            ax.add_patch(rect)
            
            # Add phase label
            mid_x = (region["start"] + region["end"]) / 2
            ax.text(mid_x, y_max * 0.98, region["name"],
                    ha='center', va='top', fontsize=10,
                    color=label_color, fontweight='bold', alpha=0.9)
        
        return
    
    # Special handling for directional_bias task
    if task_name == "directional_bias":
        # Right side: Red with varying alphas
        # Left side: Blue with varying alphas
        
        for region in regions:
            angle = region["info"].get("angle", 0)
            
            if angle == 0:
                # Forward - off white
                color = "#FEFFED"
                alpha = 0.8
                label_color = "#34495e"  # Dark grey for label
            elif angle > 0:
                # Right side (45, 90) - Green with varying alphas
                color = "#1abc9c"
                label_color = color
                if angle == 45:
                    alpha = 0.1  # Diagonal right
                else:  # angle == 90
                    alpha = 0.2  # Right
            else:
                # Left side (-45, -90) - Blue with varying alphas
                color = "#3498db"
                label_color = color
                if angle == -45:
                    alpha = 0.10  # Diagonal left 
                else:  # angle == -90
                    alpha = 0.2  # Left 
            
            rect = Rectangle(
                (region["start"], y_min),
                region["end"] - region["start"],
                y_max - y_min,
                facecolor=color,
                alpha=alpha,
                edgecolor='none', zorder=0)
            
            ax.add_patch(rect)
            
            # Add phase label
            mid_x = (region["start"] + region["end"]) / 2
            ax.text(mid_x, y_max * 0.98, region["name"],
                    ha='center', va='top', fontsize=10,
                    color=label_color, fontweight='bold', alpha=0.9)
        
        return
    
    # Handling for forward_to_right task
    if task_name == "forward_to_right":
        for region in regions:
            phase_name = region["name"]
            angle = region["info"].get("angle", 0)
            velocity = region["info"].get("velocity", 0)
            
            # Determine color based on phase
            if velocity == 0.0 or "Rest" in phase_name:
                # Rest - light yellow/cream
                color = "#d3d3d3"
                alpha = 0.4
                label_color = "#34495e"  # Grey for rest label
            elif angle == 0 and velocity > 0:
                # Forward - off white
                color = "#FEFFED"
                alpha = 0.8
                label_color = "#34495e"  # Dark grey
            elif angle == 90:
                # Right - green
                color = "#1abc9c"
                alpha = 0.15
                label_color = color

            
            rect = Rectangle(
                (region["start"], y_min),
                region["end"] - region["start"],
                y_max - y_min,
                facecolor=color,
                alpha=alpha,
                edgecolor='none', zorder=0)
            
            ax.add_patch(rect)
            
            # Add phase label
            mid_x = (region["start"] + region["end"]) / 2
            ax.text(mid_x, y_max * 0.98, region["name"],
                    ha='center', va='top', fontsize=10,
                    color=label_color, fontweight='bold', alpha=0.9)
        
        return

    # Handling for forward_to_left task
    if task_name == "forward_to_left":
        for region in regions:
            phase_name = region["name"]
            angle = region["info"].get("angle", 0)
            velocity = region["info"].get("velocity", 0)
            
            # Determine color based on phase
            if velocity == 0.0 or "Rest" in phase_name:
                # Rest - light yellow/cream
                color = "#d3d3d3"
                alpha = 0.4
                label_color = "#34495e"  # Grey for rest label
            elif angle == 0 and velocity > 0:
                # Forward - off white
                color = "#FEFFED"
                alpha = 0.8
                label_color = "#34495e"  # Dark grey
            elif angle == -90:
                # Left - blue
                color = "#3498db"
                alpha = 0.15
                label_color = color

            
            rect = Rectangle(
                (region["start"], y_min),
                region["end"] - region["start"],
                y_max - y_min,
                facecolor=color,
                alpha=alpha,
                edgecolor='none', zorder=0)
            
            ax.add_patch(rect)
            
            # Add phase label
            mid_x = (region["start"] + region["end"]) / 2
            ax.text(mid_x, y_max * 0.98, region["name"],
                    ha='center', va='top', fontsize=10,
                    color=label_color, fontweight='bold', alpha=0.9)
        
        return

    # Handling for alternating_velocity task
    if task_name == "alternating_velocity":
        for region in regions:
            phase_name = region["name"]
            velocity = region["info"].get("velocity", 0)
            
            # Determine color based on velocity
            if velocity == 0.0 or "Rest" in phase_name:
                # Rest - light yellow/cream
                color = "#d3d3d3"
                alpha = 0.4
                label_color = "#34495e"  # Grey for rest label
            elif velocity == 3.0:
                # Fast - red
                color = "#e74c3c"
                alpha = 0.25
                label_color = color
            elif velocity == 1.0:
                # Slow - Red
                color = "#e74c3c"
                alpha = 0.15
                label_color = color

            
            rect = Rectangle(
                (region["start"], y_min),
                region["end"] - region["start"],
                y_max - y_min,
                facecolor=color,
                alpha=alpha,
                edgecolor='none', zorder=0)
            
            ax.add_patch(rect)
            
            # Add phase label
            mid_x = (region["start"] + region["end"]) / 2
            ax.text(mid_x, y_max * 0.98, region["name"],
                    ha='center', va='top', fontsize=10,
                    color=label_color, fontweight='bold', alpha=0.9)
        
        return

    # Handling for rest_swim_cycles
    if task_name == "rest_swim_cycles":
        for region in regions:
            phase_name = region["name"]
            
            if "Rest" in phase_name:
                color = "#FFF9E6"  # Cream
                alpha = 0.5
                label_color = "#34495e"  # Grey
            else:
                color = "#f39c12"  # Orange for swim
                alpha = 0.15
                label_color = color
            
            rect = Rectangle(
                (region["start"], y_min),
                region["end"] - region["start"],
                y_max - y_min,
                facecolor=color,
                alpha=alpha,
                edgecolor='none', zorder=0)
            
            ax.add_patch(rect)
            
            # Add phase label
            mid_x = (region["start"] + region["end"]) / 2
            ax.text(mid_x, y_max * 0.98, region["name"],
                    ha='center', va='top', fontsize=10,
                    color=label_color, fontweight='bold', alpha=0.9)

def analyze_and_plot():
    
    df, filepath, metadata = load_data()
    if df is None:
        return

    # Get task name from metadata
    task_name = get_task_name_from_metadata(metadata)
    
    ch1_data = df['ephys_ch0'].values if 'ephys_ch0' in df.columns else None
    ch2_data = df['ephys_ch1'].values if 'ephys_ch1' in df.columns else None
    
    channels = []
    if ch1_data is not None and not np.all(ch1_data == 0):
        channels.append(('Left', ch1_data))
    if ch2_data is not None and not np.all(ch2_data == 0):
        channels.append(('Right', ch2_data))
    
    if not channels:
        return
    
    # Create figure
    n_channels = len(channels)
    fig, axes = plt.subplots(4, n_channels, figsize=(7 * n_channels, 12), squeeze=False)
    
    file_name = filepath.split("/")[-1].split("\\")[-1]
    
    # Title with task name if available
    title = f'VNR Analysis: {file_name}'
    if task_name:
        title += f'\nTask: {task_name.replace("_", " ").title()}'
    fig.suptitle(title, fontsize=14, y=0.99)
    
    processed_data = {}
    
    for ch_idx, (ch_name, raw_signal) in enumerate(channels):
        
        n_samples = len(raw_signal)
        time_axis = np.arange(n_samples) / SR
        total_duration = time_axis[-1]
        
        filtered_signal = bandpass_filter(raw_signal, SR, BANDPASS_LOW, BANDPASS_HIGH)
        std_signal = calculate_std_dev(filtered_signal, SR, WINDOW_MS)
        threshold = calculate_threshold(std_signal)
        swim_strength = calculate_swim_strength(std_signal, threshold, SR, TAU_MS)
        
        burst_indices = detect_bursts(std_signal, threshold, SR)
        bouts = segment_bouts(burst_indices, SR)
        bout_metrics = calculate_bout_metrics(bouts, SR)
        
        processed_data[ch_name] = {
            'filtered_signal': filtered_signal,
            'std_signal': std_signal,
            'threshold': threshold,
            'burst_indices': burst_indices,
            'time_axis': time_axis
        }
        
        # Plot 1: VNR Signal
        ax1 = axes[0, ch_idx]
        ax1.plot(time_axis, filtered_signal, 'k', linewidth=0.5)
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title(f'{ch_name} - VNR Signal ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz)')
        

        # Add task overlay
        if task_name:
            y_min, y_max = ax1.get_ylim()
            add_task_overlay(ax1, task_name, total_duration, y_min, y_max)
        
        # Plot 2: Std Dev
        ax2 = axes[1, ch_idx]
        ax2.plot(time_axis, std_signal, color='black', linewidth=0.5)
        ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=1.5, label='Threshold')
        
        # Add spike train above the signal
        # Get the y-axis upper limit for positioning the spike train
        y_max = np.max(std_signal) * 1.1
        spike_y_position = y_max * 1  # Position spike train slightly above max signal
        
        # Plot vertical lines (spikes) at each threshold crossing (burst detection)
        if len(burst_indices) > 0:
            burst_times = burst_indices / SR
            ax2.vlines(burst_times, ymin=spike_y_position, ymax=spike_y_position * 1.15, color='#5E2B89', linewidth= 0.7)

        
        ax2.set_ylabel('Std Dev (a.u.)')
        ax2.set_title(f'{ch_name} - Std Dev')
        ax2.legend(loc='upper right', bbox_to_anchor=(1,1.18))
        ax2.set_ylim(bottom=0, top=spike_y_position * 1.35)  # Extend y-axis to accommodate spike train

        # Add task overlay
        if task_name:
            y_min, y_max = ax2.get_ylim()
            add_task_overlay(ax2, task_name, total_duration, y_min, y_max)        
        
        # Plot 3: Swim Strength
        ax3 = axes[2, ch_idx]
        ax3.plot(time_axis, swim_strength, color='#5E2B89', linewidth=0.8, zorder=2)
        ax3.fill_between(time_axis, swim_strength, 0, color='#B48FCC', alpha=1, zorder=1)
        ax3.set_ylabel('Swim Strength (a.u.)')
        ax3.set_title(f'{ch_name} - Swim Strength')

 
        # Add task overlay
        if task_name:
            y_min, y_max = ax3.get_ylim()
            add_task_overlay(ax3, task_name, total_duration, y_min, y_max)        
 
        # Plot 4: Burst Frequency
        ax4 = axes[3, ch_idx]
        if bout_metrics:
            for bout in bout_metrics:
                ax4.bar(bout['start_time'], bout['frequency'],
                        width=bout['duration'], align='edge',
                        color='#B48FCC', alpha=1, edgecolor='#5E2B89', linewidth=0.8, zorder=2)
            
            # Mean frequency line
            mean_freq = np.mean([b['frequency'] for b in bout_metrics])
            ax4.axhline(y=mean_freq, color='r', linestyle='--', linewidth=1.5, label='Mean Frequency',  zorder=1)
            
        ax4.set_ylabel('Burst Frequency (Hz)')
        ax4.set_title(f'{ch_name} - Burst Frequency per Bout')
        ax4.set_xlabel('Time (s)')
        ax4.legend(loc='upper right', bbox_to_anchor=(1,1.18))
        ax4.set_xlim(0, total_duration)

        # Add task overlay
        if task_name:
            y_min, y_max = ax4.get_ylim()
            add_task_overlay(ax4, task_name, total_duration, y_min, y_max)
        
        ax1.sharex(ax4)
        ax2.sharex(ax4)
        ax3.sharex(ax4)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, left=0.08)
    
    # Show main figure and pause to ensure it renders before prompting for input
    plt.show(block=False)
    plt.pause(0.1)
    
    # --- Interactive Zoomed Overlay Plot ---
    if len(channels) == 2:
        ch1_name, ch2_name = channels[0][0], channels[1][0]
        
        print("\n" + "="*45)
        print("Main figure generated. Please review it.")
        while True:
            try:
                start_sec = float(input("\nEnter start time for zoomed overlay (in seconds): "))
                end_sec = float(input("Enter end time for zoomed overlay (in seconds): "))
                
                if start_sec >= end_sec:
                    print("Invalid window: start time must be less than end time. Please try again.")
                    continue
                
                start_idx = max(0, int(start_sec * SR))
                end_idx = min(len(processed_data[ch1_name]['time_axis']), int(end_sec * SR))
                
                fig_ov, (ax_ov1, ax_ov2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
                fig_ov.suptitle(f'Zoomed Overlay ({start_sec}s to {end_sec}s)', fontsize=14)
                
                t_ov = processed_data[ch1_name]['time_axis'][start_idx:end_idx]
                
                # --- Top Panel: Filtered Signal ---
                ax_ov1.plot(t_ov, processed_data[ch1_name]['filtered_signal'][start_idx:end_idx], label=ch1_name, color='black', linewidth=1.2)
                ax_ov1.plot(t_ov, processed_data[ch2_name]['filtered_signal'][start_idx:end_idx], label=ch2_name, color='#e74c3c', linewidth=1.2, alpha=0.7)
                
                ax_ov1.set_ylabel('Voltage (V)')
                ax_ov1.set_title('Amplified VNR Signal')
                ax_ov1.legend(loc='upper right')
                ax_ov1.grid(True, linestyle='--', alpha=0.5)

                # --- Bottom Panel: Std Dev and Burst Classification ---
                std1 = processed_data[ch1_name]['std_signal'][start_idx:end_idx]
                std2 = processed_data[ch2_name]['std_signal'][start_idx:end_idx]
                thresh1 = processed_data[ch1_name]['threshold']
                thresh2 = processed_data[ch2_name]['threshold']
                
                ax_ov2.plot(t_ov, std1, label=f'{ch1_name} Std', color='black', linewidth=1.2)
                ax_ov2.plot(t_ov, std2, label=f'{ch2_name} Std', color='#e74c3c', linewidth=1.2, alpha=0.7)
                
                ax_ov2.axhline(y=thresh1, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label=f'{ch1_name} Threshold')
                ax_ov2.axhline(y=thresh2, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.5, label=f'{ch2_name} Threshold')
                
                bursts1 = processed_data[ch1_name]['burst_indices']
                bursts1_win = bursts1[(bursts1 >= start_idx) & (bursts1 <= end_idx)]
                bursts2 = processed_data[ch2_name]['burst_indices']
                bursts2_win = bursts2[(bursts2 >= start_idx) & (bursts2 <= end_idx)]
                
                y_max_ov = max(np.max(std1) if len(std1) > 0 else 0, np.max(std2) if len(std2) > 0 else 0) * 1.1
                if y_max_ov == 0: y_max_ov = 1.0
                
                spike_y1, spike_y2 = y_max_ov * 1.0, y_max_ov * 1.15
                if len(bursts1_win) > 0: ax_ov2.vlines(bursts1_win / SR, ymin=spike_y1, ymax=spike_y1 * 1.1, color='black', linewidth=1.5)
                if len(bursts2_win) > 0: ax_ov2.vlines(bursts2_win / SR, ymin=spike_y2, ymax=spike_y2 * 1.1, color='#e74c3c', linewidth=1.5)
                
                ax_ov2.set_ylim(bottom=0, top=spike_y2 * 1.35)
                ax_ov2.set_xlabel('Time (s)')
                ax_ov2.set_ylabel('Std Dev (a.u.)')
                ax_ov2.set_title('Standard Deviation and Burst Detection')
                ax_ov2.legend(loc='upper right')
                ax_ov2.grid(True, linestyle='--', alpha=0.5)

                fig_ov.tight_layout()
                plt.show(block=False)
                plt.pause(0.1)
                
                again = input("\nDo you want to choose another window? (y/n): ")
                if again.lower() != 'y':
                    break
                    
            except ValueError:
                print("Invalid input. Please enter numeric values. Try again.")

    plt.show(block=True)


if __name__ == "__main__":
    analyze_and_plot()