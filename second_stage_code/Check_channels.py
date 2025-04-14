import numpy as np
import pandas as pd
from itertools import islice
from tqdm import tqdm  # Import the tqdm library

spike_amplitudes = np.load(r'spikes.amps.npy')
spike_clusters = np.load(r'spikes.clusters.npy')
cluster_peak_channels = np.load(r'clusters.peakChannel.npy')
spike_probes = np.load(r'clusters.probes.npy')
channel_locations_in_Allen_CCF = pd.read_csv(r'channels.brainLocation.tsv')

spike_peak_channel = cluster_peak_channels[spike_clusters]
spike_probe = spike_probes[spike_clusters]
spike_peak_channel = spike_peak_channel.flatten()
Channel_location_in_Allen_CCF = channel_locations_in_Allen_CCF.iloc[spike_peak_channel]

# # Limiting to the first 5 entries
# for i, (amp, channel, probe, (index, row)) in enumerate(zip(
#         spike_amplitudes[:5],
#         spike_peak_channel[:5],
#         spike_probe[:5],
#         islice(Channel_location_in_Allen_CCF.iterrows(), 5))):
#     # Extract relevant details from the row
#     location_details = row.to_dict()  # Convert the row to a dictionary
#     print(f"Spike {i}: PTP amplitude = {amp.item():.2f} µV, Peak Channel = {channel}, Probe = {probe}, Location = {location_details}")

# Prepare a list to store the output data
output_data = []

# Use tqdm for a progress bar in the loop
for i, (amp, channel, probe, (index, row)) in enumerate(tqdm(zip(
        spike_amplitudes,
        spike_peak_channel,
        spike_probe,
        Channel_location_in_Allen_CCF.iterrows()),
        desc="Processing Spikes",
        total=len(spike_amplitudes))):  # Set total for accurate progress tracking
    # Extract relevant details from the row
    location_details = row.to_dict()  # Convert the row to a dictionary

    # Append the data to the list
    output_data.append({
        "Spike Index": i,
        "PTP Amplitude (µV)": amp.item(),
        "Peak Channel": channel,
        "Probe": probe.item(),  # Convert numpy scalar to native Python type
        **location_details  # Include all details from the location
    })

# Convert the list to a DataFrame
output_df = pd.DataFrame(output_data)

# Save the DataFrame to an Excel file
output_df.to_excel(r"Spikes_PTP_Peak_Channel_Location.xlsx", index=False)

print("Data saved to 'Spikes_PTP_Peak_Channel_Location.xlsx'")


