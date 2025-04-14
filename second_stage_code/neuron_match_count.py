import os
import pickle
import numpy as np

root_dir = 
target_channel_id = 278
match_count = 0

# Make an output directory to avoid polluting the main one
output_dir =
os.makedirs(output_dir, exist_ok=True)

for subdir, dirs, files in os.walk(root_dir):
    if os.path.basename(subdir) == "filtered_ids_PTP_xc_yc":
        for file in files:
            if file.endswith(".pkl") and file.startswith("cluster_"):
                cluster_file = os.path.join(subdir, file)
                try:
                    with open(cluster_file, "rb") as f:
                        data = pickle.load(f)
                        ids = data.get("filtered_sorted_channel_ids", [])
                        ptp = data.get("filtered_sorted_PTP", [])
                        xc = data.get("filtered_sorted_Xc_coordinates", [])
                        yc = data.get("filtered_sorted_Yc_coordinates", [])
                        tx = data.get("trained_x", None)
                        ty = data.get("trained_y", None)
                        tz = data.get("trained_z", None)
                        ta = data.get("trained_alpha", None)

                        for idx, chan_id in enumerate(ids):
                            if chan_id == target_channel_id:
                                match_data = {
                                    "filtered_sorted_channel_ids": np.array([chan_id]),
                                    "filtered_sorted_PTP": np.array([ptp[idx]] if idx < len(ptp) else [None]),
                                    "filtered_sorted_Xc_coordinates": np.array([xc[idx]] if idx < len(xc) else [None]),
                                    "filtered_sorted_Yc_coordinates": np.array([yc[idx]] if idx < len(yc) else [None]),
                                    "trained_x": float(tx) if tx is not None else float("nan"),
                                    "trained_y": float(ty) if ty is not None else float("nan"),
                                    "trained_z": float(tz) if tz is not None else float("nan"),
                                    "trained_alpha": float(ta) if ta is not None else float("nan"),
                                }

                                output_path = os.path.join(output_dir, f"cluster_{match_count}.pkl")
                                with open(output_path, "wb") as out_f:
                                    pickle.dump(match_data, out_f)

                                match_count += 1
                except Exception as e:
                    print(f"Error reading {cluster_file}: {e}")

print(f"Saved {match_count} individual cluster files in: {output_dir}")
