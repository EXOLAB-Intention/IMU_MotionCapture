"""Test CSV import functionality"""
from file_io.file_handler import FileHandler

# Import CSV file
filepath = 'JJY_20260119_101434_Npose_processed.csv'
print(f"Importing {filepath}...")
data = FileHandler.import_raw_data(filepath)

print(f"\n=== Import Results ===")
print(f"Session ID: {data.session_id}")
print(f"Sensors loaded: {list(data.imu_data.keys())}")
print(f"Duration: {data.duration:.2f} seconds")

# Print details for each sensor
for location, sensor_data in data.imu_data.items():
    print(f"\n{location}:")
    print(f"  Samples: {sensor_data.n_samples}")
    print(f"  Sampling freq: {sensor_data.sampling_frequency} Hz")
    print(f"  Duration: {sensor_data.duration:.2f} s")
    print(f"  Quaternion range: [{sensor_data.quaternions.min():.4f}, {sensor_data.quaternions.max():.4f}]")
    print(f"  Acceleration range: [{sensor_data.accelerations.min():.4f}, {sensor_data.accelerations.max():.4f}]")
    print(f"  Gyroscope range: [{sensor_data.gyroscopes.min():.4f}, {sensor_data.gyroscopes.max():.4f}]")

print(f"\n=== Validation ===")
print(f"Has all sensors: {data.has_all_sensors}")
print(f"Time range: {data.get_time_range()}")
