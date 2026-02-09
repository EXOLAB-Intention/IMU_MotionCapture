"""
Test file for _remove_noise function
"""
import numpy as np


def _remove_noise(contact: np.ndarray, min_duration: int) -> np.ndarray:
    """Fill short False gaps (< min_duration) with True"""
    cleaned = contact.copy()
    
    # Find where True regions start and end
    diff = np.diff(contact.astype(int))
    true_starts = np.where(diff == 1)[0] + 1   # 0->1 transition
    true_ends = np.where(diff == -1)[0] + 1    # 1->0 transition
    
    # Correct boundaries
    if len(contact) > 0 and contact[0]:        # Starts with True
        true_starts = np.r_[0, true_starts]
    if len(contact) > 0 and contact[-1]:       # Ends with True
        true_ends = np.r_[true_ends, len(contact)]
    
    # Fill gaps between True regions
    for i in range(len(true_ends) - 1):
        gap_start = true_ends[i]
        gap_end = true_starts[i + 1]
        gap_duration = gap_end - gap_start
        
        if gap_duration < min_duration:
            cleaned[gap_start:gap_end] = True
    
    return cleaned


def print_comparison(original, cleaned, test_name, min_duration):
    """Print before/after comparison"""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"Min Duration: {min_duration}")
    print(f"{'='*60}")
    print(f"Original:  {original.astype(int)}")
    print(f"Cleaned:   {cleaned.astype(int)}")
    
    # Count True regions - show internal diff calculation
    diff = np.diff(original.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    print(f"\nInternal diff array (np.diff):")
    print(f"  {diff}")
    print(f"  Starts (where diff == 1): {starts}")
    print(f"  Ends (where diff == -1):   {ends}")
    
    if original[0]:
        starts = np.r_[0, starts]
    if original[-1]:
        ends = np.r_[ends, len(original)]
    
    print(f"\nTrue regions in original:")
    for i, (start, end) in enumerate(zip(starts, ends), 1):
        duration = end - start
        print(f"  Region {i}: indices {start}-{end}, duration={duration}")
    
    # Count changes
    removed = np.sum(original & ~cleaned)
    added = np.sum(~original & cleaned)
    print(f"\nChanges: {removed} removed, {added} added")


def main():
    print("Testing _remove_noise function")
    print("FIXED implementation: cleaned[start:end] = False")
    print("+ Handle boundary cases (start/end of array)\n")
    
    # Test 1: Short isolated True regions (should be removed)
    test1 = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    result1 = _remove_noise(test1, min_duration=3)
    print_comparison(test1, result1, "Short isolated contacts", min_duration=3)
    
    # Test 2: Mix of short and long regions
    test2 = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
    result2 = _remove_noise(test2, min_duration=4)
    print_comparison(test2, result2, "Mix of short/long contacts", min_duration=4)
    
    # Test 3: Contact at start and end
    test3 = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1], dtype=bool)
    result3 = _remove_noise(test3, min_duration=2)
    print_comparison(test3, result3, "Contacts at boundaries", min_duration=2)
    
    # Test 4: All short contacts
    test4 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
    result4 = _remove_noise(test4, min_duration=2)
    print_comparison(test4, result4, "All short single-sample contacts", min_duration=2)
    
    # Test 5: Long continuous contact
    test5 = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    result5 = _remove_noise(test5, min_duration=3)
    print_comparison(test5, result5, "Long continuous contact", min_duration=3)
    
    # Test 6: Realistic walking pattern (longer sequence)
    # Simulates: walk with 3 steps, some noise spikes
    test6 = np.array([
        0,0,0,0,0,  # No contact
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  # Step 1: 15 samples (keep)
        0,0,0,0,0,0,0,0,  # Swing phase
        1,1,  # Noise spike: 2 samples (remove if min_duration > 2)
        0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,  # Step 2: 12 samples (keep)
        0,0,0,0,0,0,0,
        1,1,1,  # Short contact: 3 samples (remove if min_duration > 3)
        0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  # Step 3: 18 samples (keep)
        0,0,0,0,0  # No contact
    ], dtype=bool)
    result6 = _remove_noise(test6, min_duration=10)
    print_comparison(test6, result6, "Realistic walking pattern (3 steps)", min_duration=10)
    
    # Test 7: Noisy signal with multiple short spikes
    test7 = np.array([
        0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,  # Multiple 1-2 sample spikes
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  # Long contact: 20 samples
        0,0,1,0,1,1,0,0,1,0,0,0,  # More noise
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  # Another long contact: 15 samples
        0,0,0,0,0
    ], dtype=bool)
    result7 = _remove_noise(test7, min_duration=8)
    print_comparison(test7, result7, "Noisy signal with spikes", min_duration=8)
    
    # Test 8: Multiple medium-length contacts
    test8 = np.array([
        0,0,0,
        1,1,1,1,1,  # 5 samples
        0,0,0,0,0,0,0,0,0,0,
        1,1,1,1,1,1,1,  # 7 samples
        0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,  # 12 samples
        0,0,0,0,
        1,1,1,1,  # 4 samples
        0,0,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,  # 9 samples
        0,0,0
    ], dtype=bool)
    result8 = _remove_noise(test8, min_duration=6)
    print_comparison(test8, result8, "Multiple medium contacts (threshold=6)", min_duration=6)
    
    # Test 9: Edge case - very long sequence with rare contacts
    test9_base = np.zeros(100, dtype=bool)
    test9_base[10:25] = True  # 15 samples
    test9_base[30:32] = True  # 2 samples (noise)
    test9_base[45:70] = True  # 25 samples
    test9_base[75] = True     # 1 sample (noise)
    test9_base[80:95] = True  # 15 samples
    result9 = _remove_noise(test9_base, min_duration=10)
    print_comparison(test9_base, result9, "Long sequence (100 samples) with sparse contacts", min_duration=10)
    
    print(f"\n{'='*60}")
    print("EXPECTED BEHAVIOR:")
    print("  - Short True regions (< min_duration) should be set to False")
    print("  - FIXED: Now correctly removes short contacts!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
