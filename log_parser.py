import json
import re
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import BarChart, Reference
import sys
import os

def get_total_time(time_value):
    if isinstance(time_value, list) and time_value:
        return time_value[0]
    elif isinstance(time_value, (int, float)):
        return time_value
    return None

def calculate_hourly_throughput(retrieval_events_df):
    hourly_counts = defaultdict(int)
    if retrieval_events_df.empty:
        return hourly_counts
    for _, event in retrieval_events_df.iterrows():
        timestamp = event.get('Timestamp')
        # each successful event in this dataframe represents one package
        num_packages = 1 
        if not pd.isna(timestamp):
            try:
                hour_bucket = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:00')
                hourly_counts[hour_bucket] += num_packages
            except (ValueError, TypeError):
                continue
    return hourly_counts

def calculate_overall_throughput(successful_retrieve_events):
    """calculates throughput, ignoring idle time longer than 30 minutes."""
    if not successful_retrieve_events:
        return 0.0, 0.0

    total_packages = len(successful_retrieve_events)
    if total_packages < 2:
        return 0.0, 0.0
    
    timestamps = sorted([pd.to_datetime(event['Timestamp']) for event in successful_retrieve_events if 'Timestamp' in event])
    if len(timestamps) < 2:
        return 0.0, 0.0

    active_duration_seconds = 0
    for i in range(len(timestamps) - 1):
        gap = (timestamps[i+1] - timestamps[i]).total_seconds()
        if gap < 10 * 60:  # 30 minutes
            active_duration_seconds += gap

    duration_hours = active_duration_seconds / 3600.0
    if duration_hours == 0:
        # if duration is zero, avoid division errors and assume a very high throughput
        if total_packages > 1:
             return total_packages * 3600.0, duration_hours
        return 0.0, 0.0

    throughput = total_packages / duration_hours
    return throughput, duration_hours

def calculate_active_time_hours(events):
    """calculates total active time, ignoring gaps longer than 30 minutes."""
    if not events or len(events) < 2:
        return 0.0
    
    timestamps = sorted([pd.to_datetime(event['Timestamp']) for event in events if 'Timestamp' in event])
    if len(timestamps) < 2:
        return 0.0

    active_duration_seconds = 0
    for i in range(len(timestamps) - 1):
        gap = (timestamps[i+1] - timestamps[i]).total_seconds()
        if gap < 30 * 60: # 30 minutes
            active_duration_seconds += gap
            
    return active_duration_seconds / 3600.0

def _normalize_error_message(error_message):
    """Normalizes a raw error message into a standardized category."""
    lower_message = error_message.lower()

    # Ignore summary-level error messages that are not root causes.
    if "retrieve finished with failed packages" in lower_message:
        return None # This is a summary message, not a root cause. Ignore it.

    # Motion & Pick/Place Failures
    if "failed to execute pick and place" in lower_message or "failed in attempt to pick and place" in lower_message:
        return "Motion: Pick/Place Execution Failed"
    if "failed to execute pick and raise" in lower_message or "failed to pick and raise" in lower_message:
        return "Motion: Pick/Raise Execution Failed"
    if "waypoint execution failed" in lower_message:
        return "Motion: Waypoint Execution Failed"
    if "not calibrated" in lower_message:
        return "Motion: System Not Calibrated"
    
    # End Effector Failures
    if "vacuum not achieved" in lower_message:
        return "End Effector: Vacuum Failure"

    # Packing Planner Failures
    if "failed to find placement" in lower_message or "unable to find a place" in lower_message:
        return "Packing: Placement Failure"

    # Vision System Failures
    if "failed to extract address" in lower_message:
        return "Vision: Address Extraction Failed"
    if "sag height was not found" in lower_message:
        return "Vision: Sag Height Not Found"
    if "failed to find package" in lower_message or "no contour" in lower_message or "cannot upload, failed to find package" in lower_message:
        return "Vision: Package Not Found"
    if "internal server error" in lower_message:
        return "Vision: Server Error"
    if "connection refused" in lower_message or "max retries exceeded" in lower_message or "read timed out" in lower_message:
        return "Vision: Connection Error"

    # OCR/Database Failures
    if "ocr timeout waiting for package" in lower_message:
        return "Database: OCR Timeout"
    if "timeout waiting for package" in lower_message:
        return "Database: Package Timeout"

    # Network/API Failures
    if "connecttimeouterror" in lower_message or "connection to cargo-robotics.com timed out" in lower_message:
        return "Network: API Connection Timeout"
    if "no route to host" in lower_message:
        return "Network: No Route to Host"
    if "error querying server for packages" in lower_message:
        return "Network: Package Query Failed"

    # General/Unknown
    if "package length is too long" in lower_message or "package height is too high" in lower_message:
        return "Physical Constraints: Package Dimension Error"
    if "error message not found" in lower_message:
        return "Unknown: No Error Message"
    
    # Fallback for any other message
    return "Uncategorized Error"

def process_log_file(log_file_path):
    """
    processes a single json log file to extract key metrics and event data.
    it also unpacks individual pick and place calls from retrieve events.
    """
    all_pickup_events = []  # NEW: Conveyable Stow events (package pickup)
    all_pick_events = []    # NEW: Execute Pick and Raise events (package placement/pick)
    all_retrieve_events = [] # UNCHANGED: this holds individual pick/place events from retrievals
    all_read_label_events = []
    file_failures = []
    
    total_error_count = 0
    retrieval_attempts = 0  # UNCHANGED: retrieval logic
    pickup_attempts = 0     # NEW: Conveyable Stow attempts
    pick_attempts = 0       # NEW: Execute Pick and Raise attempts
    error_type_counts = Counter()
    log_date = None

    # Track unique package attempts to avoid double-counting retries
    attempted_package_ids = set()

    with open(log_file_path, 'r') as f_in:
        for line in f_in:
            try:
                log_entry = json.loads(line)
                event_label = log_entry.get("event_label")
                is_success = log_entry.get("success", True)
                timestamp = log_entry.get("timestamp")

                if not log_date and timestamp:
                    log_date = pd.to_datetime(timestamp).strftime('%Y-%m-%d')

                total_time = get_total_time(log_entry.get("event_total_time"))
                
                if event_label == "Read Label Callback":
                    if is_success and total_time is not None:
                        all_read_label_events.append({
                            "Timestamp": timestamp, 
                            "Event Label": event_label, 
                            "Time (s)": total_time
                        })
                        
                elif event_label == "Conveyable Stow":
                    # NEW: This is now package pickup
                    pickup_attempts += 1
                    if is_success and total_time is not None:
                        all_pickup_events.append({
                            "Timestamp": timestamp,
                            "Event Label": "Package Pickup",
                            "Time (s)": total_time
                        })
                        
                elif event_label == "Execute Pick and Raise":
                    # NEW: This is package placement (called "pick" in dashboard)
                    pick_attempts += 1
                    if is_success and total_time is not None:
                        all_pick_events.append({
                            "Timestamp": timestamp,
                            "Event Label": "Package Pick", 
                            "Time (s)": total_time
                        })

                elif "Retrieve Packages Callback" in event_label:
                    # UNCHANGED: Keep all existing retrieval logic exactly as is
                    log_datetime = pd.to_datetime(timestamp)
                    cutoff_date_fix_logic = pd.to_datetime("2025-07-31")
                    cutoff_date_new_logic = pd.to_datetime("2025-07-23")
                    cutoff_date_old_logic = pd.to_datetime("2025-07-22")

                    if log_datetime >= cutoff_date_fix_logic:
                        # --- FIX #1 LOGIC (>= 2025-07-31) ---
                        current_attempt_ids = set()
                        failed_ids = set()

                        for log_msg in log_entry.get("logs", []):
                            attempt_match = re.search(r"Retrieving packages with IDs: \[([\d, ]+)\]", log_msg.get("message", ""))
                            if attempt_match:
                                ids_str = attempt_match.group(1)
                                if ids_str:
                                    current_attempt_ids.update([int(i.strip()) for i in ids_str.split(',')])
                            
                            failure_match = re.search(r"Retrieve finished with failed packages: \[([\d, ]+)\]", log_msg.get("message", ""))
                            if failure_match:
                                failed_ids_str = failure_match.group(1)
                                if failed_ids_str:
                                    failed_ids.update([int(i.strip()) for i in failed_ids_str.split(',')])

                        new_attempts = current_attempt_ids - attempted_package_ids
                        retrieval_attempts += len(new_attempts)
                        attempted_package_ids.update(new_attempts)
                        
                        timings = log_entry.get("timings", {})
                        for key, value in timings.items():
                            if "Pick and Place Call" in key:
                                try:
                                    pkg_id_match = re.search(r'\d+', key)
                                    if not pkg_id_match: continue
                                    
                                    pkg_id = int(pkg_id_match.group())
                                    if pkg_id not in failed_ids:
                                        retrieval_time = get_total_time(value)
                                        if retrieval_time is not None:
                                            all_retrieve_events.append({
                                                "Timestamp": timestamp,
                                                "Event Label": f"Individual Retrieval (ID: {pkg_id})",
                                                "Time (s)": retrieval_time,
                                                "Package ID": pkg_id
                                            })
                                except (AttributeError, ValueError, TypeError):
                                    continue
                    elif log_datetime >= cutoff_date_new_logic:
                        # --- NEWEST LOGIC (>= 2025-07-23, < 2025-07-31) ---
                        attempted_ids = set()
                        failed_ids = set()

                        for log_msg in log_entry.get("logs", []):
                            attempt_match = re.search(r"Retrieving packages with IDs: \[([\d, ]+)\]", log_msg.get("message", ""))
                            if attempt_match:
                                ids_str = attempt_match.group(1)
                                if ids_str:
                                    attempted_ids.update([int(i.strip()) for i in ids_str.split(',')])
                            
                            failure_match = re.search(r"Retrieve finished with failed packages: \[([\d, ]+)\]", log_msg.get("message", ""))
                            if failure_match:
                                failed_ids_str = failure_match.group(1)
                                if failed_ids_str:
                                    failed_ids.update([int(i.strip()) for i in failed_ids_str.split(',')])

                        retrieval_attempts += len(attempted_ids)
                        
                        timings = log_entry.get("timings", {})
                        for key, value in timings.items():
                            if "Pick and Place Call" in key:
                                try:
                                    pkg_id_match = re.search(r'\d+', key)
                                    if not pkg_id_match: continue
                                    
                                    pkg_id = int(pkg_id_match.group())
                                    if pkg_id not in failed_ids:
                                        retrieval_time = get_total_time(value)
                                        if retrieval_time is not None:
                                            all_retrieve_events.append({
                                                "Timestamp": timestamp,
                                                "Event Label": f"Individual Retrieval (ID: {pkg_id})",
                                                "Time (s)": retrieval_time,
                                                "Package ID": pkg_id
                                            })
                                except (AttributeError, ValueError, TypeError):
                                    continue
                    elif log_datetime >= cutoff_date_old_logic:
                        # --- PREVIOUS LOGIC (>= 2025-07-22, < 2025-07-23) ---
                        attempted_ids = []
                        for log_msg in log_entry.get("logs", []):
                            match = re.search(r"Retrieving packages with IDs: \[([\d, ]+)\]", log_msg.get("message", ""))
                            if match:
                                ids_str = match.group(1)
                                if ids_str:
                                    attempted_ids = [int(i.strip()) for i in ids_str.split(',')]
                                break
                        
                        retrieval_attempts += len(attempted_ids)
                        
                        successful_timings = log_entry.get("timings", {})
                        for key, value in successful_timings.items():
                            if "Pick and Place Call" in key:
                                try:
                                    pkg_id_match = re.search(r'\d+', key)
                                    pkg_id = int(pkg_id_match.group()) if pkg_id_match else 'N/A'
                                    retrieval_time = get_total_time(value)
                                    if retrieval_time is not None:
                                        all_retrieve_events.append({
                                            "Timestamp": timestamp,
                                            "Event Label": f"Individual Retrieval (ID: {pkg_id})",
                                            "Time (s)": retrieval_time,
                                            "Package ID": pkg_id
                                        })
                                except (AttributeError, ValueError, TypeError):
                                    continue
                    else:
                        # --- OLD LOGIC (< 2025-07-22) ---
                        for log_msg in log_entry.get("logs", []):
                            match = re.search(r"Retrieving packages with IDs: \[([\d, ]+)\]", log_msg.get("message", ""))
                            if match:
                                ids_str = match.group(1)
                                if ids_str:
                                    retrieval_attempts += len(ids_str.split(','))
                                break
                        
                        if is_success:
                            timings = log_entry.get("timings", {})
                            for key, value in timings.items():
                                if "Pick and Place Call" in key:
                                    try:
                                        pkg_id_match = re.search(r'\d+', key)
                                        pkg_id = int(pkg_id_match.group()) if pkg_id_match else 'N/A'
                                        retrieval_time = get_total_time(value)
                                        if retrieval_time is not None:
                                            all_retrieve_events.append({
                                                "Timestamp": timestamp,
                                                "Event Label": f"Individual Retrieval (ID: {pkg_id})",
                                                "Time (s)": retrieval_time,
                                                "Package ID": pkg_id
                                            })
                                    except (AttributeError, ValueError, TypeError):
                                        continue
                
                # UNCHANGED: Error handling logic
                if not is_success:
                    error_message = "Error message not found"
                    for level in ["ERROR", "WARN"]:
                        for log_msg in log_entry.get("logs", []):
                            if log_msg.get("level") == level:
                                error_message = log_msg.get("message", error_message)
                                break
                        if error_message != "Error message not found":
                            break
                    
                    if error_message == "Error message not found" and "Callback" in event_label:
                        if "Execute Pick and Place Callback" in event_label:
                            error_message = "Vacuum not achieved (inferred from callback failure)"
                    
                    normalized_error = _normalize_error_message(error_message)
                    if normalized_error:
                        total_error_count += 1
                        error_type_counts[normalized_error] += 1
                        file_failures.append({
                            "Timestamp": timestamp,
                            "Error Type": event_label or "Unknown Event",
                            "Error Message": error_message,
                            "Normalized Category": normalized_error
                        })
            except (json.JSONDecodeError, TypeError):
                continue

    # Create DataFrames
    df_pickup = pd.DataFrame(all_pickup_events)
    df_pick = pd.DataFrame(all_pick_events)
    df_retrieve_raw = pd.DataFrame(all_retrieve_events)  # UNCHANGED: retrieval events
    df_read_label = pd.DataFrame(all_read_label_events)
    df_failures = pd.DataFrame(sorted(file_failures, key=lambda x: x.get('Timestamp') or ''))

    # Create error summary
    error_summary_data = dict(error_type_counts)
    error_summary_data['Date'] = log_date
    df_error_summary = pd.DataFrame([error_summary_data])

    # Calculate statistics
    stats = defaultdict(lambda: {'avg': 'N/A', 'min': 'N/A', 'max': 'N/A'})
    for df, name in [(df_pickup, 'Pickup'), (df_pick, 'Pick'), (df_retrieve_raw, 'Retrieve'), (df_read_label, 'Read Label')]:
        if not df.empty and 'Time (s)' in df.columns and df['Time (s)'].notna().any():
            times = df['Time (s)'].dropna()
            if not times.empty:
                stats[name] = {'avg': times.mean(), 'min': times.min(), 'max': times.max()}

    # Calculate metrics
    total_packages_picked_up = len(df_pickup)
    total_packages_picked = len(df_pick)
    total_packages_retrieved = len(df_retrieve_raw)
    
    # Calculate combined stow average (sum of pickup + pick averages)
    pickup_avg = stats['Pickup']['avg'] if stats['Pickup']['avg'] != 'N/A' else 0
    pick_avg = stats['Pick']['avg'] if stats['Pick']['avg'] != 'N/A' else 0
    combined_stow_avg = (pickup_avg + pick_avg) if (pickup_avg != 'N/A' and pick_avg != 'N/A') else 'N/A'
    
    # Calculate throughput and shift times
    throughput, retrieve_shift_time_hr = calculate_overall_throughput(all_retrieve_events)
    pickup_shift_time_hr = calculate_active_time_hours(all_pickup_events)
    pick_shift_time_hr = calculate_active_time_hours(all_pick_events)

    summary_data = {
        'Date': log_date,
        'Package Pickup Avg (s)': stats['Pickup']['avg'],
        'Package Pick Avg (s)': stats['Pick']['avg'],
        'Stow Avg (s)': combined_stow_avg,  # Combined pickup + pick average
        'Retrieve Avg (s)': stats['Retrieve']['avg'],
        'Read Label Avg (s)': stats['Read Label']['avg'],
        'Packages Picked Up': total_packages_picked_up,
        'Pickup Attempts': pickup_attempts,
        'Packages Picked': total_packages_picked,
        'Pick Attempts': pick_attempts,
        'Packages Retrieved': total_packages_retrieved,  # UNCHANGED
        'Retrieval Attempts': retrieval_attempts,  # UNCHANGED
        'Throughput (pkg/hr)': throughput,
        'Total Errors': total_error_count,
        'Pickup Driver Shift Time (hr)': pickup_shift_time_hr,
        'Pick Driver Shift Time (hr)': pick_shift_time_hr,
        'Retrieve Driver Shift Time (hr)': retrieve_shift_time_hr  # UNCHANGED
    }

    return {
        "date": log_date,
        "pickup_events": df_pickup,      # NEW: Package pickup events
        "pick_events": df_pick,          # NEW: Package pick events
        "stow_events": df_pickup,        # LEGACY: For backward compatibility with report_updater
        "retrieve_events": df_retrieve_raw,  # UNCHANGED: retrieval events
        "raw_retrieve_events": df_retrieve_raw,  # UNCHANGED
        "read_label_events": df_read_label,
        "failures": df_failures,
        "summary": summary_data,
        "error_summary": df_error_summary
    }