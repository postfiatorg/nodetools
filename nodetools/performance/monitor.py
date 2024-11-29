from pathlib import Path
from typing import Optional, Dict
from .timer import Timer
from .metric_types import Metric
from functools import wraps
from datetime import datetime, timedelta
from collections import defaultdict
import json

class PerfMeasurement:
    """Handles individual performance measurements"""
    def __init__(self, process: str):
        self.process = process
        self.data = {
            'type': None,
            'value': 0,
            'unit': None
        }
        self.timer = Timer()
        self.timer.start()

    def track(self, metric_type: Metric):
        """Record a new measurement"""
        match metric_type:
            case Metric.DURATION:
                self.timer.delta()  # First call establishes the start time
            case Metric.COUNT:
                pass
            case _:
                print(f"PerfMeasurement.track: Unsupported metric type: {metric_type}")

    def end_track(self, metric_type: Metric) -> float:
        """End a measurement and return the value"""
        match metric_type:
            case Metric.DURATION:
                value = self.timer.delta() * 1000  # Convert to milliseconds
            case Metric.COUNT:
                value = 1  # Each call counts as 1
            case _:
                print(f"Unsupported metric type: {metric_type}")
                value = 0

        self.data.update({
            'type': metric_type.type_name,
            'value': value,
            'unit': metric_type.unit
        })
        return value
    
class AggregatedMeasurement:
    """Handles aggregated measurements over a window of time"""
    def __init__(self):
        self.count = 0
        self.total = 0
        self.min_value = float('inf')
        self.max_value = float('-inf')
        self.last_window_start = datetime.now()
        self.data = {
            'type': None,
            'value': 0,
            'unit': None
        }
        self.timer = Timer()
        self.timer.start()

    def track(self, metric_type: Metric):
        """Record a new measurement"""
        match metric_type:
            case Metric.DURATION:
                self.timer.delta()  # First call establishes the start time
            case Metric.COUNT:
                pass
            case _:
                print(f"AggregatedMeasurement.track: Unsupported metric type: {metric_type}")

    def end_track(self, metric_type: Metric) -> Optional[float]:
        """End a measurement and update aggregates. Returns None if still aggregating."""
        match metric_type:
            case Metric.DURATION:
                value = self.timer.delta() * 1000  # Convert to milliseconds
                self.total += value
                self.min_value = min(self.min_value, value)
                self.max_value = max(self.max_value, value)
            case Metric.COUNT:
                value = 1  # Each call counts as 1
                self.count += value
            case _:
                print(f"AggregatedMeasurement.end_track: Unsupported metric type: {metric_type}")
                return None
            
        self.data.update({
            'type': metric_type.type_name,
            'value': self.total / self.count if self.count > 0 else 0,  # Average value
            'unit': metric_type.unit
        })
        return None # Indicate no immediate logging needed
    
    def should_report(self, window_seconds: int) -> tuple[bool, Optional[dict]]:
        """Check if it's time to report aggregated metrics and return stats if it is.
        
        This method provides atomic window transition by capturing statistics at the exact
        moment the time window expires. This ensures that no measurements are lost between
        checking the window expiration and collecting statistics.
        
        Args:
            window_seconds: Number of seconds in the reporting window
            
        Returns:
            Tuple of (should_report: bool, stats: dict if reporting, None if not)
        """
        now = datetime.now()
        if (now - self.last_window_start).total_seconds() >= window_seconds:
            self.last_window_start = now
            stats = self.get_aggregate_stats()  # Capture status before reset
            # Reset statistics for new window
            self.total = 0
            self.count = 0
            self.min_value = float('inf')
            self.max_value = float('-inf')
            return True, stats
        return False, None

    def get_aggregate_stats(self) -> dict:
        """Get the current average value"""
        if self.count == 0:
            return {
                'avg': 0,
                'count': 0,
                'min': 0,
                'max': 0
            }
        
        return {
            'avg': self.total / self.count if self.count > 0 else 0,
            'count': self.count,
            'min': self.min_value if self.min_value != float('inf') else 0,
            'max': self.max_value if self.max_value != float('-inf') else 0
        }

class PerformanceMonitor:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._initialized:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, output_dir: Optional[Path] = None, time_window: Optional[int] = None, save_log: bool = False):
        if not self.__class__._initialized:
            self.output_dir = output_dir or Path.cwd() / "performance_logs"
            self.measurements: Dict[str, PerfMeasurement] = {}
            self.aggregated_measurements: Dict[str, AggregatedMeasurement] = {}
            self.time_window = time_window or 10  # Default to 10 seconds
            self.save_log = save_log
            if save_log:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            print("---------------------------------Initialized PerformanceMonitor--------------------------------")
            self.__class__._initialized = True

    def log_measurement(self, process: str, metric_type: str, stats: dict, unit: str):
        """Log a measurement to file"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "process": process,
            "metric_type": metric_type,
            "statistics": stats,
            "unit": unit
        }

        if self.save_log:
            log_file = self.output_dir / f"performance_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        print(
            f"PerformanceMonitor.log_measurement: {timestamp} - {process}: "
            f"avg={stats['avg']:.2f} {unit}, "
            f"count={stats['count']}, "
            f"min={stats['min']:.2f} {unit}, "
            f"max={stats['max']:.2f} {unit}"
        )

    @staticmethod
    def measure(process: str, *metrics: Metric, override_aggregation: bool = False):
        """Generic decorator that measures multiple metrics
        
        Args:
            process: name of the process to measure
            *metrics: variable number of metrics to measure
            override_aggregation: If True, always log individual measurements
        """
        metrics = metrics or (Metric.DURATION, Metric.COUNT)

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                monitor = PerformanceMonitor._instance

                # If no monitor is active, just run the function
                if monitor is None:
                    return func(*args, **kwargs)
                
                if override_aggregation or monitor.time_window is None:
                    # Use immediate measurements 
                    print(f"Starting measurement for {process} ({[m.type_name for m in metrics]})")
                    perf_measurement = monitor.measurements.get(process)
                    if perf_measurement is None:
                        perf_measurement = PerfMeasurement(process)
                        monitor.measurements[process] = perf_measurement

                    # Start measurement
                    for metric in metrics:
                        perf_measurement.track(metric)

                    # Execute the function
                    result = func(*args, **kwargs)

                    # End measurement and log
                    for metric in metrics:
                        value = perf_measurement.end_track(metric)
                        monitor.log_measurement(
                            process=process,
                            metric_type=metric.type_name,
                            stats={'avg': value, 'count': 1, 'min': value, 'max': value},
                            unit=metric.unit
                        )

                else:
                    # Use aggregated measurements
                    agg_measurement = monitor.aggregated_measurements.get(process)
                    if agg_measurement is None:
                        agg_measurement = AggregatedMeasurement()
                        monitor.aggregated_measurements[process] = agg_measurement

                    # Start measurement
                    for metric in metrics:
                        agg_measurement.track(metric)

                    # Execute the function
                    result = func(*args, **kwargs)

                    # Update aggregates and potentially log
                    for metric in metrics:
                        agg_measurement.end_track(metric)
                        should_report, stats = agg_measurement.should_report(monitor.time_window)
                        if should_report:
                            monitor.log_measurement(
                                process=process,
                                metric_type=metric.type_name,
                                stats=stats,
                                unit=metric.unit
                            )

                return result
            return wrapper
        return decorator

    def start(self):
        """Start the performance monitor"""
        PerformanceMonitor._instance = self
        message = f"Performance monitoring started. "
        if self.save_log:
            message += f"Performance logs will be written to {self.output_dir}"
        else:
            message += "Performance log saving is currently disabled."
        print(message)

    def stop(self):
        """Stop the performance monitor"""
        PerformanceMonitor._instance = None
        print("Performance monitoring stopped")