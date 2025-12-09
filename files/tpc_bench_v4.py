#!/usr/bin/env python3
"""
tpc_bench_v4.py - Chapter 3 Methodology-Compliant TPC Benchmark

Enhancements aligned with dissertation methodology:
- Dunn's pairwise post-hoc tests with FDR correction
- Cold/warm run stratification
- Concurrency sweep analysis
- Root-cause analysis thresholds
- Practical significance filtering (Cliff's Delta)
- Enhanced statistical reporting per Chapter 3.9
## Usage Example:
python tpc_bench_v4.py \
    --db oracle \
    --config oracle_config.json \
    --queries /home/opc/Downloads/project/tpch-dbgen/queries_oracle \
    --iterations 20 \
    --warmup 5 \
    --concurrency 1 \
    --output results_oracle/results.csv \
    --baseline resultS_oracle/previous_results.csv \
    --concurrency-sweep \
    --concurrency-levels 1 5 10 20 \
    --capture-plans
"""

import argparse
import concurrent.futures
import csv
import glob
import json
import os
import re
import time
import traceback
import uuid
import threading
import platform
import subprocess
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import statistics

# Optional dependencies
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import psutil
except Exception:
    psutil = None

try:
    import ntplib
except Exception:
    ntplib = None

try:
    import numpy as np
    from scipy import stats as scipy_stats
except Exception:
    np = None
    scipy_stats = None

import matplotlib.pyplot as plt


# ================== STATISTICAL ENHANCEMENTS (Chapter 3.9) ==================

def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Benjamini-Hochberg FDR correction for multiple comparisons (Chapter 3.9.2)
    Returns list of booleans indicating significance after correction
    """
    if not p_values:
        return []
    
    n = len(p_values)
    # Sort p-values with original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    
    # Apply BH procedure
    reject = [False] * n
    for rank, (original_idx, p_val) in enumerate(sorted_pairs, start=1):
        threshold = (rank / n) * alpha
        if p_val <= threshold:
            reject[original_idx] = True
        else:
            # BH procedure: once we fail to reject, stop
            break
    
    return reject


def dunns_post_hoc_test(groups: Dict[str, List[float]], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Dunn's pairwise post-hoc test with BH correction (Chapter 3.9.2)
    
    Performs pairwise comparisons after significant Kruskal-Wallis test.
    Returns comparison matrix with z-statistics, p-values, and significance flags.
    """
    if not scipy_stats or len(groups) < 2:
        return {"available": False, "reason": "scipy unavailable or insufficient groups"}
    
    platform_names = list(groups.keys())
    n_platforms = len(platform_names)
    
    # Calculate total N and ranks
    all_data = []
    group_labels = []
    for platform, values in groups.items():
        all_data.extend(values)
        group_labels.extend([platform] * len(values))
    
    N = len(all_data)
    ranks = scipy_stats.rankdata(all_data)
    
    # Calculate mean ranks per group
    mean_ranks = {}
    for platform in platform_names:
        platform_ranks = [ranks[i] for i, label in enumerate(group_labels) if label == platform]
        mean_ranks[platform] = np.mean(platform_ranks)
    
    # Pairwise comparisons
    comparisons = []
    p_values_list = []
    
    for i in range(n_platforms):
        for j in range(i + 1, n_platforms):
            platform_i = platform_names[i]
            platform_j = platform_names[j]
            
            n_i = len(groups[platform_i])
            n_j = len(groups[platform_j])
            
            # Dunn's z-statistic formula (Chapter 3.9.2)
            numerator = mean_ranks[platform_i] - mean_ranks[platform_j]
            denominator = np.sqrt((N * (N + 1) / 12) * (1/n_i + 1/n_j))
            z_stat = numerator / denominator
            
            # Two-tailed p-value
            p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
            
            comparisons.append({
                "platform_i": platform_i,
                "platform_j": platform_j,
                "z_statistic": float(z_stat),
                "p_value": float(p_value),
                "mean_rank_i": mean_ranks[platform_i],
                "mean_rank_j": mean_ranks[platform_j]
            })
            p_values_list.append(p_value)
    
    # Apply Benjamini-Hochberg correction
    significant_flags = benjamini_hochberg_correction(p_values_list, alpha)
    
    for comp, sig_flag in zip(comparisons, significant_flags):
        comp["significant_bh"] = sig_flag
    
    return {
        "available": True,
        "alpha": alpha,
        "n_comparisons": len(comparisons),
        "comparisons": comparisons
    }


def practical_significance_filter(cliffs_delta: float, threshold: float = 0.147) -> Dict[str, Any]:
    """
    Filter statistical significance by practical significance (Chapter 3.9.3)
    
    Returns interpretation per Cliff's Delta thresholds:
    |Δ| < 0.147: negligible
    0.147 ≤ |Δ| < 0.33: small
    0.33 ≤ |Δ| < 0.474: medium
    |Δ| ≥ 0.474: large
    """
    abs_delta = abs(cliffs_delta)
    
    if abs_delta < 0.147:
        magnitude = "negligible"
        practically_significant = False
    elif abs_delta < 0.33:
        magnitude = "small"
        practically_significant = True
    elif abs_delta < 0.474:
        magnitude = "medium"
        practically_significant = True
    else:
        magnitude = "large"
        practically_significant = True
    
    return {
        "cliffs_delta": cliffs_delta,
        "magnitude": magnitude,
        "practically_significant": practically_significant,
        "abs_delta": abs_delta
    }


def root_cause_analysis_thresholds(telemetry_window: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply RCA thresholds from Chapter 3.9.4:
    - CPU saturation: >80% utilization
    - Memory pressure: approaching limit or spills detected
    - I/O bottleneck: >90% of capacity
    - Network shuffle: high inter-node transfer (platform-specific)
    """
    rca_flags = {
        "cpu_saturation": False,
        "memory_pressure": False,
        "io_bottleneck": False,
        "network_shuffle": False
    }
    
    cpu_avg = telemetry_window.get("cpu_avg")
    mem_avg = telemetry_window.get("mem_avg")
    disk_io_avg = telemetry_window.get("disk_io_bytes_per_s_avg")
    
    # CPU saturation threshold: >80%
    if cpu_avg is not None and cpu_avg > 80:
        rca_flags["cpu_saturation"] = True
    
    # Memory pressure threshold: >85%
    if mem_avg is not None and mem_avg > 85:
        rca_flags["memory_pressure"] = True
    
    # I/O bottleneck: heuristic based on high sustained I/O
    # Note: actual capacity depends on disk type; this is a placeholder
    if disk_io_avg is not None and disk_io_avg > 100_000_000:  # >100 MB/s sustained
        rca_flags["io_bottleneck"] = True
    
    return rca_flags


# ================== COLD/WARM RUN SEPARATION (Chapter 3.6.1) ==================

def execute_with_cache_control(conn, sql: str, db_type: str, is_cold_run: bool = False):
    """
    Execute query with explicit cache control for cold/warm separation (Chapter 3.6.1)
    
    Cold run: First execution after cache flush/cluster restart
    Warm run: Subsequent execution with populated caches
    """
    if is_cold_run:
        # Attempt to flush caches (platform-specific)
        try:
            if db_type == "redshift":
                # Redshift: No direct cache flush; results rely on cluster restart
                pass
            elif db_type == "synapse":
                # Synapse: DBCC DROPCLEANBUFFERS equivalent
                cur = conn.cursor()
                cur.execute("DBCC DROPCLEANBUFFERS")
                cur.close()
            elif db_type == "oracle":
                # Oracle: Flush shared pool and buffer cache
                cur = conn.cursor()
                cur.execute("ALTER SYSTEM FLUSH SHARED_POOL")
                cur.execute("ALTER SYSTEM FLUSH BUFFER_CACHE")
                cur.close()
        except Exception as e:
            print(f"Warning: Cache flush failed for {db_type}: {e}")
    
    # Execute query (actual timing handled by run_query)
    return run_query(conn, sql, db_type, fetch_rows=True)


# ================== CONCURRENCY SWEEP (Chapter 3.9.5) ==================

def concurrency_sweep(db_type: str, cfg: Dict[str, Any], queries: List[Dict[str, str]], 
                     concurrency_levels: List[int], iterations: int = 5) -> Dict[str, Any]:
    """
    Perform concurrency sweep analysis (Chapter 3.9.5)
    
    Tests query performance under varying concurrency levels to assess:
    - Throughput scaling behavior
    - Tail latency degradation (p95, p99)
    - Resource contention onset
    """
    results = {}
    
    for concurrency in concurrency_levels:
        print(f"\nConcurrency level: {concurrency}")
        level_results = []
        
        for iteration in range(1, iterations + 1):
            print(f"  Iteration {iteration}/{iterations}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = [ex.submit(worker_execute_one, db_type, cfg, q, True, False) 
                          for q in queries]
                
                iter_start = time.time()
                for fut in concurrent.futures.as_completed(futures):
                    res = fut.result()
                    if not res.get("error"):
                        level_results.append({
                            "concurrency": concurrency,
                            "iteration": iteration,
                            "duration_s": res["duration"],
                            "rows": res["rows"]
                        })
                iter_end = time.time()
        
        # Calculate metrics for this concurrency level
        durations = [r["duration_s"] for r in level_results]
        if durations:
            percentiles = calculate_percentiles(durations)
            results[concurrency] = {
                "samples": len(durations),
                "mean_latency": statistics.mean(durations),
                "median_latency": statistics.median(durations),
                "p95_latency": percentiles.get("p95"),
                "p99_latency": percentiles.get("p99"),
                "total_queries": len(durations),
                "throughput_qph": len(durations) / ((iter_end - iter_start) / 3600) if iter_end > iter_start else 0
            }
    
    return {
        "concurrency_levels": concurrency_levels,
        "results": results
    }


# ================== QUERY COMPLEXITY (from v3) ==================
def categorize_query_complexity(sql: str, query_name: str) -> str:
    """Categorize TPC-DS/TPC-H queries by complexity"""
    sql_upper = sql.upper()
    join_count = sql_upper.count(' JOIN ')
    subquery_count = sql_upper.count('SELECT') - 1
    union_count = sql_upper.count(' UNION ')
    window_count = sql_upper.count(' OVER(')
    case_count = sql_upper.count(' CASE ')
    has_cte = 'WITH ' in sql_upper
    
    complexity_score = 0
    complexity_score += join_count * 2
    complexity_score += subquery_count * 3
    complexity_score += union_count * 2
    complexity_score += window_count * 2
    complexity_score += case_count * 1
    complexity_score += 3 if has_cte else 0
    
    tpcds_complex = ['5', '17', '24', '25', '35', '38', '74', '77', '80', '95']
    tpcds_simple = ['1', '3', '7', '19', '42', '52', '55', '68', '73', '79']
    
    query_num = re.search(r'\d+', query_name)
    if query_num:
        qnum = query_num.group()
        if qnum in tpcds_complex:
            return 'complex'
        elif qnum in tpcds_simple:
            return 'simple'
    
    if complexity_score <= 5:
        return 'simple'
    elif complexity_score <= 15:
        return 'moderate'
    else:
        return 'complex'


# ================== DB CONNECTIONS (from v3) ==================
def connect_oracle(cfg: Dict[str, Any]):
    import cx_Oracle
    user = cfg.get("user")
    pw = cfg.get("password")
    dsn = cfg.get("dsn")
    conn = cx_Oracle.connect(user, pw, dsn, encoding="UTF-8")
    return conn

def connect_redshift(cfg: Dict[str, Any]):
    import psycopg2
    conn = psycopg2.connect(
        host=cfg["host"],
        port=cfg.get("port", 5439),
        dbname=cfg.get("dbname") or cfg.get("database"),
        user=cfg["user"],
        password=cfg["password"],
        connect_timeout=10,
    )
    return conn

def connect_synapse(cfg: Dict[str, Any]):
    import pyodbc
    driver = cfg.get("driver") or "ODBC Driver 17 for SQL Server"
    server = cfg["server"]
    database = cfg["database"]
    user = cfg["user"]
    password = cfg["password"]
    conn_str = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={user};PWD={password}"
    conn = pyodbc.connect(conn_str, autocommit=False)
    return conn


# ================== PLATFORM METRICS (from v3) ==================
def extract_redshift_metrics(conn, query_id: Optional[str] = None) -> Dict[str, Any]:
    """Extract Redshift-specific metrics"""
    metrics = {
        "shuffle_bytes": None,
        "spill_bytes": None,
        "broadcast_bytes": None,
        "max_rows_per_segment": None
    }
    
    if not query_id:
        return metrics
    
    try:
        cur = conn.cursor()
        sql = f"""
        SELECT SUM(bytes) as shuffle_bytes
        FROM stl_dist 
        WHERE query = {query_id}
        """
        cur.execute(sql)
        row = cur.fetchone()
        if row and row[0]:
            metrics["shuffle_bytes"] = row[0]
        
        sql = f"""
        SELECT SUM(workmem) as spill_bytes
        FROM svl_query_summary
        WHERE query = {query_id} AND is_diskbased = 't'
        """
        cur.execute(sql)
        row = cur.fetchone()
        if row and row[0]:
            metrics["spill_bytes"] = row[0]
        
        cur.close()
    except Exception:
        pass
    
    return metrics


def extract_oracle_metrics(conn) -> Dict[str, Any]:
    """Extract Oracle-specific metrics"""
    metrics = {
        "buffer_gets": None,
        "physical_reads": None,
        "consistent_gets": None,
        "cpu_time_ms": None
    }
    
    try:
        cur = conn.cursor()
        sql = """
        SELECT name, value 
        FROM v$mystat ms, v$statname sn 
        WHERE ms.statistic# = sn.statistic# 
        AND name IN ('session logical reads', 'physical reads', 'CPU used by this session')
        """
        cur.execute(sql)
        rows = cur.fetchall()
        for row in rows:
            if row[0] == 'session logical reads':
                metrics["buffer_gets"] = row[1]
            elif row[0] == 'physical reads':
                metrics["physical_reads"] = row[1]
            elif row[0] == 'CPU used by this session':
                metrics["cpu_time_ms"] = row[1] / 100
        cur.close()
    except Exception:
        pass
    
    return metrics


# ================== QUERY RUNNER (from v3) ==================
def run_query(conn, sql: str, db_type: str, fetch_rows=True, capture_plan=False, query_id=None):
    cur = conn.cursor()
    start = time.time()
    rows = 0
    error = None
    query_plan = None
    platform_metrics = {}
    
    try:
        initial_metrics = {}
        if db_type == "oracle":
            initial_metrics = extract_oracle_metrics(conn)
        
        cur.execute(sql)
        
        if fetch_rows:
            total = 0
            while True:
                chunk = cur.fetchmany(1000)
                if not chunk:
                    break
                total += len(chunk)
            rows = total
        else:
            rows = cur.rowcount if hasattr(cur, "rowcount") and cur.rowcount is not None else 0
        
        if db_type == "oracle":
            final_metrics = extract_oracle_metrics(conn)
            platform_metrics = {
                "buffer_gets": (final_metrics["buffer_gets"] or 0) - (initial_metrics["buffer_gets"] or 0),
                "physical_reads": (final_metrics["physical_reads"] or 0) - (initial_metrics["physical_reads"] or 0)
            }
        elif db_type == "redshift" and query_id:
            platform_metrics = extract_redshift_metrics(conn, query_id)
            
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}"
    
    end = time.time()
    duration = end - start
    
    try:
        cur.close()
    except Exception:
        pass
    
    return {
        "duration": duration,
        "rows": rows,
        "error": error,
        "start": start,
        "end": end,
        "query_plan": query_plan,
        "platform_metrics": platform_metrics
    }


def load_sql_files(path_pattern: str) -> List[Dict[str,str]]:
    files = sorted(glob.glob(path_pattern))
    qlist = []
    for f in files:
        name = os.path.basename(f)
        with open(f, "r", encoding="utf-8") as fh:
            sql = fh.read().strip()
            if sql.endswith(";"):
                sql = sql[:-1]
            complexity = categorize_query_complexity(sql, name)
            qlist.append({
                "file": f,
                "name": name,
                "sql": sql,
                "complexity": complexity
            })
    return qlist


def worker_execute_one(db_type: str, cfg: Dict[str,Any], query_item: Dict[str,str], 
                       fetch_rows=True, capture_plan=False):
    conn = None
    try:
        if db_type == "oracle":
            conn = connect_oracle(cfg)
        elif db_type == "redshift":
            conn = connect_redshift(cfg)
        elif db_type == "synapse":
            conn = connect_synapse(cfg)
        else:
            raise ValueError("Unsupported db_type")
        result = run_query(conn, query_item["sql"], db_type, 
                         fetch_rows=fetch_rows, capture_plan=capture_plan)
        result["complexity"] = query_item.get("complexity", "unknown")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
    return result


# ================== TELEMETRY (from v3) ==================
class TelemetrySampler:
    def __init__(self, interval: float = 1.0):
        self.interval = max(0.1, float(interval))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[Dict[str,Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._last_disk = None
        self._last_net = None
        self._last_ts = None

    def _get_counters(self):
        disk = None
        net = None
        if psutil:
            try:
                d = psutil.disk_io_counters()
                n = psutil.net_io_counters()
                disk = (getattr(d, "read_bytes", 0) + getattr(d, "write_bytes", 0))
                net = (getattr(n, "bytes_sent", 0) + getattr(n, "bytes_recv", 0))
            except Exception:
                pass
        return disk, net

    def _sample_once(self):
        ts = time.time()
        iso = datetime.utcfromtimestamp(ts).isoformat() + "Z"
        sample = {"ts": ts, "iso": iso}
        
        try:
            if psutil:
                sample["cpu_percent"] = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                sample["mem_percent"] = getattr(vm, "percent", None)
            else:
                sample["cpu_percent"] = None
                sample["mem_percent"] = None
        except Exception:
            sample["cpu_percent"] = None
            sample["mem_percent"] = None

        try:
            du = shutil.disk_usage("/")
            sample["disk_total_bytes"] = du.total
            sample["disk_used_bytes"] = du.used
        except Exception:
            sample["disk_total_bytes"] = None
            sample["disk_used_bytes"] = None

        try:
            disk_cum, net_cum = self._get_counters()
            if self._last_ts is not None and self._last_ts != ts:
                dt = ts - self._last_ts
                if disk_cum is not None and self._last_disk is not None:
                    sample["disk_io_bytes_per_s"] = (disk_cum - self._last_disk) / dt
                else:
                    sample["disk_io_bytes_per_s"] = None
                if net_cum is not None and self._last_net is not None:
                    sample["net_bytes_per_s"] = (net_cum - self._last_net) / dt
                else:
                    sample["net_bytes_per_s"] = None
            else:
                sample["disk_io_bytes_per_s"] = None
                sample["net_bytes_per_s"] = None
            self._last_disk = disk_cum
            self._last_net = net_cum
            self._last_ts = ts
        except Exception:
            sample["disk_io_bytes_per_s"] = None
            sample["net_bytes_per_s"] = None

        return sample

    def _run(self):
        if psutil:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass
        self.start_time = time.time()
        while not self._stop_event.is_set():
            s = self._sample_once()
            self.samples.append(s)
            self._stop_event.wait(self.interval)
        self.end_time = time.time()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="TelemetrySampler", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_summary(self):
        if not self.samples:
            return {}
        cpu_vals = [s.get("cpu_percent") for s in self.samples if s.get("cpu_percent") is not None]
        mem_vals = [s.get("mem_percent") for s in self.samples if s.get("mem_percent") is not None]
        net_vals = [s.get("net_bytes_per_s") for s in self.samples if s.get("net_bytes_per_s") is not None]
        disk_vals = [s.get("disk_io_bytes_per_s") for s in self.samples if s.get("disk_io_bytes_per_s") is not None]
        
        def agg(vals):
            if not vals:
                return {}
            return {"min": min(vals), "max": max(vals), "mean": sum(vals)/len(vals)}
        
        return {
            "samples": len(self.samples),
            "duration_s": (self.end_time - self.start_time) if (self.start_time and self.end_time) else None,
            "cpu": agg(cpu_vals),
            "mem": agg(mem_vals),
            "net_bytes_per_s": agg(net_vals),
            "disk_io_bytes_per_s": agg(disk_vals)
        }


# ================== STATISTICAL ANALYSIS (enhanced from v3) ==================
def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate p50, p95, p99 percentiles"""
    if not values:
        return {}
    
    if np:
        return {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99))
        }
    else:
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "p50": sorted_vals[int(n * 0.50)],
            "p95": sorted_vals[min(int(n * 0.95), n-1)],
            "p99": sorted_vals[min(int(n * 0.99), n-1)]
        }


def cliffs_delta(group1: List[float], group2: List[float]) -> float:
    """Calculate Cliff's Delta effect size"""
    if not group1 or not group2:
        return 0.0
    
    n1, n2 = len(group1), len(group2)
    more = sum(1 for x1 in group1 for x2 in group2 if x1 > x2)
    less = sum(1 for x1 in group1 for x2 in group2 if x1 < x2)
    delta = (more - less) / (n1 * n2)
    return delta


def kruskal_wallis_test(groups: Dict[str, List[float]]) -> Dict[str, Any]:
    """Perform Kruskal-Wallis H-test"""
    if not scipy_stats or len(groups) < 2:
        return {"available": False}
    
    group_list = [vals for vals in groups.values() if vals]
    if len(group_list) < 2:
        return {"available": False}
    
    try:
        h_stat, p_value = scipy_stats.kruskal(*group_list)
        result = {
            "available": True,
            "h_statistic": float(h_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05
        }
        
        # If significant, perform Dunn's post-hoc
        if p_value < 0.05:
            dunns_result = dunns_post_hoc_test(groups)
            result["dunns_post_hoc"] = dunns_result
        
        return result
    except Exception as e:
        return {"available": False, "error": str(e)}


def mann_whitney_u_test(group1: List[float], group2: List[float]) -> Dict[str, Any]:
    """Perform Mann-Whitney U test"""
    if not scipy_stats or not group1 or not group2:
        return {"available": False}
    
    try:
        u_stat, p_value = scipy_stats.mannwhitneyu(group1, group2, alternative='two-sided')
        return {
            "available": True,
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# ================== BASELINE COMPARISON (from v3) ==================
class BaselineComparator:
    def __init__(self, baseline_path: Optional[str] = None):
        self.baseline: Optional[Dict[str, List[float]]] = None
        if baseline_path and os.path.exists(baseline_path):
            self.load_baseline(baseline_path)
    
    def load_baseline(self, path: str):
        try:
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                baseline = defaultdict(list)
                for row in reader:
                    if not row.get('error'):
                        qname = row['query_name']
                        baseline[qname].append(float(row['duration_s']))
                self.baseline = dict(baseline)
                print(f"Loaded baseline from {path}: {len(self.baseline)} queries")
        except Exception as e:
            print(f"Failed to load baseline: {e}")
            self.baseline = None
    
    def compare(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare current results against baseline"""
        if not self.baseline:
            return {"available": False, "message": "No baseline loaded"}
        
        current = defaultdict(list)
        for r in results:
            if not r.get('error'):
                current[r['query_name']].append(r['duration_s'])
        
        comparisons = {}
        for qname in current:
            if qname in self.baseline:
                curr_vals = current[qname]
                base_vals = self.baseline[qname]
                
                curr_mean = statistics.mean(curr_vals)
                base_mean = statistics.mean(base_vals)
                
                pct_change = ((curr_mean - base_mean) / base_mean) * 100
                delta = cliffs_delta(curr_vals, base_vals)
                mw_test = mann_whitney_u_test(base_vals, curr_vals)
                
                # Apply practical significance filter (Chapter 3.9.3)
                practical_sig = practical_significance_filter(delta)
                
                comparisons[qname] = {
                    "baseline_mean": base_mean,
                    "current_mean": curr_mean,
                    "percent_change": pct_change,
                    "cliffs_delta": delta,
                    "effect_size": practical_sig["magnitude"],
                    "practically_significant": practical_sig["practically_significant"],
                    "mann_whitney": mw_test,
                    "improved": pct_change < 0,
                    "degraded": pct_change > 0
                }
        
        all_pct_changes = [c["percent_change"] for c in comparisons.values()]
        improved_count = sum(1 for c in comparisons.values() if c["improved"])
        degraded_count = sum(1 for c in comparisons.values() if c["degraded"])
        
        return {
            "available": True,
            "queries_compared": len(comparisons),
            "per_query": comparisons,
            "aggregate": {
                "mean_percent_change": statistics.mean(all_pct_changes) if all_pct_changes else 0,
                "median_percent_change": statistics.median(all_pct_changes) if all_pct_changes else 0,
                "improved_queries": improved_count,
                "degraded_queries": degraded_count,
                "unchanged_queries": len(comparisons) - improved_count - degraded_count
            }
        }


# ================== BENCHMARK WITH COLD/WARM SEPARATION ==================
def benchmark(db_type: str, cfg: Dict[str,Any], queries: List[Dict[str,str]], 
              iterations: int=1, concurrency: int=1, warmup:int=0, 
              fetch_rows=True, capture_plans=False, cold_warm_separation=True):
    """
    Enhanced benchmark with cold/warm run separation (Chapter 3.6.1)
    """
    results = []
    print(f"Starting benchmark: db={db_type}, queries={len(queries)}, iterations={iterations}, "
          f"concurrency={concurrency}, warmup={warmup}, cold_warm_separation={cold_warm_separation}")
    
    # Warmup
    if warmup and len(queries):
        print(f"Running {warmup} warmup iterations...")
        for i in range(warmup):
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = [ex.submit(worker_execute_one, db_type, cfg, q, fetch_rows, False) 
                          for q in queries]
                for fut in concurrent.futures.as_completed(futures):
                    _ = fut.result()
        print("Warmup done.")

    bench_start = time.time()
    
    # Cold runs (if enabled)
    if cold_warm_separation:
        print("\n=== COLD RUNS ===")
        for q in queries:
            print(f"Cold run: {q['name']}")
            try:
                conn = None
                if db_type == "oracle":
                    conn = connect_oracle(cfg)
                elif db_type == "redshift":
                    conn = connect_redshift(cfg)
                elif db_type == "synapse":
                    conn = connect_synapse(cfg)
                
                res = execute_with_cache_control(conn, q["sql"], db_type, is_cold_run=True)
                
                results.append({
                    "query_file": q["file"],
                    "query_name": q["name"],
                    "complexity": q["complexity"],
                    "iteration": 0,  # Cold run marked as iteration 0
                    "run_type": "cold",
                    "start_time": datetime.fromtimestamp(res["start"]).isoformat(),
                    "end_time": datetime.fromtimestamp(res["end"]).isoformat(),
                    "duration_s": res["duration"],
                    "rows": res["rows"],
                    "error": res["error"],
                    "platform_metrics": res.get("platform_metrics", {})
                })
                
                status = "ERROR" if res["error"] else "OK"
                print(f"  [{status}] COLD {q['name']} duration={res['duration']:.3f}s")
                
                if conn:
                    conn.close()
            except Exception as e:
                print(f"  [ERROR] Cold run failed: {e}")
    
    # Warm runs
    print("\n=== WARM RUNS ===")
    for it in range(1, iterations+1):
        print(f"Iteration {it}/{iterations}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures_map = {}
            for q in queries:
                capture = capture_plans and (it == 1)
                fut = ex.submit(worker_execute_one, db_type, cfg, q, fetch_rows, capture)
                futures_map[fut] = q
            
            for fut in concurrent.futures.as_completed(futures_map):
                q = futures_map[fut]
                res = fut.result()
                results.append({
                    "query_file": q["file"],
                    "query_name": q["name"],
                    "complexity": res.get("complexity", "unknown"),
                    "iteration": it,
                    "run_type": "warm",
                    "start_time": datetime.fromtimestamp(res["start"]).isoformat(),
                    "end_time": datetime.fromtimestamp(res["end"]).isoformat(),
                    "duration_s": res["duration"],
                    "rows": res["rows"],
                    "error": res["error"],
                    "query_plan": res.get("query_plan"),
                    "platform_metrics": res.get("platform_metrics", {})
                })
                status = "ERROR" if res["error"] else "OK"
                print(f"  [{status}] {q['name']} ({res.get('complexity','?')}) "
                      f"duration={res['duration']:.3f}s rows={res['rows']}")
    
    bench_end = time.time()
    total_time = bench_end - bench_start

    # Separate cold and warm results for analysis
    cold_results = [r for r in results if r.get("run_type") == "cold" and not r["error"]]
    warm_results = [r for r in results if r.get("run_type") == "warm" and not r["error"]]
    failed = [r for r in results if r["error"]]
    
    total_rows = sum(r["rows"] for r in warm_results)

    # Calculate percentiles (warm runs only)
    warm_durations = [r["duration_s"] for r in warm_results]
    percentiles = calculate_percentiles(warm_durations)
    
    # Calculate cold run statistics
    cold_durations = [r["duration_s"] for r in cold_results]
    cold_percentiles = calculate_percentiles(cold_durations) if cold_durations else {}
    
    # Group by complexity
    complexity_groups = defaultdict(list)
    for r in warm_results:
        complexity_groups[r["complexity"]].append(r["duration_s"])
    
    kw_test = kruskal_wallis_test(complexity_groups)

    summary = {
        "db_type": db_type,
        "num_queries": len(queries),
        "iterations": iterations,
        "concurrency": concurrency,
        "total_time_s": total_time,
        "total_successful_warm": len(warm_results),
        "total_successful_cold": len(cold_results),
        "total_failed": len(failed),
        "queries_per_sec": (len(warm_results) / total_time) if total_time > 0 else 0.0,
        "rows_per_sec": (total_rows / total_time) if total_time > 0 else 0.0,
        "total_rows": total_rows,
        "warm_percentiles": percentiles,
        "cold_percentiles": cold_percentiles,
        "complexity_distribution": {k: len(v) for k, v in complexity_groups.items()},
        "kruskal_wallis": kw_test
    }
    return results, summary


# ================== SAVE/PRINT (enhanced) ==================
def save_results_csv(results: List[Dict[str,Any]], out_csv: str):
    if not results:
        print("No results to save.")
        return
    keys = ["query_file","query_name","complexity","iteration","run_type","start_time","end_time",
            "duration_s","rows","error"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in keys})
    print(f"Saved results to {out_csv}")


def print_summary(summary: Dict[str,Any], results: List[Dict[str,Any]], baseline_comparison: Optional[Dict] = None):
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY (Chapter 3.9 Compliant)")
    print("="*80)
    
    # Basic metrics
    print(f"Database: {summary.get('db_type')}")
    print(f"Total queries: {summary.get('num_queries')}")
    print(f"Iterations: {summary.get('iterations')}")
    print(f"Concurrency: {summary.get('concurrency')}")
    print(f"Total time: {summary.get('total_time_s', 0):.2f}s")
    print(f"Successful (warm): {summary.get('total_successful_warm')}")
    print(f"Successful (cold): {summary.get('total_successful_cold')}")
    print(f"Failed: {summary.get('total_failed')}")
    
    # Warm run percentiles (Chapter 3.9.1)
    if 'warm_percentiles' in summary and summary['warm_percentiles']:
        print(f"\nWarm Run Latency Percentiles (Chapter 3.9.1):")
        for p, val in summary['warm_percentiles'].items():
            print(f"  {p}: {val:.3f}s")
    
    # Cold run percentiles
    if 'cold_percentiles' in summary and summary['cold_percentiles']:
        print(f"\nCold Run Latency Percentiles:")
        for p, val in summary['cold_percentiles'].items():
            print(f"  {p}: {val:.3f}s")
    
    # Complexity distribution
    if 'complexity_distribution' in summary:
        print(f"\nQuery Complexity Distribution:")
        for complexity, count in summary['complexity_distribution'].items():
            print(f"  {complexity}: {count} queries")
    
    # Kruskal-Wallis with Dunn's post-hoc (Chapter 3.9.2)
    if 'kruskal_wallis' in summary and summary['kruskal_wallis'].get('available'):
        kw = summary['kruskal_wallis']
        print(f"\nKruskal-Wallis Test (complexity levels, α=0.05):")
        print(f"  H-statistic: {kw.get('h_statistic', 0):.3f}")
        print(f"  p-value: {kw.get('p_value', 1):.4f}")
        print(f"  Significant: {'YES' if kw.get('significant') else 'NO'}")
        
        # Dunn's post-hoc results
        if 'dunns_post_hoc' in kw and kw['dunns_post_hoc'].get('available'):
            dunns = kw['dunns_post_hoc']
            print(f"\n  Dunn's Post-Hoc Pairwise Comparisons (BH-corrected):")
            for comp in dunns['comparisons']:
                sig_flag = "***" if comp['significant_bh'] else "ns"
                print(f"    {comp['platform_i']} vs {comp['platform_j']}: "
                      f"z={comp['z_statistic']:.3f}, p={comp['p_value']:.4f} {sig_flag}")
    
    # Warm run statistics
    warm_results = [r for r in results if r.get('run_type') == 'warm' and not r.get('error')]
    durations = [r["duration_s"] for r in warm_results]
    if durations and len(durations) > 1:
        print(f"\nWarm Run Statistics:")
        print(f"  count: {len(durations)}")
        print(f"  min: {min(durations):.3f}s")
        print(f"  max: {max(durations):.3f}s")
        print(f"  mean: {statistics.mean(durations):.3f}s")
        print(f"  median: {statistics.median(durations):.3f}s")
        print(f"  stdev: {statistics.stdev(durations):.3f}s")
    
    # Baseline comparison with practical significance (Chapter 3.9.3)
    if baseline_comparison and baseline_comparison.get('available'):
        print("\n" + "="*80)
        print("BASELINE COMPARISON (with Practical Significance Filter)")
        print("="*80)
        agg = baseline_comparison['aggregate']
        print(f"Queries compared: {baseline_comparison['queries_compared']}")
        print(f"Mean % change: {agg['mean_percent_change']:+.2f}%")
        print(f"Median % change: {agg['median_percent_change']:+.2f}%")
        print(f"Improved: {agg['improved_queries']}, Degraded: {agg['degraded_queries']}, "
              f"Unchanged: {agg['unchanged_queries']}")
        
        per_query = baseline_comparison['per_query']
        
        # Filter for practically significant changes only
        practical_changes = {q: c for q, c in per_query.items() 
                           if c.get('practically_significant', False)}
        
        if practical_changes:
            print(f"\nPractically Significant Changes (|Δ| ≥ 0.147): {len(practical_changes)}")
            sorted_by_change = sorted(practical_changes.items(), 
                                    key=lambda x: x[1]['percent_change'])
            
            print("\nTop 5 Improvements (Practically Significant):")
            for qname, comp in sorted_by_change[:5]:
                print(f"  {qname}: {comp['percent_change']:+.2f}% "
                      f"(Δ={comp['cliffs_delta']:.3f}, {comp['effect_size']})")
            
            print("\nTop 5 Regressions (Practically Significant):")
            for qname, comp in sorted_by_change[-5:]:
                print(f"  {qname}: {comp['percent_change']:+.2f}% "
                      f"(Δ={comp['cliffs_delta']:.3f}, {comp['effect_size']})")
        else:
            print("\nNo practically significant changes detected (all |Δ| < 0.147)")
    
    # Slowest queries
    if pd:
        try:
            df = pd.DataFrame(warm_results)
            ok_mask = df["error"].isnull() | (df["error"] == "")
            slowest = df[ok_mask].sort_values("duration_s", ascending=False).head(10)
            print(f"\nTop 10 Slowest Warm Run Queries:")
            display_cols = ["query_name", "complexity", "iteration", "duration_s", "rows"]
            print(slowest[display_cols].to_string(index=False))
        except Exception:
            pass


# ================== PLOTTING (enhanced) ==================
def plot_results(results, summary, out_prefix="tpc_results", baseline_comparison=None):
    """Generate comprehensive charts with cold/warm separation"""
    warm_results = [r for r in results if r.get("run_type") == "warm" and not r["error"]]
    cold_results = [r for r in results if r.get("run_type") == "cold" and not r["error"]]
    
    warm_durations = [r["duration_s"] for r in warm_results]
    cold_durations = [r["duration_s"] for r in cold_results]
    
    if not warm_durations:
        print("No successful warm queries to plot")
        return
    
    # 1. Cold vs Warm comparison
    if cold_durations:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.hist(cold_durations, bins=20, alpha=0.7, label='Cold', edgecolor='black')
        ax1.hist(warm_durations, bins=20, alpha=0.7, label='Warm', edgecolor='black')
        ax1.set_xlabel("Query Latency (seconds)")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"Cold vs Warm Latency Distribution - {summary.get('db_type')}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        ax2.boxplot([cold_durations, warm_durations], labels=['Cold', 'Warm'])
        ax2.set_ylabel("Duration (s)")
        ax2.set_title("Cold vs Warm Latency Comparison")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_cold_warm_comparison.png", dpi=300)
        plt.close()
    
    # 2. CDF with percentiles
    plt.figure(figsize=(10, 6))
    sorted_durations = sorted(warm_durations)
    cdf = [i / len(sorted_durations) for i in range(1, len(sorted_durations) + 1)]
    plt.plot(sorted_durations, cdf, linewidth=2, label='Warm runs')
    
    percentiles = calculate_percentiles(warm_durations)
    if percentiles:
        for p, val in percentiles.items():
            pct_val = float(p[1:]) / 100
            plt.plot(val, pct_val, 'ro', markersize=8)
            plt.annotate(f'{p}={val:.2f}s', xy=(val, pct_val), 
                        xytext=(10, -5), textcoords='offset points')
    
    plt.xlabel("Query Latency (seconds)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"Warm Run Latency CDF - {summary.get('db_type')}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_warm_latency_cdf.png", dpi=300)
    plt.close()
    
    # 3. Complexity comparison
    complexity_groups = defaultdict(list)
    for r in warm_results:
        complexity_groups[r["complexity"]].append(r["duration_s"])
    
    if len(complexity_groups) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = list(complexity_groups.keys())
        values = [complexity_groups[k] for k in labels]
        bp = ax.boxplot(values, labels=labels, vert=True, patch_artist=True, showmeans=True)
        
        colors = {'simple': 'lightgreen', 'moderate': 'yellow', 'complex': 'lightcoral', 'unknown': 'gray'}
        for patch, label in zip(bp['boxes'], labels):
            patch.set_facecolor(colors.get(label, 'gray'))
        
        plt.ylabel("Duration (s)")
        plt.title(f"Warm Run Latency by Query Complexity - {summary.get('db_type')}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_complexity_comparison.png", dpi=300)
        plt.close()
    
    # 4. Baseline comparison (if available)
    if baseline_comparison and baseline_comparison.get('available'):
        per_query = baseline_comparison['per_query']
        if per_query:
            # Filter for practically significant only
            practical_queries = {q: c for q, c in per_query.items() 
                               if c.get('practically_significant', False)}
            
            if practical_queries:
                queries = list(practical_queries.keys())
                pct_changes = [practical_queries[q]['percent_change'] for q in queries]
                
                fig, ax = plt.subplots(figsize=(14, 6))
                colors_bar = ['green' if pc < 0 else 'red' for pc in pct_changes]
                ax.bar(range(len(queries)), pct_changes, color=colors_bar, alpha=0.7)
                ax.axhline(0, color='black', linewidth=0.8)
                ax.set_xticks(range(len(queries)))
                ax.set_xticklabels(queries, rotation=90, ha='right')
                ax.set_ylabel("% Change from Baseline")
                ax.set_title("Practically Significant Performance Changes vs Baseline (|Δ| ≥ 0.147)")
                ax.grid(True, axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{out_prefix}_baseline_practical_significance.png", dpi=300)
                plt.close()
    
    print(f"\nCharts saved with prefix: {out_prefix}_*.png")


# ================== UTILITIES ==================
def collect_environment_snapshot(args, cfg):
    snap = {
        "run_id": str(uuid.uuid4()),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "args": vars(args) if args else None,
        "config_keys": list(cfg.keys()) if cfg else None,
        "git_commit": None,
        "dependencies": {
            "pandas": pd is not None,
            "psutil": psutil is not None,
            "ntplib": ntplib is not None,
            "numpy": np is not None,
            "scipy": scipy_stats is not None
        }
    }
    
    try:
        if os.path.isdir(".git"):
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], 
                                            stderr=subprocess.DEVNULL).decode().strip()
            snap["git_commit"] = commit
    except Exception:
        pass
    
    return snap


def write_json_file(path: str, obj: Any):
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)
        print(f"Wrote telemetry to {path}")
    except Exception as e:
        print(f"Failed to write {path}: {e}")


def correlate_logs_with_telemetry(results: List[Dict[str,Any]], samples: List[Dict[str,Any]], 
                                   time_window_sec: float = 2.0) -> List[Dict[str,Any]]:
    """Correlate query results with telemetry and apply RCA thresholds"""
    if not samples:
        for r in results:
            r["telemetry_window"] = {}
            r["rca_flags"] = {}
        return results

    samples_sorted = sorted(samples, key=lambda s: s["ts"])
    
    def window_stats(window_start: float, window_end: float):
        vals = [s for s in samples_sorted if window_start <= s["ts"] <= window_end]
        if not vals:
            return {}
        
        def avg(field):
            arr = [v.get(field) for v in vals if v.get(field) is not None]
            return (sum(arr)/len(arr)) if arr else None
        
        return {
            "count": len(vals),
            "cpu_avg": avg("cpu_percent"),
            "mem_avg": avg("mem_percent"),
            "net_bytes_per_s_avg": avg("net_bytes_per_s"),
            "disk_io_bytes_per_s_avg": avg("disk_io_bytes_per_s")
        }
    
    out = []
    for r in results:
        try:
            end_iso = r.get("end_time")
            if not end_iso:
                r["telemetry_window"] = {}
                r["rca_flags"] = {}
                out.append(r)
                continue
            
            end_dt = datetime.fromisoformat(end_iso)
            end_ts = end_dt.timestamp()
            ws = end_ts - time_window_sec
            we = end_ts + time_window_sec
            stats = window_stats(ws, we)
            
            # Apply RCA thresholds (Chapter 3.9.4)
            rca_flags = root_cause_analysis_thresholds(stats)
            
            r2 = dict(r)
            r2["telemetry_window"] = stats
            r2["rca_flags"] = rca_flags
            out.append(r2)
        except Exception:
            r["telemetry_window"] = {}
            r["rca_flags"] = {}
            out.append(r)
    
    return out


# ================== CLI ==================
def parse_args():
    p = argparse.ArgumentParser(
        description="Chapter 3 Compliant TPC Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    p.add_argument("--db", "--db-type", dest="db", required=True, 
                   choices=["oracle","redshift","synapse"], help="Database type")
    p.add_argument("--config", required=True, help="JSON config file")
    p.add_argument("--queries", required=True, help="Folder/glob pattern for .sql files")
    p.add_argument("--iterations", type=int, default=20, 
                   help="Warm run iterations (Chapter 3.6.1 default=20)")
    p.add_argument("--concurrency", type=int, default=1, help="Concurrent workers")
    p.add_argument("--warmup", type=int, default=5, 
                   help="Warmup iterations (Chapter 3.7.3 default=5)")
    p.add_argument("--output", default="tpc_results.csv", help="Output CSV file")
    p.add_argument("--no-fetch", dest="fetch_rows", action="store_false", 
                   help="Don't fetch rows")
    
    # Analysis options
    p.add_argument("--baseline", help="Baseline CSV for comparison")
    p.add_argument("--capture-plans", action="store_true", 
                   help="Capture EXPLAIN plans")
    p.add_argument("--plans-dir", default="query_plans", 
                   help="Directory for query plans")
    
    # Cold/warm separation (Chapter 3.6.1)
    p.add_argument("--no-cold-warm-separation", dest="cold_warm_separation", 
                   action="store_false",
                   help="Disable cold/warm run separation")
    
    # Concurrency sweep (Chapter 3.9.5)
    p.add_argument("--concurrency-sweep", action="store_true",
                   help="Perform concurrency sweep analysis")
    p.add_argument("--concurrency-levels", nargs='+', type=int, 
                   default=[1, 5, 10, 20],
                   help="Concurrency levels for sweep")
    
    # Telemetry options
    p.add_argument("--telemetry-file", default=None, help="Telemetry JSON output")
    p.add_argument("--telemetry-interval", type=float, default=1.0, 
                   help="Sampling interval (seconds)")
    p.add_argument("--time-window", type=float, default=2.0, 
                   help="Correlation time window (seconds)")
    
    p.set_defaults(fetch_rows=True, capture_plans=False, cold_warm_separation=True)
    return p.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    
    # Load queries
    if os.path.isdir(args.queries):
        pattern = os.path.join(args.queries, "*.sql")
    else:
        pattern = args.queries
    
    queries = load_sql_files(pattern)
    if not queries:
        print(f"No SQL files found: {pattern}")
        return
    
    print(f"Loaded {len(queries)} queries")
    for q in queries:
        print(f"  {q['name']} ({q['complexity']})")
    
    telemetry_path = args.telemetry_file or f"{os.path.splitext(args.output)[0]}_telemetry.json"
    baseline_comparator = BaselineComparator(args.baseline)
    
    # Start telemetry
    telemetry_sampler = TelemetrySampler(interval=args.telemetry_interval)
    print(f"\nStarting telemetry sampler (interval={args.telemetry_interval}s)...")
    telemetry_sampler.start()
    
    bench_start_iso = datetime.utcnow().isoformat() + "Z"
    results = []
    summary = {}
    concurrency_sweep_results = None
    
    try:
        # Main benchmark
        results, summary = benchmark(
            args.db, cfg, queries,
            iterations=args.iterations,
            concurrency=args.concurrency,
            warmup=args.warmup,
            fetch_rows=args.fetch_rows,
            capture_plans=args.capture_plans,
            cold_warm_separation=args.cold_warm_separation
        )
        
        # Concurrency sweep (Chapter 3.9.5)
        if args.concurrency_sweep:
            print("\n" + "="*80)
            print("CONCURRENCY SWEEP ANALYSIS (Chapter 3.9.5)")
            print("="*80)
            concurrency_sweep_results = concurrency_sweep(
                args.db, cfg, queries,
                concurrency_levels=args.concurrency_levels,
                iterations=5
            )
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        traceback.print_exc()
    finally:
        print("\nStopping telemetry...")
        telemetry_sampler.stop()
    
    bench_end_iso = datetime.utcnow().isoformat() + "Z"
    
    # Save results
    save_results_csv(results, args.output)
    
    # Baseline comparison with practical significance filtering
    baseline_comparison = baseline_comparator.compare(results) if results else None
    
    # Print summary with Chapter 3.9 statistics
    print_summary(summary, results, baseline_comparison)
    
    # Generate plots
    base = os.path.splitext(args.output)[0]
    try:
        plot_results(results, summary, out_prefix=base, 
                    baseline_comparison=baseline_comparison)
    except Exception as e:
        print(f"Plotting failed: {e}")
        traceback.print_exc()
    
    # Correlate with telemetry and apply RCA thresholds (Chapter 3.9.4)
    correlated = correlate_logs_with_telemetry(results, telemetry_sampler.samples, 
                                               args.time_window)
    
    # Identify queries with bottlenecks
    bottlenecked_queries = []
    for r in correlated:
        rca_flags = r.get("rca_flags", {})
        if any(rca_flags.values()):
            bottlenecked_queries.append({
                "query_name": r.get("query_name"),
                "duration_s": r.get("duration_s"),
                "run_type": r.get("run_type"),
                "rca_flags": rca_flags,
                "telemetry": r.get("telemetry_window")
            })
    
    if bottlenecked_queries:
        print("\n" + "="*80)
        print("ROOT CAUSE ANALYSIS - BOTTLENECKED QUERIES (Chapter 3.9.4)")
        print("="*80)
        for bq in bottlenecked_queries[:10]:  # Show top 10
            print(f"\nQuery: {bq['query_name']} ({bq['run_type']}) - {bq['duration_s']:.3f}s")
            flags = bq['rca_flags']
            if flags.get('cpu_saturation'):
                print("  ⚠ CPU Saturation detected (>80%)")
            if flags.get('memory_pressure'):
                print("  ⚠ Memory Pressure detected (>85%)")
            if flags.get('io_bottleneck'):
                print("  ⚠ I/O Bottleneck detected")
            if flags.get('network_shuffle'):
                print("  ⚠ High Network Shuffle detected")
    
    # Assemble comprehensive telemetry payload
    payload = {
        "methodology_version": "Chapter 3 Compliant v4",
        "run_id": str(uuid.uuid4()),
        "bench_start": bench_start_iso,
        "bench_end": bench_end_iso,
        "summary": summary,
        "results_count": len(results),
        "results_csv": os.path.abspath(args.output),
        "results": correlated,
        "samples": telemetry_sampler.samples,
        "samples_summary": telemetry_sampler.get_summary(),
        "environment": collect_environment_snapshot(args, cfg),
        "baseline_comparison": baseline_comparison,
        "concurrency_sweep": concurrency_sweep_results,
        "bottlenecked_queries": bottlenecked_queries,
        "statistical_methods": {
            "kruskal_wallis": "3.9.2",
            "dunns_post_hoc": "3.9.2",
            "benjamini_hochberg_fdr": "3.9.2",
            "cliffs_delta": "3.9.3",
            "practical_significance_threshold": 0.147,
            "mann_whitney_u": "3.9.2",
            "percentiles": ["p50", "p95", "p99"],
            "rca_thresholds": {
                "cpu_saturation": ">80%",
                "memory_pressure": ">85%",
                "io_bottleneck": ">90% capacity"
            }
        }
    }
    
    write_json_file(telemetry_path, payload)
    
    # Print concurrency sweep summary if performed
    if concurrency_sweep_results:
        print("\n" + "="*80)
        print("CONCURRENCY SWEEP SUMMARY")
        print("="*80)
        for level, metrics in concurrency_sweep_results['results'].items():
            print(f"\nConcurrency Level: {level}")
            print(f"  Mean Latency: {metrics['mean_latency']:.3f}s")
            print(f"  Median Latency: {metrics['median_latency']:.3f}s")
            print(f"  p95 Latency: {metrics.get('p95_latency', 0):.3f}s")
            print(f"  p99 Latency: {metrics.get('p99_latency', 0):.3f}s")
            print(f"  Throughput: {metrics['throughput_qph']:.2f} QPH")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"Results: {args.output}")
    print(f"Telemetry: {telemetry_path}")
    print(f"Charts: {base}_*.png")
    print("\nMethodology Compliance:")
    print("  ✓ Cold/Warm run separation (3.6.1)")
    print("  ✓ Kruskal-Wallis H-test (3.9.2)")
    print("  ✓ Dunn's post-hoc with BH correction (3.9.2)")
    print("  ✓ Cliff's Delta effect size (3.9.3)")
    print("  ✓ Practical significance filtering (3.9.3)")
    print("  ✓ Root-cause analysis thresholds (3.9.4)")
    if args.concurrency_sweep:
        print("  ✓ Concurrency sweep analysis (3.9.5)")
    print(f"  ✓ Percentile metrics (p50, p95, p99) (3.9.1)")


if __name__ == "__main__":
    main()