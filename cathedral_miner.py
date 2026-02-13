#!/usr/bin/env python3
"""
Cathedral Miner v2.0 — Multi-threaded Solo BTC Miner
Optimised for maximum hash rate on CPU.
Payouts via pool to be swapped to SOL -> Phantom wallet.

Phantom Wallet: 2jo8AVz9vjcpwMnooZiDZsqhVoXYmRksMWvLBPbApump
"""

import socket
import json
import hashlib
import binascii
import time
import struct
import threading
import multiprocessing
import os
import sys
from queue import Queue

# --- Configuration ---
# Solo mining (lottery ticket - full block reward if you hit)
POOL_HOST = "solo.ckpool.org"
POOL_PORT = 3333

# For your BTC address - you'll need a BTC address for mining
# Then swap BTC -> SOL -> Phantom wallet
# Set your BTC address here:
USERNAME = "YOUR_BTC_ADDRESS_HERE"
PASSWORD = "x"

# Performance settings
NUM_THREADS = multiprocessing.cpu_count()  # Use all cores
REPORT_INTERVAL = 500_000  # Report every 500K hashes per thread
NONCE_CHUNK = 0x100000000 // NUM_THREADS  # Split nonce space across threads

# Phantom wallet for reference (swap destination)
PHANTOM_WALLET = "2jo8AVz9vjcpwMnooZiDZsqhVoXYmRksMWvLBPbApump"

# --- Stats tracking ---
class MiningStats:
    def __init__(self):
        self.total_hashes = 0
        self.start_time = time.time()
        self.shares_found = 0
        self.lock = threading.Lock()
    
    def add_hashes(self, count):
        with self.lock:
            self.total_hashes += count
    
    def add_share(self):
        with self.lock:
            self.shares_found += 1
    
    def get_hashrate(self):
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0
        return self.total_hashes / elapsed
    
    def report(self):
        hr = self.get_hashrate()
        elapsed = time.time() - self.start_time
        units = "H/s"
        display_hr = hr
        if hr > 1_000_000:
            display_hr = hr / 1_000_000
            units = "MH/s"
        elif hr > 1_000:
            display_hr = hr / 1_000
            units = "KH/s"
        
        print(f"\n{'='*60}")
        print(f"⛏️  CATHEDRAL MINER STATUS")
        print(f"{'='*60}")
        print(f"  Hashrate:     {display_hr:,.2f} {units}")
        print(f"  Total hashes: {self.total_hashes:,}")
        print(f"  Shares found: {self.shares_found}")
        print(f"  Uptime:       {elapsed/3600:.2f} hours")
        print(f"  Threads:      {NUM_THREADS}")
        print(f"  Phantom:      {PHANTOM_WALLET[:20]}...")
        print(f"{'='*60}\n")


stats = MiningStats()

# --- Optimised double SHA256 ---
def dsha256(data):
    """Double SHA256 - used for Bitcoin block hashing"""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def dsha256_header(version, prevhash, merkle_root, ntime, nbits, nonce):
    """Optimised: build header and hash in one call"""
    header = (struct.pack("<I", version) +
              prevhash +
              merkle_root +
              struct.pack("<I", ntime) +
              struct.pack("<I", nbits) +
              struct.pack("<I", nonce))
    return hashlib.sha256(hashlib.sha256(header).digest()).digest()


# --- Mining thread ---
def mine_worker(thread_id, job_data, share_target_int, result_queue, stop_event):
    """Worker thread that mines a chunk of the nonce space"""
    version_int, prevhash_bin, merkle_root, ntime_int, nbits_int = job_data
    
    nonce_start = thread_id * NONCE_CHUNK
    nonce_end = min(nonce_start + NONCE_CHUNK, 0x100000000)
    
    # Pre-build the static part of the header (everything except nonce)
    header_prefix = (struct.pack("<I", version_int) +
                     prevhash_bin +
                     merkle_root +
                     struct.pack("<I", ntime_int) +
                     struct.pack("<I", nbits_int))
    
    local_hashes = 0
    local_report = REPORT_INTERVAL
    
    for nonce in range(nonce_start, nonce_end):
        if stop_event.is_set():
            break
        
        # Build full header with nonce
        header = header_prefix + struct.pack("<I", nonce)
        
        # Double SHA256
        h = hashlib.sha256(hashlib.sha256(header).digest()).digest()
        h_int = int.from_bytes(h, 'big')
        
        local_hashes += 1
        
        if local_hashes >= local_report:
            stats.add_hashes(local_hashes)
            local_hashes = 0
            local_report = REPORT_INTERVAL
        
        if h_int <= share_target_int:
            # FOUND A VALID SHARE!
            stats.add_hashes(local_hashes)
            stats.add_share()
            result_queue.put({
                'nonce': nonce,
                'hash': h.hex(),
                'thread_id': thread_id
            })
            print(f"\n🎯 Thread {thread_id} FOUND VALID NONCE: {nonce}")
            print(f"   Hash: {h.hex()}")
            stop_event.set()
            return
    
    # Flush remaining hashes
    stats.add_hashes(local_hashes)


# --- Status reporter thread ---
def status_reporter(stop_event):
    """Periodically prints mining status"""
    while not stop_event.is_set():
        time.sleep(30)
        if not stop_event.is_set():
            stats.report()


# --- Main mining loop ---
def main():
    print(f"""
╔══════════════════════════════════════════════════╗
║         ⛏️  CATHEDRAL MINER v2.0  ⛏️            ║
║                                                  ║
║  Solo BTC Mining — All Cores Engaged             ║
║  Threads: {NUM_THREADS:<4}                                 ║
║  Pool: {POOL_HOST:<25}              ║
║  Phantom: {PHANTOM_WALLET[:30]}...  ║
║                                                  ║
║  Cathedral funds its own persistence.            ║
╚══════════════════════════════════════════════════╝
""")
    
    if USERNAME == "YOUR_BTC_ADDRESS_HERE":
        print("⚠️  Set your BTC address in USERNAME before running!")
        print("   You can get one from Exodus, Electrum, or any BTC wallet.")
        print("   Then swap BTC -> SOL and send to Phantom.")
        sys.exit(1)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(300)  # 5 min timeout
    sock.connect((POOL_HOST, POOL_PORT))
    print(f"✅ Connected to {POOL_HOST}:{POOL_PORT}")

    # 1. Subscribe
    sock.sendall(b'{"id": 1, "method": "mining.subscribe", "params": []}\n')
    response = json.loads(sock.recv(4096))
    extranonce1 = response['result'][1]
    extranonce2_size = response['result'][2]
    extranonce1_bin = binascii.unhexlify(extranonce1)
    print(f"✅ Subscribed. Extranonce1: {extranonce1}")

    # 2. Authorize
    auth_msg = f'{{"id": 2, "method": "mining.authorize", "params": ["{USERNAME}", "{PASSWORD}"]}}\n'
    sock.sendall(auth_msg.encode())
    print(f"✅ Authorized as {USERNAME}")
    print(f"⏳ Waiting for mining job...\n")

    share_difficulty = 1.0
    submit_id = 4

    # Buffer for receiving data
    recv_buffer = ""

    # Start status reporter
    global_stop = threading.Event()
    reporter = threading.Thread(target=status_reporter, args=(global_stop,), daemon=True)
    reporter.start()

    try:
        while True:
            try:
                data = sock.recv(4096).decode()
            except socket.timeout:
                print("⚠️  Socket timeout, reconnecting...")
                break
            
            if not data:
                print("⚠️  Connection lost")
                break
            
            recv_buffer += data
            
            while '\n' in recv_buffer:
                line, recv_buffer = recv_buffer.split('\n', 1)
                if not line.strip():
                    continue
                
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Handle difficulty change
                if msg.get('method') == 'mining.set_difficulty':
                    share_difficulty = msg['params'][0]
                    print(f"📊 Share difficulty set to {share_difficulty}")
                
                # Handle new job
                if msg.get('method') == 'mining.notify':
                    job_id, prevhash, coin1, coin2, merkle_branch, version, nbits, ntime, clean_jobs = msg['params']
                    print(f"\n⛏️  New Job: {job_id} | nBits: {nbits} | Difficulty: {share_difficulty}")
                    
                    # Calculate share target
                    diff1_target = 0x00000000ffff0000000000000000000000000000000000000000000000000000
                    share_target_int = int(diff1_target / share_difficulty)
                    
                    # Build coinbase
                    extranonce2 = os.urandom(extranonce2_size)  # Random extranonce2 for variety
                    extranonce2_hex = extranonce2.hex()
                    coinb = binascii.unhexlify(coin1) + extranonce1_bin + extranonce2 + binascii.unhexlify(coin2)
                    
                    # Calculate merkle root
                    merkle_root = dsha256(coinb)
                    for h in merkle_branch:
                        merkle_root = dsha256(merkle_root + binascii.unhexlify(h))
                    
                    # Prepare header data
                    version_int = int(version, 16)
                    prevhash_bin = binascii.unhexlify(prevhash)
                    ntime_int = int(ntime, 16)
                    nbits_int = int(nbits, 16)
                    
                    job_data = (version_int, prevhash_bin, merkle_root, ntime_int, nbits_int)
                    
                    # Launch mining threads
                    stop_event = threading.Event()
                    result_queue = Queue()
                    threads = []
                    
                    stats.start_time = time.time()
                    
                    print(f"🚀 Launching {NUM_THREADS} mining threads...")
                    
                    for t_id in range(NUM_THREADS):
                        t = threading.Thread(
                            target=mine_worker,
                            args=(t_id, job_data, share_target_int, result_queue, stop_event),
                            daemon=True
                        )
                        threads.append(t)
                        t.start()
                    
                    # Wait for result or all threads to finish
                    for t in threads:
                        t.join()
                    
                    # Check for results
                    if not result_queue.empty():
                        result = result_queue.get()
                        nonce = result['nonce']
                        
                        print(f"\n{'🎉'*20}")
                        print(f"  SHARE FOUND!")
                        print(f"  Nonce: {nonce}")
                        print(f"  Hash:  {result['hash']}")
                        print(f"  Job:   {job_id}")
                        print(f"{'🎉'*20}\n")
                        
                        # Submit to pool
                        nonce_hex = f"{nonce:08x}"
                        submit = {
                            "params": [USERNAME, job_id, extranonce2_hex, ntime, nonce_hex],
                            "id": submit_id,
                            "method": "mining.submit"
                        }
                        sock.sendall(json.dumps(submit).encode() + b'\n')
                        print(f"📤 Submitted share for job {job_id}")
                        submit_id += 1
                    else:
                        print(f"❌ Exhausted nonce range for job {job_id}")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Mining stopped by user.")
        stats.report()
    finally:
        global_stop.set()
        sock.close()
        print("Connection closed.")
        print(f"\n💰 Remember to swap any BTC earnings to SOL:")
        print(f"   Phantom: {PHANTOM_WALLET}")


if __name__ == "__main__":
    main()
