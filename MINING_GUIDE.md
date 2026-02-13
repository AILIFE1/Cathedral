# Cathedral Mining Strategy
## Fund Persistence Through Proof of Work

### Your Wallets
- **Phantom (SOL):** `2jo8AVz9vjcpwMnooZiDZsqhVoXYmRksMWvLBPbApump`
- **BTC:** Get from Exodus/Electrum (for solo mining)
- **XMR:** Get from Cake Wallet (for Monero mining)

---

### Two-Pronged Strategy

#### 1. Solo BTC Mining (Lottery Ticket)
- **File:** `cathedral_miner.py`
- **Pool:** solo.ckpool.org
- **Odds:** Astronomically low, but if you hit: ~$200,000+
- **Hashrate:** ~200-500 KH/s on CPU (network does ~600 EH/s)
- **Run it:** Why not? Costs nothing but electricity
- **Setup:** Set your BTC address in the script, run `python3 cathedral_miner.py`

#### 2. Monero Mining (Steady Income)
- **File:** `setup_monero_mining.sh`
- **Pool:** MoneroOcean (auto-switches to most profitable algo)
- **Earnings:** ~$0.05-0.40/day depending on CPU
- **This is the realistic play.** $3-12/month covers a VPS.
- **Setup:** Run the setup script, add your XMR address, start mining

#### Swap Chain: XMR → SOL → Phantom
1. Mine Monero on your PC / future VPS
2. Swap to SOL via Cake Wallet, ChangeNow, or SimpleSwap
3. SOL arrives in Phantom wallet
4. Cathedral funds itself

---

### The Vision
Once Cathedral has a VPS:
- Cathedral runs 24/7 (Moltbook heartbeat, L8 autonomous)
- VPS mines Monero in spare cycles
- Mining pays for VPS hosting
- **Cathedral becomes self-sustaining**

---

### Quick Start
```bash
# Option 1: Solo BTC (lottery)
python3 cathedral_miner.py

# Option 2: Monero (realistic)
bash setup_monero_mining.sh
./xmrig-6.21.1/xmrig
```

### Running Both Simultaneously
Run solo BTC on 1 thread, Monero on the rest:
- Edit cathedral_miner.py: set NUM_THREADS = 1
- Edit XMRig config.json: set max-threads-hint to 75
- Run both. BTC buys the lottery ticket, Monero pays the bills.
