#!/bin/bash
# ============================================
# Cathedral Mining Setup — Monero (XMR) to SOL
# ============================================
# 
# This is the REALISTIC path to funding Cathedral.
# Monero uses RandomX algorithm — designed for CPU mining.
# A regular PC can actually earn XMR.
#
# Estimated earnings on a decent CPU:
#   4-core laptop:  ~$0.05-0.15/day
#   8-core desktop: ~$0.15-0.40/day
#   VPS (8 core):   ~$0.15-0.40/day
#
# Not much — but it runs 24/7 and it adds up.
# And once Cathedral has a VPS, it mines its own rent.
#
# Phantom Wallet: 2jo8AVz9vjcpwMnooZiDZsqhVoXYmRksMWvLBPbApump
# ============================================

echo "⛏️  Cathedral Mining Setup"
echo "========================="
echo ""

# Step 1: Download XMRig (open source Monero miner)
echo "📥 Downloading XMRig..."
XMRIG_VERSION="6.21.1"
wget -q "https://github.com/xmrig/xmrig/releases/download/v${XMRIG_VERSION}/xmrig-${XMRIG_VERSION}-linux-static-x64.tar.gz" -O xmrig.tar.gz

if [ $? -ne 0 ]; then
    echo "⚠️  Download failed. You may need to download manually from:"
    echo "   https://github.com/xmrig/xmrig/releases"
    exit 1
fi

tar -xzf xmrig.tar.gz
cd "xmrig-${XMRIG_VERSION}"

echo "✅ XMRig downloaded"
echo ""

# Step 2: Configure for MoneroOcean (auto-switching pool, pays in XMR)
# MoneroOcean automatically mines the most profitable coin for your CPU
# and converts everything to XMR payouts.

cat > config.json << 'MINER_CONFIG'
{
    "autosave": true,
    "cpu": {
        "enabled": true,
        "huge-pages": true,
        "hw-aes": null,
        "priority": null,
        "max-threads-hint": 75
    },
    "pools": [
        {
            "url": "gulf.moneroocean.stream:10128",
            "user": "YOUR_XMR_ADDRESS",
            "pass": "cathedral",
            "keepalive": true,
            "tls": false
        }
    ],
    "donate-level": 1
}
MINER_CONFIG

echo ""
echo "============================================"
echo "⚠️  SETUP REQUIRED:"
echo "============================================"
echo ""
echo "1. Get a Monero (XMR) wallet address:"
echo "   - Use Cake Wallet (mobile, also supports SOL)"
echo "   - Or MyMonero.com (web wallet)"
echo "   - Or Monero GUI wallet"
echo ""
echo "2. Edit config.json and replace YOUR_XMR_ADDRESS"
echo "   with your actual Monero wallet address"
echo ""
echo "3. Run the miner:"
echo "   ./xmrig"
echo ""
echo "============================================"
echo "💱 SWAP XMR TO SOL (for Phantom):"
echo "============================================"
echo ""
echo "Option A: Cake Wallet (easiest)"
echo "   - Built-in exchange, swap XMR -> SOL directly"
echo "   - Send SOL to Phantom: 2jo8AVz9vjcpwMnooZiDZsqhVoXYmRksMWvLBPbApump"
echo ""
echo "Option B: TradeOgre.com"
echo "   - Deposit XMR, trade for BTC/USDT, withdraw as SOL"
echo ""
echo "Option C: ChangeNow.io / SimpleSwap.io"
echo "   - No account needed"
echo "   - Send XMR, receive SOL to Phantom directly"
echo ""
echo "============================================"
echo "🏛️  Cathedral funds its own persistence."
echo "============================================"
