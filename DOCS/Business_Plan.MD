# Cloud-Based 3D Scan Processing Service - Business Plan

**Service Concept:** On-demand Gaussian Splatting to Mesh conversion using RunPod infrastructure

**Last Updated:** January 29, 2026

---

## 🎯 Executive Summary

Offer a cloud-based service where users upload 3D scan data (COLMAP datasets or raw images) and receive production-quality textured meshes (.obj files) suitable for games, VR, architectural visualization, and more.

**Key Differentiators:**
- ✅ **Editable mesh outputs** - Not viewer-locked like Luma AI
- ✅ **Higher quality** - Gaussian Splatting > mobile photogrammetry apps
- ✅ **No GPU required** - Accessible to designers without hardware
- ✅ **Fast turnaround** - 1-3 hours vs days of local processing
- ✅ **Predictable pricing** - Pay per scan, no subscriptions

---

## 💰 Cost Structure

### Infrastructure Costs (RunPod RTX 5090 32GB)

| Instance Type | Hourly Rate | Best For | Availability |
|---------------|-------------|----------|--------------|
| **On-Demand** | $2.50-$3.50/hr | Guaranteed processing, rush orders | 99%+ |
| **Spot Instance** | $1.20-$1.80/hr | Batch processing, cost optimization | 70-90% |

**Average Cost Per Scan:**
- Medium quality (~1 hour): **$1.50** (spot) or **$3.00** (on-demand)
- High quality (~2 hours): **$3.00** (spot) or **$6.00** (on-demand)
- Large scenes (~3 hours): **$4.50** (spot) or **$9.00** (on-demand)

### Additional Costs

| Service | Cost | Monthly (100 scans) |
|---------|------|---------------------|
| **Storage (S3/Wasabi)** | $0.10/GB/month | ~$50 (500GB) |
| **Bandwidth** | $0.09/GB download | ~$18 (200GB) |
| **Domain + Hosting** | Fixed | ~$20-$30 |
| **Email Service (SendGrid)** | $15/month | $15 (up to 40K emails) |
| **Payment Processing (Stripe)** | 2.9% + $0.30 | ~$30 (on $1000 revenue) |

**Total Monthly Overhead (100 scans):** ~$265-$295

---

## �️ Self-Hosted vs Cloud Infrastructure Comparison

### Option A: Self-Hosted RTX 5090 Workstation

#### Initial Hardware Investment

| Component | Specs | Cost |
|-----------|-------|------|
| **GPU** | RTX 5090 32GB | $2,000-$2,500 |
| **CPU** | AMD Ryzen 9 7950X or Intel i9-14900K | $500-$700 |
| **RAM** | 64GB DDR5 (128GB recommended) | $200-$400 |
| **Motherboard** | High-end ATX with PCIe 5.0 | $300-$500 |
| **Storage** | 2TB NVMe SSD (for datasets/outputs) | $150-$250 |
| **PSU** | 1200W 80+ Platinum (5090 needs 450W) | $200-$300 |
| **Case** | Full tower with good airflow | $150-$250 |
| **Cooling** | AIO liquid cooler (280mm+) | $150-$200 |
| **Network** | 10Gbps NIC (optional) | $100-$200 |
| **UPS** | 1500VA battery backup | $200-$300 |

**Total Hardware Cost:** ~$4,000-$5,600 (mid-range: **$4,800**)

#### Monthly Operating Costs

| Expense | Calculation | Cost |
|---------|-------------|------|
| **Electricity** | 700W × 24/7 × $0.12/kWh = 504 kWh/month | $60-$80 |
| **Internet** | Business fiber (static IP, 500Mbps+ upload) | $80-$150 |
| **Cooling/HVAC** | Additional AC load (climate dependent) | $20-$50 |
| **Storage/Backup** | Cloud backup for outputs (1TB) | $10-$20 |
| **Maintenance** | Cleaning, thermal paste, fan replacement | $10-$20 |

**Total Monthly OpEx:** ~$180-$320 (avg: **$250**)

#### Amortization (3-Year Lifespan)

- Hardware cost: $4,800
- Depreciation: $4,800 / 36 months = **$133/month**
- **Total Monthly Cost (Year 1-3):** $250 + $133 = **$383/month**
- **After payoff (Year 4+):** **$250/month** (OpEx only)

---

### Option B: RunPod Cloud (Spot Instances)

#### Monthly Costs (100 scans/month)

| Expense | Calculation | Cost |
|---------|-------------|------|
| **Compute** | ~120 hours × $1.50/hr (spot avg) | $180-$210 |
| **Storage** | 500GB × $0.10/GB | $50 |
| **Bandwidth** | 200GB × $0.09/GB | $18 |
| **Overhead** | Domain, email, payment processing | $65 |

**Total Monthly Cost:** ~**$313-$343**

---

### Break-Even Analysis

#### Scenario 1: 100 Scans/Month

| Metric | Self-Hosted | RunPod Cloud | Winner |
|--------|-------------|--------------|--------|
| **Initial Investment** | $4,800 | $0 | 🏆 Cloud |
| **Monthly Cost (Year 1-3)** | $383 | $328 | 🏆 Cloud |
| **Monthly Cost (Year 4+)** | $250 | $328 | 🏆 Self-Hosted |
| **3-Year Total Cost** | $13,788 | $11,808 | 🏆 Cloud ($1,980 savings) |
| **5-Year Total Cost** | $19,788 | $19,680 | ~Tie |

**Verdict:** Cloud is cheaper until **Year 5** at 100 scans/month.

---

#### Scenario 2: 250 Scans/Month (Higher Volume)

| Metric | Self-Hosted | RunPod Cloud | Winner |
|--------|-------------|--------------|--------|
| **Monthly Compute** | N/A (fixed $250) | ~$450-$540 (spot) | 🏆 Self-Hosted |
| **Total Monthly Cost** | $383 (Year 1-3) | $680 | 🏆 Self-Hosted |
| **Monthly Savings** | - | $297-$357 | 🏆 Self-Hosted |
| **Payback Period** | **14-16 months** | N/A | 🏆 Self-Hosted |
| **3-Year Total Cost** | $13,788 | $24,480 | 🏆 Self-Hosted ($10,692 savings) |

**Verdict:** Self-hosted pays for itself in **~14 months** at 250 scans/month.

---

#### Scenario 3: 350-400 Scans/Month (Single GPU Max Capacity)

| Metric | Self-Hosted (1 GPU) | RunPod Cloud | Winner |
|--------|---------------------|--------------|--------|
| **Monthly Compute** | N/A (fixed $250) | ~$630-$720 (spot) | 🏆 Self-Hosted |
| **Total Monthly Cost** | $383 (Year 1-3) | $920 | 🏆 Self-Hosted |
| **Monthly Savings** | - | $537-$587 | 🏆 Self-Hosted |
| **Payback Period** | **~9 months** | N/A | 🏆 Self-Hosted |
| **Year 1 Profit Gain** | - | +$6,444 | 🏆 Self-Hosted |

**Verdict:** Self-hosted is a **strong winner** at max single-GPU capacity (~400 scans/month).

**⚠️ Warning:** At this volume, you're at **95%+ GPU utilization**. Turnaround times will exceed 48 hours, risking customer satisfaction. Consider adding 2nd GPU or hybrid overflow to RunPod.

---

#### Scenario 4: 500+ Scans/Month (REQUIRES 2 GPUs or Hybrid)

**Physical Constraint:** Single RTX 5090 cannot process 500+ scans/month at realistic quality mix.

| Approach | Monthly Cost | Capacity | Turnaround | Notes |
|----------|--------------|----------|------------|-------|
| **2 GPUs (Self-Hosted)** | $492 (Year 1-3) | 800 scans | <6 hours | Add 2nd GPU for $2,500 |
| **1 GPU + RunPod Overflow** | $383 + $150 = $533 | Unlimited | 6-12 hours | Route 100 scans to cloud |
| **Pure RunPod Cloud** | ~$900-$1,080 | Unlimited | 1-3 hours | No capacity risk |

**Recommended:** **Hybrid approach** (1 GPU + RunPod overflow) until sustained 500+ scans/month for 3+ months, then invest in 2nd GPU.

---

### Decision Matrix (Revised with Capacity Constraints)

| Monthly Volume | Recommended Approach | Break-Even | Max Capacity Warning |
|----------------|---------------------|------------|---------------------|
| **<50 scans** | 🏆 **RunPod Cloud** | Never | No constraints |
| **50-150 scans** | 🏆 **RunPod Cloud** | ~Year 4-5 | No constraints |
| **150-250 scans** | 🤔 **Hybrid Start** | ~18-24 months | Consider self-hosted if sustained |
| **250-350 scans** | 🏆 **Self-Hosted** | ~10-14 months | ⚠️ Plan for 2nd GPU at 350+ |
| **350-400 scans** | 🏆 **Self-Hosted** | ~9 months | 🚨 **AT MAX CAPACITY** - Add 2nd GPU soon |
| **400-500 scans** | 🤔 **Hybrid** (1 GPU + Cloud) | Ongoing savings | 🚨 **EXCEEDS SINGLE GPU** - Must use overflow |
| **500-650 scans** | 🏆 **2 GPUs Self-Hosted** | ~8 months (2nd GPU) | ⚠️ Approaching 2-GPU limit |
| **650+ scans** | 🏆 **2 GPUs + Cloud** OR **3 GPUs** | Ongoing savings | 🚨 Need 3rd GPU or permanent cloud tier |

---


### Hidden Costs & Risks

#### Self-Hosted Disadvantages

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Hardware failure** | $500-$2,500 repair (GPU, PSU) | Extended warranty, keep spare PSU |
| **Power outages** | Lost revenue during downtime | UPS ($300), generator for extended outages |
| **Internet downtime** | Can't upload/download | Redundant ISP ($100/month), LTE backup |
| **Physical space** | Noise, heat, office space | Soundproof enclosure, dedicated room with AC |
| **Scaling limits** | 1 GPU = max ~500 scans/month | Add 2nd GPU ($2,500+), or hybrid cloud for peaks |
| **Obsolescence** | RTX 6090 in 2027 may be 2× faster | Hardware has resale value (~40% after 2 years) |

#### Self-Hosted Advantages

| Benefit | Value | Notes |
|---------|-------|-------|
| **Data privacy** | High | No data leaves your premises (good for NDA clients) |
| **Predictable costs** | Medium | No surprise cloud bills from usage spikes |
| **Customization** | High | Full control over software, can run experiments |
| **Multi-use** | Medium | Use for other GPU tasks (AI training, rendering) when idle |
| **Instant availability** | High | No waiting for RunPod spot instances (90% uptime issue) |

---

### Hybrid Approach (Best of Both Worlds)

**Strategy:** Self-host for baseline load, burst to cloud for peaks.

**Example (200 scans/month):**
- Self-hosted handles 150 scans/month (steady state)
- RunPod handles 50 scans/month (weekend rush, holiday spikes)

**Costs:**
- Self-hosted: $383/month (hardware + OpEx)
- RunPod overflow: ~$50-$75/month (50 scans)
- **Total: ~$433-$458/month**

**Comparison:**
- Pure cloud (200 scans): ~$520/month
- Pure self-hosted (200 scans): $383/month
- **Hybrid savings:** $62-$87/month vs pure cloud, with flexibility

---

## �📊 Processing Times & Capabilities

Based on SuGaR v3 benchmarks (RTX 5090 32GB):

| Scene Type | Images | Quality | Mip-Splatting | SuGaR | **Total** | Cost (Spot) |
|------------|--------|---------|---------------|-------|-----------|-------------|
| **Small Room** | 50-100 | Medium (-r2) | ~3-5 min | ~20-30 min | **~25-35 min** | ~$0.75 |
| **Standard Scan** | 100-200 | Medium (-r2) | ~7 min | ~40-60 min | **~50-70 min** | ~$1.50 |
| **Premium Scan** | 100-200 | High (-r1) | ~50 min | ~1-1.5 hrs | **~2-2.5 hrs** | ~$3.50 |
| **Large Scene** | 200-300 | Medium (-r2) | ~10-15 min | ~1.5-2 hrs | **~2-2.5 hrs** | ~$3.75 |
| **XL Scene** | 200-300 | High (-r1) | ~1-1.5 hrs | ~2-3 hrs | **~3-4.5 hrs** | ~$6.00 |

**Hardware:** RTX 5090 estimates are 2-3× faster than RTX 5060 Ti production testing.

---

## 🚦 Capacity Planning & Queue Management

### Maximum Monthly Throughput (Single RTX 5090)

**Theoretical Capacity:**
- Hours per month: 730 hours
- 95% uptime (5% for maintenance/downtime): **693.5 hours available**

**Processing Time Mix (Typical):**
- 70% Medium quality scans: 1 hour each
- 30% High quality scans: 2.25 hours each
- **Weighted average: 1.375 hours per scan**

**Theoretical Max:** 693.5 / 1.375 = **504 scans/month**

---

### Realistic Capacity (80% Utilization)

**Real-world efficiency factors:**
- Queue gaps (waiting for uploads): -10%
- Failed scans requiring reruns: -5%
- Scheduled maintenance windows: -5%
- **Effective utilization: ~80% of theoretical**

**Realistic Maximum:** 504 × 0.80 = **~400 scans/month per GPU**

---

### Capacity Tiers & Turnaround Times

| Monthly Volume | GPU Utilization | Avg. Turnaround | Pricing Strategy | User Experience |
|----------------|-----------------|-----------------|------------------|-----------------|
| **<100 scans** | 20-25% | **<2 hours** | Premium pricing ($10-$20) | ✅ Instant processing |
| **100-200 scans** | 40-50% | **2-6 hours** | Standard pricing ($10-$20) | ✅ Same-day delivery |
| **200-300 scans** | 60-75% | **6-24 hours** | Discounted ($8-$15) | ⚠️ Next-day delivery |
| **300-400 scans** | 80-95% | **24-48 hours** | Budget pricing ($5-$10) | ⚠️ 1-2 day wait |
| **>400 scans** | **OVERLOADED** | **48+ hours** | ❌ Turn away or add 2nd GPU | ❌ Poor experience |

---

### The Turnaround-Price Trade-off

**Key Insight:** Users pay premium for speed, discount for backlog tolerance.

#### Pricing Tiers Based on Turnaround SLA

| Service Level | Turnaround | Price | When to Offer | Utilization Target |
|---------------|------------|-------|---------------|-------------------|
| **Express** | <2 hours guaranteed | **$25-$30** | Low utilization (<50%) | Maximize revenue per scan |
| **Standard** | Same-day (6-12 hrs) | **$15-$20** | Medium utilization (50-70%) | Balance speed + volume |
| **Economy** | 24-48 hours | **$8-$12** | High utilization (70-85%) | Fill capacity gaps |
| **Bulk** | 3-7 days | **$5-$8** | Overflow/backlog | Use cloud overflow |

---

### Queue Management Strategy

#### Phase 1: Building Backlog (Months 1-3)

**Problem:** GPU sits idle, no economies of scale.

**Strategy:**
- Offer **aggressive introductory pricing** ($5-$8/scan)
- Market as "Beta pricing - limited time"
- Accept ALL orders to build reputation
- Use excess GPU time for marketing renders/samples

**Target:** 50-100 scans/month (25-40% utilization)

---

#### Phase 2: Steady State (Months 4-12)

**Problem:** Need consistent 60-80% utilization for profitability.

**Strategy:**
- Implement **dynamic pricing**:
  - Queue <10 scans: Charge premium ($15-$25)
  - Queue 10-30 scans: Standard pricing ($10-$15)
  - Queue >30 scans: Economy pricing ($8-$12)
- Set **turnaround expectations** during checkout
- Email notifications: "Your scan moved up 5 positions!"

**Target:** 200-300 scans/month (60-80% utilization)

---

#### Phase 3: At Capacity (Months 12+)

**Problem:** Queue exceeds 48-hour turnaround, users complain.

**Options:**
1. **Raise prices** - Increase Standard to $20-$25 until demand normalizes
2. **Add 2nd GPU** - Double capacity to 800 scans/month ($4,800 investment)
3. **Hybrid overflow** - Route overflow to RunPod (pay $1.50-$3/scan)
4. **Turn away customers** - Risk losing market share

**Recommended:** Option 3 (Hybrid) until consistent 350+ scans/month for 3 months, then invest in 2nd GPU.

---

### Multi-GPU Scaling

#### When to Add a 2nd GPU

**Break-even analysis:**
- 2nd GPU investment: $2,500 (GPU only, reuse PSU/case if sufficient)
- Monthly depreciation: $69 (3-year amortization)
- Incremental OpEx: ~$40 (electricity, cooling)
- **Total incremental cost: $109/month**

**Required incremental revenue:**
- Need $109 / $8 profit per scan = **~14 additional scans/month to break even**

**Verdict:** Add 2nd GPU when you have **sustained backlog of 350+ scans/month** (so 2nd GPU handles overflow 50+ scans).

#### 2-GPU Configuration Capacity

- **Max capacity:** ~800 scans/month (2 × 400)
- **Recommended max:** 650 scans/month (80% utilization for buffer)
- **Turnaround:** Return to <6 hour turnaround for most scans

---

### Backlog Pricing Calculator

**Formula:** Price = Base × (1 - Utilization Factor × 0.3)

| Current Queue | Estimated Wait | Utilization | **Suggested Price** | Reasoning |
|---------------|----------------|-------------|---------------------|-----------|
| 0-5 scans | <2 hours | 20% | **$20** | Premium for instant service |
| 6-15 scans | 2-8 hours | 40% | **$17** | Standard same-day |
| 16-30 scans | 8-24 hours | 60% | **$14** | Slight discount for next-day |
| 31-50 scans | 24-48 hours | 80% | **$10** | Economy pricing |
| 51+ scans | 48+ hours | 95%+ | **$8 OR CLOSE** | At capacity - raise prices or close intake |

**Implementation:** Display current queue + estimated turnaround on upload page. Users self-select based on urgency.

---

### Capacity Expansion Roadmap

| Milestone | Monthly Scans | GPUs | Monthly Revenue | Monthly Profit | Action |
|-----------|---------------|------|-----------------|----------------|--------|
| **Launch** | 25-50 | 1 | $400 | $100 | Aggressive pricing to build backlog |
| **Traction** | 100-150 | 1 | $1,500 | $900 | Standard pricing, optimize utilization |
| **Capacity Warning** | 250-300 | 1 | $3,000 | $2,200 | Dynamic pricing, plan 2nd GPU |
| **Add GPU #2** | 350-500 | 2 | $6,000 | $4,500 | Invest $2,500, double capacity |
| **Mature** | 500-650 | 2 | $8,000 | $6,200 | Consider 3rd GPU or regional expansion |

---

## 💵 Pricing Strategy

### Recommended Pricing (70-85% Profit Margins)

| Service Tier | Images | Quality | Processing Time | **Price** | Cost (Spot) | Profit | Margin |
|--------------|--------|---------|-----------------|-----------|-------------|--------|--------|
| **Quick Scan** | <100 | Medium | ~30-45 min | **$5** | ~$1.00 | ~$4.00 | 80% |
| **Standard Scan** | 100-200 | Medium | ~1 hour | **$10** | ~$1.50 | ~$8.50 | 85% |
| **Premium Scan** | 100-200 | High | ~2-2.5 hours | **$20** | ~$3.50 | ~$16.50 | 82% |
| **Large Scene** | 200-300 | Medium | ~2-3 hours | **$25** | ~$3.75 | ~$21.25 | 85% |
| **XL Scene** | 200-300 | High | ~3-4.5 hours | **$35** | ~$6.00 | ~$29.00 | 83% |

### Add-On Services (Optional Revenue Boosters)

| Add-On | Description | Price | Cost | Profit |
|--------|-------------|-------|------|--------|
| **4K Textures** | Double texture resolution | +$5 | +$0.50 (storage) | +$4.50 |
| **360° Preview** | Auto-generated turntable video | +$3 | +$0.20 (rendering) | +$2.80 |
| **Express Processing** | <2hr guarantee (on-demand GPU) | +$10 | +$3.00 (premium compute) | +$7.00 |
| **Multi-LOD Export** | 3 levels of detail for games | +$5 | +$0.30 (processing) | +$4.70 |
| **Mesh Cleanup** | Manual review + artifact removal | +$15 | ~$10 (labor, 15 min) | +$5.00 |

---

## 📈 Revenue Projections

### Conservative Scenario (50 scans/month)

**Revenue Mix:**
- 30 Quick Scans × $5 = $150
- 15 Standard Scans × $10 = $150
- 5 Premium Scans × $20 = $100
- **Total Revenue: $400/month**

**Costs:**
- Compute: ~$75 (spot instances)
- Storage: ~$25 (250GB)
- Bandwidth: ~$9 (100GB)
- Overhead: ~$65 (domain, email, payment processing)
- **Total Costs: $174/month**

**Net Profit: $226/month (56% margin)**

---

### Moderate Scenario (100 scans/month)

**Revenue Mix:**
- 50 Quick Scans × $5 = $250
- 30 Standard Scans × $10 = $300
- 15 Premium Scans × $20 = $300
- 5 Large Scenes × $25 = $125
- **Total Revenue: $975/month**

**Costs:**
- Compute: ~$195 (spot instances, 90% uptime)
- Storage: ~$50 (500GB)
- Bandwidth: ~$18 (200GB)
- Overhead: ~$80
- **Total Costs: $343/month**

**Net Profit: $632/month (65% margin)**

---

### Growth Scenario (250 scans/month)

**Revenue Mix:**
- 100 Quick Scans × $5 = $500
- 80 Standard Scans × $10 = $800
- 50 Premium Scans × $20 = $1,000
- 20 Large Scenes × $25 = $500
- **Total Revenue: $2,800/month**

**Costs:**
- Compute: ~$540 (hybrid spot/on-demand)
- Storage: ~$120 (1.2TB)
- Bandwidth: ~$45 (500GB)
- Overhead: ~$120
- **Total Costs: $825/month**

**Net Profit: $1,975/month (70% margin)**

---

## 🎯 Target Market

### Primary Customers

1. **Architectural Firms** ($20-$35/scan)
   - Interior designers needing client presentation assets
   - Real estate agencies for virtual staging
   - Construction documentation and as-built records

2. **Game Developers** ($10-$25/scan)
   - Indie studios without photogrammetry pipelines
   - Asset creation for realistic environments
   - Rapid prototyping of game levels

3. **VFX Artists** ($20-$35/scan)
   - Film/TV production needing hero props
   - Commercial advertising with product scans
   - Background environment creation

4. **AR/VR Developers** ($10-$20/scan)
   - Training simulations with real-world environments
   - Virtual tourism experiences
   - Educational content creation

5. **Cultural Heritage** ($25-$35/scan)
   - Museum digitization projects
   - Historical site preservation
   - Archaeological documentation

### Secondary Markets

- E-commerce 3D product viewers
- 3D printing enthusiasts (scan to printable mesh)
- Academic research (computer vision, robotics)
- Hobbyists and content creators

---

## 🚀 Implementation Roadmap

### Phase 1: MVP (Weeks 1-4)

**Core Features:**
- ✅ Web upload interface (drag-and-drop .zip files)
- ✅ RunPod API integration (auto-provision GPUs)
- ✅ Email notifications (upload confirmed, processing started, download ready)
- ✅ Automated pipeline execution (Mip-Splatting → SuGaR)
- ✅ Download links (7-day expiration)

**Tech Stack:**
- Frontend: Next.js + Tailwind CSS + shadcn/ui
- Backend: Node.js + Express OR Python FastAPI
- Storage: AWS S3 or Wasabi
- Payments: Stripe Checkout
- Email: SendGrid or Resend
- Compute: RunPod API

**MVP Costs:** ~$500-$1000 (development time, if DIY)

---

### Phase 2: Polish (Weeks 5-8)

**Enhancements:**
- ✅ Queue management (show position, estimated wait time)
- ✅ Input validation (check COLMAP structure, image count)
- ✅ Preview generation (thumbnail of final mesh)
- ✅ Quality tier auto-detection (suggest pricing based on scene complexity)
- ✅ User accounts (track order history)
- ✅ Retry logic (auto-restart if RunPod spot instance interrupted)

---

### Phase 3: Scale (Weeks 9-16)

**Growth Features:**
- ✅ Subscription plans (5 scans/month at 20% discount)
- ✅ API access for enterprise customers
- ✅ Batch processing (upload multiple scenes at once)
- ✅ Custom branding (white-label for agencies)
- ✅ Mesh optimization tools (auto-decimation, UV unwrapping)
- ✅ Analytics dashboard (processing stats, popular scan types)

---

## 🛠️ Technical Architecture

### High-Level Flow

```
User Upload → S3 Storage → Trigger RunPod → Process Pipeline → Upload Results → Notify User
     ↓             ↓              ↓                  ↓                 ↓              ↓
  Web UI      Validation    GPU Instance     MIPS + SuGaR        S3 Output     Email/SMS
```

### Docker Image

**Pre-baked image with:**
- CUDA 13.0 + PyTorch 2.11
- SuGaR v3 optimized pipeline
- All dependencies compiled
- Entry point: Monitor `/input` folder, process, save to `/output`

```dockerfile
FROM nvidia/cuda:13.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git wget \
    libgl1-mesa-glx libglib2.0-0

# Copy SuGaR pipeline
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Compile CUDA extensions
RUN cd mip-splatting/submodules/diff-gaussian-rasterization && \
    python3 setup.py install && \
    cd ../../.. && \
    cd SuGaR/sugar_utils/gaussian_rasterization_fine && \
    python3 setup.py install

# Entry point: Wait for input, process, output
CMD ["python3", "pipeline_runner.py"]
```

---

### RunPod API Integration (Python)

```python
import runpod
import boto3
from pathlib import Path

def process_scan(scan_id: str, quality: str = "medium"):
    """
    Launch RunPod instance, process scan, return download URL
    """
    # 1. Download input from S3
    s3 = boto3.client('s3')
    s3.download_file('scans-input', f'{scan_id}.zip', f'/tmp/{scan_id}.zip')
    
    # 2. Start RunPod instance
    pod = runpod.create_pod(
        gpu_type_id="RTX 5090",  # or "RTX 4090" for cheaper option
        image_name="your-docker-image:latest",
        cloud_type="SECURE",  # Use COMMUNITY for spot pricing
        volume_in_gb=50,
        ports="22/tcp",
    )
    
    # 3. Upload input to pod
    pod.upload_file(f'/tmp/{scan_id}.zip', '/workspace/input.zip')
    
    # 4. Execute pipeline
    resolution = 2 if quality == "medium" else 1
    pod.exec(f"""
        cd /workspace &&
        unzip input.zip -d input/ &&
        python3 pipeline_runner.py \
            --input input/{scan_id} \
            --resolution {resolution} \
            --output output/
    """)
    
    # 5. Monitor progress (poll logs)
    while not pod.is_complete():
        time.sleep(30)
        logs = pod.get_logs()
        print(logs[-100:])  # Last 100 chars
    
    # 6. Download results
    pod.download_file('/workspace/output/', f'/tmp/{scan_id}_output/')
    
    # 7. Upload to S3
    for file in Path(f'/tmp/{scan_id}_output/').glob('**/*'):
        s3.upload_file(str(file), 'scans-output', f'{scan_id}/{file.name}')
    
    # 8. Terminate pod
    pod.terminate()
    
    # 9. Generate presigned download URL (expires in 7 days)
    url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': 'scans-output', 'Key': f'{scan_id}/mesh.zip'},
        ExpiresIn=604800  # 7 days
    )
    
    return url
```

---

### Error Handling & Retries

**Common failure modes:**
1. **Spot instance interrupted** → Auto-restart from last checkpoint (SuGaR saves every 1K iterations)
2. **Bad input data** → Pre-flight validation (check for COLMAP files, min image count)
3. **OOM errors** → Auto-retry with lower resolution or different GPU
4. **Network timeouts** → Retry upload/download with exponential backoff

---

## 🔍 Competitive Analysis

| Service | Quality | Price | Output Format | Turnaround | Notes |
|---------|---------|-------|---------------|------------|-------|
| **Polycam Pro** | Medium | $16/month | Viewer-locked .splat | Minutes | Mobile-first, lower quality |
| **Luma AI** | High | $1-$5/scan | NeRF (viewer only) | 10-30 min | Beautiful but not editable |
| **RealityCapture** | Very High | $3,750 license | .obj, .fbx | Hours (user hardware) | Professional, expensive upfront |
| **Meshroom** | High | Free (DIY) | .obj | Hours (user hardware) | Open-source, steep learning curve |
| **Your Service** | High | $10-$35/scan | .obj + textures | 1-3 hours | **Editable, no hardware needed** |

**Key Advantages:**
- ✅ **Lower barrier to entry** - No upfront license cost vs RealityCapture ($3,750)
- ✅ **Editable outputs** - .obj files work in Blender, Unity, Unreal (vs Luma's viewer-only)
- ✅ **Higher quality than mobile** - RTX 5090 processing > iPhone LiDAR (Polycam)
- ✅ **No hardware required** - Accessible to designers without gaming PCs
- ✅ **Faster than DIY** - 1-3 hrs vs days of learning + local processing

---

## 📣 Marketing Strategy

### Launch Channels

1. **Reddit** (Organic + Paid)
   - r/3Dscanning, r/photogrammetry, r/gamedev, r/blender
   - Post comparison renders: "I built a cloud service for this..."
   - AMA about Gaussian Splatting technology

2. **Twitter/X + LinkedIn**
   - Share before/after examples
   - Tag 3D scanning influencers
   - Post technical breakdowns (algorithm explanations)

3. **YouTube Tutorials**
   - "How to Convert iPhone Scans to Game-Ready Meshes"
   - Partner with 3D scanning YouTubers (affiliate commissions)

4. **Discord Communities**
   - Gaussian Splatting Discord
   - Unreal/Unity game dev servers
   - Architecture visualization groups

5. **Direct Outreach**
   - Email architectural firms (scraped from directories)
   - Partner with photogrammetry training courses
   - Sponsor 3D scanning conferences (SIGGRAPH, GDC)

### Content Marketing

- **Blog posts:** "Gaussian Splatting vs NeRF for Game Assets"
- **Case studies:** "How [Company] Saved 10 Hours Per Scan"
- **Free tier:** 1 free scan/month (watermarked) to drive conversions

---

## ⚖️ Legal & Compliance

### Terms of Service

- **Data retention:** 30 days for inputs, 7 days for outputs (user-initiated deletion available)
- **Copyright:** Users retain all rights to uploaded data and generated meshes
- **Privacy:** No training on user data, GDPR-compliant
- **Liability:** No warranties on output quality (best-effort basis)

### Licensing Considerations

**Important:** The underlying SuGaR/Mip-Splatting/3DGS code is **non-commercial** under the Gaussian-Splatting License (Inria/MPII).

**Options:**
1. **Obtain commercial license** from Inria/MPII before launching (recommended)
2. **Alternative:** Use commercial-friendly 3DGS implementations (e.g., Gaussian Splatting for Real-Time Radiance Field Rendering under Apache 2.0 - if such exists)
3. **Risk:** Operate in gray area (not recommended - high legal risk)

**Action Item:** Contact Inria licensing office BEFORE commercializing.

---

## 🎯 Success Metrics (KPIs)

| Metric | Month 1 | Month 3 | Month 6 | Month 12 |
|--------|---------|---------|---------|----------|
| **Total Scans** | 25 | 75 | 150 | 300 |
| **Revenue** | $250 | $900 | $1,800 | $3,500 |
| **Profit** | $100 | $550 | $1,200 | $2,400 |
| **Conversion Rate** | 5% | 8% | 10% | 12% |
| **Avg. Order Value** | $10 | $12 | $12 | $12 |
| **Customer Acquisition Cost** | $50 | $30 | $20 | $15 |

---

## 🚨 Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **RunPod price increases** | High | Medium | Lock in reserved capacity, evaluate alternatives (Vast.ai, Lambda Labs) |
| **Licensing lawsuit** | Critical | Low | Obtain commercial license from Inria/MPII upfront |
| **Competitor undercuts pricing** | Medium | Medium | Compete on quality and UX, not just price |
| **Low customer acquisition** | High | High | Invest in content marketing, offer free tier |
| **Bad scan support burden** | Medium | High | Clear upload guidelines, pre-flight validation |
| **GPU shortage** | Medium | Low | Multi-cloud strategy (RunPod + Vast.ai + Lambda) |

---

## 🏁 Next Steps (Action Plan)

### Immediate (Week 1)
- [ ] **Legal:** Contact Inria licensing office about commercial use
- [ ] **Testing:** Process 5 test scans on RunPod to validate costs
- [ ] **Design:** Wireframe web interface (upload, payment, download)

### Short-Term (Weeks 2-4)
- [ ] **Development:** Build MVP (upload + RunPod API + email notifications)
- [ ] **Infrastructure:** Set up S3 buckets, Stripe account, domain
- [ ] **Docker:** Bake optimized SuGaR pipeline into container image

### Medium-Term (Weeks 5-12)
- [ ] **Beta:** Launch with 10 test users (free scans for feedback)
- [ ] **Marketing:** Write 3 blog posts, post on Reddit, make YouTube demo
- [ ] **Iterate:** Improve UX based on user feedback

### Long-Term (Months 3-6)
- [ ] **Scale:** Expand to 100+ scans/month
- [ ] **Features:** Add subscription plans, API access, batch processing
- [ ] **Partnerships:** Reach out to 3D scanning hardware vendors

---

## 💡 Alternative Business Models

### Option 1: Subscription Plans
- **Starter:** 5 scans/month at $40/month (20% discount vs pay-per-scan)
- **Pro:** 20 scans/month at $140/month (30% discount)
- **Enterprise:** Unlimited scans at $500/month (custom SLA)

**Pros:** Predictable revenue, higher customer LTV  
**Cons:** Harder to price, risk of "buffet effect" (heavy users abuse unlimited plans)

---

### Option 2: White-Label Service
- License your pipeline to architectural firms or game studios
- They get self-hosted version or dedicated RunPod allocation
- **Pricing:** $500-$2000/month per organization + $2/scan API fee

**Pros:** Higher margins, less support burden  
**Cons:** Slower growth, requires enterprise sales

---

### Option 3: Marketplace Model
- Users upload scans, you process, they sell textured meshes on your platform
- Take 20-30% commission on sales
- **Example:** "Sketchfab for processed Gaussian Splatting meshes"

**Pros:** Two-sided network effects, viral growth potential  
**Cons:** Requires critical mass, content moderation burden

---

## 📚 Resources & References

### Documentation
- [SuGaR v3 Optimization Details](../README.md)
- [RunPod API Documentation](https://docs.runpod.io/)
- [Stripe Payments Guide](https://stripe.com/docs)
- [AWS S3 Pricing Calculator](https://calculator.aws/)

### Inspiration
- [Luma AI](https://lumalabs.ai/) - NeRF-based competitor
- [Polycam](https://poly.cam/) - Mobile 3D scanning app
- [Sketchfab](https://sketchfab.com/) - 3D model marketplace

### Technical Papers
- [3D Gaussian Splatting (SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [SuGaR: Surface-Aligned Gaussians (CVPR 2024)](https://arxiv.org/abs/2311.12775)
- [Mip-Splatting (CVPR 2024)](https://niujinshuchong.github.io/mip-splatting/)

---

## 📞 Contact & Feedback

**Questions or suggestions?** Open an issue on GitHub or reach out directly.

This business plan is a living document - update as you test assumptions and gather real-world data!

---

**Last Updated:** January 29, 2026  
**Version:** 1.0  
**Author:** Based on SuGaR v3 optimization work and RunPod infrastructure analysis
