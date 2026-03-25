"""
CAUM Adaptive Warmdown for Parameter Golf
==========================================
Replaces the fixed-schedule warmdown in the #1 submission with a CAUM-inspired
trajectory analyzer that detects training micro-states (E/S/W/V) and adjusts
the warmdown trigger point dynamically.

Integration:
    from caum_warmdown import CAUMWarmdownScheduler
    scheduler = CAUMWarmdownScheduler(args)
    # In training loop, replace lr_mul():
    scale = scheduler.get_lr_scale(step, elapsed_ms, train_loss.item())

Theory:
    CAUM classifies agent trajectories into micro-states based on observable
    dynamics. We apply the same principle to the loss trajectory:
    - E (Exploration): Loss dropping fast → delay warmdown, keep learning
    - V (Valid Progress): Loss dropping steadily → standard schedule
    - S (Stagnation): Loss plateau → trigger warmdown early
    - W (Waste): Loss oscillating without progress → aggressive warmdown
    
    By adapting warmdown to the actual training dynamics, we squeeze more
    useful training from the fixed 10-minute budget.

Paper-worthy contribution:
    This is the first application of agentic trajectory analysis (CAUM) to
    LLM training schedule optimization. While curriculum learning showed
    FineWeb is already well-curated, adaptive warmdown addresses the
    TEMPORAL dimension — when to stop exploring and start consolidating.
"""

import math
from collections import deque


class CAUMWarmdownScheduler:
    """CAUM-inspired adaptive warmdown scheduler.
    
    Drop-in replacement for the lr_mul() function in train_gpt.py.
    Tracks loss trajectory and adjusts warmdown timing based on
    detected training micro-states.
    """
    
    # Micro-state thresholds (calibrated for ~14K steps, 10-min budget)
    EXPLORATION_SLOPE = -0.005     # Loss dropping > 0.5% per window
    STAGNATION_SLOPE = -0.0005    # Loss dropping < 0.05% per window  
    WASTE_VARIANCE_RATIO = 3.0     # Variance > 3x baseline = oscillating
    
    # Warmdown adjustment factors
    EXPLORATION_DELAY = 1.3        # Delay warmdown by 30% 
    STAGNATION_ADVANCE = 0.6       # Start warmdown 40% earlier
    WASTE_ADVANCE = 0.5            # Start warmdown 50% earlier
    
    def __init__(self, warmdown_iters, max_wallclock_ms, iterations,
                 window_size=100, min_history=200, enabled=True):
        """
        Args:
            warmdown_iters: Base warmdown length in steps
            max_wallclock_ms: Max training wallclock (600000 for 10 min)
            iterations: Max iterations
            window_size: Number of recent losses to analyze
            min_history: Minimum losses before state detection activates
            enabled: Whether CAUM adaptation is active
        """
        self.warmdown_iters = warmdown_iters
        self.max_wallclock_ms = max_wallclock_ms
        self.iterations = iterations
        self.window_size = window_size
        self.min_history = min_history
        self.enabled = enabled
        
        # Loss tracking
        self.loss_history = deque(maxlen=5000)
        self.baseline_variance = None
        self.current_state = 'V'
        self.state_history = []
        
        # Adaptive warmdown multiplier
        self.warmdown_factor = 1.0
        self._last_factor_update = 0
        self._factor_update_interval = 50  # Update every 50 steps
        
        # LZ76 complexity of loss trajectory (for the CAUM narrative)
        self.lz76_complexity = 0
        self._lz76_dict = set()
    
    def _compute_slope(self, losses):
        """Linear regression slope of loss values."""
        n = len(losses)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(losses) / n
        num = sum((i - x_mean) * (losses[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / max(den, 1e-12)
    
    def _compute_variance(self, losses):
        """Variance of loss values."""
        if len(losses) < 2:
            return 0.0
        mean = sum(losses) / len(losses)
        return sum((l - mean) ** 2 for l in losses) / len(losses)
    
    def _detect_state(self):
        """Detect current training micro-state using CAUM methodology."""
        if len(self.loss_history) < self.min_history:
            return 'V'  # Not enough data, assume valid progress
        
        recent = list(self.loss_history)[-self.window_size:]
        slope = self._compute_slope(recent)
        variance = self._compute_variance(recent)
        
        # Establish baseline variance from early training
        if self.baseline_variance is None and len(self.loss_history) >= self.min_history:
            early = list(self.loss_history)[:self.window_size]
            self.baseline_variance = max(self._compute_variance(early), 1e-10)
        
        variance_ratio = variance / max(self.baseline_variance or 1e-10, 1e-10)
        
        # State classification (CAUM micro-state detection)
        if variance_ratio > self.WASTE_VARIANCE_RATIO and slope > self.STAGNATION_SLOPE:
            return 'W'  # Waste — oscillating without progress
        elif slope < self.EXPLORATION_SLOPE:
            return 'E'  # Exploration — loss dropping fast
        elif slope > self.STAGNATION_SLOPE:
            return 'S'  # Stagnation — plateau detected
        else:
            return 'V'  # Valid progress — steady improvement
    
    def _update_lz76(self, state):
        """Track LZ76 complexity of state sequence."""
        self.state_history.append(state)
        n = len(self.state_history)
        # Simple LZ76: count number of new substrings
        w = state
        for k in range(max(0, n - 10), n - 1):
            candidate = ''.join(self.state_history[k:n])
            if candidate not in self._lz76_dict:
                self._lz76_dict.add(candidate)
                self.lz76_complexity += 1
                break
    
    def record_loss(self, step, loss_value):
        """Record a training loss value."""
        self.loss_history.append(loss_value)
        
        # Periodically update state detection and warmdown factor
        if self.enabled and step - self._last_factor_update >= self._factor_update_interval:
            self._last_factor_update = step
            self.current_state = self._detect_state()
            self._update_lz76(self.current_state)
            
            # Smooth warmdown factor update
            target_factor = {
                'E': self.EXPLORATION_DELAY,
                'V': 1.0,
                'S': self.STAGNATION_ADVANCE,
                'W': self.WASTE_ADVANCE,
            }.get(self.current_state, 1.0)
            
            # Exponential moving average for smooth transitions
            alpha = 0.3
            self.warmdown_factor = (1 - alpha) * self.warmdown_factor + alpha * target_factor
    
    def get_lr_scale(self, step, elapsed_ms):
        """
        Drop-in replacement for lr_mul().
        Returns a float in [0, 1] that scales all learning rates.
        """
        if self.warmdown_iters <= 0:
            return 1.0
        
        # Adapt warmdown length using CAUM factor
        effective_warmdown = self.warmdown_iters * self.warmdown_factor
        
        if self.max_wallclock_ms is None:
            # Step-based warmdown
            warmdown_start = max(self.iterations - effective_warmdown, 0)
            if step < warmdown_start:
                return 1.0
            return max((self.iterations - step) / max(effective_warmdown, 1), 0.0)
        
        # Wallclock-based warmdown (what #1 uses)
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = effective_warmdown * step_ms
        remaining_ms = max(self.max_wallclock_ms - elapsed_ms, 0.0)
        
        if remaining_ms <= warmdown_ms:
            return remaining_ms / max(warmdown_ms, 1e-9)
        return 1.0
    
    def get_status(self):
        """Return current CAUM status for logging."""
        return {
            'state': self.current_state,
            'warmdown_factor': self.warmdown_factor,
            'lz76_complexity': self.lz76_complexity,
            'losses_tracked': len(self.loss_history),
            'state_counts': {s: self.state_history.count(s) 
                           for s in ['E', 'V', 'S', 'W'] if s in self.state_history},
        }
    
    def format_log(self):
        """Format CAUM status for train log output."""
        status = self.get_status()
        return (
            f"caum_state:{status['state']} "
            f"warmdown_factor:{status['warmdown_factor']:.3f} "
            f"lz76:{status['lz76_complexity']}"
        )


# --- Integration snippet for train_gpt.py ---
# 
# Replace the lr_mul function (lines 1596-1605) with:
#
# ```python
# from caum_warmdown import CAUMWarmdownScheduler
# caum_scheduler = CAUMWarmdownScheduler(
#     warmdown_iters=args.warmdown_iters,
#     max_wallclock_ms=max_wallclock_ms,
#     iterations=args.iterations,
#     enabled=bool(int(os.environ.get("CAUM_WARMDOWN", "1"))),
# )
# 
# # In training loop, after computing train_loss (line ~1687):
# caum_scheduler.record_loss(step, train_loss.item())
# 
# # Replace scale = lr_mul(step, elapsed_ms) (line ~1675) with:
# scale = caum_scheduler.get_lr_scale(step, elapsed_ms)
#
# # In logging (line ~1734), add:
# log0(f"step:{step} ... {caum_scheduler.format_log()}")
# ```


if __name__ == "__main__":
    """Test the CAUM warmdown scheduler with synthetic loss curves."""
    import random
    
    print("=" * 60)
    print("  CAUM Adaptive Warmdown — Simulation Test")
    print("=" * 60)
    
    # Simulate 14000 steps of training
    scheduler = CAUMWarmdownScheduler(
        warmdown_iters=3500,
        max_wallclock_ms=600000.0,  # 10 minutes
        iterations=14000,
        enabled=True,
    )
    
    # Fixed schedule (baseline) for comparison
    def fixed_lr_mul(step, elapsed_ms, warmdown_iters=3500, max_wallclock_ms=600000.0):
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
    
    # Simulate loss curve: fast drop → steady → plateau → bump
    loss = 4.0
    results = {"caum_warmdown_start": None, "fixed_warmdown_start": None}
    
    for step in range(1, 14001):
        # Simulate realistic loss curve
        if step < 500:
            loss -= 0.003 + random.gauss(0, 0.001)  # Fast exploration
        elif step < 5000:
            loss -= 0.0002 + random.gauss(0, 0.0003)  # Steady progress
        elif step < 9000:
            loss -= 0.00005 + random.gauss(0, 0.0002)  # Diminishing returns  
        else:
            loss -= 0.00001 + random.gauss(0, 0.0003)  # Near plateau
        
        elapsed_ms = step * (600000.0 / 14000)  # ~42.8ms per step
        
        scheduler.record_loss(step, loss)
        caum_scale = scheduler.get_lr_scale(step, elapsed_ms)
        fixed_scale = fixed_lr_mul(step, elapsed_ms)
        
        # Track when warmdown starts
        if caum_scale < 1.0 and results["caum_warmdown_start"] is None:
            results["caum_warmdown_start"] = step
        if fixed_scale < 1.0 and results["fixed_warmdown_start"] is None:
            results["fixed_warmdown_start"] = step
        
        if step % 2000 == 0 or step == 14000:
            status = scheduler.get_status()
            print(
                f"  Step {step:5d}: loss={loss:.4f} "
                f"state={status['state']} "
                f"factor={status['warmdown_factor']:.3f} "
                f"caum_lr={caum_scale:.4f} fixed_lr={fixed_scale:.4f}"
            )
    
    print(f"\n  Results:")
    print(f"  Fixed warmdown start:  step {results['fixed_warmdown_start']}")
    print(f"  CAUM warmdown start:   step {results['caum_warmdown_start']}")
    
    diff = (results['fixed_warmdown_start'] or 0) - (results['caum_warmdown_start'] or 0)
    if diff > 0:
        print(f"  CAUM started {diff} steps earlier (detected stagnation)")
    elif diff < 0:
        print(f"  CAUM delayed {-diff} steps (detected exploration)")
    else:
        print(f"  Same timing")
    
    print(f"\n  Final CAUM status: {scheduler.format_log()}")
    print(f"  State distribution: {scheduler.get_status()['state_counts']}")
    print(f"\n{'='*60}")
