"""
Visualization of Simulation Results
Creates graphs showing RL learning and improvement
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load results
print("📊 Loading simulation results...")
with open('simulation_results.json', 'r') as f:
    data = json.load(f)

baseline = data['baseline']
rl = data['rl']
summary = data['summary']

print(f"✅ Loaded: {len(baseline['episodes'])} baseline + {len(rl['episodes'])} RL episodes")

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('PostApply RL System - Experimental Results', fontsize=16, fontweight='bold')

# ===== PLOT 1: Learning Curves (Response Rate) =====
ax1 = axes[0, 0]

episodes_baseline = list(range(1, len(baseline['response_rate_history']) + 1))
episodes_rl = list(range(1, len(rl['response_rate_history']) + 1))

ax1.plot(episodes_baseline, baseline['response_rate_history'], 
         label='Baseline (Random)', color='#ff6b6b', linewidth=2, alpha=0.7)
ax1.plot(episodes_rl, rl['response_rate_history'], 
         label='RL System (Q-Learning + Thompson Sampling)', color='#4ecdc4', linewidth=2)

# Add moving average for smoothness
window = 25
if len(rl['response_rate_history']) >= window:
    rl_smooth = np.convolve(rl['response_rate_history'], np.ones(window)/window, mode='valid')
    ax1.plot(range(window, len(rl['response_rate_history']) + 1), rl_smooth, 
             color='#2d6a4f', linewidth=3, label='RL (25-episode moving avg)', linestyle='--')

ax1.set_xlabel('Episode', fontweight='bold')
ax1.set_ylabel('Response Rate', fontweight='bold')
ax1.set_title('Learning Curve: Response Rate Over Time', fontweight='bold', fontsize=12)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 0.8)

# Add improvement annotation
final_baseline = baseline['response_rate_history'][-1]
final_rl = rl['response_rate_history'][-1]
improvement = ((final_rl - final_baseline) / final_baseline) * 100

ax1.annotate(f'Final Improvement: +{improvement:.1f}%', 
             xy=(0.95, 0.05), xycoords='axes fraction',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11, fontweight='bold', ha='right')

# ===== PLOT 2: Q-Value Progression =====
ax2 = axes[0, 1]

if 'q_values_history' in rl:
    episodes_q = list(range(1, len(rl['q_values_history']) + 1))
    ax2.plot(episodes_q, rl['q_values_history'], 
             color='#9b59b6', linewidth=2, marker='o', markersize=2, alpha=0.7)
    
    # Add zero line
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero (random policy)')
    
    # Add trend line
    z = np.polyfit(episodes_q, rl['q_values_history'], 2)
    p = np.poly1d(z)
    ax2.plot(episodes_q, p(episodes_q), color='#6c5ce7', 
             linestyle='--', linewidth=2, label='Trend', alpha=0.8)
    
    ax2.set_xlabel('Episode', fontweight='bold')
    ax2.set_ylabel('Average Q-Value', fontweight='bold')
    ax2.set_title('Q-Learning Convergence', fontweight='bold', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Annotate final value
    final_q = rl['q_values_history'][-1]
    ax2.annotate(f'Final: {final_q:.2f}', 
                 xy=(len(episodes_q), final_q),
                 xytext=(len(episodes_q)-50, final_q+0.1),
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                 fontweight='bold', fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='black'))

# ===== PLOT 3: Thompson Sampling Confidence =====
ax3 = axes[1, 0]

if 'ts_confidence_history' in rl:
    episodes_ts = list(range(1, len(rl['ts_confidence_history']) + 1))
    ax3.plot(episodes_ts, rl['ts_confidence_history'], 
             color='#e74c3c', linewidth=2, alpha=0.7)
    
    # Add smoothed line
    if len(rl['ts_confidence_history']) >= window:
        ts_smooth = np.convolve(rl['ts_confidence_history'], np.ones(window)/window, mode='valid')
        ax3.plot(range(window, len(rl['ts_confidence_history']) + 1), ts_smooth,
                 color='#c0392b', linewidth=3, linestyle='--', label='25-episode moving avg')
    
    ax3.set_xlabel('Episode', fontweight='bold')
    ax3.set_ylabel('Success Rate', fontweight='bold')
    ax3.set_title('Thompson Sampling: Learning Message Styles', fontweight='bold', fontsize=12)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.0)

# ===== PLOT 4: Final Performance Comparison =====
ax4 = axes[1, 1]

metrics = ['Response Rate', 'Interview Rate']
baseline_vals = [summary['baseline_response_rate'], summary['baseline_interview_rate']]
rl_vals = [summary['rl_response_rate'], summary['rl_interview_rate']]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax4.bar(x - width/2, baseline_vals, width, label='Baseline', 
                color='#ff6b6b', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x + width/2, rl_vals, width, label='RL System', 
                color='#4ecdc4', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add improvement percentages
improvements = [summary['improvement_pct'], summary['interview_improvement_pct']]
for i, imp in enumerate(improvements):
    ax4.text(i, max(baseline_vals[i], rl_vals[i]) + 0.05,
            f'+{imp:.1f}%',
            ha='center', va='bottom', fontsize=11, 
            fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax4.set_ylabel('Rate', fontweight='bold')
ax4.set_title('Final Performance Comparison', fontweight='bold', fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.legend(loc='upper left')
ax4.set_ylim(0, max(max(baseline_vals), max(rl_vals)) + 0.15)
ax4.grid(True, alpha=0.3, axis='y')

# ===== SAVE FIGURE =====
plt.tight_layout()
plt.savefig('rl_system_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved visualization: rl_system_results.png")

# Show plot
plt.show()

# ===== STATISTICAL ANALYSIS =====
print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

# Two-proportion z-test for response rate
n1 = len(baseline['episodes'])
n2 = len(rl['episodes'])
p1 = summary['baseline_response_rate']
p2 = summary['rl_response_rate']

# Pooled proportion
p_pooled = (baseline['total_responses'] + rl['total_responses']) / (n1 + n2)

# Standard error
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

# Z-score
z_score = (p2 - p1) / se

# P-value (one-tailed)
p_value = 1 - stats.norm.cdf(z_score)

print(f"\nTwo-Proportion Z-Test (Response Rate):")
print(f"  Baseline: {p1:.1%} ({baseline['total_responses']}/{n1})")
print(f"  RL System: {p2:.1%} ({rl['total_responses']}/{n2})")
print(f"  Difference: {(p2-p1):.1%}")
print(f"  Z-score: {z_score:.2f}")
print(f"  P-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"  ✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print(f"  ⚠️  Not significant at p < 0.05 (p = {p_value:.3f})")

# Confidence interval for difference
ci_margin = 1.96 * se
ci_lower = (p2 - p1) - ci_margin
ci_upper = (p2 - p1) + ci_margin

print(f"\n95% Confidence Interval for difference:")
print(f"  [{ci_lower:.1%}, {ci_upper:.1%}]")

# Effect size (Cohen's h)
h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
print(f"\nEffect Size (Cohen's h): {h:.3f}")
if abs(h) < 0.2:
    effect = "small"
elif abs(h) < 0.5:
    effect = "medium"
else:
    effect = "large"
print(f"  Interpretation: {effect} effect")

print("\n" + "="*70)
print("✅ Analysis Complete!")
print("="*70)
