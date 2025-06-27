from statsmodels.stats.power import TTestIndPower

def print_required_sample_size(effect_size, alpha, power, alternative):
    analysis = TTestIndPower()
    try:
        result = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative=alternative
        )
        print(f"N per group (effect size={effect_size}, alpha={alpha}, power={power}, alternative='{alternative}'): {float(result):.2f}")
    except Exception as e:
        print(f"[ERROR] Could not compute sample size for alternative='{alternative}': {e}")

# Voorbeelden
print_required_sample_size(effect_size=0.5, alpha=0.05, power=0.99, alternative='two-sided')
print_required_sample_size(effect_size=0.5, alpha=0.05, power=0.99, alternative='larger')
print_required_sample_size(effect_size=0.5, alpha=0.05, power=0.8, alternative='larger')
print_required_sample_size(effect_size=0.5, alpha=0.05, power=0.8, alternative='two-sided')
