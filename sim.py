import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.stats as stats

# === PARAMÈTRES GÉNÉRAUX === #
N_AGENTS = 50
N_GENERATIONS = 50
N_INTERACTIONS = 10
N_RUNS = 30
ENERGY_START = 100
COOPERATION_COST = 5
COOPERATION_BENEFIT = 10
PUNISHMENT_COST = 2
PUNISHMENT_EFFECT = -5
MEMORY_LIMIT = 5

# === CONFIGURATION PAR INTERFACE === #
st.title("Simulation de coopération chez des agents primates")

def full_param_dict(**kwargs):
    keys = ["use_memory", "use_punishment", "use_reputation", "use_kin_selection", "use_noise", "use_variable_altruism"]
    return {k: kwargs.get(k, False) for k in keys}

conditions = {
    "Contrôle (aucun mécanisme)": full_param_dict(),
    "Mémoire sociale": full_param_dict(use_memory=True),
    "Punition": full_param_dict(use_punishment=True),
    "Réputation": full_param_dict(use_reputation=True),
    "Parenté": full_param_dict(use_kin_selection=True),
    "Erreur de stratégie": full_param_dict(use_noise=True),
    "Altruisme variable": full_param_dict(use_variable_altruism=True),
    "Tous activés": full_param_dict(
        use_memory=True, use_punishment=True, use_reputation=True,
        use_kin_selection=True, use_noise=True, use_variable_altruism=True
    )
}

selected_conditions = st.multiselect("Choisissez les conditions à comparer :", list(conditions.keys()), default=list(conditions.keys())[:2])

# === CLASSE AGENT === #
class PrimateAgent:
    def __init__(self, id, parent1=None, parent2=None, params=None):
        self.id = id
        self.energy = ENERGY_START
        self.params = params or {}

        if parent1 and parent2:
            base_strategy = random.choice([parent1.genetic_strategy, parent2.genetic_strategy])
            if random.random() < 0.1:
                base_strategy = 'cooperate' if base_strategy == 'defect' else 'defect'
            self.genetic_strategy = base_strategy
        else:
            self.genetic_strategy = None

        self.behavior_strategy = self.genetic_strategy

        if self.params.get("use_variable_altruism"):
            if parent1 and parent2 and hasattr(parent1, "altruism") and hasattr(parent2, "altruism"):
                mean_altruism = (parent1.altruism + parent2.altruism) / 2
                self.altruism = min(1.0, max(0.0, mean_altruism + random.uniform(-0.05, 0.05)))
            else:
                self.altruism = random.uniform(0.3, 0.7)

        if self.params.get("use_kin_selection"):
            self.family_id = parent1.family_id if parent1 else random.randint(0, 10)

        if self.params.get("use_reputation"):
            self.reputation = 0.5
        if self.params.get("use_memory"):
            self.memory = {}
        if self.params.get("use_noise"):
            self.error_rate = 0.05

    def decide(self, partner):
        if self.params.get("use_noise") and random.random() < self.error_rate:
            return random.choice(['cooperate', 'defect'])

        if self.params.get("use_memory") and hasattr(self, 'memory') and partner.id in self.memory:
            if self.memory[partner.id].count('betray') > 0:
                return 'defect'

        if self.params.get("use_reputation") and hasattr(partner, 'reputation'):
            if partner.reputation > 0.6:
                return 'cooperate'
            elif partner.reputation < 0.3:
                return 'defect'

        if self.params.get("use_kin_selection") and hasattr(self, 'family_id') and hasattr(partner, 'family_id'):
            if self.family_id == partner.family_id:
                return 'cooperate'

        if self.params.get("use_variable_altruism") and hasattr(self, 'altruism'):
            return 'cooperate' if random.random() < self.altruism else 'defect'

        return self.behavior_strategy

    def update_memory(self, partner_id, action):
        if self.params.get("use_memory"):
            if partner_id not in self.memory:
                self.memory[partner_id] = []
            self.memory[partner_id].append(action)
            if len(self.memory[partner_id]) > MEMORY_LIMIT:
                self.memory[partner_id] = self.memory[partner_id][-MEMORY_LIMIT:]

# === SIMULATION PRINCIPALE ===
def run_simulation(params):
    strategies = ['cooperate'] * (N_AGENTS // 2) + ['defect'] * (N_AGENTS // 2)
    random.shuffle(strategies)
    agents = [PrimateAgent(i, params=params) for i in range(N_AGENTS)]
    for agent, strategy in zip(agents, strategies):
        agent.genetic_strategy = strategy
        agent.behavior_strategy = strategy

    history = []
    for _ in range(N_GENERATIONS):
        n_coop = 0
        for _ in range(N_INTERACTIONS):
            a, b = random.sample(agents, 2)
            a_action = a.decide(b)
            b_action = b.decide(a)

            if a_action == 'cooperate' and b_action == 'cooperate':
                a.energy += COOPERATION_BENEFIT - COOPERATION_COST
                b.energy += COOPERATION_BENEFIT - COOPERATION_COST
                n_coop += 2
            elif a_action == 'cooperate' and b_action == 'defect':
                a.energy -= COOPERATION_COST
                b.energy += COOPERATION_BENEFIT
                n_coop += 1
                if params.get("use_punishment"):
                    b.energy += PUNISHMENT_EFFECT
                    a.energy -= PUNISHMENT_COST
            elif a_action == 'defect' and b_action == 'cooperate':
                b.energy -= COOPERATION_COST
                a.energy += COOPERATION_BENEFIT
                n_coop += 1
                if params.get("use_punishment"):
                    a.energy += PUNISHMENT_EFFECT
                    b.energy -= PUNISHMENT_COST

            if a_action == 'cooperate' and b_action == 'defect' and hasattr(a, "memory"):
                a.update_memory(b.id, 'betray')
            if b_action == 'cooperate' and a_action == 'defect' and hasattr(b, "memory"):
                b.update_memory(a.id, 'betray')
            if a_action == 'cooperate' and b_action == 'cooperate' and hasattr(a, "memory"):
                a.update_memory(b.id, 'coop')
                b.update_memory(a.id, 'coop')

        history.append(n_coop / (N_INTERACTIONS * 2))

        agents.sort(key=lambda x: x.energy, reverse=True)
        top_half = agents[:N_AGENTS // 2]
        new_agents = []
        for _ in range(N_AGENTS // 2):
            p1, p2 = random.sample(top_half, 2)
            child = PrimateAgent(id=random.randint(1000, 9999), parent1=p1, parent2=p2, params=params)
            new_agents.extend([p1, child])
        agents = new_agents

    return history

# === COMPARAISON MULTIPLE ===
def run_comparisons():
    progress_bar = st.progress(0)
    log_output = st.empty()
    full_log = ""
    results = {}
    for i, label in enumerate(selected_conditions):
        params = conditions[label]
        log_output.text(f"Simulation en cours : {label}")
        runs = []
        for r in range(N_RUNS):
            run = run_simulation(params)
            runs.append(run)
            full_log += f"Condition {label}, run {r+1}/{N_RUNS} terminé"
            progress_bar.progress((i + r / N_RUNS) / len(selected_conditions))
        results[label] = np.array(runs)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(N_GENERATIONS)

    for label, data in results.items():
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.set_title("Comparaison des conditions - Taux de coopération")
    ax.set_xlabel("Générations")
    ax.set_ylabel("Taux de coopération")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.text_area("Journal complet des exécutions", full_log, height=300)

    base_label = list(results.keys())[0]
    base = results[base_label][:, -1]
    for label, data in list(results.items())[1:]:
        t_stat, p_val = stats.ttest_ind(base, data[:, -1])
        st.markdown(f"**{label} vs {base_label}** : p = {p_val:.4f}")

# === LANCEMENT ===
if st.button("Lancer les simulations comparatives"):
    run_comparisons()
