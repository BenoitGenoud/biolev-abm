import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random

# === PARAMÈTRES GÉNÉRAUX === #
N_AGENTS = 50
N_GENERATIONS = 50
N_INTERACTIONS = 10
ENERGY_START = 100
COOPERATION_COST = 5
COOPERATION_BENEFIT = 10
PUNISHMENT_COST = 2
PUNISHMENT_EFFECT = -5
MEMORY_LIMIT = 5

# === CONFIGURATION PAR INTERFACE === #
st.title("Simulation de coopération chez des agents primates")

st.sidebar.header("Activer des mécanismes de soutien")
use_memory = st.sidebar.checkbox("Mémoire sociale", value=False)
use_punishment = st.sidebar.checkbox("Punition", value=False)
use_reputation = st.sidebar.checkbox("Réputation", value=False)
use_kin_selection = st.sidebar.checkbox("Parenté", value=False)
use_noise = st.sidebar.checkbox("Erreur de stratégie", value=False)
use_variable_altruism = st.sidebar.checkbox("Altruisme variable", value=False)

# === CLASSE AGENT === #
class PrimateAgent:
    def __init__(self, id, parent1=None, parent2=None):
        self.id = id
        self.energy = ENERGY_START

        if parent1 and parent2:
            base_strategy = random.choice([parent1.genetic_strategy, parent2.genetic_strategy])
            if random.random() < 0.1:
                base_strategy = 'cooperate' if base_strategy == 'defect' else 'defect'
            self.genetic_strategy = base_strategy
        else:
            self.genetic_strategy = None  # sera défini manuellement dans la première génération

        self.behavior_strategy = self.genetic_strategy

        if use_variable_altruism:
            if parent1 and parent2 and hasattr(parent1, "altruism") and hasattr(parent2, "altruism"):
                mean_altruism = (parent1.altruism + parent2.altruism) / 2
                self.altruism = min(1.0, max(0.0, mean_altruism + random.uniform(-0.05, 0.05)))
            else:
                self.altruism = random.uniform(0.3, 0.7)

        if use_kin_selection:
            self.family_id = parent1.family_id if parent1 else random.randint(0, 10)

        if use_reputation:
            self.reputation = 0.5
        if use_memory:
            self.memory = {}
        if use_noise:
            self.error_rate = 0.05

    def decide(self, partner):
        if use_noise and random.random() < self.error_rate:
            return random.choice(['cooperate', 'defect'])

        if use_memory and partner.id in self.memory:
            if self.memory[partner.id].count('betray') > 0:
                return 'defect'

        if use_reputation:
            if partner.reputation > 0.6:
                return 'cooperate'
            elif partner.reputation < 0.3:
                return 'defect'

        if use_kin_selection and self.family_id == partner.family_id:
            return 'cooperate'

        if use_variable_altruism:
            if random.random() < self.altruism:
                return 'cooperate'
            else:
                return 'defect'

        return self.behavior_strategy

    def update_memory(self, partner_id, action):
        if use_memory:
            if partner_id not in self.memory:
                self.memory[partner_id] = []
            self.memory[partner_id].append(action)
            if len(self.memory[partner_id]) > MEMORY_LIMIT:
                self.memory[partner_id] = self.memory[partner_id][-MEMORY_LIMIT:]

# === SIMULATION === #
def run_simulation():
    # Initialisation équilibrée à 50/50 entre stratégies
    strategies = ['cooperate'] * (N_AGENTS // 2) + ['defect'] * (N_AGENTS // 2)
    random.shuffle(strategies)
    agents = [PrimateAgent(i) for i in range(N_AGENTS)]
    for agent, strategy in zip(agents, strategies):
        agent.genetic_strategy = strategy
        agent.behavior_strategy = strategy  # phénotype aligné initialement

    cooperation_history = []

    for generation in range(N_GENERATIONS):
        n_coop = 0

        for _ in range(N_INTERACTIONS):
            a, b = random.sample(agents, 2)
            action_a = a.decide(b)
            action_b = b.decide(a)

            if action_a == 'cooperate' and action_b == 'cooperate':
                a.energy += COOPERATION_BENEFIT - COOPERATION_COST
                b.energy += COOPERATION_BENEFIT - COOPERATION_COST
                if use_reputation:
                    a.reputation = min(1, a.reputation + 0.05)
                    b.reputation = min(1, b.reputation + 0.05)
                if use_memory:
                    a.update_memory(b.id, 'coop')
                    b.update_memory(a.id, 'coop')
                n_coop += 2
            elif action_a == 'cooperate' and action_b == 'defect':
                a.energy -= COOPERATION_COST
                b.energy += COOPERATION_BENEFIT
                if use_reputation:
                    a.reputation = max(0, a.reputation - 0.05)
                    b.reputation = min(1, b.reputation + 0.02)
                if use_memory:
                    a.update_memory(b.id, 'betray')
                    b.update_memory(a.id, 'coop')
                n_coop += 1
                if use_punishment:
                    b.energy += PUNISHMENT_EFFECT
                    a.energy -= PUNISHMENT_COST
            elif action_a == 'defect' and action_b == 'cooperate':
                b.energy -= COOPERATION_COST
                a.energy += COOPERATION_BENEFIT
                if use_reputation:
                    b.reputation = max(0, b.reputation - 0.05)
                    a.reputation = min(1, a.reputation + 0.02)
                if use_memory:
                    b.update_memory(a.id, 'betray')
                    a.update_memory(b.id, 'coop')
                n_coop += 1
                if use_punishment:
                    a.energy += PUNISHMENT_EFFECT
                    b.energy -= PUNISHMENT_COST
            else:
                if use_memory:
                    a.update_memory(b.id, 'betray')
                    b.update_memory(a.id, 'betray')

        agents.sort(key=lambda x: x.energy, reverse=True)
        top_half = agents[:N_AGENTS // 2]
        new_agents = []
        for _ in range(N_AGENTS // 2):
            parent1, parent2 = random.sample(top_half, 2)
            child = PrimateAgent(id=random.randint(1000, 9999), parent1=parent1, parent2=parent2)
            new_agents.extend([parent1, child])
        agents = new_agents

        cooperation_rate = n_coop / (N_INTERACTIONS * 2)
        cooperation_history.append(cooperation_rate)

    return cooperation_history

# === LANCEMENT DE LA SIMULATION === #
if st.button("Lancer la simulation"):
    history = run_simulation()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history)
    ax.set_title("Évolution de la coopération")
    ax.set_xlabel("Générations")
    ax.set_ylabel("Taux de coopérateurs")
    ax.grid(True)
    st.pyplot(fig)
