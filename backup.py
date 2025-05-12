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

# === CLASSE AGENT === #
class PrimateAgent:
    def __init__(self, id, parent=None):
        self.id = id
        self.energy = ENERGY_START
        if use_reputation:
            self.reputation = 0.5
        if use_memory:
            self.memory = {}

        if parent:
            self.strategy = parent.strategy if random.random() > 0.1 else (
                'cooperate' if parent.strategy == 'defect' else 'defect')
        else:
            self.strategy = random.choice(['cooperate', 'defect'])

    def decide(self, partner):
        if use_memory and partner.id in self.memory:
            betrayals = self.memory[partner.id].count('betray')
            if betrayals > 0:
                return 'defect'

        if use_reputation and partner.reputation > 0.6:
            return 'cooperate'
        if use_reputation and partner.reputation < 0.3:
            return 'defect'

        return self.strategy

    def update_memory(self, partner_id, action):
        if use_memory:
            if partner_id not in self.memory:
                self.memory[partner_id] = []
            self.memory[partner_id].append(action)
            if len(self.memory[partner_id]) > MEMORY_LIMIT:
                self.memory[partner_id] = self.memory[partner_id][-MEMORY_LIMIT:]

# === SIMULATION === #
def run_simulation():
    agents = [PrimateAgent(i) for i in range(N_AGENTS)]
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
        for parent in top_half:
            child = PrimateAgent(id=random.randint(1000, 9999), parent=parent)
            new_agents.append(parent)
            new_agents.append(child)
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