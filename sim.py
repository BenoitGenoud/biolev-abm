# Import des dépendances
import streamlit as st              # Pour le front
import matplotlib.pyplot as plt     # Pour tracer des graphiques
import numpy as np                  # Pour le rapport statistique
import random                       # Pour générer des nombres pseudo-aléatoires
import scipy.stats as stats         # Pour le rapport statistique

# Paramètres généraux de la simulation
N_AGENTS = 50                       # Le nombre d'agents dans la population
N_GENERATIONS = 50                  # Le nombre de générations
N_INTERACTIONS = 10                 # Le nombre d'interactions par génération
N_RUNS = 30                         # Le nombre d'itération par condition
ENERGY_START = 100                  # L'énergie initiale de chaque agent
COOPERATION_COST = 5                # Le coût en énergie d'un acte coopératif
COOPERATION_BENEFIT = 10            # Le gain en énergie pour celui qui bénéficie de la coopération
PUNISHMENT_COST = 2                 # Le coût en énergie pour punir un agent
PUNISHMENT_EFFECT = -5              # Le pénalité en énergie subie par un agent qui se fait punir
MEMORY_LIMIT = 5                    # La limite de mémoire, donc le nombre de souvenirs d'interactions

# Interface Streamlit
st.title("Simulation de coopération chez des agents primates")

# Nouveau : choix du mode décisionnel
mode_decision = st.selectbox("Mode décisionnel", ["Hiérarchique", "Cumulatif"])

# On définit le dictionnaire de paramètres de base (tout "false")
def full_param_dict(**kwargs):
    keys = ["use_memory", "use_punishment", "use_reputation", "use_kin_selection", "use_noise", "use_variable_altruism"]
    return {k: kwargs.get(k, False) for k in keys}

# Liste des différentes conditions expérimentales
conditions = {
    "Aucun mécanisme": full_param_dict(),
    "Mémoire sociale": full_param_dict(use_memory=True),
    "Punition": full_param_dict(use_punishment=True),
    "Réputation": full_param_dict(use_reputation=True),
    "Parenté": full_param_dict(use_kin_selection=True),
    "Erreur de stratégie": full_param_dict(use_noise=True),
    "Altruisme variable": full_param_dict(use_variable_altruism=True),
    "Tous les mécanismes": full_param_dict(
        use_memory=True, use_punishment=True, use_reputation=True,
        use_kin_selection=True, use_noise=True, use_variable_altruism=True
    )
}

# Choix des conditions par l'utilisateur
selected_conditions = st.multiselect("Conditions à comparer", list(conditions.keys()), default=["Aucun mécanisme", "Tous les mécanismes"])

# Définition d'un objet "agent primate"
class PrimateAgent:
    def __init__(self, id, parent1=None, parent2=None, params=None):
        self.id = id
        self.energy = ENERGY_START      # énergie initiale
        self.params = params or {}

        # Héritage de la stratégie génétique
        if parent1 and parent2:
            base_strategy = random.choice([parent1.genetic_strategy, parent2.genetic_strategy])
            if random.random() < 0.1:
                base_strategy = 'cooperate' if base_strategy == 'defect' else 'defect'
            self.genetic_strategy = base_strategy
        else:
            self.genetic_strategy = None

        self.behavior_strategy = self.genetic_strategy

        # Niveau d'altruisme initial
        if self.params.get("use_variable_altruism"):
            self.altruism = random.uniform(0.3, 0.7)

        # Famille pour la sélection par parenté
        if self.params.get("use_kin_selection"):
            self.family_id = parent1.family_id if parent1 else random.randint(0, 10)

        # Réputation initiale
        if self.params.get("use_reputation"):
            self.reputation = 0.5

        # Mémoire initiale des interactions
        if self.params.get("use_memory"):
            self.memory = {}

        # Taux d'erreur stratégique
        if self.params.get("use_noise"):
            self.error_rate = 0.05

    # Fonction décisionnelle (avec choix hiérarchique ou cumulatif)
    def decide(self, partner):
        # Mode hiérarchique : évaluation séquentielle prioritaire
        if mode_decision == "Hiérarchique":
            if self.params.get("use_noise") and random.random() < self.error_rate:
                return random.choice(['cooperate', 'defect'])
            if self.params.get("use_memory") and partner.id in self.memory and 'betray' in self.memory[partner.id]:
                return 'defect'
            if self.params.get("use_reputation") and hasattr(partner, 'reputation'):
                if partner.reputation > 0.6:
                    return 'cooperate'
                elif partner.reputation < 0.3:
                    return 'defect'
            if self.params.get("use_kin_selection") and hasattr(self, 'family_id') and hasattr(partner, 'family_id') and self.family_id == partner.family_id:
                return 'cooperate'
            if self.params.get("use_variable_altruism"):
                return 'cooperate' if random.random() < self.altruism else 'defect'
            return self.behavior_strategy

        # Mode cumulatif : décision probabiliste cumulée
        elif mode_decision == "Cumulatif":
            if self.params.get("use_noise") and random.random() < self.error_rate:
                return random.choice(['cooperate', 'defect'])
            score = 0
            if self.params.get("use_memory") and partner.id in self.memory and 'betray' in self.memory[partner.id]:
                score -= 1
            if self.params.get("use_reputation"):
                score += 1 if partner.reputation > 0.6 else -1 if partner.reputation < 0.3 else 0
            if self.params.get("use_kin_selection") and self.family_id == partner.family_id:
                score += 1
            if self.params.get("use_variable_altruism"):
                score += (self.altruism - 0.5) * 2
            # Liste des mécanismes activés
            active_mechanisms = any([
                self.params.get("use_memory"),
                self.params.get("use_reputation"),
                self.params.get("use_kin_selection"),
                self.params.get("use_variable_altruism")
            ])

            # Si aucun mécanisme actif => comportement strict selon génotype
            if not active_mechanisms:
                return self.behavior_strategy
            
            score += 0.5 if self.behavior_strategy == 'cooperate' else -0.5
            prob = 1 / (1 + np.exp(-score))
            return 'cooperate' if random.random() < prob else 'defect'
    
    # Fonction permettant de mettre à jour la mémoire d'un agent après une interaction
    def update_memory(self, partner_id, action):
        if self.params.get("use_memory"):
            if partner_id not in self.memory:
                self.memory[partner_id] = []
            self.memory[partner_id].append(action)
            if len(self.memory[partner_id]) > MEMORY_LIMIT:
                self.memory[partner_id] = self.memory[partner_id][-MEMORY_LIMIT:]

# Fonction qui gère la simulation
def run_simulation(params):
    # Création des agents initiaux avec moitié coopérants, moitié tricheurs
    strategies = ['cooperate'] * (N_AGENTS // 2) + ['defect'] * (N_AGENTS // 2)
    random.shuffle(strategies)
    agents = [PrimateAgent(i, params=params) for i in range(N_AGENTS)]
    for agent, strategy in zip(agents, strategies):
        agent.genetic_strategy = strategy
        agent.behavior_strategy = strategy

    history = [] # Historique du taux de coopération

    # On boucle x fois, avec x = le nombre de générations
    for _ in range(N_GENERATIONS):
        n_coop = 0      # compteur d'actes de coopération

        # Phase d'interactions
        # On boucle x fois, avec x = le nombre d'interactions par générations
        for _ in range(N_INTERACTIONS):
            a, b = random.sample(agents, 2)     # Interaction entre 2 agents, aléatoirement sélectionnés
            a_action = a.decide(b)
            b_action = b.decide(a)

            # En fonction des décisions des agents, on met à jour leur quantité d'énergie
            # Si les deux coopèrent : énergie = énergie + bénéfice - coût
            if a_action == 'cooperate' and b_action == 'cooperate':
                a.energy += COOPERATION_BENEFIT - COOPERATION_COST
                b.energy += COOPERATION_BENEFIT - COOPERATION_COST
                n_coop += 2
            # Si un coopère et l'autre trahit, on a respectivement :
            # énergie = énergie - coût // énergie = énergie + bénéfice
            elif a_action == 'cooperate' and b_action == 'defect':
                a.energy -= COOPERATION_COST
                b.energy += COOPERATION_BENEFIT
                n_coop += 1
                # Si on utilise le mécanisme de punition, on l'applique
                if params.get("use_punishment"):
                    b.energy += PUNISHMENT_EFFECT
                    a.energy -= PUNISHMENT_COST
            # Si l’un coopère et l’autre trahit : même logique que ci-dessus, mais inversée
            elif a_action == 'defect' and b_action == 'cooperate':
                b.energy -= COOPERATION_COST
                a.energy += COOPERATION_BENEFIT
                n_coop += 1
                if params.get("use_punishment"):
                    a.energy += PUNISHMENT_EFFECT
                    b.energy -= PUNISHMENT_COST

            # Si on utilise la réputation, elle est mise à jour dynamiquement
            if params.get("use_reputation"):
                if a_action == 'cooperate':
                    a.reputation = min(1.0, a.reputation + 0.05)
                else:
                    a.reputation = max(0.0, a.reputation - 0.05)

                if b_action == 'cooperate':
                    b.reputation = min(1.0, b.reputation + 0.05)
                else:
                    b.reputation = max(0.0, b.reputation - 0.05)

            # On met en mémoire les interactions
            if a_action == 'cooperate' and b_action == 'defect' and hasattr(a, "memory"):
                a.update_memory(b.id, 'betray')
            if b_action == 'cooperate' and a_action == 'defect' and hasattr(b, "memory"):
                b.update_memory(a.id, 'betray')
            if a_action == 'cooperate' and b_action == 'cooperate' and hasattr(a, "memory"):
                a.update_memory(b.id, 'coop')
                b.update_memory(a.id, 'coop')

        # Moyenne des coopérations
        history.append(n_coop / (N_INTERACTIONS * 2))
        
        # Phase de reproduction
        # Les agents qui ont le plus d'énergie se reproduisent (top 50%)
        agents.sort(key=lambda x: x.energy, reverse=True)
        top_half = agents[:N_AGENTS // 2]
        new_agents = []
        for _ in range(N_AGENTS // 2):
            p1, p2 = random.sample(top_half, 2)
            child = PrimateAgent(id=random.randint(1000, 9999), parent1=p1, parent2=p2, params=params)
            new_agents.extend([p1, child])
        agents = new_agents

    return history

# Comparaison des conditions
def run_comparisons():
    progress_bar = st.progress(0)
    log_status = st.empty()
    full_log = ""
    results = {}
    total_iterations = 0

    for i, label in enumerate(selected_conditions):
        params = conditions[label]
        log_status.text(f"Simulation en cours : {label}")
        runs = []
        for r in range(N_RUNS):
            run = run_simulation(params)
            runs.append(run)
            total_iterations += 1
            full_log += f"Condition {label}, run {r+1}/{N_RUNS} terminé\n"
            log_status.text(full_log)
            progress_bar.progress((i + r / N_RUNS) / len(selected_conditions))
        results[label] = np.array(runs)

    progress_bar.empty()
    log_status.success(f"Simulation terminée. Nombre total d'itérations : {total_iterations}")

    # Affichage du graphique de résultats
    st.subheader("Graphique")
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

    # Rapport statistique : comparaison du taux de coopération final entre conditions
    st.subheader("Rapport statistique")
    base_label = list(results.keys())[0]
    base = results[base_label][:, -1]
    for label, data in list(results.items())[1:]:
        other = data[:, -1]
        t_stat, p_val = stats.ttest_ind(base, other)            # Test T
        u_stat, u_p_val = stats.mannwhitneyu(base, other)       # Mann Whitney U-Test

        st.markdown(f"**{label} vs {base_label}** :")
        st.markdown(f"- t-test : p = {p_val:.4f}")
        st.markdown(f"- Mann-Whitney U-test : p = {u_p_val:.4f}")

        # Affichage de la moyenne, écart-type et intervalle de confiance
        mean_other = np.mean(other)
        std_other = np.std(other)
        ci_low = mean_other - 1.96 * std_other / np.sqrt(N_RUNS)
        ci_high = mean_other + 1.96 * std_other / np.sqrt(N_RUNS)
        st.markdown(f"- Moyenne : {mean_other:.3f}, Écart-type : {std_other:.3f}, IC95%: [{ci_low:.3f}, {ci_high:.3f}]")

    # Logs
    st.subheader("Logs")
    with st.expander("Logs"):
        st.text_area("Journal complet des exécutions", full_log, height=300)

# Ajout du bouton de lancement
if st.button("Lancer les simulations comparatives"):
    run_comparisons()
