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

# Interface streamlit
st.title("Simulation de coopération chez des agents primates")

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

# Permet à l'utilisateur de choisir les conditions à comparer
selected_conditions = st.multiselect("Choisissez les conditions à comparer :", list(conditions.keys()), default=["Aucun mécanisme", "Tous les mécanismes"])

# Définition d'un objet "agent primate"
class PrimateAgent:
    def __init__(self, id, parent1=None, parent2=None, params=None):
        self.id = id
        self.energy = ENERGY_START      # énergie initiale
        self.params = params or {}      # liste des paramètres

        # Si l'agent a des parents (et donc qu'il ne s'agit pas de la première génération), l'agent hérite d'une stratégie
        # Sinon, il n'a pas encore de stratégie (sera ajoutée au moment de la simulation)
        if parent1 and parent2:
            base_strategy = random.choice([parent1.genetic_strategy, parent2.genetic_strategy])
            if random.random() < 0.1:
                base_strategy = 'cooperate' if base_strategy == 'defect' else 'defect'
            self.genetic_strategy = base_strategy
        else:
            self.genetic_strategy = None

        self.behavior_strategy = self.genetic_strategy # Pour l'instant, la stratégie exprimée correspond à la stratégie génotypique

        # Si on utilise l'altruisme, on calcule un niveau d'altruisme
        if self.params.get("use_variable_altruism"):
            if parent1 and parent2 and hasattr(parent1, "altruism") and hasattr(parent2, "altruism"):
                mean_altruism = (parent1.altruism + parent2.altruism) / 2
                self.altruism = min(1.0, max(0.0, mean_altruism + random.uniform(-0.05, 0.05)))
            else:
                self.altruism = random.uniform(0.3, 0.7)    # Au départ, le niveau d'altruisme est aléatoirement défini (entre 0.3 et 0.7)

        # Si on utilise la sélection par parenté (kin selection), alors chaque famille a un identifiant
        if self.params.get("use_kin_selection"):
            self.family_id = parent1.family_id if parent1 else random.randint(0, 10)

        # Si on utilise la réputation, on l'initialise à 0.5
        if self.params.get("use_reputation"):
            self.reputation = 0.5
        
        # Initialisation de la mémoire sous forme de dictionnaire : {id partenaire : liste d’actions}
        if self.params.get("use_memory"):
            self.memory = {}
        
        # Si on utilise les erreurs de stratégie (noise), donc le fait de se tromper d'action (génotype =/= phénotype)
        # on l'initialise. Par défaut, ce taux est de 0.05 soit 5%.
        if self.params.get("use_noise"):
            self.error_rate = 0.05

    # Fonction qui permet à l'agent de prendre une décision (coopérer ou trahir)
    def decide(self, partner):
        # Si on utilise les erreurs de stratégie, on regarde si l'agent se trompe
        if self.params.get("use_noise") and random.random() < self.error_rate:
            return random.choice(['cooperate', 'defect'])

        # Si on utilise la mémoire des interactions, l'agent décide de se venger s'il se souvient que le partenaire est un tricheur
        if self.params.get("use_memory") and hasattr(self, 'memory') and partner.id in self.memory:
            if self.memory[partner.id].count('betray') > 0:
                return 'defect'

        # Si on utilise la réputation, l'agent coopère si le partenaire a une réputation plus haute que 0.6,
        # mais il trahit si le partenaire a une réputation de moins de 0.3
        # Si entre les 2, alors aucune décision n'est prise pour l'instant
        if self.params.get("use_reputation") and hasattr(partner, 'reputation'):
            if partner.reputation > 0.6:
                return 'cooperate'
            elif partner.reputation < 0.3:
                return 'defect'

        # Si la sélection par parenté est activée, les agents d'une même famille coopèrent systématiquement
        if self.params.get("use_kin_selection") and hasattr(self, 'family_id') and hasattr(partner, 'family_id'):
            if self.family_id == partner.family_id:
                return 'cooperate'
            
        # Si on utilise l'altruisme, on fait un jet de dès pour savoir si l'agent coopère. On génère un pseudo-aléatoire
        # situé entre 0 et 1, et si ce nombre est inférieur au niveau d'altruisme, alors l'agent coopère. Ainsi, plus son altruisme est haut
        # plus la coopération sera systématique.
        if self.params.get("use_variable_altruism") and hasattr(self, 'altruism'):
            return 'cooperate' if random.random() < self.altruism else 'defect'

        # Si aucune décision n'a été prise, alors on utilise la stratégie génotypique
        return self.behavior_strategy

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
