# -*- coding: utf-8 -*-

# 1. --- IMPORTS ---
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# --- Suppress convergence warnings from pymoo for cleaner output ---
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# 2. --- DATA GENERATION MODULE (as provided by user) ---
def generate_simulation_data(
        num_users: int,
        num_jobs: int,
        dim: int = 32,
        scenario: str = 'red_sea',
        seed: Optional[int] = None
) -> Tuple[List[List[Dict]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generate all simulation data based on specified market scenario (Chapter 4.1.1).
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Generate user and job vectors
    if scenario == 'blue_sea':
        user_vectors = np.random.rand(num_users, dim)
        job_vectors = np.random.rand(num_jobs, dim)
    elif scenario == 'red_sea':
        hot_spot = np.random.rand(1, dim)
        num_hot_jobs = int(0.2 * num_jobs)
        hot_job_vectors = np.random.normal(loc=hot_spot, scale=0.1, size=(num_hot_jobs, dim))
        cold_job_vectors = np.random.rand(num_jobs - num_hot_jobs, dim)
        job_vectors = np.vstack([hot_job_vectors, cold_job_vectors])
        np.random.shuffle(job_vectors)
        num_hot_users = int(0.8 * num_users)
        hot_user_vectors = np.random.normal(loc=hot_spot, scale=0.1, size=(num_hot_users, dim))
        cold_user_vectors = np.random.rand(num_users - num_hot_users, dim)
        user_vectors = np.vstack([hot_user_vectors, cold_user_vectors])
        np.random.shuffle(user_vectors)
    elif scenario == 'multi_polar':
        num_polars = 5
        hot_spots = [np.random.rand(1, dim) for _ in range(num_polars)]
        num_hot_jobs_per_spot = int(0.10 * num_jobs)
        job_vectors_list = [np.random.rand(num_jobs - len(hot_spots) * num_hot_jobs_per_spot, dim)]
        for spot in hot_spots:
            job_vectors_list.append(np.random.normal(loc=spot, scale=0.1, size=(num_hot_jobs_per_spot, dim)))
        job_vectors = np.vstack(job_vectors_list)
        np.random.shuffle(job_vectors)
        num_hot_users_per_spot = int(0.18 * num_users)
        user_vectors_list = [np.random.rand(num_users - len(hot_spots) * num_hot_users_per_spot, dim)]
        for spot in hot_spots:
            user_vectors_list.append(np.random.normal(loc=spot, scale=0.1, size=(num_hot_users_per_spot, dim)))
        user_vectors = np.vstack(user_vectors_list)
        np.random.shuffle(user_vectors)
    else:
        raise ValueError("Unknown scenario.")

    # 2. Generate preference matrix and job similarity matrix
    preference_matrix = cosine_similarity(user_vectors, job_vectors)
    job_similarity_matrix = cosine_similarity(job_vectors)

    # 3. Generate sorted preference lists for each user
    user_preferences = []
    for user_idx in range(num_users):
        scores = preference_matrix[user_idx]
        sorted_jobs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        user_pref_list = [{'job_id': job_id, 'score': score} for job_id, score in sorted_jobs]
        user_preferences.append(user_pref_list)

    # Add company info for diversity calculation
    num_companies = int(num_jobs / 2)  # Assume 2 jobs per company on average
    company_ids = np.random.randint(0, num_companies, size=num_jobs)
    # Simulate company popularity (some companies get more applications)
    # We use a power-law like distribution for popularity ranks
    company_popularity = np.random.power(0.5, size=num_companies)
    company_rank = np.argsort(np.argsort(-company_popularity)) + 1

    job_meta = {
        'company_id': company_ids,
        'company_rank': company_rank[company_ids],
        'num_companies': num_companies
    }

    return user_preferences, preference_matrix, job_similarity_matrix, user_vectors, job_vectors, job_meta


# 3. --- METRIC CALCULATION HELPERS (Chapter 4.2) ---
def calculate_gini(x: np.ndarray) -> float:
    """Calculates the Gini coefficient for a given distribution (e.g., applications per job)."""
    if np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x, dtype=float)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def calculate_ndcg_at_k(true_rank: int, k: int) -> float:
    """Calculates NDCG@k for a single item."""
    if true_rank is None or true_rank >= k:
        return 0.0
    # The rank passed is 0-indexed, so we add 2 for the log denominator (pos 1 -> rank 0 -> log(0+2))
    dcg = 1.0 / np.log2(true_rank + 2)
    # The ideal position is rank 1 (index 0). So the denominator should be log2(0 + 2) or log2(1 + 1).
    idcg = 1.0 / np.log2(1 + 1)  # Corrected line
    return dcg / idcg

def calculate_mjvd(matched_job_vectors: np.ndarray) -> float:
    """Calculates Matched Job Vector Diversity."""
    if len(matched_job_vectors) < 2:
        return 0.0
    # Use 1 - cosine_similarity as the distance metric
    sim_matrix = cosine_similarity(matched_job_vectors)
    distances = 1 - sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
    return np.mean(distances)


# 4. --- AGENT/MODEL IMPLEMENTATION (Chapter 3 & 4.3) ---

class BaseAgent:
    """Abstract base class for all recommendation agents."""

    def __init__(self, num_users, num_jobs, user_prefs, pref_matrix, job_sim_matrix, job_vectors, job_meta):
        self.num_users = num_users
        self.num_jobs = num_jobs
        self.user_preferences = user_prefs
        self.preference_matrix = pref_matrix
        self.job_similarity_matrix = job_sim_matrix
        self.job_vectors = job_vectors
        self.job_meta = job_meta
        self.model_name = "BaseAgent"

    def get_recommendation(self, user_id: int, market_state: Dict) -> int:
        """
        Main method to get a recommendation for a user.
        Must be implemented by all subclasses.
        """
        raise NotImplementedError

    def train(self, historical_data):
        """Optional training method for models that need it (e.g., LiJAR)."""
        pass


class TopKGreedyAgent(BaseAgent):
    """Baseline: Recommends the user's most preferred available job."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "Top-k Greedy"

    def get_recommendation(self, user_id: int, market_state: Dict) -> int:
        # Recommends the highest-ranked job from the user's original preference list.
        # In this simulation, this is equivalent to their first choice.
        return self.user_preferences[user_id][0]['job_id']


class DegreeCentralityAgent(BaseAgent):
    """Baseline: A simple heuristic balancing preference and network degree."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "Degree Centrality"
        self.lambda_dc = 0.5  # Hyperparameter to balance preference vs. centrality
        # Pre-calculate degree centrality
        # Build adjacency matrix based on a similarity threshold
        adj_matrix = (self.job_similarity_matrix > 0.8).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        self.degree_centrality = np.sum(adj_matrix, axis=1)
        # Normalize to [0, 1]
        if np.max(self.degree_centrality) > 0:
            self.degree_centrality = self.degree_centrality / np.max(self.degree_centrality)

    def get_recommendation(self, user_id: int, market_state: Dict) -> int:
        scores = []
        user_match_scores = self.preference_matrix[user_id]
        for job_id in range(self.num_jobs):
            # Score(j) = MatchScore(u, j) - lambda * Degree(j)
            score = user_match_scores[job_id] - self.lambda_dc * self.degree_centrality[job_id]
            scores.append((job_id, score))

        # Return the job_id with the highest score
        recommended_job_id = max(scores, key=lambda x: x[1])[0]
        return recommended_job_id


class LiJARForecastAgent(BaseAgent):
    """Baseline: Predicts future congestion and penalizes recommendations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "LiJAR-Forecast"
        self.lambda_lijar = 0.5  # Hyperparameter
        self.model = LinearRegression()
        self.is_trained = False

    def train(self, historical_data: Dict[str, np.ndarray]):
        """Trains the linear regression model on historical data."""
        X = historical_data['job_vectors']
        y = historical_data['final_applications']
        self.model.fit(X, y)
        self.is_trained = True
        # print(f"LiJAR-Forecast model trained.") # Silenced for cleaner main run output

    def get_recommendation(self, user_id: int, market_state: Dict) -> int:
        if not self.is_trained:
            # Fallback to Top-K if not trained
            return self.user_preferences[user_id][0]['job_id']

        # Predict future congestion for all jobs
        predicted_congestion = self.model.predict(self.job_vectors)
        # Normalize prediction
        if np.max(predicted_congestion) > 0:
            predicted_congestion = predicted_congestion / np.max(predicted_congestion)

        scores = []
        user_match_scores = self.preference_matrix[user_id]
        for job_id in range(self.num_jobs):
            # Score(j) = MatchScore(u, j) - lambda * PredictedCongestion(j)
            score = user_match_scores[job_id] - self.lambda_lijar * predicted_congestion[job_id]
            scores.append((job_id, score))

        recommended_job_id = max(scores, key=lambda x: x[1])[0]
        return recommended_job_id


class MGOAgent(BaseAgent):
    """The proposed MGO-Agent."""

    def __init__(self, ablation: Optional[str] = None, pagerank_alpha=0.15, **kwargs):
        super().__init__(**kwargs)
        self.ablation = ablation
        self.pagerank_alpha = pagerank_alpha
        self.model_name = f"MGO-Agent ({ablation})" if ablation else "MGO-Agent (Full)"

        if self.ablation == "No-NSGAIII": self.model_name = "MGO-No-NSGAIII"
        if self.ablation == "No-Pagerank": self.model_name = "MGO-No-Pagerank"
        if self.ablation == "No-Diversity": self.model_name = "MGO-No-Diversity"

        # Hyperparameters from the paper
        self.qual_threshold = 0.3  # Feasibility boundary
        self.risk_aversion_k = 0.5  # For fail_count -> risk_aversion
        self.diversity_weight = 0.1  # w_div in selection formula

    # --- 3.2.1 Perception Module ---
    def _perception_module(self, market_state: Dict) -> np.ndarray:
        """Calculates Networked Congestion Pressure (CP)."""
        # Ablation: Use direct count instead of PageRank
        if self.ablation == "No-Pagerank":
            applications = market_state['applications_per_job']
            capacity = market_state['initial_capacity']
            # Add a small epsilon to avoid division by zero
            return applications / (capacity + 1e-6)

        # 1. Build capacity-aware adjacency matrix W
        W = np.zeros((self.num_jobs, self.num_jobs))
        remaining_capacity = market_state['job_capacity']
        for i in range(self.num_jobs):
            for j in range(self.num_jobs):
                if i != j and self.job_similarity_matrix[i, j] > 0.7:  # Similarity threshold
                    W[i, j] = self.job_similarity_matrix[i, j] * remaining_capacity[j]

        # 2. Create column-normalized transition matrix M
        col_sums = W.sum(axis=0)
        # Handle columns that sum to zero to avoid division by zero
        non_zero_cols = col_sums > 0
        M = np.zeros_like(W)
        M[:, non_zero_cols] = W[:, non_zero_cols] / col_sums[non_zero_cols]

        # Handle dangling nodes (columns that sum to 0) by creating uniform transition
        dangling_nodes = col_sums == 0
        if np.any(dangling_nodes):
            M[:, dangling_nodes] = 1.0 / self.num_jobs

        # 3. Personalized PageRank calculation
        # Here we use the global_ppr variant as the default
        v = market_state['applications_per_job'].copy().astype(float)
        if v.sum() > 0:
            v = v / v.sum()
        else:  # If no applications yet, use uniform
            v = np.ones(self.num_jobs) / self.num_jobs

        cp = np.ones(self.num_jobs) / self.num_jobs  # Initial CP
        for _ in range(20):  # Iterate to convergence
            cp_new = (1 - self.pagerank_alpha) * (M.T @ cp) + self.pagerank_alpha * v
            # Check for convergence
            if np.linalg.norm(cp_new - cp, 1) < 1e-6:
                break
            cp = cp_new

        return cp

    # --- 3.2.2 Decision Module ---
    def _decision_module(self, user_id: int, cp_vector: np.ndarray) -> List[Dict]:
        """Generates the Pareto front using NSGA-III."""
        user_match_scores = self.preference_matrix[user_id]

        # 1. Feasibility Boundary Setting (Constraint Handling)
        feasible_jobs_indices = [j for j in range(self.num_jobs) if user_match_scores[j] >= self.qual_threshold]
        if not feasible_jobs_indices:
            # If no job is feasible, return an empty array
            return []

        # Ablation: If not using NSGA-III, use a simple weighted sum
        if self.ablation == "No-NSGAIII":
            scores = []
            for job_id in feasible_jobs_indices:
                # Weighted sum of normalized objectives. Here we use a simple heuristic.
                f1 = user_match_scores[job_id]
                f2 = cp_vector[job_id]
                # Combine them (f1 is max, f2 is min)
                score = f1 - 0.5 * f2
                scores.append((job_id, score))
            best_job_id = max(scores, key=lambda x: x[1])[0]
            # Return a "front" with just this single best solution
            return [{
                "job_id": best_job_id,
                "f1_qual": user_match_scores[best_job_id],
                "f2_cong": cp_vector[best_job_id],
                "f3_div": 0
            }]

        # --- Pymoo Problem Definition for NSGA-III ---
        class MultiObjectiveProblem(Problem):
            def __init__(self, mgo_agent: 'MGOAgent', user_id: int, cp: np.ndarray, feasible_indices: List[int]):
                self.agent = mgo_agent
                self.user_match_scores = self.agent.preference_matrix[user_id]
                self.cp = cp
                self.feasible_indices = feasible_indices
                self.num_objectives = 2 if self.agent.ablation == "No-Diversity" else 3

                super().__init__(n_var=1, n_obj=self.num_objectives, n_constr=0, xl=0, xu=len(feasible_indices) - 1,
                                 type_var=int)

            def _evaluate(self, x, out, *args, **kwargs):
                # x is a matrix of shape (n_solutions, n_var=1)
                # Cast index to int during evaluation
                job_indices = [self.feasible_indices[int(idx[0])] for idx in x]

                # f1: Maximize MatchQuality -> Minimize -MatchQuality
                f1 = -self.user_match_scores[job_indices]

                # f2: Minimize CongestionPressure
                f2 = self.cp[job_indices]

                objectives = [f1, f2]

                if self.num_objectives == 3:
                    # f3: Maximize Diversity -> Minimize -Diversity
                    ranks = self.agent.job_meta['company_rank'][job_indices]
                    num_companies = self.agent.job_meta['num_companies']
                    f3_raw = (ranks - 1) / (num_companies - 1)
                    f3 = -f3_raw
                    objectives.append(f3)

                out["F"] = np.column_stack(objectives)

        problem = MultiObjectiveProblem(self, user_id, cp_vector, feasible_jobs_indices)

        # Setup NSGA-III algorithm
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
        algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)
        termination = get_termination("n_gen", 40)

        res = minimize(problem, algorithm, termination, seed=1, verbose=False)

        # Return the Pareto front solutions. Store job_id along with objectives.
        pareto_front = []
        if res.X is not None and res.F is not None:
            # De-duplicate solutions (sometimes NSGA3 returns duplicate points)
            unique_designs, unique_indices = np.unique(res.X.astype(int), return_index=True)
            for i in unique_indices:
                # Cast index to int when processing results
                job_idx_in_feasible = int(res.X[i][0])
                original_job_id = feasible_jobs_indices[job_idx_in_feasible]
                objectives = res.F[i]
                pareto_front.append({
                    "job_id": original_job_id,
                    "f1_qual": -objectives[0],  # Convert back to maximization value
                    "f2_cong": objectives[1],
                    "f3_div": -objectives[2] if problem.n_obj == 3 else 0
                })

        return pareto_front

    # --- 3.2.3 Selection Module ---
    def _selection_module(self, user_id: int, fail_count: int, pareto_front: List[Dict]) -> int:
        """Selects the final action from the Pareto front."""
        if not pareto_front:
            # Fallback to Top-K if front is empty
            return self.user_preferences[user_id][0]['job_id']

        # 1. Dynamic Behavior Modeling
        risk_aversion = 1 - np.exp(-self.risk_aversion_k * fail_count)

        # 2. Weighted Sorting and Final Decision
        # Normalize f1 and f2 scores in the pareto front to [0, 1] for fair weighting
        f1_scores = np.array([s['f1_qual'] for s in pareto_front])
        f2_scores = np.array([s['f2_cong'] for s in pareto_front])

        # Handle cases with a single point on the front where normalization would be 0/0
        norm_f1 = (f1_scores - f1_scores.min()) / (f1_scores.max() - f1_scores.min() + 1e-9)
        norm_f2 = (f2_scores - f2_scores.min()) / (f2_scores.max() - f2_scores.min() + 1e-9)

        final_scores = []
        for i, solution in enumerate(pareto_front):
            # Score = (1-risk) * norm(f1) - risk * norm(f2) + w_div * f3
            score = (1 - risk_aversion) * norm_f1[i] - \
                    risk_aversion * norm_f2[i] + \
                    self.diversity_weight * solution['f3_div']
            final_scores.append((solution['job_id'], score))

        recommended_job_id = max(final_scores, key=lambda x: x[1])[0]

        # For visualization (Fig 5.4), we need to save the state
        if hasattr(self, 'visualization_hook') and self.visualization_hook == user_id:
            self._save_visualization_data(pareto_front, risk_aversion, recommended_job_id)

        return recommended_job_id

    def get_recommendation(self, user_id: int, market_state: Dict) -> int:
        """The full MGO-Agent pipeline."""
        # 1. Perception
        congestion_pressure = self._perception_module(market_state)
        # 2. Decision
        pareto_front = self._decision_module(user_id, congestion_pressure)
        # 3. Selection
        fail_count = market_state['user_fail_counts'][user_id]
        recommendation = self._selection_module(user_id, fail_count, pareto_front)

        return recommendation

    def _save_visualization_data(self, front, risk, final_choice_id):
        """Hook to save data for Figure 5.4."""
        self.viz_data = {
            "front": front,
            "risk_aversion": risk,
            "final_choice_id": final_choice_id
        }


# 5. --- SIMULATION ENGINE ---
class Simulation:
    """Manages the round-based simulation process."""

    def __init__(self, agent: BaseAgent, user_prefs, pref_matrix, job_sim, u_vec, j_vec, job_meta,
                 num_users, num_jobs, seed, accept_theta=0.7, accept_k=10.0,
                 fixed_accept_prob=None, max_rounds=10):
        # Core components
        self.agent = agent
        self.user_preferences = user_prefs
        self.preference_matrix = pref_matrix
        self.job_similarity_matrix = job_sim
        self.user_vectors = u_vec
        self.job_vectors = j_vec
        self.job_meta = job_meta
        self.num_users = num_users
        self.num_jobs = num_jobs
        self.seed = seed

        # Parameters
        self.max_rounds = max_rounds
        self.accept_theta = accept_theta
        self.accept_k = accept_k
        self.fixed_accept_prob = fixed_accept_prob

        # State variables
        self.job_capacity = np.ones(num_jobs, dtype=int) * int(num_users * 1.5 / num_jobs)
        self.initial_capacity = self.job_capacity.copy()
        self.user_status = np.full(num_users, -1, dtype=int)
        self.user_fail_counts = np.zeros(num_users, dtype=int)
        self.applications_total = np.zeros(num_jobs, dtype=int)

        # Logging
        self.gini_evolution = []
        # NEW: Detailed log for each successful match
        self.match_log = []
        # Precompute user-side ranks for faster lookup
        self.user_initial_ranks = {u: {p['job_id']: i for i, p in enumerate(self.user_preferences[u])} for u in
                                   range(self.num_users)}

    def _run_single_round(self, round_num: int):
        """Runs the two phases of a single simulation round."""
        unmatched_users = np.where(self.user_status == -1)[0]
        if not len(unmatched_users):
            return

        # --- Phase 1: Intention Gathering & Guidance ---
        market_state = {
            'job_capacity': self.job_capacity.copy(),
            'initial_capacity': self.initial_capacity,
            'user_fail_counts': self.user_fail_counts.copy(),
            'applications_per_job': self.applications_total.copy()
        }
        intentions = []
        start_time = time.time()

        for user_id in unmatched_users:
            recommended_job = self.agent.get_recommendation(user_id, market_state)
            match_quality = self.preference_matrix[user_id, recommended_job]
            if self.fixed_accept_prob is not None:
                acceptance_prob = self.fixed_accept_prob
            else:
                acceptance_prob = 1 / (1 + np.exp(-self.accept_k * (match_quality - self.accept_theta)))

            intended_job = recommended_job if np.random.rand() < acceptance_prob else self.user_preferences[user_id][0][ 'job_id']
            intentions.append((user_id, intended_job))

        self.runtime = (time.time() - start_time) / len(unmatched_users) if len(unmatched_users) > 0 else 0

        # --- Phase 2: Market Clearing & Matching ---
        applications_this_round = {j: [] for j in range(self.num_jobs)}
        for user_id, job_id in intentions:
            applications_this_round[job_id].append(user_id)
            self.applications_total[job_id] += 1

        for job_id in range(self.num_jobs):
            applicants = applications_this_round[job_id]
            if not applicants or self.job_capacity[job_id] == 0:
                continue

            applicant_scores = [(uid, self.preference_matrix[uid, job_id]) for uid in applicants]
            sorted_applicants = sorted(applicant_scores, key=lambda x: x[1], reverse=True)

            num_to_hire = min(len(sorted_applicants), self.job_capacity[job_id])
            for i in range(num_to_hire):
                hired_user_id, match_score = sorted_applicants[i]

                # Check if user is already matched in this same round to another job (rare but possible)
                if self.user_status[hired_user_id] != -1:
                    continue

                self.user_status[hired_user_id] = job_id
                self.job_capacity[job_id] -= 1

                # NEW: Log detailed match info
                self.match_log.append({
                    'user_id': hired_user_id,
                    'job_id': job_id,
                    'match_score': match_score,
                    'user_side_rank': self.user_initial_ranks[hired_user_id].get(job_id),
                    'job_side_rank': i  # 0-indexed rank among applicants for this job
                })

        for user_id in unmatched_users:
            if self.user_status[user_id] == -1:
                self.user_fail_counts[user_id] += 1

        self.gini_evolution.append(calculate_gini(self.applications_total))

    def run(self) -> Dict[str, Any]:
        """Runs the full simulation for all rounds."""
        if isinstance(self.agent, LiJARForecastAgent):
            dummy_sim = Simulation(
                TopKGreedyAgent(num_users=self.num_users, num_jobs=self.num_jobs, user_prefs=self.user_preferences,
                                pref_matrix=self.preference_matrix, job_sim_matrix=self.job_similarity_matrix,
                                job_vectors=self.job_vectors, job_meta=self.job_meta), self.user_preferences,
                self.preference_matrix, self.job_similarity_matrix, self.user_vectors, self.job_vectors, self.job_meta,
                self.num_users, self.num_jobs, self.seed)
            dummy_sim.run()
            historical_data = {
                'job_vectors': self.job_vectors,
                'final_applications': dummy_sim.applications_total
            }
            self.agent.train(historical_data)

        for i in range(self.max_rounds):
            self._run_single_round(i)
            if np.all(self.user_status != -1):
                break

        return self._calculate_final_metrics()

    def _calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculates all performance metrics from Chapter 4.2 after the simulation."""
        match_df = pd.DataFrame(self.match_log)
        num_successful = len(match_df)

        if num_successful == 0:
            # Return a dictionary of zeros if no matches occurred
            fairness_user_zeros = {"top_ndcg": 0, "top_rate": 0, "bot_ndcg": 0, "bot_rate": 0, "top_ahq": 0,
                                   "bot_ahq": 0, "top_rank": 0, "bot_rank": 0}
            fairness_job_zeros = {"top_fill": 0, "top_ahq": 0, "bot_fill": 0, "bot_ahq": 0, "top_rank": 0,
                                  "bot_rank": 0}
            return {"GMR": 0, "Gini": calculate_gini(self.applications_total), "Avg. NDCG@10": 0, "Job AHQ": 0,
                    "User AHQ": 0,
                    "Avg. User-side Rank": 0, "Avg. Job-side Rank": 0, "MJVD": 0, "Runtime": self.runtime * 1000,
                    "gini_evo": self.gini_evolution, "fairness_user": fairness_user_zeros,
                    "fairness_job": fairness_job_zeros}

        # --- Standard Metrics ---
        gmr = num_successful / self.num_users
        final_gini = calculate_gini(self.applications_total)

        # NEW: Calculate all new metrics from the match_df
        match_df['ndcg'] = match_df['user_side_rank'].apply(lambda r: calculate_ndcg_at_k(r, k=10))
        avg_ndcg = match_df['ndcg'].mean()

        # In our symmetric model, Job AHQ and User AHQ are the same value (match_score)
        job_ahq = match_df['match_score'].mean()
        user_ahq = job_ahq

        avg_user_rank = (match_df['user_side_rank'] + 1).mean()  # Convert 0-indexed to 1-based
        avg_job_rank = (match_df['job_side_rank'] + 1).mean()  # Convert 0-indexed to 1-based

        matched_vectors = self.job_vectors[match_df['job_id'].unique()]
        mjvd = calculate_mjvd(matched_vectors)

        # --- Fairness Analysis ---
        user_avg_scores = np.mean(self.preference_matrix, axis=1)
        top_20_user_thresh = np.percentile(user_avg_scores, 80)
        bot_20_user_thresh = np.percentile(user_avg_scores, 20)
        top_users = np.where(user_avg_scores >= top_20_user_thresh)[0]
        bot_users = np.where(user_avg_scores <= bot_20_user_thresh)[0]

        top_user_matches_df = match_df[match_df['user_id'].isin(top_users)]
        bot_user_matches_df = match_df[match_df['user_id'].isin(bot_users)]

        fairness_user = {
            "top_rate": len(top_user_matches_df) / len(top_users) if len(top_users) > 0 else 0,
            "bot_rate": len(bot_user_matches_df) / len(bot_users) if len(bot_users) > 0 else 0,
            "top_ndcg": top_user_matches_df['ndcg'].mean() if not top_user_matches_df.empty else 0,
            "bot_ndcg": bot_user_matches_df['ndcg'].mean() if not bot_user_matches_df.empty else 0,
            "top_ahq": top_user_matches_df['match_score'].mean() if not top_user_matches_df.empty else 0,
            # User AHQ for top users
            "bot_ahq": bot_user_matches_df['match_score'].mean() if not bot_user_matches_df.empty else 0,
            "top_rank": (top_user_matches_df['user_side_rank'] + 1).mean() if not top_user_matches_df.empty else 0,
            "bot_rank": (bot_user_matches_df['user_side_rank'] + 1).mean() if not bot_user_matches_df.empty else 0
        }

        job_avg_scores = np.mean(self.preference_matrix, axis=0)
        top_20_job_thresh = np.percentile(job_avg_scores, 80)
        bot_20_job_thresh = np.percentile(job_avg_scores, 20)
        top_jobs = np.where(job_avg_scores >= top_20_job_thresh)[0]
        bot_jobs = np.where(job_avg_scores <= bot_20_job_thresh)[0]

        top_job_matches_df = match_df[match_df['job_id'].isin(top_jobs)]
        bot_job_matches_df = match_df[match_df['job_id'].isin(bot_jobs)]

        fairness_job = {
            "top_fill": len(top_job_matches_df) / len(top_jobs) if len(top_jobs) > 0 else 0,
            "bot_fill": len(bot_job_matches_df) / len(bot_jobs) if len(bot_jobs) > 0 else 0,
            "top_ahq": top_job_matches_df['match_score'].mean() if not top_job_matches_df.empty else 0,
            # Job AHQ for top jobs
            "bot_ahq": bot_job_matches_df['match_score'].mean() if not bot_job_matches_df.empty else 0,
            "top_rank": (top_job_matches_df['job_side_rank'] + 1).mean() if not top_job_matches_df.empty else 0,
            "bot_rank": (bot_job_matches_df['job_side_rank'] + 1).mean() if not bot_job_matches_df.empty else 0
        }

        return {
            "GMR": gmr, "Gini": final_gini, "Avg. NDCG@10": avg_ndcg,
            "Job AHQ": job_ahq, "User AHQ": user_ahq,
            "Avg. User-side Rank": avg_user_rank, "Avg. Job-side Rank": avg_job_rank,
            "MJVD": mjvd, "Runtime": self.runtime * 1000,
            "gini_evo": self.gini_evolution,
            "fairness_user": fairness_user,
            "fairness_job": fairness_job
        }

# 6. --- EXPERIMENT ORCHESTRATION ---
def run_experiment(agent_class, agent_params, sim_params, seeds):
    """Helper to run a simulation across multiple seeds and aggregate results."""
    all_results = []
    pbar = tqdm(seeds, desc=f"Running {agent_class.__name__}")

    for i, seed in enumerate(pbar):
        sim_params['seed'] = seed
        np.random.seed(seed)
        user_prefs, pref_matrix, job_sim, u_vec, j_vec, job_meta = generate_simulation_data(
            num_users=sim_params['num_users'], num_jobs=sim_params['num_jobs'],
            scenario=sim_params['scenario'], seed=seed)
        agent_args = {"num_users": sim_params['num_users'], "num_jobs": sim_params['num_jobs'],
                      "user_prefs": user_prefs, "pref_matrix": pref_matrix,
                      "job_sim_matrix": job_sim, "job_vectors": j_vec, "job_meta": job_meta}
        agent_args.update(agent_params)
        agent = agent_class(**agent_args)
        if i == 0: pbar.set_description(f"Running {agent.model_name}")
        sim_args = {"agent": agent, "user_prefs": user_prefs, "pref_matrix": pref_matrix,
                    "job_sim": job_sim, "u_vec": u_vec, "j_vec": j_vec, "job_meta": job_meta,
                    "num_users": sim_params['num_users'], "num_jobs": sim_params['num_jobs'],
                    "seed": seed, "max_rounds": sim_params.get('max_rounds', 10),
                    "fixed_accept_prob": sim_params.get('fixed_accept_prob', None)}
        simulation = Simulation(**sim_args)
        result = simulation.run()
        all_results.append(result)

    # Aggregate results
    aggregated = {}
    # NEW: Add new metrics to the aggregation list
    keys_to_agg = ["GMR", "Gini", "Avg. NDCG@10", "Job AHQ", "User AHQ",
                   "Avg. User-side Rank", "Avg. Job-side Rank", "MJVD", "Runtime"]
    # NEW: Add new fairness keys
    fairness_keys_user = ["top_rate", "bot_rate", "top_ndcg", "bot_ndcg", "top_ahq", "bot_ahq", "top_rank", "bot_rank"]
    fairness_keys_job = ["top_fill", "bot_fill", "top_ahq", "bot_ahq", "top_rank", "bot_rank"]

    for key in keys_to_agg:
        values = [r.get(key, 0) for r in all_results]  # Use .get for safety
        mean = np.mean(values)
        ci = 1.96 * (np.std(values) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0
        aggregated[key] = (mean, ci)

    aggregated['fairness_user'] = {}
    for key in fairness_keys_user:
        values = [r.get('fairness_user', {}).get(key, 0) for r in all_results]
        mean = np.mean(values)
        ci = 1.96 * (np.std(values) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0
        aggregated['fairness_user'][key] = (mean, ci)

    aggregated['fairness_job'] = {}
    for key in fairness_keys_job:
        values = [r.get('fairness_job', {}).get(key, 0) for r in all_results]
        mean = np.mean(values)
        ci = 1.96 * (np.std(values) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0
        aggregated['fairness_job'][key] = (mean, ci)

    gini_evos = [r['gini_evo'] for r in all_results if 'gini_evo' in r]
    if gini_evos:
        max_len = max(len(evo) for evo in gini_evos)
        padded_evos = [np.pad(evo, (0, max_len - len(evo)), 'edge') for evo in gini_evos]
        aggregated['gini_evo'] = np.mean(padded_evos, axis=0)
    else:
        aggregated['gini_evo'] = []

    # NEW: Save the raw results for this experiment run to a CSV
    # This will be useful for creating detailed plots later
    raw_df = pd.DataFrame(all_results)
    # Flatten nested dictionaries for easier CSV storage
    if not raw_df.empty:
        for col in ['fairness_user', 'fairness_job']:
            if col in raw_df.columns:
                nested_df = pd.json_normalize(raw_df[col]).add_prefix(f'{col}.')
                raw_df = raw_df.drop(col, axis=1).join(nested_df)

    # We will save this raw_df in the main block after each experiment
    aggregated['raw_df'] = raw_df

    return aggregated


if __name__ == '__main__':
    # --- GLOBAL CONFIGURATION ---
    SEEDS = [27, 83, 4, 59, 98]

    NUM_USERS = 200
    NUM_JOBS = 40
    MAX_ROUNDS = 10
    OUTPUT_DIR = "Exp_data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_run_results = {}

    print("--- STARTING MGO-AGENT FULL EXPERIMENTAL RUN ---")

    # --- Exp 1, 3, 4: Core Red Sea Benchmark & Ablations ---
    print("\n[1/5] Running Core Benchmark in 'Red Sea' Market...")
    sim_params_redsea = {"num_users": NUM_USERS, "num_jobs": NUM_JOBS, "scenario": "red_sea", "max_rounds": MAX_ROUNDS}
    models_to_run = {
        "MGO-Agent (Full)": (MGOAgent, {}), "Top-k Greedy": (TopKGreedyAgent, {}),
        "Degree Centrality": (DegreeCentralityAgent, {}), "LiJAR-Forecast": (LiJARForecastAgent, {}),
        "MGO-No-Pagerank": (MGOAgent, {"ablation": "No-Pagerank"}),
        "MGO-No-NSGAIII": (MGOAgent, {"ablation": "No-NSGAIII"}),
        "MGO-No-Diversity": (MGOAgent, {"ablation": "No-Diversity"}),
    }
    benchmark_raw_dfs = {}
    for name, (agent_class, params) in models_to_run.items():
        results = run_experiment(agent_class, params, sim_params_redsea, SEEDS)
        all_run_results[name] = results
        benchmark_raw_dfs[name] = results.pop('raw_df', pd.DataFrame())  # Pop raw_df to keep results dict clean

    # NEW: Save raw data for benchmark
    pd.concat(benchmark_raw_dfs, names=['Model']).reset_index().to_csv(
        os.path.join(OUTPUT_DIR, "raw_data_benchmark.csv"), index=False)

    # --- Exp 2: Robustness Across Markets ---
    print("\n[2/5] Running Robustness Checks in 'Blue Sea' & 'Multi-polar' Markets...")
    robustness_results = {}
    robustness_raw_dfs = {}
    for scenario in ["blue_sea", "multi_polar"]:
        sim_params = {"num_users": NUM_USERS, "num_jobs": NUM_JOBS, "scenario": scenario, "max_rounds": MAX_ROUNDS}
        robustness_results[scenario] = {}
        robustness_raw_dfs[scenario] = {}
        for name in ["MGO-Agent (Full)", "Top-k Greedy"]:
            agent_class, params = models_to_run[name]
            results = run_experiment(agent_class, params, sim_params, SEEDS)
            robustness_results[scenario][name] = results
            robustness_raw_dfs[scenario][name] = results.pop('raw_df', pd.DataFrame())

    # NEW: Save raw data for robustness
    pd.concat({(s, m): df for s, models in robustness_raw_dfs.items() for m, df in models.items()},
              names=['Scenario', 'Model']).reset_index().to_csv(os.path.join(OUTPUT_DIR, "raw_data_robustness.csv"),
                                                                index=False)

    # --- Exp 5: Parameter Sensitivity (Alpha) ---
    print("\n[3/5] Running Parameter Sensitivity Analysis for PageRank Alpha...")
    alpha_results = {}
    alpha_raw_dfs = {}
    alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    for alpha in alphas:
        agent_params = {"pagerank_alpha": alpha}
        results = run_experiment(MGOAgent, agent_params, sim_params_redsea, SEEDS)
        alpha_results[alpha] = results
        alpha_raw_dfs[alpha] = results.pop('raw_df', pd.DataFrame())

    # NEW: Save raw data for alpha sensitivity
    pd.concat(alpha_raw_dfs, names=['alpha']).reset_index().to_csv(
        os.path.join(OUTPUT_DIR, "raw_data_alpha_sensitivity.csv"), index=False)

    # --- Exp 6: User Trust Sensitivity (p_accept) ---
    print("\n[4/5] Running User Trust Sensitivity Analysis for Acceptance Rate...")
    p_accept_results = {}
    p_accept_raw_dfs = {}
    p_accepts = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for p in p_accepts:
        sim_p = sim_params_redsea.copy()
        sim_p["fixed_accept_prob"] = p
        results = run_experiment(MGOAgent, {}, sim_p, SEEDS)
        p_accept_results[p] = results
        p_accept_raw_dfs[p] = results.pop('raw_df', pd.DataFrame())

    # NEW: Save raw data for p_accept sensitivity
    pd.concat(p_accept_raw_dfs, names=['p_accept']).reset_index().to_csv(
        os.path.join(OUTPUT_DIR, "raw_data_p_accept_sensitivity.csv"), index=False)

    # --- Exp 7: Visualization Data for Pareto Front ---
    # (This section remains unchanged as it already generates specific data for one plot)
    print("\n[5/5] Generating Visualization Data for a Single Decision...")
    viz_seed = 42
    np.random.seed(viz_seed)
    user_prefs_viz, pref_matrix_viz, job_sim_viz, u_vec_viz, j_vec_viz, job_meta_viz = generate_simulation_data(
        NUM_USERS, NUM_JOBS, scenario='red_sea', seed=viz_seed)
    viz_agent = MGOAgent(num_users=NUM_USERS, num_jobs=NUM_JOBS, user_prefs=user_prefs_viz, pref_matrix=pref_matrix_viz,
                         job_sim_matrix=job_sim_viz, job_vectors=j_vec_viz, job_meta=job_meta_viz)
    user_avg_scores = np.mean(pref_matrix_viz, axis=1)
    viz_user_id = np.argsort(user_avg_scores)[-int(NUM_USERS * 0.1)]
    setattr(viz_agent, 'visualization_hook', viz_user_id)
    market_state_dummy = {'job_capacity': np.ones(NUM_JOBS, dtype=int) * int(NUM_USERS * 1.5 / NUM_JOBS),
                          'initial_capacity': np.ones(NUM_JOBS, dtype=int) * int(NUM_USERS * 1.5 / NUM_JOBS),
                          'user_fail_counts': np.zeros(NUM_USERS, dtype=int),
                          'applications_per_job': np.random.randint(0, 10, size=NUM_JOBS)}
    market_state_dummy['user_fail_counts'][viz_user_id] = 0
    viz_agent.get_recommendation(viz_user_id, market_state_dummy)
    viz_data_low_risk = viz_agent.viz_data
    market_state_dummy['user_fail_counts'][viz_user_id] = 5
    viz_agent.get_recommendation(viz_user_id, market_state_dummy)
    viz_data_high_risk = viz_agent.viz_data
    feasible_jobs_indices_viz = [j for j in range(NUM_JOBS) if
                                 pref_matrix_viz[viz_user_id, j] >= viz_agent.qual_threshold]
    cp_viz = viz_agent._perception_module(market_state_dummy)
    background_points = [(pref_matrix_viz[viz_user_id, j], cp_viz[j]) for j in feasible_jobs_indices_viz]
    # NEW: Save viz data to json
    with open(os.path.join(OUTPUT_DIR, 'raw_data_figure_5_4.json'), 'w') as f:
        json.dump({
            "background_points": background_points,
            "viz_data_low_risk": viz_data_low_risk,
            "viz_data_high_risk": viz_data_high_risk
        }, f)

    print("\n--- ALL SIMULATIONS COMPLETE. GENERATING RESULTS... ---")

    # --- 7. RESULTS PROCESSING AND OUTPUT ---

    # --- Table 5.1 ---
    # NEW: Add new columns
    table_5_1_cols = ["GMR", "Gini", "Avg. NDCG@10", "Job AHQ", "User AHQ", "Avg. User-side Rank", "Avg. Job-side Rank",
                      "MJVD", "Runtime"]
    table_5_1_names = ["GMR ↑", "Gini ↓", "Avg. NDCG@10 ↑", "Job AHQ ↑", "User AHQ ↑", "Avg. User Rank ↓",
                       "Avg. Job Rank ↓", "MJVD ↑", "Runtime (ms) ↓"]
    df1 = pd.DataFrame({name: res for name, res in all_run_results.items() if
                        name in models_to_run and not name.startswith("MGO-No")}).T
    df1 = df1[table_5_1_cols]
    df1.index.name = "Model"
    df1.columns = table_5_1_names
    for col in df1.columns: df1[col] = df1[col].apply(
        lambda x: f"{x[0]:.3f} ± {x[1]:.3f}" if not col.startswith("Runtime") else f"{x[0]:.1f} ± {x[1]:.1f}")

    # --- Table 5.3 ---
    # NEW: Add new columns
    table_5_3_cols = ["GMR", "Gini", "Avg. NDCG@10", "Job AHQ", "User AHQ", "Avg. User-side Rank", "Avg. Job-side Rank",
                      "MJVD", "Runtime"]
    table_5_3_names = ["GMR ↑", "Gini ↓", "Avg. NDCG@10 ↑", "Job AHQ ↑", "User AHQ ↑", "Avg. User Rank ↓",
                       "Avg. Job Rank ↓", "MJVD ↑", "Runtime (ms) ↓"]
    df3 = pd.DataFrame({name: res for name, res in all_run_results.items() if name.startswith("MGO-")}).T
    df3 = df3.reindex(["MGO-Agent (Full)", "MGO-No-Pagerank", "MGO-No-NSGAIII", "MGO-No-Diversity"])
    df3 = df3[table_5_3_cols]
    df3.index.name = "Model Variant"
    df3.columns = table_5_3_names
    for col in df3.columns: df3[col] = df3[col].apply(
        lambda x: f"{x[0]:.3f} ± {x[1]:.3f}" if not col.startswith("Runtime") else f"{x[0]:.1f} ± {x[1]:.1f}")

    # --- Table 5.2 ---
    # NEW: Add new columns
    rows = []
    for scenario, models in robustness_results.items():
        for model_name, results in models.items():
            row = {"Market Scenario": scenario.replace("_", " ").title(), "Model": model_name}
            for metric, name in [("GMR", "GMR ↑"), ("Gini", "Gini ↓"), ("Avg. NDCG@10", "Avg. NDCG@10 ↑"),
                                 ("Job AHQ", "Job AHQ ↑"), ("User AHQ", "User AHQ ↑"),
                                 ("Avg. User-side Rank", "Avg. User Rank ↓"),
                                 ("Avg. Job-side Rank", "Avg. Job Rank ↓")]:
                row[name] = f"{results[metric][0]:.3f}"
            rows.append(row)
    df2 = pd.DataFrame(rows).set_index(["Market Scenario", "Model"])

    # --- Tables 5.4 & 5.5 ---
    # NEW: Add new columns
    rows_t4, rows_t5 = [], []
    for model_name in ["Top-k Greedy", "MGO-Agent (Full)"]:
        res_user = all_run_results[model_name]['fairness_user']
        res_job = all_run_results[model_name]['fairness_job']
        rows_t4.append({"Model": model_name, "Tier": "High-Competitiveness (Top 20%)",
                        "Avg. NDCG@10 ↑": f"{res_user['top_ndcg'][0]:.3f}",
                        "Match Success Rate ↑": f"{res_user['top_rate'][0]:.3f}",
                        "User AHQ ↑": f"{res_user['top_ahq'][0]:.3f}",
                        "Avg. User Rank ↓": f"{res_user['top_rank'][0]:.2f}"})
        rows_t4.append({"Model": model_name, "Tier": "Lower-Competitiveness (Bot 20%)",
                        "Avg. NDCG@10 ↑": f"{res_user['bot_ndcg'][0]:.3f}",
                        "Match Success Rate ↑": f"{res_user['bot_rate'][0]:.3f}",
                        "User AHQ ↑": f"{res_user['bot_ahq'][0]:.3f}",
                        "Avg. User Rank ↓": f"{res_user['bot_rank'][0]:.2f}"})
        rows_t5.append(
            {"Model": model_name, "Tier": "Top-Tier Jobs (Top 20%)", "Job Fill Rate ↑": f"{res_job['top_fill'][0]:.3f}",
             "Avg. Hiring Quality (AHQ) ↑": f"{res_job['top_ahq'][0]:.3f}",
             "Avg. Job Rank ↓": f"{res_job['top_rank'][0]:.2f}"})
        rows_t5.append({"Model": model_name, "Tier": "Long-Tail Jobs (Bot 20%)",
                        "Job Fill Rate ↑": f"{res_job['bot_fill'][0]:.3f}",
                        "Avg. Hiring Quality (AHQ) ↑": f"{res_job['bot_ahq'][0]:.3f}",
                        "Avg. Job Rank ↓": f"{res_job['bot_rank'][0]:.2f}"})
    df4 = pd.DataFrame(rows_t4).set_index(["Model", "Tier"])
    df5 = pd.DataFrame(rows_t5).set_index(["Model", "Tier"])

    # --- Save and Print Tables ---
    df1.to_csv(os.path.join(OUTPUT_DIR, "table_5_1.csv"))
    df2.to_csv(os.path.join(OUTPUT_DIR, "table_5_2.csv"))
    df3.to_csv(os.path.join(OUTPUT_DIR, "table_5_3.csv"))
    df4.to_csv(os.path.join(OUTPUT_DIR, "table_5_4.csv"))
    df5.to_csv(os.path.join(OUTPUT_DIR, "table_5_5.csv"))
    print("\n\n--- RESULTS ---")
    print("\n--- Table 5.1: Overall Performance Comparison in 'Red Sea' Market ---");
    print(df1.to_string())
    print("\n--- Table 5.3: Ablation Study of MGO-Agent Core Components ---");
    print(df3.to_string())
    print("\n--- Table 5.2: Robustness Analysis Across Market Scenarios ---");
    print(df2.to_string())
    print("\n--- Table 5.4: Impact on Different Tiers of Job Seekers ---");
    print(df4.to_string())
    print("\n--- Table 5.5: Impact on Different Tiers of Jobs ---");
    print(df5.to_string())

    # --- Figure 5.1 ---
    plt.figure(figsize=(10, 6));
    [plt.plot(all_run_results[name]['gini_evo'], label=name, marker='o', markersize=4) for name in
     ["Top-k Greedy", "LiJAR-Forecast", "MGO-Agent (Full)"]];
    plt.title("Figure 5.1: Congestion Gini Coefficient Dynamics by Simulation Round", fontsize=16);
    plt.xlabel("Simulation Round", fontsize=12);
    plt.ylabel("Congestion Gini Coefficient", fontsize=12);
    plt.legend();
    plt.grid(True, linestyle='--', alpha=0.6);
    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_5_1.png"))

    # --- Figure 5.2 ---
    plt.figure(figsize=(10, 6));
    alphas_plot = sorted(alpha_results.keys());
    gmr_vals = np.array([alpha_results[a]['GMR'][0] for a in alphas_plot]);
    gini_vals = np.array([alpha_results[a]['Gini'][0] for a in alphas_plot]);
    ndcg_vals = np.array([alpha_results[a]['Avg. NDCG@10'][0] for a in alphas_plot]);
    norm_gmr = (gmr_vals - gmr_vals.min()) / (gmr_vals.max() - gmr_vals.min() + 1e-9);
    norm_1_gini = (1 - gini_vals - (1 - gini_vals).min()) / ((1 - gini_vals).max() - (1 - gini_vals).min() + 1e-9);
    norm_ndcg = (ndcg_vals - ndcg_vals.min()) / (ndcg_vals.max() - ndcg_vals.min() + 1e-9);
    plt.plot(alphas_plot, norm_gmr, label="Normalized GMR", marker='s');
    plt.plot(alphas_plot, norm_1_gini, label="Normalized (1 - Gini)", marker='x');
    plt.plot(alphas_plot, norm_ndcg, label="Normalized Avg. NDCG@10", marker='o');
    plt.title(r"Figure 5.2: Impact of PageRank $\alpha$ on Key Metrics", fontsize=16);
    plt.xlabel(r"Teleportation Probability $\alpha$", fontsize=12);
    plt.ylabel("Normalized Metric Value", fontsize=12);
    plt.legend();
    plt.grid(True, linestyle='--', alpha=0.6);
    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_5_2.png"))

    # --- Figure 5.3 ---
    fig, ax1 = plt.subplots(figsize=(10, 6));
    p_accepts_plot = sorted(p_accept_results.keys());
    gmr_p_vals = [p_accept_results[p]['GMR'][0] for p in p_accepts_plot];
    gini_p_vals = [p_accept_results[p]['Gini'][0] for p in p_accepts_plot];
    line1, = ax1.plot(p_accepts_plot, gmr_p_vals, 'b-', marker='o', label="Global Match Rate (GMR)");
    ax1.set_xlabel("Average User Acceptance Rate ($p_{accept}$)", fontsize=12);
    ax1.set_ylabel("Global Match Rate (GMR)", color='b', fontsize=12);
    ax1.tick_params('y', colors='b');
    ax2 = ax1.twinx();
    line2, = ax2.plot(p_accepts_plot, gini_p_vals, 'r-', marker='s', label="Congestion Gini Coefficient");
    ax2.set_ylabel("Congestion Gini Coefficient", color='r', fontsize=12);
    ax2.tick_params('y', colors='r');
    fig.suptitle("Figure 5.3: Impact of User Acceptance Rate on System Efficiency & Equity", fontsize=16);
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]);
    ax2.legend(handles=[line1, line2], loc='center right');
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_5_3.png"))

    # --- NEW: Figure 5.3.1 ---
    plt.figure(figsize=(10, 6));
    p_accepts_plot = sorted(p_accept_results.keys());
    gmr_vals_p = np.array([p_accept_results[p]['GMR'][0] for p in p_accepts_plot]);
    gini_vals_p = np.array([p_accept_results[p]['Gini'][0] for p in p_accepts_plot]);
    ndcg_vals_p = np.array([p_accept_results[p]['Avg. NDCG@10'][0] for p in p_accepts_plot]);
    norm_gmr_p = (gmr_vals_p - gmr_vals_p.min()) / (gmr_vals_p.max() - gmr_vals_p.min() + 1e-9);
    norm_1_gini_p = (1 - gini_vals_p - (1 - gini_vals_p).min()) / (
                (1 - gini_vals_p).max() - (1 - gini_vals_p).min() + 1e-9);
    norm_ndcg_p = (ndcg_vals_p - ndcg_vals_p.min()) / (ndcg_vals_p.max() - ndcg_vals_p.min() + 1e-9);
    plt.plot(p_accepts_plot, norm_gmr_p, label="Normalized GMR", marker='s');
    plt.plot(p_accepts_plot, norm_1_gini_p, label="Normalized (1 - Gini)", marker='x');
    plt.plot(p_accepts_plot, norm_ndcg_p, label="Normalized Avg. NDCG@10", marker='o');
    plt.title(r"Figure 5.3.1: Impact of User Acceptance Rate on All Key Metrics", fontsize=16);
    plt.xlabel(r"Average User Acceptance Rate ($p_{accept}$)", fontsize=12);
    plt.ylabel("Normalized Metric Value", fontsize=12);
    plt.legend();
    plt.grid(True, linestyle='--', alpha=0.6);
    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_5_3_1.png"))

    # --- Figure 5.4 ---
    if viz_data_low_risk and viz_data_low_risk.get('front'):
        plt.figure(figsize=(10, 8));
        bg_x = [p[0] for p in background_points];
        bg_y = [p[1] for p in background_points];
        plt.scatter(bg_x, bg_y, c='gray', alpha=0.2, s=20, label="Feasible Candidates");
        front_data = sorted(viz_data_low_risk['front'], key=lambda p: p['f1_qual']);
        front_x = [s['f1_qual'] for s in front_data];
        front_y = [s['f2_cong'] for s in front_data];
        plt.plot(front_x, front_y, 'b-o', markersize=6, label="Pareto Front (NSGA-III Result)");
        low_risk_choice = next(
            s for s in viz_data_low_risk['front'] if s['job_id'] == viz_data_low_risk['final_choice_id']);
        high_risk_choice = next(
            s for s in viz_data_high_risk['front'] if s['job_id'] == viz_data_high_risk['final_choice_id']);
        plt.scatter(low_risk_choice['f1_qual'], low_risk_choice['f2_cong'], c='green', s=250, marker='*',
                    edgecolors='black', zorder=5, label=f"Low-Risk Choice (fail_count=0)");
        plt.scatter(high_risk_choice['f1_qual'], high_risk_choice['f2_cong'], c='red', s=250, marker='*',
                    edgecolors='black', zorder=5, label=f"High-Risk-Aversion Choice (fail_count=5)");
        plt.title("Figure 5.4: MGO-Agent Decision Visualization on the Pareto Front", fontsize=16);
        plt.xlabel("Matching Quality ($f_1$) →", fontsize=12);
        plt.ylabel("Congestion Pressure ($f_2$) →", fontsize=12);
        plt.legend(loc="upper right");
        plt.grid(True, linestyle='--', alpha=0.6);
        plt.tight_layout();
        plt.savefig(os.path.join(OUTPUT_DIR, "figure_5_4.png"))
    else:
        print("\nSkipping Figure 5.4 generation: No Pareto Front data was captured.")

    print(f"\nAll tables and figures saved to '{OUTPUT_DIR}' directory.")
    print("\n✅ All tasks completed successfully.")