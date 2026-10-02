"""Reproducible, outcome-free chemical maps and Gaussian GTM.

GTM follows Bishop, Svensen & Williams (1998): an equal-prior Gaussian
mixture constrained by a smooth RBF manifold, fitted with regularized EM.
Gaussian noise is applied to continuous PCA scores, not asserted to be a
Bernoulli model of fingerprint bits. Targets are never used in map fitting.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from dataclasses import asdict, dataclass

import numpy as np
from rdkit import rdBase
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .features import build_features


@dataclass(frozen=True)
class GTMConfig:
    nodes: int = 15
    rbfs: int = 5
    width: float = 1.0  # in units of RBF-centre spacing
    regularization: float = 0.01
    max_iterations: int = 150
    tolerance: float = 1e-5


class GaussianGTM:
    """Small dense reference implementation; predict never refits the manifold."""

    def __init__(self, config: GTMConfig | None = None):
        self.config = config or GTMConfig()
        c = self.config
        if not (c.nodes >= 3 and 2 <= c.rbfs < c.nodes and c.width > 0 and
                c.regularization >= 0 and c.max_iterations > 0 and c.tolerance > 0):
            raise ValueError("Invalid GTM grid, RBF, width, regularization or convergence settings")
        axis = np.linspace(-1, 1, c.nodes)
        self.grid = np.array([(x, y) for y in axis for x in axis])
        centres_axis = np.linspace(-1, 1, c.rbfs)
        centres = np.array([(x, y) for y in centres_axis for x in centres_axis])
        sigma = c.width * 2 / (c.rbfs - 1)
        self.phi = np.column_stack([np.exp(-cdist(self.grid, centres, 'sqeuclidean') / (2 * sigma**2)), np.ones(len(self.grid))])

    def _posterior(self, values):
        distance = cdist(values, self.phi @ self.weights, 'sqeuclidean')
        log_joint = -0.5 * self.beta * distance + values.shape[1] / 2 * np.log(self.beta / (2 * np.pi)) - np.log(len(self.grid))
        log_density = logsumexp(log_joint, axis=1)
        return np.exp(log_joint - log_density[:, None]), log_density, distance

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or min(values.shape) < 3 or not np.isfinite(values).all():
            raise ValueError("GTM fitting requires at least three finite rows and dimensions")
        if np.var(values, axis=0).sum() < 1e-12:
            raise ValueError("GTM needs non-constant reference chemistry")
        pca = PCA(n_components=min(3, values.shape[1]), svd_solver='full').fit(values)
        if pca.explained_variance_[0] < 1e-12:
            raise ValueError("GTM needs non-constant reference chemistry")
        plane = (self.grid * np.sqrt(pca.explained_variance_[:2])) @ pca.components_[:2] + pca.mean_
        self.weights = np.linalg.lstsq(self.phi, plane, rcond=None)[0]
        separation = cdist(plane, plane, 'sqeuclidean')
        np.fill_diagonal(separation, np.inf)
        variance = max(float(pca.explained_variance_[2]), float(np.mean(separation.min(axis=1))) / 2, 1e-6)
        self.beta = 1 / variance
        penalty = np.eye(self.phi.shape[1]); penalty[-1, -1] = 0  # unpenalized bias
        self.objective_history = []
        self.converged = False
        self.stop_reason = 'iteration_limit'
        for iteration in range(self.config.max_iterations + 1):
            responsibility, density, _ = self._posterior(values)
            objective = float(density.sum() - self.config.regularization / 2 * np.sum(self.weights[:-1]**2))
            if self.objective_history and objective < self.objective_history[-1] - 1e-6 * (1 + abs(self.objective_history[-1])):
                self.weights, self.beta = previous_weights, previous_beta
                self.stop_reason = 'numerical_objective_decrease'
                break
            self.objective_history.append(objective)
            if len(self.objective_history) > 1 and abs(objective - self.objective_history[-2]) <= self.config.tolerance * (1 + abs(self.objective_history[-2])):
                self.converged, self.stop_reason = True, 'relative_objective_tolerance'
                break
            if iteration == self.config.max_iterations:
                break
            previous_weights, previous_beta = self.weights.copy(), self.beta
            population = responsibility.sum(axis=0)
            lhs = self.phi.T @ (population[:, None] * self.phi) + self.config.regularization / self.beta * penalty
            rhs = self.phi.T @ responsibility.T @ values
            self.weights = np.linalg.lstsq(lhs, rhs, rcond=1e-12)[0]
            new_distance = cdist(values, self.phi @ self.weights, 'sqeuclidean')
            weighted_error = np.sum(responsibility * new_distance)
            self.beta = float(np.clip(values.size / max(weighted_error, 1e-12), 1e-10, 1e10))
        self.iterations = len(self.objective_history) - 1
        return self

    def project(self, values):
        values = np.asarray(values, dtype=float)
        if not hasattr(self, 'weights') or values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Fit GTM first and supply finite rows")
        responsibility, density, _ = self._posterior(values)
        entropy = -np.sum(responsibility * np.log(np.maximum(responsibility, 1e-300)), axis=1) / np.log(len(self.grid))
        return {'coordinates': responsibility @ self.grid, 'responsibilities': responsibility,
                'mode': responsibility.argmax(axis=1), 'entropy': entropy, 'log_density': density}


def _continuous_features(raw, reference, representation):
    """Fit imputation/scaling on reference chemistry, keep fingerprint bits unscaled."""
    blocks = []
    if representation in {'morgan', 'morgan_rdkit'}:
        fingerprints = raw[:, :1024].astype(float)
        total_variance = np.var(fingerprints[reference], axis=0).sum()
        blocks.append(fingerprints / np.sqrt(max(total_variance, 1e-12)))
    if representation in {'rdkit_descriptors', 'morgan_rdkit'}:
        descriptors = raw[:, 1024:] if representation == 'morgan_rdkit' else raw
        imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(descriptors[reference])
        imputed = imputer.transform(descriptors)
        scaler = StandardScaler().fit(imputed[reference])
        scaled = scaler.transform(imputed)
        active = scaler.var_ > 1e-12
        blocks.append(scaled[:, active] / np.sqrt(max(1, active.sum())))
    values = np.column_stack(blocks)
    dimensions = min(32, len(reference) - 1, values.shape[1])
    if dimensions < 3:
        raise ValueError('At least four reference molecules and three variable dimensions are required')
    pca = PCA(n_components=dimensions, svd_solver='full').fit(values[reference])
    transformed = pca.transform(values)
    # Scalar standardization keeps the reference's total PCA variance at D.
    scale = np.sqrt(max(float(np.var(transformed[reference], axis=0).mean()), 1e-12))
    return transformed / scale, {'pca_dimensions': dimensions, 'pca_variance_retained': float(pca.explained_variance_ratio_.sum()),
                                 'scaling': 'Reference imputation/scaling; fingerprint bits not variance-scaled; equal-total-variance feature blocks; PCA scores'}


def build_map_atlas(records, *, methods=('tsne', 'gtm', 'pca'), representations=('morgan', 'rdkit_descriptors', 'morgan_rdkit'), seed=42, perplexity=35, cache_dir=None, gtm_config=None):
    """Build maps once per unique chemical entity and reuse them across folds/tasks.

    GTM fits reference development chemistry when supplied; otherwise validation
    chemistry, then explicitly disclosed all-chemistry exploration as a last resort.
    t-SNE is transductive. It has no claimed out-of-sample transform.
    """
    methods, representations = tuple(dict.fromkeys(methods)), tuple(dict.fromkeys(representations))
    if not methods or set(methods) - {'tsne', 'gtm', 'pca'}:
        raise ValueError('Map methods must be tsne, gtm or pca')
    if not representations or set(representations) - {'morgan', 'rdkit_descriptors', 'morgan_rdkit'}:
        raise ValueError('Unknown chemical representation')
    molecules = {r['chemical_id']: r['canonical_smiles'] for r in records}
    ids = sorted(molecules)
    smiles = [molecules[i] for i in ids]
    if len(ids) < 4:
        return {'chemical_ids': ids, 'views': [], 'note': 'Fewer than four molecules: only the basic placeholder/PCA map is available.'}
    reference_ids = {r['chemical_id'] for r in records if r['split'] == 'development'}
    scope = 'development chemistry (pooled reference atlas, not a fold-specific applicability domain)'
    if len(reference_ids) < 4:
        reference_ids = {r['chemical_id'] for r in records if r['split'] == 'validation'}
        scope = 'validation chemistry; exploratory atlas, no held-out novelty claim'
    if len(reference_ids) < 4:
        reference_ids = set(ids)
        scope = 'all displayed chemistry including test; transductive exploration only'
    reference = np.array([i for i, key in enumerate(ids) if key in reference_ids])
    # Keep fitting cost bounded and repeatable without outcomes.
    if len(reference) > 2000:
        reference = np.sort(np.random.default_rng(seed).choice(reference, 2000, replace=False))
        scope += '; deterministic 2000-molecule reference sample'
    config = gtm_config or GTMConfig()
    import sklearn, scipy
    signature = hashlib.sha256(json.dumps({'version': 1, 'smiles': smiles, 'reference': reference.tolist(), 'methods': methods,
        'representations': representations, 'seed': seed, 'perplexity': perplexity, 'gtm': asdict(config),
        'rdkit': rdBase.rdkitVersion, 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}, sort_keys=True).encode()).hexdigest()
    cache_path = Path(cache_dir) / f'{signature}.json' if cache_dir else None
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        cached['reference_scope'] = scope
        return cached
    views = []
    with threadpool_limits(limits=2):
        for representation in representations:
            raw = build_features(smiles, representation=representation, descriptor_set='full').values
            continuous, preprocessing = _continuous_features(raw, reference, representation)
            for method in methods:
                view = {'id': f'{representation}:{method}', 'representation': representation, 'method': method,
                        'axes': [f'{method.upper()} 1', f'{method.upper()} 2'], 'preprocessing': preprocessing}
                if method == 'gtm':
                    model = GaussianGTM(config).fit(continuous[reference])
                    projected = model.project(continuous)
                    view.update(coordinates=projected['coordinates'].round(6).tolist(), grid=model.grid.tolist(),
                        responsibilities_b64=base64.b64encode(projected['responsibilities'].astype('<f4').tobytes()).decode(),
                        mode=projected['mode'].tolist(), entropy=projected['entropy'].round(6).tolist(),
                        log_density=projected['log_density'].round(6).tolist(),
                        fit={'config': asdict(config), 'iterations': model.iterations, 'converged': model.converged,
                             'stop_reason': model.stop_reason, 'objective_history': model.objective_history, 'beta': model.beta},
                        note='Gaussian GTM on continuous PCA scores; posterior mean coordinates. Projection entropy is not predictive uncertainty.')
                elif method == 'pca':
                    view['coordinates'] = continuous[:, :2].round(6).tolist()
                    view['note'] = 'Reference-fitted PCA, projected onto its first two components.'
                else:
                    if representation == 'morgan':
                        bits = raw.astype(np.float32)
                        intersection = bits @ bits.T
                        counts = bits.sum(axis=1)
                        union = counts[:, None] + counts[None, :] - intersection
                        distances = 1 - np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0)
                        np.fill_diagonal(distances, 0)
                        coordinates = TSNE(n_components=2, metric='precomputed', init='random', perplexity=min(perplexity, len(ids)-1), random_state=seed).fit_transform(distances)
                        view['note'] = 'Transductive t-SNE on original Morgan Tanimoto distances; global distances are not quantitative.'
                    else:
                        coordinates = TSNE(n_components=2, init='pca', perplexity=min(perplexity, len(ids)-1), random_state=seed).fit_transform(continuous)
                        view['note'] = 'Transductive t-SNE on reference-preprocessed continuous features; global distances are not quantitative.'
                    view['coordinates'] = coordinates.round(6).tolist()
                views.append(view)
    atlas = {'chemical_ids': ids, 'views': views, 'reference_scope': scope, 'reference_molecules': len(reference),
             'reference_chemical_ids': [ids[i] for i in reference], 'signature': signature}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(atlas, allow_nan=False, separators=(',', ':')))
    return atlas
