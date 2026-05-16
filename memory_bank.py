"""
Molecule Memory Bank for AdaRePO Self-Distillation.

Maintains a per-query bank of high-value generated molecules that can
replace the original reference as the guidance target.  This implements
the "learner becomes the improved oracle" concept from RPI:

  References(q, step_n) = {m_ref} ∪ B_n(q)

When the best entry in B(q) surpasses m_ref, the guidance loss trains
toward the model's own best generation (self-distillation).
"""
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BankEntry:
    """A single memory bank entry."""
    smiles: str
    reward: float
    step: int


class MoleculeMemoryBank:
    """
    Per-query memory bank of high-value generated molecules.

    Promotion criterion:
        v(o_i) > v(m_ref) + delta  AND  sim(m_i, m_ref) > sim_min  AND  valid(m_i)

    Active reference selection:
        m_star(q) = argmax_{m ∈ {m_ref} ∪ B(q)} v(m)
    """

    def __init__(
        self,
        max_size_per_query: int = 5,
        promotion_margin: float = 0.1,
        similarity_min: float = 0.3,
        max_age: int = 1000,
    ):
        self.max_size = max_size_per_query
        self.promotion_margin = promotion_margin
        self.similarity_min = similarity_min
        self.max_age = max_age
        self._bank: Dict[str, List[BankEntry]] = {}
        # lazy import so the module can be loaded without rdkit
        self._rdkit_ready = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_rdkit(self):
        if not self._rdkit_ready:
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem, DataStructs
                self._Chem = Chem
                self._AllChem = AllChem
                self._DataStructs = DataStructs
                self._rdkit_ready = True
            except ImportError:
                raise ImportError(
                    "rdkit is required for MoleculeMemoryBank. "
                    "Install with: pip install rdkit"
                )

    def _is_valid(self, smiles: str) -> bool:
        self._ensure_rdkit()
        try:
            mol = self._Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False

    def _tanimoto(self, smi_a: str, smi_b: str, n_bits: int = 1024) -> float:
        """Tanimoto similarity between two SMILES using Morgan fingerprints."""
        self._ensure_rdkit()
        try:
            mol_a = self._Chem.MolFromSmiles(smi_a)
            mol_b = self._Chem.MolFromSmiles(smi_b)
            if mol_a is None or mol_b is None:
                return 0.0
            fp_a = self._AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=n_bits)
            fp_b = self._AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=n_bits)
            return float(self._DataStructs.TanimotoSimilarity(fp_a, fp_b))
        except Exception:
            return 0.0

    @staticmethod
    def _query_hash(query: str) -> str:
        return hashlib.md5(query.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def try_promote(
        self,
        query: str,
        smiles: str,
        reward: float,
        ref_smiles: str,
        ref_reward: float,
        step: int,
    ) -> bool:
        """
        Attempt to promote a generated molecule into the memory bank.

        Returns True if the molecule was added.
        """
        # Check promotion criteria
        if reward <= ref_reward + self.promotion_margin:
            return False
        if not self._is_valid(smiles):
            return False
        sim = self._tanimoto(smiles, ref_smiles)
        if sim < self.similarity_min:
            return False

        qh = self._query_hash(query)
        entries = self._bank.get(qh, [])

        # Don't add exact duplicates
        for e in entries:
            if e.smiles == smiles:
                if reward > e.reward:
                    e.reward = reward
                    e.step = step
                return False  # updated existing, not a new addition

        entry = BankEntry(smiles=smiles, reward=reward, step=step)

        if len(entries) < self.max_size:
            entries.append(entry)
        else:
            # Replace the lowest-reward entry if new one is better
            min_idx = min(range(len(entries)), key=lambda i: entries[i].reward)
            if reward > entries[min_idx].reward:
                entries[min_idx] = entry
            else:
                return False

        self._bank[qh] = entries
        return True

    def get_best_reference(
        self,
        query: str,
        ref_smiles: str,
        ref_reward: float,
        current_step: int,
    ) -> Tuple[str, float]:
        """
        Select the best guidance target from {m_ref} ∪ B(q).

        Returns (best_smiles, best_reward).
        If no bank entry beats the reference, returns the reference.
        """
        qh = self._query_hash(query)
        entries = self._bank.get(qh, [])

        # Evict stale entries
        if self.max_age > 0:
            entries = [e for e in entries if (current_step - e.step) <= self.max_age]
            self._bank[qh] = entries

        best_smiles = ref_smiles
        best_reward = ref_reward

        for e in entries:
            if e.reward > best_reward:
                best_smiles = e.smiles
                best_reward = e.reward

        return best_smiles, best_reward

    def evict_stale(self, current_step: int):
        """Remove all entries older than max_age across all queries."""
        if self.max_age <= 0:
            return
        for qh in list(self._bank.keys()):
            self._bank[qh] = [
                e for e in self._bank[qh]
                if (current_step - e.step) <= self.max_age
            ]
            if not self._bank[qh]:
                del self._bank[qh]

    @property
    def total_size(self) -> int:
        return sum(len(v) for v in self._bank.values())

    @property
    def num_queries(self) -> int:
        return len(self._bank)

    def stats(self) -> dict:
        """Return summary statistics for logging."""
        if not self._bank:
            return {
                "memory_bank/total_entries": 0,
                "memory_bank/num_queries": 0,
                "memory_bank/avg_reward": 0.0,
                "memory_bank/max_reward": 0.0,
            }
        all_rewards = [e.reward for entries in self._bank.values() for e in entries]
        return {
            "memory_bank/total_entries": len(all_rewards),
            "memory_bank/num_queries": len(self._bank),
            "memory_bank/avg_reward": sum(all_rewards) / len(all_rewards),
            "memory_bank/max_reward": max(all_rewards),
        }
