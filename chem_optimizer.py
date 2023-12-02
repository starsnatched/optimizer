import gym
from gym import spaces
from rdkit import Chem
from rdkit.Chem import AllChem
from stable_baselines3 import PPO
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import QED
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Draw

class MoleculeEnv(gym.Env):
    """
    A molecule design environment for Reinforcement Learning using OpenAI Gym.
    """
    def __init__(self, elements, max_atoms=10):
        super(MoleculeEnv, self).__init__()
        self.max_atoms = max_atoms

        self.elements = elements

        self.action_space = spaces.Discrete(len(elements))

        self.observation_space = spaces.MultiDiscrete([len(elements)] * 10)
        
        self.init_molecule()
        
    def init_molecule(self):
        self.molecule = Chem.RWMol()

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(Chem.MolToSmiles(self.molecule))
        elif mode == 'img':
            mol_img = Draw.MolToImage(self.molecule)
            return mol_img

    def step(self, action):
        element = self.elements[action]
        
        atom = Chem.Atom(element)
        index = self.molecule.AddAtom(atom)

        state = self.observation_space.sample()  
        reward = self.evaluate_molecule(self.molecule)
        
        done = self.molecule.GetNumAtoms() >= self.max_atoms
        
        if not done:
            if self.molecule.GetNumAtoms() > 1:
                try:
                    Chem.SanitizeMol(self.molecule)
                except:
                    reward = -2
                    done = True

        state = self._get_state()
        
        return state, reward, done, {}
    
    def _get_state(self):
        state = [0] * self.max_atoms

        atom_counts = {element: 0 for element in self.elements}
        for atom in self.molecule.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in atom_counts:
                atom_counts[symbol] += 1

        idx = 0
        for element in self.elements:
            for _ in range(atom_counts[element]):
                state[idx] = self.elements.index(element)
                idx += 1

                if idx >= self.max_atoms:
                    break
            if idx >= self.max_atoms:
                break
            
        return state
        
    def evaluate_molecule(self, molecule):
        mol = Chem.Mol(molecule)
    
        if mol is None or mol.GetNumAtoms() == 0:
            return -1
    
        try:
            Chem.SanitizeMol(mol)
    
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
            hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            qed_score = QED.qed(mol)

            violation_count = sum([
                mw > 500,
                logp > 5,
                hbd > 5,
                hba > 10
            ])

            rule_of_five_penalty = violation_count * -1

            reward = qed_score + rule_of_five_penalty

            return reward
        except:
            return -2

    def reset(self):
        self.init_molecule()
        return self.observation_space.sample()

    def render(self, mode='human', close=False):
        return Chem.MolToSmiles(self.molecule)


ELEMENTS = ['C', 'O', 'H', 'N']

env = MoleculeEnv(ELEMENTS, max_atoms=10)

model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=50000)

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print(f'Step: {action}, Reward: {rewards}, New Observation: {obs}')

smiles = env.render()
print(f"Designed Molecule SMILES: {smiles}")

mol = Chem.MolFromSmiles(smiles)

img = Draw.MolToImage(mol)
img.save("molecule.png")