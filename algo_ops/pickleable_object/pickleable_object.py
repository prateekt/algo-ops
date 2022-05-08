import os
import pickle


class PickleableObject:
    def to_pickle(self, out_pkl_path: str) -> None:
        """
        Pickles object.

        param out_pkl_path: Path to output pickle file.
        """
        with open(out_pkl_path, "wb") as fout:
            pickle.dump(self, fout)
        assert os.path.exists(out_pkl_path)

    @classmethod
    def load_from_pickle(cls, pkl_path: str) -> "PickleableObject":
        """
        Loads object from pickled file.

        param pkl_path: Path to pkl file containing object

        Return:
            loaded_reference: The loaded object
        """
        with open(pkl_path, "rb") as fin:
            loaded_obj = pickle.load(fin)
        assert isinstance(loaded_obj, cls)
        return loaded_obj
