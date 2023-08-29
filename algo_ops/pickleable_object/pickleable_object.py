import bz2
import gzip
import lzma
import blosc
import os
import pickle
from typing import Optional


class PickleableObject:
    def to_pickle(self, out_pkl_path: str, compression: Optional[str] = None) -> None:
        """
        Pickles object.

        param out_pkl_path: Path to output pickle file.
        param compression: Compression to use. Can be None, "gzip", "bz2", "lzma", or "blosc".
        """
        if compression is None:
            with open(out_pkl_path, "wb") as out:
                pickle.dump(self, out)
        elif compression == "gzip":
            with gzip.open(out_pkl_path, "wb") as out:
                pickle.dump(self, out)
        elif compression == "bz2":
            with bz2.BZ2File(out_pkl_path, "wb") as out:
                pickle.dump(self, out)
        elif compression == "lzma":
            with lzma.open(out_pkl_path, "wb") as out:
                pickle.dump(self, out)
        elif compression == "blosc":
            with open(out_pkl_path, "wb") as out:
                out.write(blosc.compress(pickle.dumps(self)))
        else:
            raise ValueError("Invalid compression: {}".format(compression))
        assert os.path.exists(out_pkl_path)

    @classmethod
    def load_from_pickle(
        cls, pkl_path: str, compression: Optional[str] = None
    ) -> "PickleableObject":
        """
        Loads object from pickled file.

        param pkl_path: Path to pkl file containing object
        param compression: Compression to use. Can be None, "gzip", "bz2", or "lzma".

        Return:
            loaded_reference: The loaded object
        """
        if compression is None:
            with open(pkl_path, "rb") as fin:
                loaded_obj = pickle.load(fin)
        elif compression == "gzip":
            with gzip.open(pkl_path, "rb") as fin:
                loaded_obj = pickle.load(fin)
        elif compression == "bz2":
            with bz2.BZ2File(pkl_path, "rb") as fin:
                loaded_obj = pickle.load(fin)
        elif compression == "lzma":
            with lzma.open(pkl_path, "rb") as fin:
                loaded_obj = pickle.load(fin)
        elif compression == "blosc":
            with open(pkl_path, "rb") as fin:
                loaded_obj = pickle.loads(blosc.decompress(fin.read()))
        else:
            raise ValueError("Invalid compression: {}".format(compression))
        assert isinstance(loaded_obj, cls)
        return loaded_obj
