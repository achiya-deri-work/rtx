import hashlib
import numpy as np
import pytest

from benchmarks.prepare_precision20m_corpus import prepare


def test_prefix_preserves_training_hash_dtype_and_existing_files(tmp_path):
    source=tmp_path/'source.npy'; output=tmp_path/'prefix.npy'
    tokens=np.arange(257,dtype=np.uint16)
    np.save(source,tokens)
    result=prepare(source,output,100)
    actual=np.load(output)
    assert actual.dtype==tokens.dtype
    np.testing.assert_array_equal(actual,tokens[:100])
    assert result['token_sha256']==hashlib.sha256(tokens[:100].tobytes()).hexdigest()
    with pytest.raises(FileExistsError):prepare(source,output,100)


@pytest.mark.parametrize('values,count',[(np.arange(3),4),(np.array([-1]),1),
    (np.array([257]),1),(np.ones((2,2),dtype=int),1),(np.array([1.]),1)])
def test_bad_corpus_is_not_written(tmp_path,values,count):
    source=tmp_path/'source.npy';output=tmp_path/'prefix.npy'
    np.save(source,values)
    with pytest.raises(ValueError):prepare(source,output,count)
    assert not output.exists()
