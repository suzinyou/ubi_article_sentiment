import re
import numpy as np

space = re.compile('▁')
close = re.compile(r'(?<=[\uAC00-\uD7AF\w])\u2581(?=[\'""·”’,\.\?!\)\]\}@])', re.UNICODE)
_open = re.compile(r'(?<=[·\'"‘“\(\[@])\u2581(?=[\uAC00-\uD7AF\w])', re.UNICODE)
close_cont = re.compile(r'(?<=[\uAC00-\uD7AF\w][”’\"\'@])\u2581(?=[\uAC00-\uD7AF\w])', re.UNICODE)
dots = re.compile(r'(?<=[.·])\u2581(?=[.·])', re.UNICODE)


def recover_sentence_from_tokens(tokens):
    joined = ''.join(tokens)
    r = _open.sub('', joined)
    r = close.sub('', r)
    r = close_cont.sub('', r)
    r = dots.sub('', r)
    r = space.sub(' ', r)
    return r


def generate_overlapping_segments(array, length, overlap):
    array = np.array(array)
    i = 0
    while i < array.shape[-1] - overlap:
        if array.ndim > 1:
            segment = array[:, i:i + length]
        else:
            segment = array[i:i + length]
        i += length - overlap

        yield segment


def f(x, length, overlap):
    non_overlap = length - overlap
    n = x // non_overlap
    r = x % non_overlap
    if r == overlap:
        return n, 0
    elif r == 0:
        return n - 1, non_overlap
    else:
        return n, (x - non_overlap * n) % length


def generate_valid_lengths(valid_lengths, length, overlap):
    if valid_lengths.ndim > 0:
        res = []
        for vl in valid_lengths:
            n_segments, remainder = f(vl, length, overlap)
            full_segment_lengths = np.repeat(length, n_segments)
            if remainder == 0:
                arr = full_segment_lengths.astype(np.int32)
            else:
                arr = np.r_[full_segment_lengths, remainder].astype(np.int32)
            res.append(arr)
    else:
        n_segments, remainder = f(valid_lengths, length, overlap)
        full_segment_lengths = np.repeat(length, n_segments)
        if remainder == 0:
            res = full_segment_lengths.astype(np.int32)
        else:
            res = np.r_[full_segment_lengths, remainder].astype(np.int32)

    return res


