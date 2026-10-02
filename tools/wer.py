import re, sys

def words(t): return re.sub(r"[^\w\s']", "", t.lower()).split()

def edits(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]

ref = [words(l.rstrip("\n").split("\t")[-1]) for l in open(sys.argv[1])]
hyp = [words(l.rstrip("\n").split("\t")[-1]) for l in open(sys.argv[2])]
e = [edits(r, h) for r, h in zip(ref, hyp)]
n = sum(map(len, ref))
print(f"clips {len(ref)} words {n} edits {sum(e)} rate {100 * sum(e) / n:.2f}% identical_clips {sum(x == 0 for x in e)} worst_clip {max(x / max(len(r), 1) for x, r in zip(e, ref)):.3f}")
