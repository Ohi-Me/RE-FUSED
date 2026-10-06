"""Print SHA-256 and size of every file in a folder (sorted), to show that a rerun reproduces files byte for byte."""
import hashlib, os, sys
root = sys.argv[1]
for name in sorted(os.listdir(root)):
    p = os.path.join(root, name)
    if os.path.isfile(p):
        print(hashlib.sha256(open(p, "rb").read()).hexdigest(), os.path.getsize(p), name)
