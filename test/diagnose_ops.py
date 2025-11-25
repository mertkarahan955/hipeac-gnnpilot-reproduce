import torch
import traceback
import sys

so = "../KG_GNN/build/libKGGNN.so"
print("Attempting to load:", so)
try:
    torch.ops.load_library(so)
    print("Loaded", so)
except Exception as e:
    print("Failed to load:", e)
    traceback.print_exc()

# List torch.ops namespaces (top-level attributes)
namespaces = [n for n in dir(torch.ops) if not n.startswith("__")]
print("\nTop-level torch.ops namespaces (count={}):".format(len(namespaces)))
print(namespaces)

# For each namespace, show up to 50 members
for ns in namespaces:
    try:
        attr = getattr(torch.ops, ns)
        members = [m for m in dir(attr) if not m.startswith("__")]
        if members:
            print(f"\nNamespace: {ns} (members count={len(members)})")
            print(members[:50])
    except Exception:
        pass

# Specifically check KGGNN
print("\nChecking torch.ops.KGGNN availability:")
if hasattr(torch.ops, 'KGGNN'):
    k = torch.ops.KGGNN
    kmembers = [m for m in dir(k) if not m.startswith('__')]
    print('KGGNN members ({}):'.format(len(kmembers)))
    print(kmembers)
else:
    print('torch.ops.KGGNN not present')

print('\nDone')
