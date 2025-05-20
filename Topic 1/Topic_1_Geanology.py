import pandas as pd
from collections import defaultdict, deque, Counter

# Reload with header=None and set correct column names manually
genealogy_df = pd.read_csv('Dataset4-genealogy.csv', delimiter=';', header=None, names=['ParentSerialNumber', 'ChildSerialNumber'])

parent_to_children = defaultdict(list)
for _, row in genealogy_df.iterrows():
    parent_to_children[row['ParentSerialNumber']].append(row['ChildSerialNumber'])

all_parents = set(genealogy_df['ParentSerialNumber'])
all_children = set(genealogy_df['ChildSerialNumber'])
roots = list(all_parents - all_children)

depth_counter = Counter()
queue = deque((node, 1) for node in roots)
visited = set()
while queue:
    current, depth = queue.popleft()
    if current in visited:
        continue
    visited.add(current)
    depth_counter[depth] += 1
    for child in parent_to_children.get(current, []):
        queue.append((child, depth + 1))

print(dict(depth_counter))


parent_to_children = defaultdict(list)
for _, row in genealogy_df.iterrows():
    parent_to_children[row['ParentSerialNumber']].append(row['ChildSerialNumber'])

all_parents = set(genealogy_df['ParentSerialNumber'])
all_children = set(genealogy_df['ChildSerialNumber'])
roots = list(all_parents - all_children)

# Capture full path from root to nodes at depth 4
paths = []
queue = deque((node, [node]) for node in roots)

while queue:
    current, path = queue.popleft()
    if len(path) == 4:
        paths.append(path)
    for child in parent_to_children.get(current, []):
        queue.append((child, path + [child]))

# Print the full path(s) to depth 4 nodes
for p in paths:
    print(p, "\n")

# Count number of children per parent
children_count = genealogy_df.groupby('ParentSerialNumber')['ChildSerialNumber'].count()

# Count how many parents each child has
parents_count = genealogy_df.groupby('ChildSerialNumber')['ParentSerialNumber'].count()

# Count frequencies of how many parents/children entities have
children_distribution = Counter(children_count)
parents_distribution = Counter(parents_count)

# Print results
print("Distribution of number of children per parent:")
print(dict(children_distribution))

print("\nDistribution of number of parents per child:")
print(dict(parents_distribution))
