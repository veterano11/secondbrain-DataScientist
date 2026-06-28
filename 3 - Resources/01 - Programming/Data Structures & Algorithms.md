---
tags: [programming, algorithms, foundational]
status: growing
created: 2026-06-27
---

# Data Structures & Algorithms

## 1. Why This Matters

Data structures are about **organizing data for efficient access and modification**. The difference between O(1) and O(n) lookup might not matter on 100 rows, but on 100 million rows it is the difference between milliseconds and hours.

In data science, understanding data structures helps you:
- Choose the right container for your data (list, set, dict, array)
- Know why pandas is fast (it uses NumPy arrays under the hood)
- Design efficient feature engineering loops
- Understand database indexing (hash indexes, B-trees)
- Pass coding interviews — but more importantly, write efficient production code

---

## 2. Big O Notation — The Language of Performance

Big O describes how runtime or memory grows **as input size increases**.

### 2.1 The Intuition

Do not count exact operations — count **how the algorithm scales**.

| Notation | Name | If input doubles... |
|---|---|---|
| O(1) | Constant | Time stays the same |
| O(log n) | Logarithmic | Time increases by 1 step |
| O(n) | Linear | Time doubles |
| O(n log n) | Linearithmic | Time slightly more than doubles |
| O(n²) | Quadratic | Time quadruples |
| O(2ⁿ) | Exponential | Time becomes enormous |

### 2.2 Concrete Examples

```python
# O(1) — constant time
def get_first(items):
    return items[0]          # direct memory access

# O(n) — linear time
def contains(items, target):
    for item in items:       # might check all items
        if item == target:
            return True
    return False

# O(n²) — quadratic time
def pairwise_distance(points):
    result = []
    for p1 in points:        # n times
        for p2 in points:    # n times = n² total
            result.append(dist(p1, p2))
    return result

# O(log n) — logarithmic time
def binary_search(sorted_list, target):
    low, high = 0, len(sorted_list) - 1
    while low <= high:       # halves the search space each iteration
        mid = (low + high) // 2
        if sorted_list[mid] == target:
            return mid
        elif sorted_list[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
```

### 2.3 Why Big O Matters in Data Science

| Operation | Data Structure | Big O | Example |
|---|---|---|---|
| Lookup by key | dict | O(1) | Feature lookup, caching |
| Lookup by index | list | O(1) | Accessing training samples |
| Membership test | set | O(1) | Checking if ID exists |
| Membership test | list | O(n) | Slow for large lists |
| Sort | list | O(n log n) | Sorting features by importance |
| Find max/min | list | O(n) | Finding best/worst case |

The lesson: if you need to check membership frequently, use a **set** (O(1)) not a **list** (O(n)).

---

## 3. Essential Data Structures

### 3.1 Array / List

- **Memory**: contiguous block of memory
- **Access**: O(1) by index (direct memory address calculation)
- **Insert/delete at end**: O(1) amortized
- **Insert/delete at beginning**: O(n) (shift all subsequent elements)

**In Python**: `list` is a dynamic array. When it runs out of space, it allocates ~1.125× more memory and copies everything — hence "amortized" O(1) for append.

```python
arr = [1, 2, 3]
arr.append(4)            # O(1) amortized
arr.insert(0, 0)         # O(n) — shift everything
```

**NumPy arrays**: fixed type, contiguous memory, cache-friendly. Much faster for numerical operations because operations are vectorized (run in C).

```python
arr = np.array([1, 2, 3])
arr * 2                  # vectorized — runs in C, not Python
```

### 3.2 Hash Table (dict / set)

**How it works**: a hash function maps keys to array indices. Collisions are handled by chaining (multiple keys at the same index stored in a linked list).

- **Insert**: O(1) average, O(n) worst-case (many collisions)
- **Lookup**: O(1) average, O(n) worst-case
- **Delete**: O(1) average

**The magic**: good hash functions distribute keys evenly, making worst-case practically impossible.

```python
lookup = {"alice": 25, "bob": 30, "charlie": 35}
lookup["alice"]           # O(1) — hash "alice", jump to that slot

# set is a dict with only keys (no values)
ids = {101, 102, 103}
102 in ids                # O(1)
```

### 3.3 Stack (LIFO)

Last-In-First-Out. In Python, a `list` works perfectly as a stack.

```python
stack = []
stack.append(1)           # push
stack.append(2)
stack.pop()               # 2 — last in, first out
```

**Use cases**: depth-first search, undo operations, parentheses matching.

### 3.4 Queue (FIFO)

First-In-First-Out. Use `collections.deque` for O(1) append/pop from both ends.

```python
from collections import deque

queue = deque()
queue.append(1)            # enqueue (right side)
queue.append(2)
queue.popleft()            # 1 — first in, first out
```

**Use cases**: breadth-first search, task scheduling, streaming data.

### 3.5 Tree

A hierarchical structure with a root and children.

**Binary Search Tree (BST)**:
- Left child < parent < right child
- Search: O(log n) average, O(n) worst-case (unbalanced)
- Balanced variants (AVL, Red-Black) guarantee O(log n)

In data science, trees appear as **decision trees** (and their ensembles: Random Forest, XGBoost). [[Object-Oriented Programming]] covers implementing tree structures with classes.

### 3.6 Heap (Priority Queue)

Always gives you the smallest (min-heap) or largest (max-heap) element.

```python
import heapq

data = [5, 3, 7, 1, 9]
heapq.heapify(data)        # min-heap: [1, 3, 7, 5, 9]
heapq.heappop(data)        # 1 (smallest)
heapq.heappush(data, 2)    # O(log n) insert
```

**Use cases**: finding top-k elements, Dijkstra's algorithm, priority scheduling.

---

## 4. Key Algorithms for Data Science

### 4.1 Sorting

Python uses **Timsort** (O(n log n) worst-case), a hybrid of merge sort and insertion sort optimized for real-world data (which often has partial order).

```python
sorted_list = sorted(unsorted_list)       # returns new list
unsorted_list.sort()                      # in-place sort
```

### 4.2 Searching

- **Linear search**: O(n) — unsorted data
- **Binary search**: O(log n) — requires sorted data

```python
import bisect

sorted_data = [1, 3, 5, 7, 9]
pos = bisect.bisect_left(sorted_data, 6)  # 3 (between 5 and 7)
```

### 4.3 Dynamic Programming

"Solve a problem by breaking it into overlapping subproblems and solving each once."

**Fibonacci — naive (exponential) vs DP (linear)**:

```python
# Without DP: O(2ⁿ) — recomputes same values exponentially
def fib_naive(n):
    return n if n <= 1 else fib_naive(n-1) + fib_naive(n-2)

# With memoization: O(n) — each value computed once
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo(n):
    return n if n <= 1 else fib_memo(n-1) + fib_memo(n-2)
```

**In data science**: edit distance (NLP), Viterbi algorithm (HMM), dynamic time warping (time series), sequence alignment (bioinformatics). [[Functional Programming]]'s pure functions align naturally with DP's stateless subproblems.

---

## 5. Common Mistakes

1. **Using `list` where `set` is appropriate**: if you only need membership tests, `set` is O(1) vs `list` O(n). On 1M items, the difference is microseconds vs seconds.

2. **Not understanding amortized cost**: `list.append()` is O(1) on average. But occasionally it is O(n) when resizing. This matters for latency-sensitive applications.

3. **Assuming O(n) is always bad**: for small n (n < 1000), O(n) may be faster than O(log n) with high constant factors. Profile before optimizing. [[Code Quality]] covers profiling tools for measuring real-world performance.

4. **Recursion depth**: Python has a recursion limit (~1000). Use iteration for deep recursion. Dynamic programming usually avoids recursion anyway.

---

## 6. Check Your Understanding

1. You need to check if a specific user ID exists in a collection of 10 million IDs. Do you use a list or a set? Why?
2. What is the Big O of `dict.get(key)`? How does a hash table achieve this?
3. A decision tree trained on 1000 samples takes 0.1s. Approximately how long for 1,000,000 samples? (Hint: think about tree complexity)
4. When would O(n) be faster than O(1) in practice?
5. Why does `list.append()` document itself as "amortized O(1)"? What does "amortized" mean?

---

## 7. Summary

Data structures are about tradeoffs: arrays give fast access but slow insertion; hash tables give fast lookup but use more memory; trees give ordered access but have overhead. Big O notation quantifies these tradeoffs. The most practical takeaway: use `dict`/`set` for fast lookups, `list` for ordered sequences, `deque` for queues, and `heapq` for priority. Understanding these choices separates efficient data code from slow code.

---

## 8. Where to Go Next

- [[Python Fundamentals]] — Lists, dicts, sets in practice
- [[Python for Data Science]] — NumPy arrays are optimized for numerical data
- [[Supervised Learning]] — Decision trees are binary trees
- [[Object-Oriented Programming]] — Implementing data structures as classes
- [[Functional Programming]] — Pure functions for algorithm design
