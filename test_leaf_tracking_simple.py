#!/usr/bin/env python3
"""
Simple unit test for leaf node tracking logic.
"""

# Test the leaf tracking logic in isolation

class SimpleTreeNode:
    def __init__(self):
        self.children = {}
        self.parent = None
        self.lock_ref = 0
        self.id = id(self)

    def __repr__(self):
        return f"Node({self.id}, children={len(self.children)}, lock={self.lock_ref})"

class SimpleRadixCache:
    def __init__(self, fast_eviction=True):
        self.fast_eviction = fast_eviction
        self.root = SimpleTreeNode()
        self.root.lock_ref = 1  # Root is always locked
        self.leaf_nodes = set() if fast_eviction else None

    def _add_leaf_node(self, node):
        """Add a node to the leaf set if it's an evictable leaf."""
        if self.fast_eviction:
            if len(node.children) == 0 and node.lock_ref == 0:
                self.leaf_nodes.add(node)

    def _remove_leaf_node(self, node):
        """Remove a node from the leaf set."""
        if self.fast_eviction:
            self.leaf_nodes.discard(node)

    def _collect_leaves_slow(self):
        """Original O(n) implementation."""
        ret_list = []
        stack = list(self.root.children.values())

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                if cur_node.lock_ref == 0:
                    ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _collect_leaves_fast(self):
        """Fast O(1) implementation."""
        if self.fast_eviction:
            return list(self.leaf_nodes)
        else:
            return self._collect_leaves_slow()

    def insert_node(self, parent, child_key):
        """Insert a new leaf node under parent."""
        new_node = SimpleTreeNode()
        new_node.parent = parent
        parent.children[child_key] = new_node

        # Update leaf tracking
        self._remove_leaf_node(parent)  # parent is no longer a leaf
        self._add_leaf_node(new_node)  # new_node is a new leaf

        return new_node

    def delete_leaf(self, node):
        """Delete a leaf node."""
        assert len(node.children) == 0, "Can only delete leaf nodes"

        self._remove_leaf_node(node)

        parent = node.parent
        # Find and remove from parent
        for k, v in list(parent.children.items()):
            if v == node:
                del parent.children[k]
                break

        # If parent now has no children and is unlocked, it becomes a leaf
        if len(parent.children) == 0 and parent.lock_ref == 0:
            self._add_leaf_node(parent)

    def lock_node(self, node):
        """Lock a node (increment lock_ref)."""
        if node.lock_ref == 0:
            self._remove_leaf_node(node)
        node.lock_ref += 1

    def unlock_node(self, node):
        """Unlock a node (decrement lock_ref)."""
        assert node.lock_ref > 0
        node.lock_ref -= 1

        # If node just became unlocked and is a leaf, add to leaf set
        if node.lock_ref == 0 and len(node.children) == 0:
            self._add_leaf_node(node)

def test_basic_operations():
    """Test basic leaf tracking operations."""
    print("=" * 80)
    print("Test: Basic Operations")
    print("=" * 80)

    cache = SimpleRadixCache(fast_eviction=True)

    # Test 1: Insert a node
    print("\nTest 1: Insert a leaf node")
    node1 = cache.insert_node(cache.root, "A")
    fast_leaves = set(cache._collect_leaves_fast())
    slow_leaves = set(cache._collect_leaves_slow())

    print(f"  Fast leaves: {len(fast_leaves)}")
    print(f"  Slow leaves: {len(slow_leaves)}")
    assert fast_leaves == slow_leaves, "Leaf sets don't match after insert!"
    assert node1 in fast_leaves, "New node not in leaf set!"
    print("  ✓ Passed")

    # Test 2: Insert multiple nodes
    print("\nTest 2: Insert multiple leaf nodes")
    node2 = cache.insert_node(cache.root, "B")
    node3 = cache.insert_node(cache.root, "C")

    fast_leaves = set(cache._collect_leaves_fast())
    slow_leaves = set(cache._collect_leaves_slow())

    print(f"  Fast leaves: {len(fast_leaves)}")
    print(f"  Slow leaves: {len(slow_leaves)}")
    assert fast_leaves == slow_leaves, "Leaf sets don't match!"
    assert len(fast_leaves) == 3, f"Expected 3 leaves, got {len(fast_leaves)}"
    print("  ✓ Passed")

    # Test 3: Insert child under leaf (leaf becomes internal node)
    print("\nTest 3: Insert child under leaf node")
    node1_child = cache.insert_node(node1, "A1")

    fast_leaves = set(cache._collect_leaves_fast())
    slow_leaves = set(cache._collect_leaves_slow())

    print(f"  Fast leaves: {len(fast_leaves)}")
    print(f"  Slow leaves: {len(slow_leaves)}")
    assert fast_leaves == slow_leaves, "Leaf sets don't match!"
    assert node1 not in fast_leaves, "node1 should not be a leaf anymore!"
    assert node1_child in fast_leaves, "node1_child should be a leaf!"
    print("  ✓ Passed")

    # Test 4: Delete a leaf
    print("\nTest 4: Delete a leaf node")
    cache.delete_leaf(node2)

    fast_leaves = set(cache._collect_leaves_fast())
    slow_leaves = set(cache._collect_leaves_slow())

    print(f"  Fast leaves: {len(fast_leaves)}")
    print(f"  Slow leaves: {len(slow_leaves)}")
    assert fast_leaves == slow_leaves, "Leaf sets don't match!"
    assert node2 not in fast_leaves, "Deleted node still in leaf set!"
    print("  ✓ Passed")

    # Test 5: Lock a leaf node
    print("\nTest 5: Lock a leaf node")
    cache.lock_node(node3)

    fast_leaves = set(cache._collect_leaves_fast())
    slow_leaves = set(cache._collect_leaves_slow())

    print(f"  Fast leaves: {len(fast_leaves)}")
    print(f"  Slow leaves: {len(slow_leaves)}")
    assert fast_leaves == slow_leaves, "Leaf sets don't match!"
    assert node3 not in fast_leaves, "Locked node should not be in leaf set!"
    print("  ✓ Passed")

    # Test 6: Unlock a leaf node
    print("\nTest 6: Unlock a leaf node")
    cache.unlock_node(node3)

    fast_leaves = set(cache._collect_leaves_fast())
    slow_leaves = set(cache._collect_leaves_slow())

    print(f"  Fast leaves: {len(fast_leaves)}")
    print(f"  Slow leaves: {len(slow_leaves)}")
    assert fast_leaves == slow_leaves, "Leaf sets don't match!"
    assert node3 in fast_leaves, "Unlocked leaf should be in leaf set!"
    print("  ✓ Passed")

    # Test 7: Delete last child (parent becomes leaf)
    print("\nTest 7: Delete last child (parent becomes leaf)")
    cache.delete_leaf(node1_child)

    fast_leaves = set(cache._collect_leaves_fast())
    slow_leaves = set(cache._collect_leaves_slow())

    print(f"  Fast leaves: {len(fast_leaves)}")
    print(f"  Slow leaves: {len(slow_leaves)}")
    assert fast_leaves == slow_leaves, "Leaf sets don't match!"
    assert node1 in fast_leaves, "node1 should be a leaf again!"
    print("  ✓ Passed")

    print("\n✓ All basic operation tests passed!")

def test_complex_scenario():
    """Test a more complex scenario."""
    print("\n" + "=" * 80)
    print("Test: Complex Scenario")
    print("=" * 80)

    cache = SimpleRadixCache(fast_eviction=True)

    # Build a tree
    #      root
    #     /  |  \
    #    A   B   C
    #   / \      |
    #  A1 A2     C1

    nodeA = cache.insert_node(cache.root, "A")
    nodeB = cache.insert_node(cache.root, "B")
    nodeC = cache.insert_node(cache.root, "C")

    nodeA1 = cache.insert_node(nodeA, "A1")
    nodeA2 = cache.insert_node(nodeA, "A2")
    nodeC1 = cache.insert_node(nodeC, "C1")

    print("\nInitial tree structure:")
    print("  Leaves should be: A1, A2, B, C1")

    fast_leaves = set(cache._collect_leaves_fast())
    slow_leaves = set(cache._collect_leaves_slow())

    print(f"  Fast leaves: {len(fast_leaves)}")
    print(f"  Slow leaves: {len(slow_leaves)}")

    expected_leaves = {nodeA1, nodeA2, nodeB, nodeC1}
    assert fast_leaves == slow_leaves == expected_leaves, "Leaf sets don't match expected!"
    print("  ✓ Correct")

    # Lock some nodes
    print("\nLocking A1 and B...")
    cache.lock_node(nodeA1)
    cache.lock_node(nodeB)

    fast_leaves = set(cache._collect_leaves_fast())
    slow_leaves = set(cache._collect_leaves_slow())

    expected_leaves = {nodeA2, nodeC1}
    assert fast_leaves == slow_leaves == expected_leaves, "Leaf sets don't match after locking!"
    print(f"  Leaves now: {len(fast_leaves)} (should be A2, C1)")
    print("  ✓ Correct")

    # Delete A2
    print("\nDeleting A2...")
    cache.delete_leaf(nodeA2)

    fast_leaves = set(cache._collect_leaves_fast())
    slow_leaves = set(cache._collect_leaves_slow())

    print(f"  Fast leaves: {len(fast_leaves)}")
    print(f"  Slow leaves: {len(slow_leaves)}")
    assert fast_leaves == slow_leaves, "Leaf sets don't match!"
    print("  ✓ Correct")

    # Unlock A1 - should become a leaf again
    print("\nUnlocking A1...")
    cache.unlock_node(nodeA1)

    fast_leaves = set(cache._collect_leaves_fast())
    slow_leaves = set(cache._collect_leaves_slow())

    assert nodeA1 in fast_leaves, "A1 should be a leaf again!"
    print(f"  Leaves now: {len(fast_leaves)}")
    print("  ✓ Correct")

    print("\n✓ Complex scenario test passed!")

if __name__ == "__main__":
    print("Leaf Node Tracking Unit Tests")
    print("=" * 80)

    test_basic_operations()
    test_complex_scenario()

    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)
