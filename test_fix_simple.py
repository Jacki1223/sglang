#!/usr/bin/env python3
"""
Simple test to verify initialization order fix.
"""

# Test the initialization order issue is fixed
print("Testing initialization order fix...")
print()

class Parent:
    def __init__(self):
        print("  Parent.__init__() called")
        self.reset()  # This will call child's reset() if overridden

    def reset(self):
        print("  Parent.reset() called")

class BrokenChild(Parent):
    """This will fail - tries to use _data before it's initialized."""
    def __init__(self):
        print("  BrokenChild.__init__() starting...")
        super().__init__()  # Calls Parent.__init__() which calls reset()
        self._data = []  # Initialized AFTER super().__init__()
        print("  BrokenChild.__init__() completed")

    def reset(self):
        print("  BrokenChild.reset() called")
        super().reset()
        self._data.clear()  # ERROR: _data doesn't exist yet!

class FixedChild(Parent):
    """This works - initializes _data BEFORE calling super().__init__()."""
    def __init__(self):
        print("  FixedChild.__init__() starting...")
        self._data = []  # Initialized BEFORE super().__init__()
        super().__init__()  # Now reset() can safely use _data
        print("  FixedChild.__init__() completed")

    def reset(self):
        print("  FixedChild.reset() called")
        super().reset()
        self._data.clear()  # OK: _data exists

# Test broken version
print("1. Testing BrokenChild (should fail):")
print("-" * 50)
try:
    broken = BrokenChild()
    print("  ❌ Should have failed!")
except AttributeError as e:
    print(f"  ✓ Expected error: {e}")

print()

# Test fixed version
print("2. Testing FixedChild (should work):")
print("-" * 50)
try:
    fixed = FixedChild()
    print("  ✓ Success! No error.")
except Exception as e:
    print(f"  ❌ Unexpected error: {e}")

print()
print("="*60)
print("This demonstrates the fix applied to PersistentHeapRadixCache:")
print("  - Initialize _eviction_heap BEFORE calling super().__init__()")
print("  - This way, when parent's __init__() calls reset(),")
print("    the child's _eviction_heap already exists")
print("="*60)
