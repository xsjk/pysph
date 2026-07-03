import unittest

from pysph.base.tree.tree import LEAF_DFS_TEMPLATE, POINT_DFS_TEMPLATE


class DepthFirstTraversalTemplateTests(unittest.TestCase):
    def test_backtrack_guard_checks_stack_index_before_child_stack(self):
        values = {
            "setup": "",
            "common_operation": "",
            "node_operation": "",
            "leaf_operation": "",
            "output_expr": "",
            "max_depth": 21,
            "k": 8,
        }
        for template in (LEAF_DFS_TEMPLATE, POINT_DFS_TEMPLATE):
            with self.subTest(template=template[:20]):
                operation = template % values
                self.assertIn(
                    "while (idx >= 0 && child_stack[idx] >= 8-1)",
                    operation,
                )
                self.assertNotIn(
                    "while (child_stack[idx] >= 8-1 && idx >= 0)",
                    operation,
                )


if __name__ == "__main__":
    unittest.main()
