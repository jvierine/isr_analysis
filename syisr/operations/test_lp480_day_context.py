import unittest
from lp480_day_context import boundary
class BoundaryTests(unittest.TestCase):
    def test_real_boundary_and_reset(self):
        self.assertEqual(boundary(86400,[82800,84600]),84600)
        self.assertEqual(boundary(86400,[82800]),82800)
        self.assertIsNone(boundary(86400,[81000]))
        self.assertIsNone(boundary(172800,[84600]))
        self.assertIsNone(boundary(86400,[]))
        self.assertIsNone(boundary(None,[84600]))
        self.assertIsNone(boundary(86400,[86400]))
if __name__=='__main__':unittest.main()
