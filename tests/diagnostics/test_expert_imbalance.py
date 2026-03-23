import unittest
import tempfile
import os
import numpy as np

from tpu_inference.observability.analyze_imbalance import parse_log_file, simulate_sharding

class TestExpertImbalanceAnalysis(unittest.TestCase):

    def test_simulate_sharding_baseline_vs_lpt(self):
        """
        테스트 시나리오: 4개의 Expert(각각 토큰 처리량 10, 20, 30, 40)를 2개의 Shard로 나눌 때,
        Baseline(순차적 할당)과 LPT(Greedy Bin-Packing)의 불균형 지표(Rho, CV)가 정확히 계산되는지 확인.
        """
        group_sizes = np.array([10, 20, 30, 40])
        num_shards = 2
        
        # [1] Baseline (Contiguous):
        # Shard 0 = 10 + 20 = 30
        # Shard 1 = 30 + 40 = 70
        # Mean = 50, Max = 70 -> Rho = 1.4
        # Std = 20 -> CV = 20 / 50 = 0.4
        
        # [2] LPT (Greedy Bin-Packing):
        # 정렬: [40, 30, 20, 10]
        # Shard 0: 40 -> 40 + 10 = 50
        # Shard 1: 30 -> 30 + 20 = 50
        # Mean = 50, Max = 50 -> Rho = 1.0
        # Std = 0 -> CV = 0.0
        
        res_baseline, res_lpt = simulate_sharding(group_sizes, num_shards)
        
        self.assertIsNotNone(res_baseline)
        self.assertIsNotNone(res_lpt)
        
        rho_b, cv_b = res_baseline
        rho_l, cv_l = res_lpt
        
        self.assertAlmostEqual(rho_b, 1.4, places=2)
        self.assertAlmostEqual(cv_b, 0.4, places=2)
        
        self.assertAlmostEqual(rho_l, 1.0, places=2)
        self.assertAlmostEqual(cv_l, 0.0, places=2)

    def test_simulate_sharding_invalid_shards(self):
        """Expert 수보다 Shard 수가 많을 경우의 예외 처리 확인"""
        group_sizes = np.array([10, 20])
        res_baseline, res_lpt = simulate_sharding(group_sizes, 4) # 2 Experts, 4 Shards
        self.assertIsNone(res_baseline)
        self.assertIsNone(res_lpt)

    def test_parse_log_file(self):
        """임시 텍스트(로그) 파일을 생성하여 정규식 및 파싱 로직이 정상 작동하는지 확인"""
        log_content = """
        MoE Token Imbalance Profile:
        Layer: 0
        [PREFILL]
        --- Current Step ---
        Group Sizes: [2, 4, 6, 8]
        
        Final Accumulated MoE Token Imbalance Profile:
        Layer: 0
        [PREFILL]
        --- Accumulated ---
        Group Sizes: [20, 40, 60, 80]
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(log_content)
            temp_path = f.name
            
        try:
            current_data, final_data, step_count = parse_log_file(temp_path)
            self.assertEqual(step_count, 1)
            np.testing.assert_array_equal(current_data[1][0]['PREFILL'], np.array([2, 4, 6, 8]))
            np.testing.assert_array_equal(final_data[0]['PREFILL'], np.array([20, 40, 60, 80]))
        finally:
            os.remove(temp_path)

    def test_simulate_sharding_ep16(self):
        """
        테스트 시나리오: 제공된 256개의 group size 배열을 16개의 Shard(ep=16)로 나누었을 때,
        Baseline과 LPT 각각에 대한 불균형 지표(Rho, CV)를 계산하고 출력.
        """
        group_sizes = np.array([
            668, 1208, 1304, 622, 141, 1433, 1312, 1449, 1028, 459, 835, 60, 837, 1031, 517, 618,
            382, 676, 124, 392, 553, 374, 214, 1018, 775, 1343, 352, 924, 353, 292, 1289, 1239,
            271, 1558, 91, 2666, 1225, 1103, 223, 879, 349, 742, 3648, 228, 1716, 1157, 336, 611,
            609, 343, 820, 2034, 951, 1504, 1059, 857, 1081, 649, 1054, 1180, 891, 693, 1421, 863,
            678, 1592, 453, 665, 3030, 1687, 1888, 85, 1332, 406, 1298, 889, 1651, 468, 2400, 272,
            2496, 407, 570, 621, 368, 829, 1164, 1996, 2275, 439, 1079, 1094, 1131, 895, 318, 388,
            1120, 1103, 1323, 1164, 1327, 2253, 960, 1594, 733, 1002, 845, 1580, 1046, 1264, 788, 1033,
            441, 251, 945, 238, 1223, 1641, 953, 1616, 957, 401, 2431, 1069, 782, 991, 767, 1983,
            769, 307, 2078, 1032, 1436, 2538, 704, 104, 920, 863, 339, 1138, 802, 272, 607, 986,
            805, 125, 1681, 1590, 1629, 1544, 2089, 385, 1467, 2043, 1102, 988, 928, 1315, 1254, 705,
            650, 565, 1054, 204, 1208, 690, 508, 1284, 382, 3370, 776, 1283, 999, 131, 430, 355,
            1217, 773, 1633, 437, 743, 706, 2198, 566, 793, 482, 1233, 144, 372, 820, 564, 1304,
            1348, 1097, 1729, 1561, 1119, 900, 1220, 1394, 517, 7019, 986, 1612, 754, 181, 636, 1035,
            860, 1356, 1594, 1139, 746, 422, 1734, 1844, 597, 1256, 970, 474, 384, 654, 1730, 521,
            720, 194, 539, 927, 269, 329, 1456, 667, 864, 1253, 1826, 154, 786, 544, 853, 2093,
            590, 1939, 547, 782, 997, 3132, 2021, 477, 1571, 1312, 1739, 1034, 331, 1778, 930, 296
        ])
        num_shards = 16
        
        res_baseline, res_lpt = simulate_sharding(group_sizes, num_shards)
        self.assertIsNotNone(res_baseline)
        self.assertIsNotNone(res_lpt)
        
        rho_b, cv_b = res_baseline
        rho_l, cv_l = res_lpt
        
        print(f"\n[EP {num_shards} Sharding Results]")
        print(f"  - Baseline -> Max/Mean: {rho_b:.4f}, CV: {cv_b:.4f}")
        print(f"  - LPT      -> Max/Mean: {rho_l:.4f}, CV: {cv_l:.4f}")

        self.assertTrue(rho_b >= 1.0)
        self.assertTrue(rho_l >= 1.0)
        # LPT should generally be more balanced or equal to Baseline
        self.assertTrue(rho_l <= rho_b)

if __name__ == '__main__':
    unittest.main()
