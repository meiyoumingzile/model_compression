robnet_large_v1 = robnet_large_v2 = [[3, 3,
                                      3, 3,
                                      3, 2,
                                      3, 2,
                                      1, 3,
                                      2, 2,
                                      2, 3]]

arch_code = [[3, 1,#超网络结构，00是0卷积，01是可分离卷积，10是1卷积，11是残差卷积
                3, 3,
                1, 3,# 1, 3,
                2, 2,
                1, 3,
                3, 0,
                3, 2],
               [3, 0,
                2, 1,
                2, 0,
                1, 1,
                0, 2,
                3, 1,
                3, 0],
               [3, 3,
                3, 2,
                2, 3,
                2, 0,
                1, 0,
                1, 1,
                0, 3],
               [1, 3,
                1, 2,
                2, 1,
                1, 2,
                1, 3,
                0, 0,
                2, 1],
               [3, 3,
                2, 0,
                1, 1,
                1, 2,
                1, 2,
                0, 1,
                2, 3],
               [3, 1,
                3, 3,
                3, 2,
                1, 3,
                0, 1,
                2, 3,
                2, 3],
               [3, 3,
                3, 1,
                1, 2,
                3, 1,
                2, 1,
                1, 1,
                2, 1],
               [2, 3,
                3, 3,
                2, 3,
                3, 3,
                2, 3,
                2, 3,
                1, 0],
               [3, 3,
                3, 3,
                3, 2,
                2, 3,
                1, 2,
                0, 1,
                2, 3],
               [1, 3,
                2, 0,
                3, 3,
                1, 3,
                2, 3,
                3, 3,
                3, 0],
               [3, 1,
                3, 0,
                0, 0,
                3, 0,
                2, 1,
                0, 2,
                1, 3],
               [2, 3,
                2, 3,
                3, 1,
                1, 3,
                1, 3,
                2, 3,
                1, 0],
               [3, 3,
                1, 1,
                2, 0,
                1, 3,
                3, 2,
                1, 1,
                0, 2],
               [2, 3,
                3, 3,
                2, 2,
                3, 2,
                2, 3,
                3, 2,
                1, 1],
               [2, 3,
                3, 3,
                3, 2,
                0, 2,
                1, 1,
                3, 1,
                1, 3],
               [3, 3,
                0, 3,
                3, 2,
                0, 2,
                1, 0,
                3, 2,
                3, 3],
               [3, 3,
                2, 3,
                3, 1,
                3, 1,
                1, 1,
                1, 1,
                3, 3],
               [1, 3,
                2, 3,
                3, 2,
                3, 0,
                3, 3,
                0, 1,
                0, 2],
               [3, 1,
                3, 3,
                3, 3,
                3, 1,
                3, 0,
                3, 2,
                3, 3],
               [3, 3,
                2, 2,
                3, 3,
                2, 0,
                0, 1,
                0, 2,
                2, 3]]
