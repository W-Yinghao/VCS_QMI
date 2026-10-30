Publication copy (sample names aliased; numbers and conclusions unchanged).

# R27-D candidate-source audit (FIT, K=5) -- numbers only

Per set = (condition/checkpoint, training peer T; searching peer = the other). hit = top-5 identity hit; reach = true counterpart present in the eligible bank (re-computed under exclusion). Queries per set are 8 FIT batches (not independent images; distinct observed images listed).

| state | peer | mode | dir | n | distinct imgs | reach | A_slot hit | A_unique hit | B_direct hit | C_mix hit | A_u hit/reach | B hit/reach | C hit/reach | A&B / A-only / B-only / none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f40_s1001/prefix_reconstructed | A | normal | i2t | 1024 | 1014 | 896 | 892 | 892 | 418 | 892 | 0.9955 | 0.4665 | 0.9955 | 417 / 475 / 1 / 131 |
| f40_s1001/prefix_reconstructed | A | normal | t2i | 1024 | 1014 | 919 | 216 | 224 | 248 | 270 | 0.2437 | 0.2699 | 0.2938 | 118 / 106 / 130 / 670 |
| f40_s1001/prefix_reconstructed | A | exclude_same_source | i2t | 1024 | 1014 | 39 | - | 0 | 9 | 2 | 0.0000 | 0.2308 | 0.0513 | 0 / 0 / 9 / 1015 |
| f40_s1001/prefix_reconstructed | A | exclude_same_source | t2i | 1024 | 1014 | 919 | - | 222 | 248 | 268 | 0.2416 | 0.2699 | 0.2916 | 118 / 104 / 130 / 672 |
| f40_s1001/prefix_reconstructed | B | normal | i2t | 1024 | 1005 | 919 | 911 | 911 | 428 | 913 | 0.9913 | 0.4657 | 0.9935 | 426 / 485 / 2 / 111 |
| f40_s1001/prefix_reconstructed | B | normal | t2i | 1024 | 1005 | 912 | 222 | 231 | 238 | 276 | 0.2533 | 0.2610 | 0.3026 | 108 / 123 / 130 / 663 |
| f40_s1001/prefix_reconstructed | B | exclude_same_source | i2t | 1024 | 1005 | 44 | - | 0 | 7 | 5 | 0.0000 | 0.1591 | 0.1136 | 0 / 0 / 7 / 1017 |
| f40_s1001/prefix_reconstructed | B | exclude_same_source | t2i | 1024 | 1005 | 912 | - | 231 | 238 | 276 | 0.2533 | 0.2610 | 0.3026 | 108 / 123 / 130 / 663 |
| f40_s1001/a1_iface_last | A | normal | i2t | 1024 | 1006 | 934 | 927 | 927 | 661 | 928 | 0.9925 | 0.7077 | 0.9936 | 660 / 267 / 1 / 96 |
| f40_s1001/a1_iface_last | A | normal | t2i | 1024 | 1006 | 935 | 283 | 289 | 356 | 381 | 0.3091 | 0.3807 | 0.4075 | 192 / 97 / 164 / 571 |
| f40_s1001/a1_iface_last | A | exclude_same_source | i2t | 1024 | 1006 | 75 | - | 0 | 16 | 12 | 0.0000 | 0.2133 | 0.1600 | 0 / 0 / 16 / 1008 |
| f40_s1001/a1_iface_last | A | exclude_same_source | t2i | 1024 | 1006 | 933 | - | 276 | 354 | 371 | 0.2958 | 0.3794 | 0.3976 | 186 / 90 / 168 / 580 |
| f40_s1001/a1_iface_last | B | normal | i2t | 1024 | 1008 | 934 | 923 | 923 | 668 | 925 | 0.9882 | 0.7152 | 0.9904 | 666 / 257 / 2 / 99 |
| f40_s1001/a1_iface_last | B | normal | t2i | 1024 | 1008 | 922 | 255 | 267 | 327 | 336 | 0.2896 | 0.3547 | 0.3644 | 174 / 93 / 153 / 604 |
| f40_s1001/a1_iface_last | B | exclude_same_source | i2t | 1024 | 1008 | 85 | - | 0 | 24 | 16 | 0.0000 | 0.2824 | 0.1882 | 0 / 0 / 24 / 1000 |
| f40_s1001/a1_iface_last | B | exclude_same_source | t2i | 1024 | 1008 | 920 | - | 255 | 323 | 327 | 0.2772 | 0.3511 | 0.3554 | 167 / 88 / 156 / 613 |
| f60_s1001/prefix_reconstructed | A | normal | i2t | 1024 | 1007 | 802 | 780 | 780 | 326 | 785 | 0.9726 | 0.4065 | 0.9788 | 320 / 460 / 6 / 238 |
| f60_s1001/prefix_reconstructed | A | normal | t2i | 1024 | 1007 | 820 | 193 | 195 | 183 | 228 | 0.2378 | 0.2232 | 0.2780 | 92 / 103 / 91 / 738 |
| f60_s1001/prefix_reconstructed | A | exclude_same_source | i2t | 1024 | 1007 | 60 | - | 0 | 13 | 8 | 0.0000 | 0.2167 | 0.1333 | 0 / 0 / 13 / 1011 |
| f60_s1001/prefix_reconstructed | A | exclude_same_source | t2i | 1024 | 1007 | 820 | - | 195 | 183 | 228 | 0.2378 | 0.2232 | 0.2780 | 92 / 103 / 91 / 738 |
| f60_s1001/prefix_reconstructed | B | normal | i2t | 1024 | 1004 | 834 | 822 | 822 | 397 | 825 | 0.9856 | 0.4760 | 0.9892 | 393 / 429 / 4 / 198 |
| f60_s1001/prefix_reconstructed | B | normal | t2i | 1024 | 1004 | 830 | 202 | 206 | 224 | 246 | 0.2482 | 0.2699 | 0.2964 | 114 / 92 / 110 / 708 |
| f60_s1001/prefix_reconstructed | B | exclude_same_source | i2t | 1024 | 1004 | 49 | - | 0 | 12 | 9 | 0.0000 | 0.2449 | 0.1837 | 0 / 0 / 12 / 1012 |
| f60_s1001/prefix_reconstructed | B | exclude_same_source | t2i | 1024 | 1004 | 830 | - | 206 | 224 | 246 | 0.2482 | 0.2699 | 0.2964 | 114 / 92 / 110 / 708 |
| f60_s1001/a1_iface_last | A | normal | i2t | 1024 | 1009 | 875 | 854 | 854 | 636 | 856 | 0.9760 | 0.7269 | 0.9783 | 632 / 222 / 4 / 166 |
| f60_s1001/a1_iface_last | A | normal | t2i | 1024 | 1009 | 884 | 249 | 253 | 305 | 327 | 0.2862 | 0.3450 | 0.3699 | 164 / 89 / 141 / 630 |
| f60_s1001/a1_iface_last | A | exclude_same_source | i2t | 1024 | 1009 | 118 | - | 0 | 28 | 18 | 0.0000 | 0.2373 | 0.1525 | 0 / 0 / 28 / 996 |
| f60_s1001/a1_iface_last | A | exclude_same_source | t2i | 1024 | 1009 | 882 | - | 249 | 304 | 324 | 0.2823 | 0.3447 | 0.3673 | 163 / 86 / 141 / 634 |
| f60_s1001/a1_iface_last | B | normal | i2t | 1024 | 1010 | 903 | 877 | 877 | 657 | 878 | 0.9712 | 0.7276 | 0.9723 | 655 / 222 / 2 / 145 |
| f60_s1001/a1_iface_last | B | normal | t2i | 1024 | 1010 | 879 | 231 | 239 | 288 | 304 | 0.2719 | 0.3276 | 0.3458 | 149 / 90 / 139 / 646 |
| f60_s1001/a1_iface_last | B | exclude_same_source | i2t | 1024 | 1010 | 139 | - | 0 | 34 | 24 | 0.0000 | 0.2446 | 0.1727 | 0 / 0 / 34 / 990 |
| f60_s1001/a1_iface_last | B | exclude_same_source | t2i | 1024 | 1010 | 879 | - | 234 | 288 | 298 | 0.2662 | 0.3276 | 0.3390 | 149 / 85 / 139 / 651 |

## Pooled over the two peers (counts summed)

| state | mode | dir | method | n | reach | hit | hit/n | hit/reach | mean GT fraction | MRR@5 | rows <5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f40_s1001/prefix_reconstructed | normal | i2t | A_slot_reference | 2048 | 1815 | 1803 | 0.8804 | 0.9934 | 0.3474 | 0.8755 | 0 |
| f40_s1001/prefix_reconstructed | normal | i2t | A_unique | 2048 | 1815 | 1803 | 0.8804 | 0.9934 | 0.3474 | 0.8755 | 0 |
| f40_s1001/prefix_reconstructed | normal | i2t | B_direct | 2048 | 1815 | 846 | 0.4131 | 0.4661 | 0.1067 | 0.2854 | 0 |
| f40_s1001/prefix_reconstructed | normal | i2t | C_mix | 2048 | 1815 | 1805 | 0.8813 | 0.9945 | 0.3421 | 0.8757 | 0 |
| f40_s1001/prefix_reconstructed | normal | t2i | A_slot_reference | 2048 | 1831 | 438 | 0.2139 | 0.2392 | 0.2139 | 0.1411 | 0 |
| f40_s1001/prefix_reconstructed | normal | t2i | A_unique | 2048 | 1831 | 455 | 0.2222 | 0.2485 | 0.2222 | 0.1439 | 0 |
| f40_s1001/prefix_reconstructed | normal | t2i | B_direct | 2048 | 1831 | 486 | 0.2373 | 0.2654 | 0.2373 | 0.1527 | 0 |
| f40_s1001/prefix_reconstructed | normal | t2i | C_mix | 2048 | 1831 | 546 | 0.2666 | 0.2982 | 0.2666 | 0.1545 | 0 |
| f40_s1001/prefix_reconstructed | exclude_same_source | i2t | A_unique | 2048 | 83 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| f40_s1001/prefix_reconstructed | exclude_same_source | i2t | B_direct | 2048 | 83 | 16 | 0.0078 | 0.1928 | 0.0016 | 0.0036 | 0 |
| f40_s1001/prefix_reconstructed | exclude_same_source | i2t | C_mix | 2048 | 83 | 7 | 0.0034 | 0.0843 | 0.0007 | 0.0008 | 0 |
| f40_s1001/prefix_reconstructed | exclude_same_source | t2i | A_unique | 2048 | 1831 | 453 | 0.2212 | 0.2474 | 0.2212 | 0.1429 | 0 |
| f40_s1001/prefix_reconstructed | exclude_same_source | t2i | B_direct | 2048 | 1831 | 486 | 0.2373 | 0.2654 | 0.2373 | 0.1527 | 0 |
| f40_s1001/prefix_reconstructed | exclude_same_source | t2i | C_mix | 2048 | 1831 | 544 | 0.2656 | 0.2971 | 0.2656 | 0.1535 | 0 |
| f40_s1001/a1_iface_last | normal | i2t | A_slot_reference | 2048 | 1868 | 1850 | 0.9033 | 0.9904 | 0.3570 | 0.8945 | 0 |
| f40_s1001/a1_iface_last | normal | i2t | A_unique | 2048 | 1868 | 1850 | 0.9033 | 0.9904 | 0.3570 | 0.8945 | 0 |
| f40_s1001/a1_iface_last | normal | i2t | B_direct | 2048 | 1868 | 1329 | 0.6489 | 0.7115 | 0.1922 | 0.5167 | 0 |
| f40_s1001/a1_iface_last | normal | i2t | C_mix | 2048 | 1868 | 1853 | 0.9048 | 0.9920 | 0.3551 | 0.8948 | 0 |
| f40_s1001/a1_iface_last | normal | t2i | A_slot_reference | 2048 | 1857 | 538 | 0.2627 | 0.2897 | 0.2627 | 0.1815 | 0 |
| f40_s1001/a1_iface_last | normal | t2i | A_unique | 2048 | 1857 | 556 | 0.2715 | 0.2994 | 0.2715 | 0.1842 | 0 |
| f40_s1001/a1_iface_last | normal | t2i | B_direct | 2048 | 1857 | 683 | 0.3335 | 0.3678 | 0.3335 | 0.2256 | 0 |
| f40_s1001/a1_iface_last | normal | t2i | C_mix | 2048 | 1857 | 717 | 0.3501 | 0.3861 | 0.3501 | 0.2028 | 0 |
| f40_s1001/a1_iface_last | exclude_same_source | i2t | A_unique | 2048 | 160 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| f40_s1001/a1_iface_last | exclude_same_source | i2t | B_direct | 2048 | 160 | 40 | 0.0195 | 0.2500 | 0.0039 | 0.0128 | 0 |
| f40_s1001/a1_iface_last | exclude_same_source | i2t | C_mix | 2048 | 160 | 28 | 0.0137 | 0.1750 | 0.0027 | 0.0032 | 0 |
| f40_s1001/a1_iface_last | exclude_same_source | t2i | A_unique | 2048 | 1853 | 531 | 0.2593 | 0.2866 | 0.2593 | 0.1728 | 0 |
| f40_s1001/a1_iface_last | exclude_same_source | t2i | B_direct | 2048 | 1853 | 677 | 0.3306 | 0.3654 | 0.3306 | 0.2234 | 0 |
| f40_s1001/a1_iface_last | exclude_same_source | t2i | C_mix | 2048 | 1853 | 698 | 0.3408 | 0.3767 | 0.3408 | 0.1921 | 0 |
| f60_s1001/prefix_reconstructed | normal | i2t | A_slot_reference | 2048 | 1636 | 1602 | 0.7822 | 0.9792 | 0.2630 | 0.7754 | 0 |
| f60_s1001/prefix_reconstructed | normal | i2t | A_unique | 2048 | 1636 | 1602 | 0.7822 | 0.9792 | 0.2630 | 0.7754 | 0 |
| f60_s1001/prefix_reconstructed | normal | i2t | B_direct | 2048 | 1636 | 723 | 0.3530 | 0.4419 | 0.0889 | 0.2531 | 0 |
| f60_s1001/prefix_reconstructed | normal | i2t | C_mix | 2048 | 1636 | 1610 | 0.7861 | 0.9841 | 0.2623 | 0.7763 | 0 |
| f60_s1001/prefix_reconstructed | normal | t2i | A_slot_reference | 2048 | 1650 | 395 | 0.1929 | 0.2394 | 0.1929 | 0.1338 | 0 |
| f60_s1001/prefix_reconstructed | normal | t2i | A_unique | 2048 | 1650 | 401 | 0.1958 | 0.2430 | 0.1958 | 0.1346 | 0 |
| f60_s1001/prefix_reconstructed | normal | t2i | B_direct | 2048 | 1650 | 407 | 0.1987 | 0.2467 | 0.1987 | 0.1255 | 0 |
| f60_s1001/prefix_reconstructed | normal | t2i | C_mix | 2048 | 1650 | 474 | 0.2314 | 0.2873 | 0.2314 | 0.1428 | 0 |
| f60_s1001/prefix_reconstructed | exclude_same_source | i2t | A_unique | 2048 | 109 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| f60_s1001/prefix_reconstructed | exclude_same_source | i2t | B_direct | 2048 | 109 | 25 | 0.0122 | 0.2294 | 0.0024 | 0.0083 | 0 |
| f60_s1001/prefix_reconstructed | exclude_same_source | i2t | C_mix | 2048 | 109 | 17 | 0.0083 | 0.1560 | 0.0017 | 0.0020 | 0 |
| f60_s1001/prefix_reconstructed | exclude_same_source | t2i | A_unique | 2048 | 1650 | 401 | 0.1958 | 0.2430 | 0.1958 | 0.1346 | 0 |
| f60_s1001/prefix_reconstructed | exclude_same_source | t2i | B_direct | 2048 | 1650 | 407 | 0.1987 | 0.2467 | 0.1987 | 0.1255 | 0 |
| f60_s1001/prefix_reconstructed | exclude_same_source | t2i | C_mix | 2048 | 1650 | 474 | 0.2314 | 0.2873 | 0.2314 | 0.1428 | 0 |
| f60_s1001/a1_iface_last | normal | i2t | A_slot_reference | 2048 | 1778 | 1731 | 0.8452 | 0.9736 | 0.3067 | 0.8306 | 0 |
| f60_s1001/a1_iface_last | normal | i2t | A_unique | 2048 | 1778 | 1731 | 0.8452 | 0.9736 | 0.3071 | 0.8308 | 0 |
| f60_s1001/a1_iface_last | normal | i2t | B_direct | 2048 | 1778 | 1293 | 0.6313 | 0.7272 | 0.1802 | 0.5064 | 0 |
| f60_s1001/a1_iface_last | normal | i2t | C_mix | 2048 | 1778 | 1734 | 0.8467 | 0.9753 | 0.3078 | 0.8312 | 0 |
| f60_s1001/a1_iface_last | normal | t2i | A_slot_reference | 2048 | 1763 | 480 | 0.2344 | 0.2723 | 0.2344 | 0.1650 | 0 |
| f60_s1001/a1_iface_last | normal | t2i | A_unique | 2048 | 1763 | 492 | 0.2402 | 0.2791 | 0.2402 | 0.1677 | 0 |
| f60_s1001/a1_iface_last | normal | t2i | B_direct | 2048 | 1763 | 593 | 0.2896 | 0.3364 | 0.2896 | 0.1956 | 0 |
| f60_s1001/a1_iface_last | normal | t2i | C_mix | 2048 | 1763 | 631 | 0.3081 | 0.3579 | 0.3081 | 0.1840 | 0 |
| f60_s1001/a1_iface_last | exclude_same_source | i2t | A_unique | 2048 | 257 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| f60_s1001/a1_iface_last | exclude_same_source | i2t | B_direct | 2048 | 257 | 62 | 0.0303 | 0.2412 | 0.0062 | 0.0205 | 0 |
| f60_s1001/a1_iface_last | exclude_same_source | i2t | C_mix | 2048 | 257 | 42 | 0.0205 | 0.1634 | 0.0041 | 0.0049 | 0 |
| f60_s1001/a1_iface_last | exclude_same_source | t2i | A_unique | 2048 | 1761 | 483 | 0.2358 | 0.2743 | 0.2358 | 0.1635 | 0 |
| f60_s1001/a1_iface_last | exclude_same_source | t2i | B_direct | 2048 | 1761 | 592 | 0.2891 | 0.3362 | 0.2891 | 0.1955 | 0 |
| f60_s1001/a1_iface_last | exclude_same_source | t2i | C_mix | 2048 | 1761 | 622 | 0.3037 | 0.3532 | 0.3037 | 0.1799 | 0 |
