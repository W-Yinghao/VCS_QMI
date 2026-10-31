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
| f40_s1001/bn_cross_last | A | normal | i2t | 1024 | 1009 | 943 | 931 | 931 | 654 | 932 | 0.9873 | 0.6935 | 0.9883 | 653 / 278 / 1 / 92 |
| f40_s1001/bn_cross_last | A | normal | t2i | 1024 | 1009 | 941 | 245 | 254 | 331 | 334 | 0.2699 | 0.3518 | 0.3549 | 169 / 85 / 162 / 608 |
| f40_s1001/bn_cross_last | A | exclude_same_source | i2t | 1024 | 1009 | 75 | - | 0 | 15 | 12 | 0.0000 | 0.2000 | 0.1600 | 0 / 0 / 15 / 1009 |
| f40_s1001/bn_cross_last | A | exclude_same_source | t2i | 1024 | 1009 | 940 | - | 246 | 331 | 326 | 0.2617 | 0.3521 | 0.3468 | 165 / 81 / 166 / 612 |
| f40_s1001/bn_cross_last | B | normal | i2t | 1024 | 1018 | 933 | 922 | 922 | 669 | 924 | 0.9882 | 0.7170 | 0.9904 | 666 / 256 / 3 / 99 |
| f40_s1001/bn_cross_last | B | normal | t2i | 1024 | 1018 | 936 | 272 | 277 | 312 | 343 | 0.2959 | 0.3333 | 0.3665 | 176 / 101 / 136 / 611 |
| f40_s1001/bn_cross_last | B | exclude_same_source | i2t | 1024 | 1018 | 74 | - | 0 | 15 | 10 | 0.0000 | 0.2027 | 0.1351 | 0 / 0 / 15 / 1009 |
| f40_s1001/bn_cross_last | B | exclude_same_source | t2i | 1024 | 1018 | 935 | - | 271 | 310 | 338 | 0.2898 | 0.3316 | 0.3615 | 173 / 98 / 137 / 616 |
| f40_s1001/cn_mix_last | A | normal | i2t | 1024 | 1012 | 922 | 911 | 911 | 630 | 911 | 0.9881 | 0.6833 | 0.9881 | 629 / 282 / 1 / 112 |
| f40_s1001/cn_mix_last | A | normal | t2i | 1024 | 1012 | 929 | 271 | 274 | 325 | 350 | 0.2949 | 0.3498 | 0.3767 | 175 / 99 / 150 / 600 |
| f40_s1001/cn_mix_last | A | exclude_same_source | i2t | 1024 | 1012 | 74 | - | 0 | 19 | 6 | 0.0000 | 0.2568 | 0.0811 | 0 / 0 / 19 / 1005 |
| f40_s1001/cn_mix_last | A | exclude_same_source | t2i | 1024 | 1012 | 927 | - | 268 | 325 | 346 | 0.2891 | 0.3506 | 0.3732 | 173 / 95 / 152 / 604 |
| f40_s1001/cn_mix_last | B | normal | i2t | 1024 | 1012 | 932 | 926 | 926 | 672 | 926 | 0.9936 | 0.7210 | 0.9936 | 671 / 255 / 1 / 97 |
| f40_s1001/cn_mix_last | B | normal | t2i | 1024 | 1012 | 917 | 287 | 298 | 344 | 366 | 0.3250 | 0.3751 | 0.3991 | 197 / 101 / 147 / 579 |
| f40_s1001/cn_mix_last | B | exclude_same_source | i2t | 1024 | 1012 | 73 | - | 0 | 16 | 6 | 0.0000 | 0.2192 | 0.0822 | 0 / 0 / 16 / 1008 |
| f40_s1001/cn_mix_last | B | exclude_same_source | t2i | 1024 | 1012 | 916 | - | 287 | 343 | 357 | 0.3133 | 0.3745 | 0.3897 | 193 / 94 / 150 / 587 |
| f60_s1001/bn_cross_last | A | normal | i2t | 1024 | 1006 | 894 | 875 | 875 | 642 | 878 | 0.9787 | 0.7181 | 0.9821 | 638 / 237 / 4 / 145 |
| f60_s1001/bn_cross_last | A | normal | t2i | 1024 | 1006 | 887 | 219 | 228 | 290 | 312 | 0.2570 | 0.3269 | 0.3517 | 150 / 78 / 140 / 656 |
| f60_s1001/bn_cross_last | A | exclude_same_source | i2t | 1024 | 1006 | 112 | - | 0 | 26 | 17 | 0.0000 | 0.2321 | 0.1518 | 0 / 0 / 26 / 998 |
| f60_s1001/bn_cross_last | A | exclude_same_source | t2i | 1024 | 1006 | 885 | - | 220 | 289 | 306 | 0.2486 | 0.3266 | 0.3458 | 146 / 74 / 143 / 661 |
| f60_s1001/bn_cross_last | B | normal | i2t | 1024 | 1001 | 898 | 875 | 875 | 649 | 875 | 0.9744 | 0.7227 | 0.9744 | 647 / 228 / 2 / 147 |
| f60_s1001/bn_cross_last | B | normal | t2i | 1024 | 1001 | 896 | 245 | 258 | 308 | 323 | 0.2879 | 0.3438 | 0.3605 | 164 / 94 / 144 / 622 |
| f60_s1001/bn_cross_last | B | exclude_same_source | i2t | 1024 | 1001 | 133 | - | 0 | 28 | 17 | 0.0000 | 0.2105 | 0.1278 | 0 / 0 / 28 / 996 |
| f60_s1001/bn_cross_last | B | exclude_same_source | t2i | 1024 | 1001 | 895 | - | 250 | 307 | 318 | 0.2793 | 0.3430 | 0.3553 | 160 / 90 / 147 / 627 |
| f60_s1001/cn_mix_last | A | normal | i2t | 1024 | 1014 | 887 | 849 | 849 | 621 | 853 | 0.9572 | 0.7001 | 0.9617 | 615 / 234 / 6 / 169 |
| f60_s1001/cn_mix_last | A | normal | t2i | 1024 | 1014 | 877 | 234 | 248 | 285 | 305 | 0.2828 | 0.3250 | 0.3478 | 155 / 93 / 130 / 646 |
| f60_s1001/cn_mix_last | A | exclude_same_source | i2t | 1024 | 1014 | 131 | - | 0 | 21 | 16 | 0.0000 | 0.1603 | 0.1221 | 0 / 0 / 21 / 1003 |
| f60_s1001/cn_mix_last | A | exclude_same_source | t2i | 1024 | 1014 | 876 | - | 245 | 284 | 301 | 0.2797 | 0.3242 | 0.3436 | 155 / 90 / 129 / 650 |
| f60_s1001/cn_mix_last | B | normal | i2t | 1024 | 1012 | 894 | 857 | 857 | 642 | 859 | 0.9586 | 0.7181 | 0.9609 | 639 / 218 / 3 / 164 |
| f60_s1001/cn_mix_last | B | normal | t2i | 1024 | 1012 | 897 | 250 | 252 | 273 | 304 | 0.2809 | 0.3043 | 0.3389 | 156 / 96 / 117 / 655 |
| f60_s1001/cn_mix_last | B | exclude_same_source | i2t | 1024 | 1012 | 141 | - | 0 | 27 | 21 | 0.0000 | 0.1915 | 0.1489 | 0 / 0 / 27 / 997 |
| f60_s1001/cn_mix_last | B | exclude_same_source | t2i | 1024 | 1012 | 896 | - | 246 | 273 | 301 | 0.2746 | 0.3047 | 0.3359 | 154 / 92 / 119 / 659 |

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
| f40_s1001/bn_cross_last | normal | i2t | A_slot_reference | 2048 | 1876 | 1853 | 0.9048 | 0.9877 | 0.3581 | 0.8953 | 0 |
| f40_s1001/bn_cross_last | normal | i2t | A_unique | 2048 | 1876 | 1853 | 0.9048 | 0.9877 | 0.3581 | 0.8953 | 0 |
| f40_s1001/bn_cross_last | normal | i2t | B_direct | 2048 | 1876 | 1323 | 0.6460 | 0.7052 | 0.1877 | 0.5090 | 0 |
| f40_s1001/bn_cross_last | normal | i2t | C_mix | 2048 | 1876 | 1856 | 0.9062 | 0.9893 | 0.3547 | 0.8957 | 0 |
| f40_s1001/bn_cross_last | normal | t2i | A_slot_reference | 2048 | 1877 | 517 | 0.2524 | 0.2754 | 0.2524 | 0.1778 | 0 |
| f40_s1001/bn_cross_last | normal | t2i | A_unique | 2048 | 1877 | 531 | 0.2593 | 0.2829 | 0.2593 | 0.1799 | 0 |
| f40_s1001/bn_cross_last | normal | t2i | B_direct | 2048 | 1877 | 643 | 0.3140 | 0.3426 | 0.3140 | 0.2111 | 0 |
| f40_s1001/bn_cross_last | normal | t2i | C_mix | 2048 | 1877 | 677 | 0.3306 | 0.3607 | 0.3306 | 0.1969 | 0 |
| f40_s1001/bn_cross_last | exclude_same_source | i2t | A_unique | 2048 | 149 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| f40_s1001/bn_cross_last | exclude_same_source | i2t | B_direct | 2048 | 149 | 30 | 0.0146 | 0.2013 | 0.0029 | 0.0088 | 0 |
| f40_s1001/bn_cross_last | exclude_same_source | i2t | C_mix | 2048 | 149 | 22 | 0.0107 | 0.1477 | 0.0021 | 0.0025 | 0 |
| f40_s1001/bn_cross_last | exclude_same_source | t2i | A_unique | 2048 | 1875 | 517 | 0.2524 | 0.2757 | 0.2524 | 0.1731 | 0 |
| f40_s1001/bn_cross_last | exclude_same_source | t2i | B_direct | 2048 | 1875 | 641 | 0.3130 | 0.3419 | 0.3130 | 0.2097 | 0 |
| f40_s1001/bn_cross_last | exclude_same_source | t2i | C_mix | 2048 | 1875 | 664 | 0.3242 | 0.3541 | 0.3242 | 0.1902 | 0 |
| f40_s1001/cn_mix_last | normal | i2t | A_slot_reference | 2048 | 1854 | 1837 | 0.8970 | 0.9908 | 0.3569 | 0.8870 | 0 |
| f40_s1001/cn_mix_last | normal | i2t | A_unique | 2048 | 1854 | 1837 | 0.8970 | 0.9908 | 0.3569 | 0.8870 | 0 |
| f40_s1001/cn_mix_last | normal | i2t | B_direct | 2048 | 1854 | 1302 | 0.6357 | 0.7023 | 0.1894 | 0.5019 | 0 |
| f40_s1001/cn_mix_last | normal | i2t | C_mix | 2048 | 1854 | 1837 | 0.8970 | 0.9908 | 0.3521 | 0.8870 | 0 |
| f40_s1001/cn_mix_last | normal | t2i | A_slot_reference | 2048 | 1846 | 558 | 0.2725 | 0.3023 | 0.2725 | 0.1905 | 0 |
| f40_s1001/cn_mix_last | normal | t2i | A_unique | 2048 | 1846 | 572 | 0.2793 | 0.3099 | 0.2793 | 0.1928 | 0 |
| f40_s1001/cn_mix_last | normal | t2i | B_direct | 2048 | 1846 | 669 | 0.3267 | 0.3624 | 0.3267 | 0.2201 | 0 |
| f40_s1001/cn_mix_last | normal | t2i | C_mix | 2048 | 1846 | 716 | 0.3496 | 0.3879 | 0.3496 | 0.2095 | 0 |
| f40_s1001/cn_mix_last | exclude_same_source | i2t | A_unique | 2048 | 147 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| f40_s1001/cn_mix_last | exclude_same_source | i2t | B_direct | 2048 | 147 | 35 | 0.0171 | 0.2381 | 0.0034 | 0.0075 | 0 |
| f40_s1001/cn_mix_last | exclude_same_source | i2t | C_mix | 2048 | 147 | 12 | 0.0059 | 0.0816 | 0.0012 | 0.0013 | 0 |
| f40_s1001/cn_mix_last | exclude_same_source | t2i | A_unique | 2048 | 1843 | 555 | 0.2710 | 0.3011 | 0.2710 | 0.1852 | 0 |
| f40_s1001/cn_mix_last | exclude_same_source | t2i | B_direct | 2048 | 1843 | 668 | 0.3262 | 0.3625 | 0.3262 | 0.2198 | 0 |
| f40_s1001/cn_mix_last | exclude_same_source | t2i | C_mix | 2048 | 1843 | 703 | 0.3433 | 0.3814 | 0.3433 | 0.2024 | 0 |
| f60_s1001/bn_cross_last | normal | i2t | A_slot_reference | 2048 | 1792 | 1750 | 0.8545 | 0.9766 | 0.3139 | 0.8393 | 0 |
| f60_s1001/bn_cross_last | normal | i2t | A_unique | 2048 | 1792 | 1750 | 0.8545 | 0.9766 | 0.3142 | 0.8396 | 0 |
| f60_s1001/bn_cross_last | normal | i2t | B_direct | 2048 | 1792 | 1291 | 0.6304 | 0.7204 | 0.1789 | 0.4882 | 0 |
| f60_s1001/bn_cross_last | normal | i2t | C_mix | 2048 | 1792 | 1753 | 0.8560 | 0.9782 | 0.3141 | 0.8399 | 0 |
| f60_s1001/bn_cross_last | normal | t2i | A_slot_reference | 2048 | 1783 | 464 | 0.2266 | 0.2602 | 0.2266 | 0.1601 | 0 |
| f60_s1001/bn_cross_last | normal | t2i | A_unique | 2048 | 1783 | 486 | 0.2373 | 0.2726 | 0.2373 | 0.1632 | 0 |
| f60_s1001/bn_cross_last | normal | t2i | B_direct | 2048 | 1783 | 598 | 0.2920 | 0.3354 | 0.2920 | 0.1996 | 0 |
| f60_s1001/bn_cross_last | normal | t2i | C_mix | 2048 | 1783 | 635 | 0.3101 | 0.3561 | 0.3101 | 0.1805 | 0 |
| f60_s1001/bn_cross_last | exclude_same_source | i2t | A_unique | 2048 | 245 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| f60_s1001/bn_cross_last | exclude_same_source | i2t | B_direct | 2048 | 245 | 54 | 0.0264 | 0.2204 | 0.0053 | 0.0153 | 0 |
| f60_s1001/bn_cross_last | exclude_same_source | i2t | C_mix | 2048 | 245 | 34 | 0.0166 | 0.1388 | 0.0033 | 0.0038 | 0 |
| f60_s1001/bn_cross_last | exclude_same_source | t2i | A_unique | 2048 | 1780 | 470 | 0.2295 | 0.2640 | 0.2295 | 0.1558 | 0 |
| f60_s1001/bn_cross_last | exclude_same_source | t2i | B_direct | 2048 | 1780 | 596 | 0.2910 | 0.3348 | 0.2910 | 0.1991 | 0 |
| f60_s1001/bn_cross_last | exclude_same_source | t2i | C_mix | 2048 | 1780 | 624 | 0.3047 | 0.3506 | 0.3047 | 0.1736 | 0 |
| f60_s1001/cn_mix_last | normal | i2t | A_slot_reference | 2048 | 1781 | 1706 | 0.8330 | 0.9579 | 0.3019 | 0.8172 | 0 |
| f60_s1001/cn_mix_last | normal | i2t | A_unique | 2048 | 1781 | 1706 | 0.8330 | 0.9579 | 0.3019 | 0.8175 | 0 |
| f60_s1001/cn_mix_last | normal | i2t | B_direct | 2048 | 1781 | 1263 | 0.6167 | 0.7092 | 0.1747 | 0.4934 | 0 |
| f60_s1001/cn_mix_last | normal | i2t | C_mix | 2048 | 1781 | 1712 | 0.8359 | 0.9613 | 0.3032 | 0.8182 | 0 |
| f60_s1001/cn_mix_last | normal | t2i | A_slot_reference | 2048 | 1774 | 484 | 0.2363 | 0.2728 | 0.2363 | 0.1618 | 0 |
| f60_s1001/cn_mix_last | normal | t2i | A_unique | 2048 | 1774 | 500 | 0.2441 | 0.2818 | 0.2441 | 0.1649 | 0 |
| f60_s1001/cn_mix_last | normal | t2i | B_direct | 2048 | 1774 | 558 | 0.2725 | 0.3145 | 0.2725 | 0.1811 | 0 |
| f60_s1001/cn_mix_last | normal | t2i | C_mix | 2048 | 1774 | 609 | 0.2974 | 0.3433 | 0.2974 | 0.1774 | 0 |
| f60_s1001/cn_mix_last | exclude_same_source | i2t | A_unique | 2048 | 272 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| f60_s1001/cn_mix_last | exclude_same_source | i2t | B_direct | 2048 | 272 | 48 | 0.0234 | 0.1765 | 0.0047 | 0.0170 | 0 |
| f60_s1001/cn_mix_last | exclude_same_source | i2t | C_mix | 2048 | 272 | 37 | 0.0181 | 0.1360 | 0.0036 | 0.0043 | 0 |
| f60_s1001/cn_mix_last | exclude_same_source | t2i | A_unique | 2048 | 1772 | 491 | 0.2397 | 0.2771 | 0.2397 | 0.1611 | 0 |
| f60_s1001/cn_mix_last | exclude_same_source | t2i | B_direct | 2048 | 1772 | 557 | 0.2720 | 0.3143 | 0.2720 | 0.1807 | 0 |
| f60_s1001/cn_mix_last | exclude_same_source | t2i | C_mix | 2048 | 1772 | 602 | 0.2939 | 0.3397 | 0.2939 | 0.1738 | 0 |
