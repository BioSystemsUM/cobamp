\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: vp0_ind + vn0_ind + vp1_ind + vn1_ind + vp2_ind + vn2_ind + vp3_ind
      + vn3_ind + vp4_ind + vn4_ind + vp5_ind + vn5_ind + vp6_ind + vn6_ind
      + vp7_ind + vn7_ind + vp8_ind + vn8_ind
Subject To
 Ci0#0:                      u0 + vp0 - vn0 >= 0
 Ci1#1:                      - u0 + u1 + u2 + vp1 - vn1 >= 0
 Ci2#2:                      - u1 + vp2 - vn2 >= 0
 Ci3#3:                      - u0 - u2 + u5 + vp4 - vn4 >= 0
 Ci4#4:                      u3 + vp5 - vn5 >= 0
 Ci5#5:                      - u0 - u3 + u4 + vp6 - vn6 >= 0
 Ci6#6:                      - u4 + u5 + vp7 - vn7 >= 0
 Ci7#7:                      - u5 + vp8 - vn8 - w0 >= 0
 Cr0#8:                      u2 + vp3 - vn3  = 0
 Cb#9:                       - w0 + C  = 0
 Cvp0_ind_helper0#10:        vp0_ind + vp0_inda  = 1
 Cvp0_ind_helper1#11:        - vp0_inda + vp0_indb >= 0
 Cvp0_ind_helper2#12:        - vp0_ind + vp0_indc  = 0
 Cvp0_ind_helper3#13:        - vp0_indc + vp0_indd >= 0
 Cvn0_ind_helper0#14:        vn0_ind + vn0_inda  = 1
 Cvn0_ind_helper1#15:        - vn0_inda + vn0_indb >= 0
 Cvn0_ind_helper2#16:        - vn0_ind + vn0_indc  = 0
 Cvn0_ind_helper3#17:        - vn0_indc + vn0_indd >= 0
 Cvp1_ind_helper0#18:        vp1_ind + vp1_inda  = 1
 Cvp1_ind_helper1#19:        - vp1_inda + vp1_indb >= 0
 Cvp1_ind_helper2#20:        - vp1_ind + vp1_indc  = 0
 Cvp1_ind_helper3#21:        - vp1_indc + vp1_indd >= 0
 Cvn1_ind_helper0#22:        vn1_ind + vn1_inda  = 1
 Cvn1_ind_helper1#23:        - vn1_inda + vn1_indb >= 0
 Cvn1_ind_helper2#24:        - vn1_ind + vn1_indc  = 0
 Cvn1_ind_helper3#25:        - vn1_indc + vn1_indd >= 0
 Cvp2_ind_helper0#26:        vp2_ind + vp2_inda  = 1
 Cvp2_ind_helper1#27:        - vp2_inda + vp2_indb >= 0
 Cvp2_ind_helper2#28:        - vp2_ind + vp2_indc  = 0
 Cvp2_ind_helper3#29:        - vp2_indc + vp2_indd >= 0
 Cvn2_ind_helper0#30:        vn2_ind + vn2_inda  = 1
 Cvn2_ind_helper1#31:        - vn2_inda + vn2_indb >= 0
 Cvn2_ind_helper2#32:        - vn2_ind + vn2_indc  = 0
 Cvn2_ind_helper3#33:        - vn2_indc + vn2_indd >= 0
 Cvp3_ind_helper0#34:        vp3_ind + vp3_inda  = 1
 Cvp3_ind_helper1#35:        - vp3_inda + vp3_indb >= 0
 Cvp3_ind_helper2#36:        - vp3_ind + vp3_indc  = 0
 Cvp3_ind_helper3#37:        - vp3_indc + vp3_indd >= 0
 Cvn3_ind_helper0#38:        vn3_ind + vn3_inda  = 1
 Cvn3_ind_helper1#39:        - vn3_inda + vn3_indb >= 0
 Cvn3_ind_helper2#40:        - vn3_ind + vn3_indc  = 0
 Cvn3_ind_helper3#41:        - vn3_indc + vn3_indd >= 0
 Cvp4_ind_helper0#42:        vp4_ind + vp4_inda  = 1
 Cvp4_ind_helper1#43:        - vp4_inda + vp4_indb >= 0
 Cvp4_ind_helper2#44:        - vp4_ind + vp4_indc  = 0
 Cvp4_ind_helper3#45:        - vp4_indc + vp4_indd >= 0
 Cvn4_ind_helper0#46:        vn4_ind + vn4_inda  = 1
 Cvn4_ind_helper1#47:        - vn4_inda + vn4_indb >= 0
 Cvn4_ind_helper2#48:        - vn4_ind + vn4_indc  = 0
 Cvn4_ind_helper3#49:        - vn4_indc + vn4_indd >= 0
 Cvp5_ind_helper0#50:        vp5_ind + vp5_inda  = 1
 Cvp5_ind_helper1#51:        - vp5_inda + vp5_indb >= 0
 Cvp5_ind_helper2#52:        - vp5_ind + vp5_indc  = 0
 Cvp5_ind_helper3#53:        - vp5_indc + vp5_indd >= 0
 Cvn5_ind_helper0#54:        vn5_ind + vn5_inda  = 1
 Cvn5_ind_helper1#55:        - vn5_inda + vn5_indb >= 0
 Cvn5_ind_helper2#56:        - vn5_ind + vn5_indc  = 0
 Cvn5_ind_helper3#57:        - vn5_indc + vn5_indd >= 0
 Cvp6_ind_helper0#58:        vp6_ind + vp6_inda  = 1
 Cvp6_ind_helper1#59:        - vp6_inda + vp6_indb >= 0
 Cvp6_ind_helper2#60:        - vp6_ind + vp6_indc  = 0
 Cvp6_ind_helper3#61:        - vp6_indc + vp6_indd >= 0
 Cvn6_ind_helper0#62:        vn6_ind + vn6_inda  = 1
 Cvn6_ind_helper1#63:        - vn6_inda + vn6_indb >= 0
 Cvn6_ind_helper2#64:        - vn6_ind + vn6_indc  = 0
 Cvn6_ind_helper3#65:        - vn6_indc + vn6_indd >= 0
 Cvp7_ind_helper0#66:        vp7_ind + vp7_inda  = 1
 Cvp7_ind_helper1#67:        - vp7_inda + vp7_indb >= 0
 Cvp7_ind_helper2#68:        - vp7_ind + vp7_indc  = 0
 Cvp7_ind_helper3#69:        - vp7_indc + vp7_indd >= 0
 Cvn7_ind_helper0#70:        vn7_ind + vn7_inda  = 1
 Cvn7_ind_helper1#71:        - vn7_inda + vn7_indb >= 0
 Cvn7_ind_helper2#72:        - vn7_ind + vn7_indc  = 0
 Cvn7_ind_helper3#73:        - vn7_indc + vn7_indd >= 0
 Cvp8_ind_helper0#74:        vp8_ind + vp8_inda  = 1
 Cvp8_ind_helper1#75:        - vp8_inda + vp8_indb >= 0
 Cvp8_ind_helper2#76:        - vp8_ind + vp8_indc  = 0
 Cvp8_ind_helper3#77:        - vp8_indc + vp8_indd >= 0
 Cvn8_ind_helper0#78:        vn8_ind + vn8_inda  = 1
 Cvn8_ind_helper1#79:        - vn8_inda + vn8_indb >= 0
 Cvn8_ind_helper2#80:        - vn8_ind + vn8_indc  = 0
 Cvn8_ind_helper3#81:        - vn8_indc + vn8_indd >= 0
 _E0#82:                     vp0_ind + vn0_ind <= 1
 _E1#83:                     vp1_ind + vn1_ind <= 1
 _E2#84:                     vp2_ind + vn2_ind <= 1
 _E3#85:                     vp3_ind + vn3_ind <= 1
 _E4#86:                     vp4_ind + vn4_ind <= 1
 _E5#87:                     vp5_ind + vn5_ind <= 1
 _E6#88:                     vp6_ind + vn6_ind <= 1
 _E7#89:                     vp7_ind + vn7_ind <= 1
 _E8#90:                     vp8_ind + vn8_ind <= 1
 KShortestSizeConstraint#91: vp0_ind + vn0_ind + vp1_ind + vn1_ind + vp2_ind
                             + vn2_ind + vp3_ind + vn3_ind + vp4_ind + vn4_ind
                             + vp5_ind + vn5_ind + vp6_ind + vn6_ind + vp7_ind
                             + vn7_ind + vp8_ind + vn8_ind >= 1
 cut0#92:                    vp0_ind + vn0_ind <= 0
 cut0#93:                    vp8_ind + vn8_ind <= 0
 cut0#94:                    vp4_ind + vn4_ind + vp7_ind + vn7_ind <= 1
 cut0#95:                    vp4_ind + vn4_ind + vp6_ind + vn6_ind <= 1
 cut0#96:                    vp4_ind + vn4_ind + vp5_ind + vn5_ind <= 1
 cut0#97:                    vp2_ind + vn2_ind + vp3_ind + vn3_ind + vp7_ind
                             + vn7_ind <= 2
 cut0#98:                    vp2_ind + vn2_ind + vp3_ind + vn3_ind + vp6_ind
                             + vn6_ind <= 2
 cut0#99:                    vp2_ind + vn2_ind + vp3_ind + vn3_ind + vp5_ind
                             + vn5_ind <= 2
 cut0#100:                   vp1_ind + vn1_ind + vp3_ind + vn3_ind + vp7_ind
                             + vn7_ind <= 2
 cut0#101:                   vp1_ind + vn1_ind + vp3_ind + vn3_ind + vp6_ind
                             + vn6_ind <= 2
 cut0#102:                   vp1_ind + vn1_ind + vp3_ind + vn3_ind + vp5_ind
                             + vn5_ind <= 2
 Cvp0_ind_ind1: vp0_ind = 0 -> vp0  = 1
 Cvp0_ind_ind2: vp0_ind = 1 -> vp0 - C >= 0
 Cvn0_ind_ind1: vn0_ind = 0 -> vn0  = 1
 Cvn0_ind_ind2: vn0_ind = 1 -> vn0 - C >= 0
 Cvp1_ind_ind1: vp1_ind = 0 -> vp1  = 1
 Cvp1_ind_ind2: vp1_ind = 1 -> vp1 - C >= 0
 Cvn1_ind_ind1: vn1_ind = 0 -> vn1  = 1
 Cvn1_ind_ind2: vn1_ind = 1 -> vn1 - C >= 0
 Cvp2_ind_ind1: vp2_ind = 0 -> vp2  = 1
 Cvp2_ind_ind2: vp2_ind = 1 -> vp2 - C >= 0
 Cvn2_ind_ind1: vn2_ind = 0 -> vn2  = 1
 Cvn2_ind_ind2: vn2_ind = 1 -> vn2 - C >= 0
 Cvp3_ind_ind1: vp3_ind = 0 -> vp3  = 1
 Cvp3_ind_ind2: vp3_ind = 1 -> vp3 - C >= 0
 Cvn3_ind_ind1: vn3_ind = 0 -> vn3  = 1
 Cvn3_ind_ind2: vn3_ind = 1 -> vn3 - C >= 0
 Cvp4_ind_ind1: vp4_ind = 0 -> vp4  = 1
 Cvp4_ind_ind2: vp4_ind = 1 -> vp4 - C >= 0
 Cvn4_ind_ind1: vn4_ind = 0 -> vn4  = 1
 Cvn4_ind_ind2: vn4_ind = 1 -> vn4 - C >= 0
 Cvp5_ind_ind1: vp5_ind = 0 -> vp5  = 1
 Cvp5_ind_ind2: vp5_ind = 1 -> vp5 - C >= 0
 Cvn5_ind_ind1: vn5_ind = 0 -> vn5  = 1
 Cvn5_ind_ind2: vn5_ind = 1 -> vn5 - C >= 0
 Cvp6_ind_ind1: vp6_ind = 0 -> vp6  = 1
 Cvp6_ind_ind2: vp6_ind = 1 -> vp6 - C >= 0
 Cvn6_ind_ind1: vn6_ind = 0 -> vn6  = 1
 Cvn6_ind_ind2: vn6_ind = 1 -> vn6 - C >= 0
 Cvp7_ind_ind1: vp7_ind = 0 -> vp7  = 1
 Cvp7_ind_ind2: vp7_ind = 1 -> vp7 - C >= 0
 Cvn7_ind_ind1: vn7_ind = 0 -> vn7  = 1
 Cvn7_ind_ind2: vn7_ind = 1 -> vn7 - C >= 0
 Cvp8_ind_ind1: vp8_ind = 0 -> vp8  = 1
 Cvp8_ind_ind2: vp8_ind = 1 -> vp8 - C >= 0
 Cvn8_ind_ind1: vn8_ind = 0 -> vn8  = 1
 Cvn8_ind_ind2: vn8_ind = 1 -> vn8 - C >= 0
Bounds
      u0 Free
      u1 Free
      u2 Free
      u3 Free
      u4 Free
      u5 Free
      C >= 1
 0 <= vp0_ind <= 1
 0 <= vn0_ind <= 1
 0 <= vp1_ind <= 1
 0 <= vn1_ind <= 1
 0 <= vp2_ind <= 1
 0 <= vn2_ind <= 1
 0 <= vp3_ind <= 1
 0 <= vn3_ind <= 1
 0 <= vp4_ind <= 1
 0 <= vn4_ind <= 1
 0 <= vp5_ind <= 1
 0 <= vn5_ind <= 1
 0 <= vp6_ind <= 1
 0 <= vn6_ind <= 1
 0 <= vp7_ind <= 1
 0 <= vn7_ind <= 1
 0 <= vp8_ind <= 1
 0 <= vn8_ind <= 1
 0 <= vp0_inda <= 1
 0 <= vp0_indb <= 1
 0 <= vp0_indc <= 1
 0 <= vp0_indd <= 1
 0 <= vn0_inda <= 1
 0 <= vn0_indb <= 1
 0 <= vn0_indc <= 1
 0 <= vn0_indd <= 1
 0 <= vp1_inda <= 1
 0 <= vp1_indb <= 1
 0 <= vp1_indc <= 1
 0 <= vp1_indd <= 1
 0 <= vn1_inda <= 1
 0 <= vn1_indb <= 1
 0 <= vn1_indc <= 1
 0 <= vn1_indd <= 1
 0 <= vp2_inda <= 1
 0 <= vp2_indb <= 1
 0 <= vp2_indc <= 1
 0 <= vp2_indd <= 1
 0 <= vn2_inda <= 1
 0 <= vn2_indb <= 1
 0 <= vn2_indc <= 1
 0 <= vn2_indd <= 1
 0 <= vp3_inda <= 1
 0 <= vp3_indb <= 1
 0 <= vp3_indc <= 1
 0 <= vp3_indd <= 1
 0 <= vn3_inda <= 1
 0 <= vn3_indb <= 1
 0 <= vn3_indc <= 1
 0 <= vn3_indd <= 1
 0 <= vp4_inda <= 1
 0 <= vp4_indb <= 1
 0 <= vp4_indc <= 1
 0 <= vp4_indd <= 1
 0 <= vn4_inda <= 1
 0 <= vn4_indb <= 1
 0 <= vn4_indc <= 1
 0 <= vn4_indd <= 1
 0 <= vp5_inda <= 1
 0 <= vp5_indb <= 1
 0 <= vp5_indc <= 1
 0 <= vp5_indd <= 1
 0 <= vn5_inda <= 1
 0 <= vn5_indb <= 1
 0 <= vn5_indc <= 1
 0 <= vn5_indd <= 1
 0 <= vp6_inda <= 1
 0 <= vp6_indb <= 1
 0 <= vp6_indc <= 1
 0 <= vp6_indd <= 1
 0 <= vn6_inda <= 1
 0 <= vn6_indb <= 1
 0 <= vn6_indc <= 1
 0 <= vn6_indd <= 1
 0 <= vp7_inda <= 1
 0 <= vp7_indb <= 1
 0 <= vp7_indc <= 1
 0 <= vp7_indd <= 1
 0 <= vn7_inda <= 1
 0 <= vn7_indb <= 1
 0 <= vn7_indc <= 1
 0 <= vn7_indd <= 1
 0 <= vp8_inda <= 1
 0 <= vp8_indb <= 1
 0 <= vp8_indc <= 1
 0 <= vp8_indd <= 1
 0 <= vn8_inda <= 1
 0 <= vn8_indb <= 1
 0 <= vn8_indc <= 1
 0 <= vn8_indd <= 1
Binaries
 vp0_ind  vn0_ind  vp1_ind  vn1_ind  vp2_ind  vn2_ind  vp3_ind  vn3_ind 
 vp4_ind  vn4_ind  vp5_ind  vn5_ind  vp6_ind  vn6_ind  vp7_ind  vn7_ind 
 vp8_ind  vn8_ind  vp0_inda  vp0_indb  vp0_indc  vp0_indd  vn0_inda  vn0_indb 
 vn0_indc  vn0_indd  vp1_inda  vp1_indb  vp1_indc  vp1_indd  vn1_inda 
 vn1_indb  vn1_indc  vn1_indd  vp2_inda  vp2_indb  vp2_indc  vp2_indd 
 vn2_inda  vn2_indb  vn2_indc  vn2_indd  vp3_inda  vp3_indb  vp3_indc 
 vp3_indd  vn3_inda  vn3_indb  vn3_indc  vn3_indd  vp4_inda  vp4_indb 
 vp4_indc  vp4_indd  vn4_inda  vn4_indb  vn4_indc  vn4_indd  vp5_inda 
 vp5_indb  vp5_indc  vp5_indd  vn5_inda  vn5_indb  vn5_indc  vn5_indd 
 vp6_inda  vp6_indb  vp6_indc  vp6_indd  vn6_inda  vn6_indb  vn6_indc 
 vn6_indd  vp7_inda  vp7_indb  vp7_indc  vp7_indd  vn7_inda  vn7_indb 
 vn7_indc  vn7_indd  vp8_inda  vp8_indb  vp8_indc  vp8_indd  vn8_inda 
 vn8_indb  vn8_indc  vn8_indd 
End
